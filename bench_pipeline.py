#!/usr/bin/env python3
"""
Product-Keyword Linking Pipeline — 병목 분석 + 최적화 벤치마크

각 단계(embed / qdrant search / ES bulk)를 개별 측정하고,
최적화 전후 처리량을 비교.

최적화:
  1. Qdrant batch query — 64건 순차 → 1회 배치 HTTP 호출
  2. AsyncQdrantIndexer — sync → async (이벤트 루프 직접)
  3. 배치 크기 / 워커 수 최적 탐색

사용:
  python bench_pipeline.py --limit 100
  python bench_pipeline.py --limit 500 --workers 8
"""

import argparse
import asyncio
import time
from pathlib import Path

from elasticsearch.helpers import async_bulk

from es_indexer import ESIndexer
from kserve_embed_client import EmbeddingClient, RURI_QUERY_PREFIX
from qdrant_indexer import AsyncQdrantIndexer, QdrantIndexer

QUERY_PREFIX = RURI_QUERY_PREFIX

PRODUCT_KEYWORDS_SCHEMA = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "product_name": {
                "type": "text",
                "fields": {"raw": {"type": "keyword"}},
            },
            "keywords": {
                "type": "nested",
                "properties": {
                    "keyword": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
                    "keyword_id": {"type": "integer"},
                    "score": {"type": "float"},
                },
            },
        }
    },
}


# ============================================================
# Phase 1: 각 단계 독립 벤치마크
# ============================================================
async def bench_stages(products: list[str], args):
    """각 단계를 독립적으로 측정"""
    print("\n" + "=" * 60)
    print("  Phase 1: 단계별 독립 벤치마크")
    print("=" * 60)

    n = len(products)
    prefixed = [QUERY_PREFIX + p for p in products]

    # ── Stage A: KServe Embedding ──
    embedder = EmbeddingClient(args.kserve_url, args.model)
    t0 = time.perf_counter()
    embeddings = embedder.embed(prefixed)
    embed_ms = (time.perf_counter() - t0) * 1000
    embed_rps = n / (embed_ms / 1000)
    print(f"\n  [A] KServe Embedding ({n}건)")
    print(f"      {embed_ms:.0f}ms → {embed_rps:,.0f} texts/sec")
    print(f"      shape: {embeddings.shape}")

    # ── Stage B-1: Qdrant 순차 검색 (기존 방식) ──
    qdrant = QdrantIndexer(args.qdrant_url, args.collection, 768)
    t0 = time.perf_counter()
    results_seq = []
    for emb in embeddings:
        resp = qdrant.search(emb.tolist(), args.top_k)
        results_seq.append(resp.points)
    search_seq_ms = (time.perf_counter() - t0) * 1000
    search_seq_rps = n / (search_seq_ms / 1000)
    print(f"\n  [B-1] Qdrant 순차 검색 ({n}건 × top-{args.top_k})")
    print(f"        {search_seq_ms:.0f}ms → {search_seq_rps:,.0f} queries/sec")

    # ── Stage B-2: Qdrant Sync 배치 검색 (최적화) ──
    t0 = time.perf_counter()
    vectors = [emb.tolist() for emb in embeddings]
    results_batch_resp = qdrant.search_batch(vectors, args.top_k)
    search_batch_ms = (time.perf_counter() - t0) * 1000
    search_batch_rps = n / (search_batch_ms / 1000)
    speedup = search_seq_ms / search_batch_ms if search_batch_ms > 0 else 0
    print(f"\n  [B-2] Qdrant Sync 배치 검색 ({n}건 × top-{args.top_k}, 1회 HTTP)")
    print(f"        {search_batch_ms:.0f}ms → {search_batch_rps:,.0f} queries/sec")
    print(f"        순차 대비 {speedup:.1f}x 빠름")

    # ── Stage B-3: AsyncQdrantIndexer 배치 검색 ──
    async_indexer = AsyncQdrantIndexer(args.qdrant_url, args.collection, 768)
    t0 = time.perf_counter()
    results_async_resp = await async_indexer.search_batch(vectors, args.top_k)
    search_async_ms = (time.perf_counter() - t0) * 1000
    search_async_rps = n / (search_async_ms / 1000)
    print(f"\n  [B-3] AsyncQdrantIndexer 배치 검색 ({n}건 × top-{args.top_k})")
    print(f"        {search_async_ms:.0f}ms → {search_async_rps:,.0f} queries/sec")
    await async_indexer.close()

    # ── Stage C: ES Bulk Indexing ──
    es = ESIndexer(args.es_url, "bench_pipeline_test")
    await es.create_index(PRODUCT_KEYWORDS_SCHEMA)

    # results_batch_resp에서 결과 추출
    actions = []
    for i, product in enumerate(products):
        points = results_batch_resp[i].points
        doc = {
            "product_name": product,
            "keywords": [
                {"keyword": pt.payload["keyword"], "keyword_id": pt.id,
                 "score": round(pt.score, 6)}
                for pt in points
            ],
        }
        actions.append({"_index": es.index_name, "_id": product, "_source": doc})

    t0 = time.perf_counter()
    await async_bulk(es.es, actions, chunk_size=len(actions), raise_on_error=False)
    es_ms = (time.perf_counter() - t0) * 1000
    es_rps = n / (es_ms / 1000)
    print(f"\n  [C] ES Bulk Indexing ({n}건, nested ×{args.top_k})")
    print(f"      {es_ms:.0f}ms → {es_rps:,.0f} docs/sec")

    await es.finalize()
    await es.close()

    # ── 요약 ──
    print("\n  ─── 단계별 요약 ───")
    print(f"  {'단계':<30} {'시간':>8}  {'처리량':>12}")
    print(f"  {'─'*30} {'─'*8}  {'─'*12}")
    for label, ms, rps in [
        ("KServe Embedding", embed_ms, embed_rps),
        ("Qdrant 순차 검색", search_seq_ms, search_seq_rps),
        ("Qdrant Sync 배치 (최적화)", search_batch_ms, search_batch_rps),
        ("Qdrant Async 배치 (최적화)", search_async_ms, search_async_rps),
        ("ES Bulk Indexing", es_ms, es_rps),
    ]:
        print(f"  {label:<30} {ms:>7.0f}ms  {rps:>10,.0f}/s")

    # 이론적 최대 (embed + batch search + ES)
    optimized_total_ms = embed_ms + search_batch_ms + es_ms
    optimized_rps = n / (optimized_total_ms / 1000)
    original_total_ms = embed_ms + search_seq_ms + es_ms
    original_rps = n / (original_total_ms / 1000)
    print(f"\n  이론적 단일 배치 (순차):   {original_total_ms:.0f}ms → {original_rps:,.0f}/s")
    print(f"  이론적 단일 배치 (최적화): {optimized_total_ms:.0f}ms → {optimized_rps:,.0f}/s")

    return embeddings


# ============================================================
# Phase 2: 최적화된 파이프라인 E2E 벤치마크
# ============================================================
async def bench_optimized_pipeline(products: list[str], args):
    """최적화 적용 후 E2E 처리량 측정 — AsyncQdrantIndexer 사용"""
    print("\n" + "=" * 60)
    print("  Phase 2: 최적화 파이프라인 E2E")
    print("=" * 60)

    n = len(products)
    embedder = EmbeddingClient(args.kserve_url, args.model)
    async_indexer = AsyncQdrantIndexer(args.qdrant_url, args.collection, 768)
    es = ESIndexer(args.es_url, "bench_pipeline_opt")
    await es.create_index(PRODUCT_KEYWORDS_SCHEMA)

    semaphore = asyncio.Semaphore(args.workers)

    processed = 0
    total_embed_ms = 0
    total_search_ms = 0
    total_es_ms = 0

    async def process_batch_optimized(batch_products: list[str]):
        nonlocal processed, total_embed_ms, total_search_ms, total_es_ms
        async with semaphore:
            loop = asyncio.get_running_loop()

            # Step 1: Embed (sync → default executor)
            prefixed = [QUERY_PREFIX + p for p in batch_products]
            t0 = time.perf_counter()
            embeddings = await loop.run_in_executor(None, embedder.embed, prefixed)
            embed_ms = (time.perf_counter() - t0) * 1000

            # Step 2: Qdrant async 배치 검색
            vectors = [emb.tolist() for emb in embeddings]
            t1 = time.perf_counter()
            results = await async_indexer.search_batch(vectors, args.top_k)
            search_ms = (time.perf_counter() - t1) * 1000

            # Step 3: ES bulk (async)
            actions = []
            for i, product in enumerate(batch_products):
                points = results[i].points
                doc = {
                    "product_name": product,
                    "keywords": [
                        {"keyword": pt.payload["keyword"], "keyword_id": pt.id,
                         "score": round(pt.score, 6)}
                        for pt in points
                    ],
                }
                actions.append({
                    "_index": es.index_name, "_id": product, "_source": doc,
                })

            t2 = time.perf_counter()
            await async_bulk(
                es.es, actions, chunk_size=len(actions), raise_on_error=False
            )
            es_ms = (time.perf_counter() - t2) * 1000

            processed += len(batch_products)
            total_embed_ms += embed_ms
            total_search_ms += search_ms
            total_es_ms += es_ms

    # 배치 분할 + 실행
    batch_size = args.batch_size
    batches = [products[i:i + batch_size] for i in range(0, n, batch_size)]

    print(f"\n  설정: {n}건, batch={batch_size}, workers={args.workers}, top_k={args.top_k}")

    t_start = time.perf_counter()
    await asyncio.gather(*[process_batch_optimized(b) for b in batches])
    wall_sec = time.perf_counter() - t_start
    wall_rps = n / wall_sec

    await es.finalize()
    count = await es.count()
    await es.close()
    await async_indexer.close()

    print(f"\n  결과:")
    print(f"    Wall time:   {wall_sec:.2f}초")
    print(f"    처리량:      {wall_rps:,.0f} products/sec")
    print(f"    문서 수:     {count:,}")
    print(f"    Embed 합계:  {total_embed_ms / 1000:.1f}초 (코루틴 time)")
    print(f"    Search 합계: {total_search_ms / 1000:.1f}초 (코루틴 time)")
    print(f"    ES 합계:     {total_es_ms / 1000:.1f}초 (코루틴 time)")

    return wall_rps


# ============================================================
# Phase 3: 기존 파이프라인 E2E (비교용)
# ============================================================
async def bench_original_pipeline(products: list[str], args):
    """기존(비최적화) 파이프라인 E2E — 순차 검색 + sync QdrantIndexer"""
    print("\n" + "=" * 60)
    print("  Phase 3: 기존 파이프라인 E2E (비교 기준)")
    print("=" * 60)

    n = len(products)
    embedder = EmbeddingClient(args.kserve_url, args.model)
    qdrant = QdrantIndexer(args.qdrant_url, args.collection, 768)
    es = ESIndexer(args.es_url, "bench_pipeline_orig")
    await es.create_index(PRODUCT_KEYWORDS_SCHEMA)

    semaphore = asyncio.Semaphore(args.workers)
    processed = 0

    async def process_batch_original(batch_products: list[str]):
        nonlocal processed
        async with semaphore:
            loop = asyncio.get_running_loop()

            def embed_and_search():
                prefixed = [QUERY_PREFIX + p for p in batch_products]
                embeddings = embedder.embed(prefixed)
                matches = []
                for emb in embeddings:
                    resp = qdrant.search(emb.tolist(), args.top_k)
                    matches.append(resp.points)
                return matches

            all_matches = await loop.run_in_executor(None, embed_and_search)

            actions = []
            for i, product in enumerate(batch_products):
                doc = {
                    "product_name": product,
                    "keywords": [
                        {"keyword": pt.payload["keyword"], "keyword_id": pt.id,
                         "score": round(pt.score, 6)}
                        for pt in all_matches[i]
                    ],
                }
                actions.append({
                    "_index": es.index_name, "_id": product, "_source": doc,
                })

            await async_bulk(
                es.es, actions, chunk_size=len(actions), raise_on_error=False
            )
            processed += len(batch_products)

    batch_size = args.batch_size
    batches = [products[i:i + batch_size] for i in range(0, n, batch_size)]

    print(f"\n  설정: {n}건, batch={batch_size}, workers={args.workers}, top_k={args.top_k}")

    t_start = time.perf_counter()
    await asyncio.gather(*[process_batch_original(b) for b in batches])
    wall_sec = time.perf_counter() - t_start
    wall_rps = n / wall_sec

    await es.finalize()
    count = await es.count()
    await es.close()

    print(f"\n  결과:")
    print(f"    Wall time: {wall_sec:.2f}초")
    print(f"    처리량:    {wall_rps:,.0f} products/sec")
    print(f"    문서 수:   {count:,}")

    return wall_rps


# ============================================================
# Phase 4: 워커 수별 스케일링 테스트
# ============================================================
async def bench_scaling(products: list[str], args):
    """workers 수별 처리량 비교 — AsyncQdrantIndexer 사용"""
    print("\n" + "=" * 60)
    print("  Phase 4: 워커 수별 스케일링 (최적화 파이프라인)")
    print("=" * 60)

    n = len(products)
    embedder = EmbeddingClient(args.kserve_url, args.model)
    results = []

    for workers in [1, 2, 4, 8]:
        async_indexer = AsyncQdrantIndexer(args.qdrant_url, args.collection, 768)
        es = ESIndexer(args.es_url, f"bench_scale_{workers}w")
        await es.create_index(PRODUCT_KEYWORDS_SCHEMA)

        semaphore = asyncio.Semaphore(workers)

        async def process(batch_products, _sem=semaphore,
                          _ai=async_indexer, _es=es):
            async with _sem:
                loop = asyncio.get_running_loop()
                prefixed = [QUERY_PREFIX + p for p in batch_products]
                embeddings = await loop.run_in_executor(None, embedder.embed, prefixed)
                vectors = [e.tolist() for e in embeddings]
                qdrant_results = await _ai.search_batch(vectors, args.top_k)
                actions = []
                for i, product in enumerate(batch_products):
                    pts = qdrant_results[i].points
                    actions.append({
                        "_index": _es.index_name, "_id": product,
                        "_source": {
                            "product_name": product,
                            "keywords": [
                                {"keyword": pt.payload["keyword"],
                                 "keyword_id": pt.id, "score": round(pt.score, 6)}
                                for pt in pts
                            ],
                        },
                    })
                await async_bulk(_es.es, actions, chunk_size=len(actions),
                                 raise_on_error=False)

        batches = [products[i:i + args.batch_size] for i in range(0, n, args.batch_size)]
        t0 = time.perf_counter()
        await asyncio.gather(*[process(b) for b in batches])
        wall = time.perf_counter() - t0
        rps = n / wall

        await es.finalize()
        await es.close()
        await async_indexer.close()

        results.append((workers, wall, rps))
        print(f"  workers={workers:>2}  {wall:.2f}s  {rps:>6,.0f} products/sec")

    print(f"\n  {'workers':>8}  {'wall':>8}  {'throughput':>14}  {'scaling':>8}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*14}  {'─'*8}")
    base_rps = results[0][2]
    for w, wall, rps in results:
        print(f"  {w:>8}  {wall:>7.2f}s  {rps:>12,.0f}/s  {rps/base_rps:>7.2f}x")


# ============================================================
# Main
# ============================================================
async def run(args):
    products = Path(args.products).read_text("utf-8").strip().split("\n")
    if args.limit > 0:
        products = products[:args.limit]
    print(f"  대상: {len(products):,}건 (products_400k.txt)")

    # Phase 1: 단계별 벤치마크
    await bench_stages(products, args)

    # Phase 2: 최적화 E2E
    opt_rps = await bench_optimized_pipeline(products, args)

    # Phase 3: 기존 E2E
    orig_rps = await bench_original_pipeline(products, args)

    # Phase 4: 스케일링
    await bench_scaling(products, args)

    # ── 최종 요약 ──
    print("\n" + "=" * 60)
    print("  최종 요약")
    print("=" * 60)
    speedup = opt_rps / orig_rps if orig_rps > 0 else 0
    print(f"  기존 파이프라인:     {orig_rps:>8,.0f} products/sec")
    print(f"  최적화 파이프라인:   {opt_rps:>8,.0f} products/sec")
    print(f"  향상:               {speedup:.2f}x")
    print()

    # 10,000/sec 가능성 분석
    print("  ─── 10,000 products/sec 달성 분석 ───")
    if opt_rps >= 10000:
        print(f"  ✓ 현재 설정으로 달성 가능! ({opt_rps:,.0f}/s)")
    else:
        needed_factor = 10000 / opt_rps
        print(f"  현재 최적화 처리량: {opt_rps:,.0f}/s")
        print(f"  목표 대비:          {needed_factor:.1f}x 부족")
        print(f"  병목:               KServe 임베딩 (GPU 추론)")
        print(f"  달성 방법:")
        print(f"    - GPU 서버 {needed_factor:.0f}대 병렬 (수평 확장)")
        print(f"    - H200/A100 GPU로 교체 (10-30x 추론 성능)")
        print(f"    - 임베딩 캐싱 (동일 상품 재처리 회피)")
        print(f"    - 모델 양자화 (INT8/FP16 → 2-3x 가속)")


def main():
    parser = argparse.ArgumentParser(description="Pipeline throughput benchmark")
    parser.add_argument("--products", default="data/products_400k.txt")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--kserve_url", default="http://localhost:8080")
    parser.add_argument("--qdrant_url", default="http://localhost:6333")
    parser.add_argument("--es_url", default="http://localhost:9200")
    parser.add_argument("--model", default="ruri_v3")
    parser.add_argument("--collection", default="keywords")
    args = parser.parse_args()

    print("=" * 60)
    print("  Pipeline Throughput Benchmark")
    print("=" * 60)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
