#!/usr/bin/env python3
# link_products_to_keywords.py
"""
Product → KServe Embedding → Qdrant keyword search → Elasticsearch indexing

상품명을 임베딩하여 Qdrant의 키워드 컬렉션에서 유사 키워드 top-K를 조회한 뒤,
결과를 Elasticsearch에 nested 구조로 저장.

사전 조건:
  - Elasticsearch + Qdrant: docker compose up -d
  - KServe embedding server 실행 중
  - Qdrant keywords 컬렉션 인덱싱 완료

실행:
  # 테스트 (10건)
  python link_products_to_keywords.py --limit 10

  # 본 실행 (100k건)
  python link_products_to_keywords.py --limit 100000 --workers 4

  # Parquet 파일 읽기
  python link_products_to_keywords.py --parquet data/products.parquet --limit 10000

  # Parquet 디렉토리 읽기 (여러 파일)
  python link_products_to_keywords.py --parquet data/products_partitioned/ --limit 50000

  # Realtime 모드 (기존 인덱스 보존)
  python link_products_to_keywords.py --mode realtime --limit 1000

  # ES 9 클러스터 + fingerprint
  python link_products_to_keywords.py --limit 100 \\
      --es_nodes https://es01:9200 https://es02:9200 \\
      --es_fingerprint "B1:2A:96:..." \\
      --es_username elastic --es_password changeme
"""

import argparse
import asyncio
import logging
import time
from pathlib import Path

from elasticsearch.helpers import async_bulk

from es_indexer import Config as ESConfig, ESIndexer
from kserve_embed_client import EmbeddingClient, RURI_QUERY_PREFIX
from pipeline_commons import AsyncFailureLogger, ParquetReader, batch_iter, timer
from qdrant_indexer import AsyncQdrantIndexer

QUERY_PREFIX = RURI_QUERY_PREFIX

logger = logging.getLogger("link_products")

# ============================================================
# ES 스키마 — nested 키워드 매핑
# ============================================================
PRODUCT_KEYWORDS_SCHEMA = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "product_name": {
                "type": "text",
                "analyzer": "standard",
                "fields": {"raw": {"type": "keyword"}},
            },
            "keywords": {
                "type": "nested",
                "properties": {
                    "keyword": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {"raw": {"type": "keyword"}},
                    },
                    "keyword_id": {"type": "integer"},
                    "score": {"type": "float"},
                },
            },
        }
    },
}


# ============================================================
# 진행 통계
# ============================================================
class _Stats:
    LOG_INTERVAL = 1000

    def __init__(self, total: int):
        self.total = total
        self.processed = 0
        self.retries = 0
        self.failed_docs = 0
        self.failed_batches = 0
        self._start = time.perf_counter()
        self._interval_start = time.perf_counter()
        self._interval_count = 0

    def update(self, count: int, embed_ms: float, search_ms: float, es_ms: float):
        self.processed += count
        self._interval_count += count
        n = self.processed
        elapsed = time.perf_counter() - self._start
        avg_rps = n / elapsed if elapsed > 0 else 0

        should_log = n <= 100 or n % self.LOG_INTERVAL == 0 or n == self.total
        if should_log:
            now = time.perf_counter()
            interval_sec = now - self._interval_start
            interval_rps = self._interval_count / interval_sec if interval_sec > 0 else 0
            self._interval_start = now
            self._interval_count = 0
            pct = n / self.total * 100
            msg = (
                f"[{pct:5.1f}%] {n:>7,}/{self.total:,}  "
                f"embed={embed_ms:.0f}ms  search={search_ms:.0f}ms  es={es_ms:.0f}ms  "
                f"avg={avg_rps:,.0f} p/s  interval={interval_rps:,.0f} p/s"
            )
            print(f"         {msg}")
            logger.info(msg)

    def record_retry(self):
        self.retries += 1

    def record_failure(self, doc_count: int):
        self.failed_docs += doc_count
        self.failed_batches += 1

    @property
    def wall_sec(self) -> float:
        return time.perf_counter() - self._start


# ============================================================
# 배치 처리 코루틴 — 재시도 + Dead Letter
# ============================================================
async def _process_batch(
    batch_idx: int,
    products: list[str],
    embedder: EmbeddingClient,
    qdrant: AsyncQdrantIndexer,
    es_indexer: ESIndexer,
    top_k: int,
    semaphore: asyncio.Semaphore,
    stats: _Stats,
    max_retries: int,
    retry_delay: float,
    failure_logger: AsyncFailureLogger,
    payload_key: str = "keyword",
):
    """
    1. embed(products) via executor → 2. qdrant.search_batch() async → 3. async_bulk to ES
    실패 시 지수 백오프 재시도, 최종 실패 시 실패 로깅.
    """
    async with semaphore:
        loop = asyncio.get_running_loop()
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 2):  # 1 = 최초, 2~N+1 = 재시도
            try:
                # Step 1: 임베딩 (sync → default executor)
                prefixed = [QUERY_PREFIX + p for p in products]
                with timer() as t_embed:
                    embeddings = await loop.run_in_executor(None, embedder.embed, prefixed)

                # Step 2: Qdrant 비동기 배치 검색
                vectors = [emb.tolist() for emb in embeddings]
                with timer() as t_search:
                    batch_results = await qdrant.search_batch(vectors, top_k)
                all_matches = [resp.points for resp in batch_results]

                # Step 3: ES bulk 인덱싱 (async)
                actions = []
                for i, product in enumerate(products):
                    doc = {
                        "product_name": product,
                        "keywords": [
                            {
                                "keyword": pt.payload.get(payload_key, ""),
                                "keyword_id": pt.id,
                                "score": round(pt.score, 6),
                            }
                            for pt in all_matches[i]
                        ],
                    }
                    actions.append({
                        "_index": es_indexer.index_name,
                        "_id": product,
                        "_source": doc,
                    })

                with timer() as t_es:
                    _success, errors = await async_bulk(
                        es_indexer.es, actions, chunk_size=len(actions), raise_on_error=False
                    )

                if errors:
                    logger.warning(f"Batch {batch_idx}: {len(errors)} ES bulk errors")

                stats.update(len(products), t_embed.ms, t_search.ms, t_es.ms)
                return  # 성공

            except Exception as e:
                last_error = e
                is_last = attempt > max_retries
                if is_last:
                    break
                delay = retry_delay * (2 ** (attempt - 1))
                stats.record_retry()
                logger.warning(
                    f"Batch {batch_idx} 실패 (시도 {attempt}/{max_retries + 1}), "
                    f"{delay:.1f}초 후 재시도: {e}"
                )
                await asyncio.sleep(delay)

        # 모든 재시도 소진 → 실패 로깅
        logger.error(
            f"Batch {batch_idx} 최종 실패 ({len(products)}건, "
            f"{max_retries + 1}회 시도): {last_error}"
        )
        stats.record_failure(len(products))
        await failure_logger.log_failure(
            batch_idx, last_error,
            data_info={"count": len(products), "products": products},
        )


# ============================================================
# 메인 파이프라인
# ============================================================
async def _run(args):
    # ── 로그 설정 ──
    data_path = args.parquet if args.parquet else args.products
    log_dir = data_path.parent.parent / "logs" if data_path.parent.name == "data" else Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"link_products_{ts}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    print(f"  Log: {log_file}")

    # [1/5] 상품 로드
    print("\n  [1/5] 상품 로드")

    if args.parquet:
        # Parquet 파일 또는 디렉토리 읽기
        reader = ParquetReader(
            args.parquet,
            chunk_size=args.parquet_chunk_size,
            text_column=args.parquet_column,
            limit=args.limit,
        )
        products = []
        for _, chunk_products in reader.iter_chunks():
            products.extend(chunk_products)
        total = len(products)
        source = f"Parquet: {args.parquet.name}"
        if args.parquet.is_dir():
            source += f" ({len(reader.parquet_files)} files)"
        print(f"         {source} → {total:,}건")
    else:
        # 텍스트 파일 읽기 (기존 방식)
        products = args.products.read_text(encoding="utf-8").strip().split("\n")
        if args.limit > 0:
            products = products[: args.limit]
        total = len(products)
        print(f"         텍스트 파일: {args.products.name} → {total:,}건")

    logger.info(f"products={total}, batch={args.batch_size}, workers={args.workers}, top_k={args.top_k}")

    # [2/5] 클라이언트 초기화
    print("\n  [2/5] 클라이언트 초기화")
    embedder = EmbeddingClient(args.kserve_url, args.model)
    qdrant = AsyncQdrantIndexer(args.qdrant_url, args.collection, 768)

    es_config = ESConfig(
        es_url=args.es_url,
        es_nodes=args.es_nodes,
        es_fingerprint=args.es_fingerprint,
        es_username=args.es_username,
        es_password=args.es_password,
        es_api_key=args.es_api_key,
        index_name=args.index,
    )
    es = ESIndexer.from_config(es_config)

    qdrant_count = await qdrant.get_count()
    es_display = args.es_nodes[0] if args.es_nodes else args.es_url
    node_info = f" (+{len(args.es_nodes)-1} nodes)" if args.es_nodes and len(args.es_nodes) > 1 else ""
    print(f"         KServe: {args.kserve_url}/v2/models/{args.model}")
    print(f"         Qdrant: {args.collection} ({qdrant_count:,} vectors)")
    print(f"         ES:     {es_display}{node_info}/{args.index}")

    # [3/5] ES 인덱스 생성
    print(f"\n  [3/5] ES 인덱스: {args.index} (mode={args.mode})")
    if args.mode == "bulk":
        await es.create_index(PRODUCT_KEYWORDS_SCHEMA)
        print("         생성 완료 (refresh=-1, translog=async)")
    else:
        created = await es.ensure_index(PRODUCT_KEYWORDS_SCHEMA)
        if created:
            print("         신규 생성")
        else:
            existing = await es.count()
            print(f"         기존 유지 ({existing:,}건)")

    # [4/5] 배치 처리
    print(
        f"\n  [4/5] 파이프라인 실행 "
        f"(workers={args.workers}, batch={args.batch_size}, top_k={args.top_k}, "
        f"retries={args.max_retries})"
    )
    stats = _Stats(total)
    semaphore = asyncio.Semaphore(args.workers)

    # 실패 로그 경로 결정
    if args.dead_letter_path:
        fl_path = args.dead_letter_path
    else:
        fl_path = log_file.with_suffix(".failures.jsonl")
    failure_logger = AsyncFailureLogger(fl_path, enabled=True)

    batches = list(batch_iter(products, args.batch_size))
    await asyncio.gather(*[
        _process_batch(
            bid, batch, embedder, qdrant, es,
            args.top_k, semaphore, stats,
            args.max_retries, args.retry_delay, failure_logger,
            payload_key=args.payload_key,
        )
        for bid, batch in batches
    ])
    wall = stats.wall_sec

    # [5/5] 최적화 + 검증
    print("\n  [5/5] 최적화 + 검증")
    if args.mode == "bulk":
        await es.finalize()
    else:
        await es.refresh()

    count = await es.count()
    summary_lines = [
        f"도큐먼트 수: {count:,}",
        f"Wall time: {wall:.1f}초",
        f"처리량: {total / wall:,.0f} products/sec",
    ]
    if stats.retries > 0:
        summary_lines.append(f"재시도 횟수: {stats.retries}회")
    if stats.failed_docs > 0:
        summary_lines.append(f"실패 문서: {stats.failed_docs:,}건 ({stats.failed_batches} 배치)")
        summary_lines.append(f"실패 로그: {failure_logger.log_path}")
    for line in summary_lines:
        print(f"         {line}")
        logger.info(line)

    # 샘플 출력
    sample = await es.es.search(
        index=args.index,
        query={"match_all": {}},
        size=1,
    )
    hits = sample["hits"]["hits"]
    if hits:
        doc = hits[0]["_source"]
        print(f"\n  --- 샘플 결과 ---")
        print(f"  상품: {doc['product_name']}")
        for kw in doc["keywords"][:5]:
            print(f"    id={kw['keyword_id']:>6}  score={kw['score']:.4f}  {kw['keyword']}")
        if len(doc["keywords"]) > 5:
            print(f"    ... 외 {len(doc['keywords']) - 5}건")

    await qdrant.close()
    await es.close()
    print("\n  완료!")
    logger.info("완료")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Product → Embedding → Qdrant top-K keywords → Elasticsearch nested indexing"
    )
    # ── 데이터 소스 ──
    parser.add_argument("--products", type=Path, default=Path("data/products_400k.txt"))
    parser.add_argument("--parquet", type=Path, default=None, help="Parquet 파일 또는 디렉토리 경로 (설정 시 --products 무시)")
    parser.add_argument("--parquet_chunk_size", type=int, default=10000, help="Parquet 청크 크기 (행 수)")
    parser.add_argument("--parquet_column", default="product_name", help="상품명이 들어있는 컬럼명")

    parser.add_argument("--limit", type=int, default=100_000, help="0=전체 (default: 100000)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=30)

    parser.add_argument("--kserve_url", default="http://localhost:8080")
    parser.add_argument("--qdrant_url", default="http://localhost:6333")
    parser.add_argument("--es_url", default="http://localhost:9200")

    parser.add_argument("--model", default="ruri_v3")
    parser.add_argument("--collection", default="keywords")
    parser.add_argument("--index", default="product_keywords")
    parser.add_argument("--payload_key", default="keyword",
                        help="Qdrant payload 필드 이름 (default: keyword)")

    parser.add_argument(
        "--mode", choices=["bulk", "realtime"], default="bulk",
        help="bulk=인덱스 재생성, realtime=기존 보존",
    )

    # ── 재시도 / 실패 처리 ──
    retry = parser.add_argument_group("재시도 / 실패 처리")
    retry.add_argument(
        "--max_retries", type=int, default=3,
        help="배치 실패 시 최대 재시도 횟수 (0=재시도 없음, default: 3)",
    )
    retry.add_argument(
        "--retry_delay", type=float, default=1.0,
        help="첫 재시도 대기 시간 (초, 이후 ×2 지수 백오프, default: 1.0)",
    )
    retry.add_argument(
        "--dead_letter_path", type=Path, default=None,
        help="실패 문서 JSONL 파일 경로 (미지정 시 logs/ 에 자동 생성)",
    )

    # ── ES 클러스터 연결 ──
    cluster = parser.add_argument_group("ES 클러스터 연결 (ES 9+)")
    cluster.add_argument(
        "--es_nodes", nargs="+", default=None,
        help="클러스터 노드 URL 목록 (설정 시 --es_url 무시)",
    )
    cluster.add_argument(
        "--es_fingerprint", default=None,
        help="TLS 인증서 SHA-256 fingerprint (--es_nodes 사용 시 필수)",
    )
    cluster.add_argument("--es_username", default=None, help="Basic Auth 사용자명")
    cluster.add_argument("--es_password", default=None, help="Basic Auth 비밀번호")
    cluster.add_argument(
        "--es_api_key", default=None,
        help="API Key (--es_username/--es_password 대신 사용)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Product → Keyword Linking Pipeline")
    print("=" * 60)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
