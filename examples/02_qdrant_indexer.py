#!/usr/bin/env python3
"""
qdrant-indexer 패키지 예시 — Mock으로 전체 파이프라인 코드 경로 검증

테스트 항목:
  1. Config 기본값 + 커스텀 생성 (BaseConfig 상속)
  2. QdrantIndexer — create_collection, upsert_batch, search, search_batch
  3. Searcher — search, search_batch (embed + Qdrant 연동)
  4. PipelineStats 통계 로직 (공용 유틸리티)
  5. _process_batch 비동기 배치 처리 (embed via executor + upsert async + 재시도)
"""

import asyncio
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np

from qdrant_indexer import Config, QdrantIndexer, AsyncQdrantIndexer, Searcher, SearchResult
from qdrant_indexer.pipeline import _process_batch
from kserve_embed_client import (
    AsyncFailureLogger,
    EmbeddingClient,
    PipelineStats,
    RetryConfig,
    FailureLogger,
)


def test_config():
    """Config: 기본값 + 커스텀 설정"""
    print("=" * 60)
    print("[1] Config — 기본값 + 커스텀")
    print("=" * 60)

    # 기본값
    c = Config()
    assert c.kserve_url == "http://localhost:8080"
    assert c.model_name == "ruri_v3"
    assert c.qdrant_url == "http://localhost:6333"
    assert c.collection_name == "keywords"
    assert c.vector_dim == 768
    assert c.batch_size == 64
    assert c.workers == 4
    assert c.max_retries == 3
    assert c.retry_exponential is True
    assert c.log_failures is True
    assert c.bulk_mode is False
    assert c.bulk_indexing_threshold == 20000
    assert c.payload_key == "keyword"
    print(f"  기본값: kserve={c.kserve_url}, qdrant={c.qdrant_url}, dim={c.vector_dim}")
    print(f"  처리: batch={c.batch_size}, workers={c.workers}")
    print(f"  재시도: max={c.max_retries}, exponential={c.retry_exponential}")
    print(f"  벌크: bulk_mode={c.bulk_mode}, indexing_threshold={c.bulk_indexing_threshold}")

    # 커스텀
    c2 = Config(
        kserve_url="http://gpu-server:8080",
        qdrant_url="http://qdrant-cluster:6333",
        collection_name="products",
        vector_dim=1024,
        batch_size=128,
        workers=16,
        limit=10000,
        parquet_path=Path("/data/products.parquet"),
    )
    assert c2.vector_dim == 1024
    assert c2.collection_name == "products"
    assert c2.parquet_path == Path("/data/products.parquet")
    print(f"  커스텀: dim={c2.vector_dim}, collection={c2.collection_name}")
    print(f"  PASS\n")


def test_qdrant_indexer():
    """QdrantIndexer: Mock Qdrant Client로 CRUD 검증"""
    print("=" * 60)
    print("[2] QdrantIndexer — create, upsert, search, search_batch")
    print("=" * 60)

    with patch("qdrant_indexer.indexer.QdrantClient") as MockClient:
        mock_client = MockClient.return_value

        # create_collection
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        indexer = QdrantIndexer("http://localhost:6333", "test_col", 768)
        indexer.create_collection()
        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == "test_col"
        print(f"  create_collection: OK")

        # 기존 컬렉션 삭제 후 재생성
        mock_existing = MagicMock()
        mock_existing.name = "test_col"
        mock_collections.collections = [mock_existing]
        mock_client.get_collections.return_value = mock_collections

        indexer.create_collection()
        mock_client.delete_collection.assert_called_with("test_col")
        print(f"  기존 컬렉션 삭제 + 재생성: OK")

        # ensure_collection (이미 존재 → False)
        mock_client.reset_mock()
        mock_collections.collections = [mock_existing]
        result = indexer.ensure_collection()
        assert result is False
        mock_client.create_collection.assert_not_called()
        print(f"  ensure_collection (존재): False, create 미호출  OK")

        # ensure_collection (없음 → True + 신규 생성)
        mock_client.reset_mock()
        mock_collections.collections = []
        result = indexer.ensure_collection()
        assert result is True
        mock_client.create_collection.assert_called_once()
        print(f"  ensure_collection (신규): True, create 호출  OK")

        # upsert_batch
        keywords = ["東京", "大阪", "京都"]
        embeddings = np.random.randn(3, 768)
        indexer.upsert_batch(start_id=100, keywords=keywords, embeddings=embeddings)
        upsert_call = mock_client.upsert.call_args
        points = upsert_call.kwargs["points"]
        assert len(points) == 3
        assert points[0].id == 100
        assert points[0].payload == {"keyword": "東京"}
        assert points[2].id == 102
        assert len(points[0].vector) == 768
        print(f"  upsert_batch(3건): ids=[100,101,102], payloads OK")

        # search (단건)
        mock_search_result = MagicMock()
        mock_point = MagicMock()
        mock_point.score = 0.95
        mock_point.payload = {"keyword": "東京 ラーメン"}
        mock_point.id = 42
        mock_search_result.points = [mock_point]
        mock_client.query_points.return_value = mock_search_result

        result = indexer.search([0.1] * 768, top_k=5)
        assert result.points[0].score == 0.95
        assert result.points[0].payload["keyword"] == "東京 ラーメン"
        search_call = mock_client.query_points.call_args
        assert search_call.kwargs["limit"] == 5
        print(f"  search: score={result.points[0].score}, keyword='{result.points[0].payload['keyword']}'  OK")

        # search_batch
        mock_batch_result = [mock_search_result, mock_search_result]
        mock_client.query_batch_points.return_value = mock_batch_result

        batch_result = indexer.search_batch([[0.1] * 768, [0.2] * 768], top_k=3)
        assert len(batch_result) == 2
        batch_call = mock_client.query_batch_points.call_args
        assert len(batch_call.kwargs["requests"]) == 2
        print(f"  search_batch(2 queries): {len(batch_result)}건 결과  OK")

        # count
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 40000
        mock_client.get_collection.return_value = mock_collection_info
        assert indexer.count == 40000
        print(f"  count: {indexer.count}  OK")

        # create_collection(bulk_mode=True) — indexing_threshold=0 설정 검증
        mock_client.reset_mock()
        mock_collections.collections = []
        indexer.create_collection(bulk_mode=True)
        create_call = mock_client.create_collection.call_args
        opt_config = create_call.kwargs.get("optimizers_config")
        assert opt_config is not None
        assert opt_config.indexing_threshold == 0
        print(f"  create_collection(bulk_mode=True): indexing_threshold=0  OK")

        # create_collection(bulk_mode=False) — optimizers_config 미전달 검증
        mock_client.reset_mock()
        indexer.create_collection(bulk_mode=False)
        create_call = mock_client.create_collection.call_args
        assert "optimizers_config" not in create_call.kwargs
        print(f"  create_collection(bulk_mode=False): optimizers_config 없음  OK")

        # finalize — update_collection + 폴링 검증
        mock_client.reset_mock()
        mock_info = MagicMock()
        mock_info.optimizer_status.status = "ok"
        mock_info.points_count = 40000
        mock_client.get_collection.return_value = mock_info

        indexer.finalize(indexing_threshold=20000)
        update_call = mock_client.update_collection.call_args
        assert update_call.kwargs["collection_name"] == "test_col"
        assert update_call.kwargs["optimizers_config"].indexing_threshold == 20000
        print(f"  finalize(20000): update_collection 호출  OK")

        # finalize — 폴링 대기 시뮬레이션 (indexing → ok)
        mock_client.reset_mock()
        mock_indexing = MagicMock()
        mock_indexing.optimizer_status.status = "indexing"
        mock_indexing.points_count = 30000
        mock_ok = MagicMock()
        mock_ok.optimizer_status.status = "ok"
        mock_ok.points_count = 40000
        mock_client.get_collection.side_effect = [mock_indexing, mock_ok]

        indexer.finalize(indexing_threshold=15000)
        assert mock_client.get_collection.call_count == 2
        print(f"  finalize 폴링: indexing → ok (2회 조회)  OK")

    print("  PASS\n")


def test_searcher():
    """Searcher: embed + Qdrant search 연동"""
    print("=" * 60)
    print("[3] Searcher — embed → search 파이프라인")
    print("=" * 60)

    with patch("qdrant_indexer.searcher.EmbeddingClient") as MockEmbed, \
         patch("qdrant_indexer.searcher.QdrantIndexer") as MockQdrant:

        mock_embedder = MockEmbed.return_value
        mock_indexer = MockQdrant.return_value

        # Mock embed: (1, 768)
        mock_embedder.embed.return_value = np.random.randn(1, 768)

        # Mock search result
        mock_points = []
        for i, (score, kw) in enumerate([
            (0.95, "東京 ラーメン 人気"),
            (0.91, "東京 つけ麺"),
            (0.88, "東京 味噌ラーメン"),
        ]):
            p = MagicMock()
            p.score = score
            p.payload = {"keyword": kw}
            p.id = i + 1
            mock_points.append(p)

        mock_response = MagicMock()
        mock_response.points = mock_points
        mock_indexer.search.return_value = mock_response

        searcher = Searcher(Config())
        results = searcher.search("東京 ラーメン", top_k=3)

        # 검증: prefix 추가
        embed_call = mock_embedder.embed.call_args[0][0]
        assert embed_call == ["検索クエリ: 東京 ラーメン"]
        print(f"  쿼리 prefix: '{embed_call[0]}'  OK")

        # 검증: SearchResult 변환
        assert len(results) == 3
        assert isinstance(results[0], SearchResult)
        assert results[0].rank == 1
        assert results[0].score == 0.95
        assert results[0].keyword == "東京 ラーメン 人気"
        print(f"  결과 #{results[0].rank}: score={results[0].score:.2f} '{results[0].keyword}'")
        print(f"  결과 #{results[1].rank}: score={results[1].score:.2f} '{results[1].keyword}'")
        print(f"  결과 #{results[2].rank}: score={results[2].score:.2f} '{results[2].keyword}'")

        # search_batch
        mock_embedder.embed.return_value = np.random.randn(2, 768)
        mock_batch = [mock_response, mock_response]
        mock_indexer.search_batch.return_value = mock_batch

        batch_results = searcher.search_batch(["東京", "大阪"], top_k=3)
        assert len(batch_results) == 2
        assert "東京" in batch_results
        assert "大阪" in batch_results

        # prefix 확인
        embed_call = mock_embedder.embed.call_args[0][0]
        assert embed_call == ["検索クエリ: 東京", "検索クエリ: 大阪"]
        print(f"  search_batch: 2 queries → {len(batch_results)} results  OK")

        # custom prefix
        mock_embedder.embed.return_value = np.random.randn(1, 768)
        searcher.search("テスト", prefix="検索文書: ")
        embed_call = mock_embedder.embed.call_args[0][0]
        assert embed_call == ["検索文書: テスト"]
        print(f"  custom prefix: '{embed_call[0]}'  OK")

        # SearchResult.payload 필드 검증
        assert results[0].payload == {"keyword": "東京 ラーメン 人気"}
        print(f"  SearchResult.payload: {results[0].payload}  OK")

    # custom payload_key
    with patch("qdrant_indexer.searcher.EmbeddingClient") as MockEmbed, \
         patch("qdrant_indexer.searcher.QdrantIndexer") as MockQdrant:

        mock_embedder = MockEmbed.return_value
        mock_indexer = MockQdrant.return_value
        mock_embedder.embed.return_value = np.random.randn(1, 768)

        # payload에 "keyword" 없고 "product_name"만 있는 경우
        mock_point = MagicMock()
        mock_point.score = 0.92
        mock_point.payload = {"product_name": "商品X", "price": 5000}
        mock_point.id = 42
        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_indexer.search.return_value = mock_response

        searcher2 = Searcher(Config(payload_key="product_name"), payload_key="product_name")
        results2 = searcher2.search("商品", top_k=1)

        assert results2[0].keyword == "商品X"  # payload_key로 추출
        assert results2[0].payload["price"] == 5000
        print(f"  payload_key='product_name': keyword='{results2[0].keyword}', price={results2[0].payload['price']}  OK")

    print("  PASS\n")


def test_pipeline_stats():
    """PipelineStats: 스레드 안전 통계 + 로깅 조건 (공용 유틸리티)"""
    print("=" * 60)
    print("[4] PipelineStats — 통계 + RPS 계산")
    print("=" * 60)

    stats = PipelineStats(total=1000)
    assert stats.processed == 0
    assert stats.total == 1000

    # 업데이트 시뮬레이션
    stats.update(count=64, embed_ms=15.0, upsert_ms=8.0)
    assert stats.processed == 64
    assert stats.get_timing("embed_ms") == 15.0
    assert stats.get_timing("upsert_ms") == 8.0
    print(f"  1차 업데이트: processed={stats.processed}, embed_ms={stats.get_timing('embed_ms')}")

    stats.update(count=36, embed_ms=12.0, upsert_ms=6.0)
    assert stats.processed == 100  # 100건 → should_log 트리거
    assert stats.get_timing("embed_ms") == 27.0
    print(f"  2차 업데이트: processed={stats.processed}, embed_ms={stats.get_timing('embed_ms')}")

    # wall_sec
    time.sleep(0.05)
    assert stats.wall_sec > 0.05
    print(f"  wall_sec: {stats.wall_sec:.3f}s  OK")

    # record_retry / record_failure
    stats.record_retry()
    assert stats.retries == 1
    stats.record_failure(10)
    assert stats.failed_count == 10
    assert stats.failed_batches == 1
    print(f"  retries={stats.retries}, failed={stats.failed_count}  OK")

    print("  PASS\n")


def test_async_process_batch():
    """_process_batch: 비동기 배치 처리 (embed via executor + upsert async + 재시도)"""
    print("=" * 60)
    print("[5] _process_batch — 비동기 embed → upsert 배치 처리")
    print("=" * 60)

    async def _run_tests():
        # Mock embedder (sync — run_in_executor에서 실행됨)
        mock_embedder = MagicMock(spec=EmbeddingClient)
        mock_embedder.embed.return_value = np.random.randn(3, 768)

        # Mock async indexer
        mock_indexer = AsyncMock(spec=AsyncQdrantIndexer)

        config = Config(max_retries=2, retry_backoff=0.01, retry_exponential=False)
        stats = PipelineStats(total=100)
        semaphore = asyncio.Semaphore(4)

        with tempfile.TemporaryDirectory() as td:
            fl = AsyncFailureLogger(Path(td) / "fail.jsonl", enabled=True)

            # 1. 정상 배치 처리
            await _process_batch(
                batch_id=0,
                keywords=["a", "b", "c"],
                embedder=mock_embedder,
                indexer=mock_indexer,
                semaphore=semaphore,
                stats=stats,
                config=config,
                failure_logger=fl,
            )

            mock_embedder.embed.assert_called_with(["a", "b", "c"])
            mock_indexer.upsert_batch.assert_called_once()
            upsert_args = mock_indexer.upsert_batch.call_args
            assert upsert_args.kwargs["start_id"] == 0
            assert upsert_args.kwargs["keywords"] == ["a", "b", "c"]
            assert stats.processed == 3
            print(f"  정상 배치: processed={stats.processed}, embed 호출 OK, upsert 호출 OK")

            # 2. 실패 → 재시도 → 성공
            mock_indexer.reset_mock()
            mock_indexer.upsert_batch.side_effect = [
                ConnectionError("timeout"),
                None,  # 2번째 성공
            ]

            await _process_batch(
                batch_id=100,
                keywords=["x", "y"],
                embedder=mock_embedder,
                indexer=mock_indexer,
                semaphore=semaphore,
                stats=stats,
                config=config,
                failure_logger=fl,
            )

            assert mock_indexer.upsert_batch.call_count == 2
            assert stats.retries == 1
            print(f"  재시도 후 성공: upsert 호출 {mock_indexer.upsert_batch.call_count}회, retries={stats.retries}  OK")

            # 3. 모든 재시도 실패 → failure_logger에 기록
            mock_indexer.reset_mock()
            mock_indexer.upsert_batch.side_effect = RuntimeError("DB crashed")

            await _process_batch(
                batch_id=200,
                keywords=["fail1", "fail2"],
                embedder=mock_embedder,
                indexer=mock_indexer,
                semaphore=semaphore,
                stats=stats,
                config=config,
                failure_logger=fl,
            )

            # _process_batch는 예외를 던지지 않고 failure_logger에 기록
            assert stats.failed_count == 2
            assert stats.failed_batches == 1
            fail_log = (Path(td) / "fail.jsonl").read_text().strip().split("\n")
            assert len(fail_log) == 1  # 최종 실패 1건
            assert "DB crashed" in fail_log[0]
            print(f"  최종 실패: AsyncFailureLogger에 {len(fail_log)}건, failed_count={stats.failed_count}  OK")

    asyncio.run(_run_tests())
    print("  PASS\n")


def test_custom_payloads():
    """QdrantIndexer: 커스텀 ids + payloads 파라미터 검증"""
    print("=" * 60)
    print("[6] QdrantIndexer — 커스텀 ids + payloads")
    print("=" * 60)

    with patch("qdrant_indexer.indexer.QdrantClient") as MockClient:
        mock_client = MockClient.return_value
        indexer = QdrantIndexer("http://localhost:6333", "test_col", 768)

        # 커스텀 ids + payloads
        embeddings = np.random.randn(3, 768)
        indexer.upsert_batch(
            start_id=0,
            keywords=["ignored"],
            embeddings=embeddings,
            ids=["uuid-a", "uuid-b", "uuid-c"],
            payloads=[
                {"product_name": "商品A", "category": "食品", "price": 1000},
                {"product_name": "商品B", "category": "飲料", "price": 2000},
                {"product_name": "商品C", "category": "菓子", "price": 3000},
            ],
        )
        points = mock_client.upsert.call_args.kwargs["points"]
        assert len(points) == 3
        assert points[0].id == "uuid-a"
        assert points[0].payload["product_name"] == "商品A"
        assert points[1].id == "uuid-b"
        assert points[2].payload["price"] == 3000
        print(f"  upsert_batch(custom): ids=[uuid-a..c], payloads OK")

        # 하위호환: 기존 keyword 모드
        mock_client.reset_mock()
        indexer.upsert_batch(
            start_id=100,
            keywords=["東京", "大阪"],
            embeddings=np.random.randn(2, 768),
        )
        points = mock_client.upsert.call_args.kwargs["points"]
        assert points[0].id == 100
        assert points[0].payload == {"keyword": "東京"}
        assert points[1].id == 101
        print(f"  upsert_batch(기본): 하위호환 OK")

        # ids만 커스텀 (payloads는 keyword 기반)
        mock_client.reset_mock()
        indexer.upsert_batch(
            start_id=0,
            keywords=["東京", "大阪"],
            embeddings=np.random.randn(2, 768),
            ids=[999, 1000],
        )
        points = mock_client.upsert.call_args.kwargs["points"]
        assert points[0].id == 999
        assert points[0].payload == {"keyword": "東京"}
        print(f"  ids만 커스텀: id=999, payload=keyword 기반  OK")

        # ids 길이 불일치 → ValueError
        try:
            indexer.upsert_batch(
                start_id=0, keywords=[], embeddings=np.random.randn(3, 768),
                ids=["a", "b"],
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "ids" in str(e)
            print(f"  ids 길이 불일치 → ValueError: OK")

        # payloads 길이 불일치 → ValueError
        try:
            indexer.upsert_batch(
                start_id=0, keywords=[], embeddings=np.random.randn(3, 768),
                payloads=[{"a": 1}],
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "payloads" in str(e)
            print(f"  payloads 길이 불일치 → ValueError: OK")

        # 커스텀 payload_key
        mock_client.reset_mock()
        indexer.upsert_batch(
            start_id=0,
            keywords=["商品A", "商品B"],
            embeddings=np.random.randn(2, 768),
            payload_key="product_name",
        )
        points = mock_client.upsert.call_args.kwargs["points"]
        assert points[0].payload == {"product_name": "商品A"}
        assert points[1].payload == {"product_name": "商品B"}
        print(f"  payload_key='product_name': OK")

    print("  PASS\n")


def test_async_custom_payloads():
    """AsyncQdrantIndexer: 커스텀 ids + payloads 파라미터 검증 (비동기)"""
    print("=" * 60)
    print("[7] AsyncQdrantIndexer — 커스텀 ids + payloads")
    print("=" * 60)

    async def _run():
        with patch("qdrant_indexer.indexer.AsyncQdrantClient"):
            indexer = AsyncQdrantIndexer("http://localhost:6333", "test_col", 768)
            indexer.client = AsyncMock()

            # 커스텀 ids + payloads
            embeddings = np.random.randn(3, 768)
            await indexer.upsert_batch(
                start_id=0,
                keywords=[],
                embeddings=embeddings,
                ids=[1001, 1002, 1003],
                payloads=[
                    {"text": "hello", "lang": "en"},
                    {"text": "world", "lang": "en"},
                    {"text": "こんにちは", "lang": "ja"},
                ],
            )
            points = indexer.client.upsert.call_args.kwargs["points"]
            assert points[0].id == 1001
            assert points[0].payload["text"] == "hello"
            assert points[2].payload["lang"] == "ja"
            print(f"  async upsert_batch(custom): ids + payloads OK")

            # 하위호환
            indexer.client.reset_mock()
            await indexer.upsert_batch(
                start_id=200,
                keywords=["a", "b"],
                embeddings=np.random.randn(2, 768),
            )
            points = indexer.client.upsert.call_args.kwargs["points"]
            assert points[0].id == 200
            assert points[0].payload == {"keyword": "a"}
            print(f"  async upsert_batch(기본): 하위호환 OK")

            # ids 길이 불일치 → ValueError
            try:
                await indexer.upsert_batch(
                    start_id=0, keywords=[], embeddings=np.random.randn(3, 768),
                    ids=["a"],
                )
                assert False, "Should have raised ValueError"
            except ValueError:
                print(f"  async ids 길이 불일치 → ValueError: OK")

    asyncio.run(_run())
    print("  PASS\n")


if __name__ == "__main__":
    test_config()
    test_qdrant_indexer()
    test_searcher()
    test_pipeline_stats()
    test_async_process_batch()
    test_custom_payloads()
    test_async_custom_payloads()

    print("=" * 60)
    print("ALL qdrant-indexer EXAMPLES PASSED")
    print("=" * 60)
