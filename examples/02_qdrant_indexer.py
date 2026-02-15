#!/usr/bin/env python3
"""
qdrant-indexer 패키지 예시 — Mock으로 전체 파이프라인 코드 경로 검증

테스트 항목:
  1. Config 기본값 + 커스텀 생성 (BaseConfig 상속)
  2. QdrantIndexer — create_collection, upsert_batch, search, search_batch
  3. Searcher — search, search_batch (embed + Qdrant 연동)
  4. PipelineStats 통계 로직 (공용 유틸리티)
  5. pipeline._create_batch_processor 팩토리 패턴
"""

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

from qdrant_indexer import Config, QdrantIndexer, AsyncQdrantIndexer, Searcher, SearchResult
from qdrant_indexer.pipeline import _create_batch_processor
from kserve_embed_client import EmbeddingClient, PipelineStats, RetryConfig, FailureLogger


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
    print(f"  기본값: kserve={c.kserve_url}, qdrant={c.qdrant_url}, dim={c.vector_dim}")
    print(f"  처리: batch={c.batch_size}, workers={c.workers}")
    print(f"  재시도: max={c.max_retries}, exponential={c.retry_exponential}")

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


def test_batch_processor():
    """_create_batch_processor: 팩토리 패턴 + 재시도 통합"""
    print("=" * 60)
    print("[5] _create_batch_processor — embed → upsert 배치 처리")
    print("=" * 60)

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = np.random.randn(3, 768)

    mock_indexer = MagicMock()

    stats = PipelineStats(total=100)

    with tempfile.TemporaryDirectory() as td:
        fl = FailureLogger(Path(td) / "fail.jsonl", enabled=True)
        rc = RetryConfig(max_retries=2, initial_backoff=0.01)

        process = _create_batch_processor(mock_embedder, mock_indexer, stats, fl, rc)

        # 정상 배치 처리 (positional args — ThreadPoolExecutor와 동일)
        process(0, ["a", "b", "c"])
        mock_embedder.embed.assert_called_with(["a", "b", "c"])
        mock_indexer.upsert_batch.assert_called_once()
        upsert_args = mock_indexer.upsert_batch.call_args
        assert upsert_args.kwargs["start_id"] == 0
        assert upsert_args.kwargs["keywords"] == ["a", "b", "c"]
        assert stats.processed == 3
        print(f"  정상 배치: processed={stats.processed}, embed 호출 OK, upsert 호출 OK")

        # 실패 → 재시도 → 성공
        mock_indexer.reset_mock()
        mock_indexer.upsert_batch.side_effect = [
            ConnectionError("timeout"),
            None,  # 2번째 성공
        ]

        process2 = _create_batch_processor(mock_embedder, mock_indexer, stats, fl, rc)
        process2(100, ["x", "y"])
        assert mock_indexer.upsert_batch.call_count == 2
        print(f"  재시도 후 성공: upsert 호출 {mock_indexer.upsert_batch.call_count}회  OK")

        # 모든 재시도 실패
        mock_indexer.reset_mock()
        mock_indexer.upsert_batch.side_effect = RuntimeError("DB crashed")

        process3 = _create_batch_processor(mock_embedder, mock_indexer, stats, fl, rc)
        try:
            process3(200, ["fail1", "fail2"])
            assert False, "Should have raised"
        except RuntimeError:
            pass

        fail_log = (Path(td) / "fail.jsonl").read_text().strip().split("\n")
        assert len(fail_log) >= 2
        print(f"  최종 실패: FailureLogger에 {len(fail_log)}건 기록  OK")

    print("  PASS\n")


if __name__ == "__main__":
    test_config()
    test_qdrant_indexer()
    test_searcher()
    test_pipeline_stats()
    test_batch_processor()

    print("=" * 60)
    print("ALL qdrant-indexer EXAMPLES PASSED")
    print("=" * 60)
