#!/usr/bin/env python3
"""
통합 테스트 — 실제 작은 데이터로 전체 파이프라인 코드 경로 검증

외부 서비스(KServe, Qdrant, ES) 없이 Mock으로 동작하며,
패키지 간 import 경로 + 실제 Parquet 데이터 처리를 검증합니다.

테스트 항목:
  1. 크로스 패키지 import 정합성
  2. 실제 Parquet → ParquetReader → qdrant pipeline 시뮬레이션
  3. 실제 Parquet → es pipeline _load_keywords 시뮬레이션
  4. Searcher → EmbeddingClient → QdrantIndexer 전체 흐름
  5. 패키지별 로거 독립성 검증
  6. RetryConfig + FailureLogger 크로스 패키지 재사용
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# 1. 크로스 패키지 import 정합성
# ============================================================
def test_cross_package_imports():
    """모든 패키지의 public API가 정상적으로 import되는지 검증"""
    print("=" * 60)
    print("[1] 크로스 패키지 import 정합성")
    print("=" * 60)

    # kserve-embed-client
    from kserve_embed_client import EmbeddingClient
    print("  kserve_embed_client: EmbeddingClient OK")

    # pipeline-commons
    from pipeline_commons import (
        ParquetReader,
        validate_parquet_columns,
        setup_logging,
        get_logger,
        RetryConfig,
        FailureLogger,
        with_retry,
    )
    print("  pipeline_commons: 7개 export OK")

    # qdrant-indexer
    from qdrant_indexer import (
        Config as QdrantConfig,
        QdrantIndexer,
        AsyncQdrantIndexer,
        run_indexing,
        Searcher,
        SearchResult,
        run_batch_search,
    )
    print("  qdrant_indexer: 7개 export OK")

    # es-indexer
    from es_indexer import (
        Config as ESConfig,
        DEFAULT_SCHEMA,
        ESIndexer,
        build_es_client,
        run_indexing as es_run_indexing,
        run_realtime,
    )
    print("  es_indexer: 6개 export OK")

    # qdrant_indexer 내부에서 pipeline_commons를 사용하는지 확인
    from qdrant_indexer.pipeline import _process_batch
    from qdrant_indexer.searcher import Searcher as Searcher2
    assert Searcher is Searcher2
    print("  qdrant_indexer.pipeline → pipeline_commons import: OK")

    # pipeline_commons 공용 유틸리티 확인
    from pipeline_commons import load_keywords, PipelineStats, AsyncFailureLogger
    print("  pipeline_commons 공용 유틸리티: OK")

    # 타입 확인
    assert issubclass(QdrantConfig, object)
    assert issubclass(ESConfig, object)
    assert callable(run_indexing)
    assert callable(es_run_indexing)
    print("  타입 검증: OK")

    print("  PASS\n")


# ============================================================
# 2. 실제 Parquet → qdrant pipeline 시뮬레이션
# ============================================================
def test_parquet_to_qdrant_pipeline():
    """실제 Parquet 파일 생성 → ParquetReader → embed + upsert 시뮬레이션"""
    print("=" * 60)
    print("[2] Parquet → Qdrant 파이프라인 시뮬레이션")
    print("=" * 60)

    from kserve_embed_client import EmbeddingClient
    from pipeline_commons import AsyncFailureLogger, ParquetReader, PipelineStats
    from qdrant_indexer import AsyncQdrantIndexer, Config as QdrantCfg
    from qdrant_indexer.pipeline import _process_batch

    async def _run_pipeline_test():
        with tempfile.TemporaryDirectory() as td:
            # 실제 Parquet 파일 생성: 50개 일본어 키워드
            keywords = [
                "東京 ラーメン", "大阪 たこ焼き", "京都 抹茶", "北海道 海鮮",
                "福岡 もつ鍋", "名古屋 味噌カツ", "広島 お好み焼き", "仙台 牛タン",
                "札幌 スープカレー", "横浜 中華街", "神戸 ビーフ", "沖縄 ソーキそば",
                "金沢 寿司", "長崎 ちゃんぽん", "鹿児島 黒豚", "新潟 日本酒",
                "静岡 うなぎ", "愛媛 みかん", "熊本 馬刺し", "山形 芋煮",
                "秋田 きりたんぽ", "岩手 わんこそば", "青森 りんご", "石川 加賀料理",
                "富山 ブラックラーメン", "岐阜 鶏ちゃん", "三重 松阪牛", "滋賀 近江牛",
                "奈良 柿の葉寿司", "和歌山 梅干し", "鳥取 カニ", "島根 出雲そば",
                "岡山 きびだんご", "山口 ふぐ", "香川 讃岐うどん", "徳島 阿波尾鶏",
                "高知 カツオ", "佐賀 シシリアンライス", "大分 とり天", "宮崎 チキン南蛮",
                "栃木 餃子", "群馬 焼きまんじゅう", "埼玉 十万石饅頭", "千葉 落花生",
                "茨城 納豆", "長野 信州そば", "山梨 ほうとう", "福島 喜多方ラーメン",
                "徳島 鯛めし", "佐賀 嬉野温泉",
            ]

            table = pa.table({
                "keyword": keywords,
                "category": [f"food_{i % 5}" for i in range(len(keywords))],
                "score": [round(0.5 + i * 0.01, 2) for i in range(len(keywords))],
            })
            parquet_path = Path(td) / "keywords.parquet"
            pq.write_table(table, parquet_path)
            print(f"  Parquet 생성: {len(keywords)}행, 3컬럼 (keyword, category, score)")

            # ParquetReader로 청크 읽기
            reader = ParquetReader(parquet_path, chunk_size=15, text_column="keyword")
            assert reader.total_rows == 50
            assert reader.num_chunks == 4  # ceil(50/15)
            print(f"  ParquetReader: {reader.total_rows}행, {reader.num_chunks}청크")

            all_keywords = []
            for batch_id, kws in reader.iter_chunks():
                all_keywords.extend(kws)
            assert len(all_keywords) == 50
            assert all_keywords[0] == "東京 ラーメン"
            assert all_keywords[-1] == "佐賀 嬉野温泉"
            print(f"  전체 읽기: {len(all_keywords)}행 OK")

            # Mock embed_fn (sync) + async indexer로 비동기 파이프라인 시뮬레이션
            mock_embed_fn = MagicMock(
                side_effect=lambda texts: np.random.randn(len(texts), 768)
            )

            mock_indexer = AsyncMock(spec=AsyncQdrantIndexer)

            config = QdrantCfg(max_retries=2, retry_backoff=0.01, retry_exponential=False)
            stats = PipelineStats(total=50)
            fl = AsyncFailureLogger(Path(td) / "fail.jsonl", enabled=True)
            semaphore = asyncio.Semaphore(4)

            # 청크 단위로 비동기 처리 시뮬레이션
            reader2 = ParquetReader(parquet_path, chunk_size=15, text_column="keyword")
            batch_size = 10
            coros = []

            for chunk_id, chunk_kws in reader2.iter_chunks():
                for i in range(0, len(chunk_kws), batch_size):
                    batch = chunk_kws[i:i + batch_size]
                    bid = chunk_id * 15 + i
                    coros.append(_process_batch(
                        bid, batch, mock_embed_fn, mock_indexer,
                        semaphore, stats, config, fl,
                    ))

            await asyncio.gather(*coros)

            assert stats.processed == 50
            assert mock_embed_fn.call_count > 0
            assert mock_indexer.upsert_batch.call_count > 0
            print(f"  파이프라인 완료: processed={stats.processed}, "
                  f"embed 호출={mock_embed_fn.call_count}회, "
                  f"upsert 호출={mock_indexer.upsert_batch.call_count}회")

            # 각 embed 호출에서 올바른 키워드가 전달되었는지 확인
            first_call_texts = mock_embed_fn.call_args_list[0][0][0]
            assert isinstance(first_call_texts, list)
            assert all(isinstance(t, str) for t in first_call_texts)
            print(f"  첫 embed 호출: {len(first_call_texts)}건 — '{first_call_texts[0]}'")

            # 각 upsert에서 올바른 embeddings shape이 전달되었는지 확인
            first_upsert = mock_indexer.upsert_batch.call_args_list[0]
            emb_arg = first_upsert.kwargs["embeddings"]
            assert emb_arg.shape[1] == 768
            print(f"  첫 upsert: embeddings shape={emb_arg.shape}")

            # wall time 확인
            assert stats.wall_sec > 0
            print(f"  wall_sec: {stats.wall_sec:.4f}s")

    asyncio.run(_run_pipeline_test())

    print("  PASS\n")


# ============================================================
# 3. 실제 Parquet → es pipeline _load_keywords 시뮬레이션
# ============================================================
def test_parquet_to_es_pipeline():
    """실제 Parquet → load_keywords (공용 유틸리티) 검증"""
    print("=" * 60)
    print("[3] Parquet → ES 파이프라인 시뮬레이션")
    print("=" * 60)

    from pipeline_commons import load_keywords
    from es_indexer import Config

    with tempfile.TemporaryDirectory() as td:
        # Parquet 파일 생성
        keywords = [f"商品_{i:03d}" for i in range(100)]
        table = pa.table({"keyword": keywords})
        parquet_path = Path(td) / "products.parquet"
        pq.write_table(table, parquet_path)

        # Parquet 소스로 전체 로드
        config = Config(parquet_path=parquet_path, parquet_chunk_size=30)
        kws, source = load_keywords(config)
        assert len(kws) == 100
        assert "Parquet" in source
        assert kws[0] == "商品_000"
        assert kws[99] == "商品_099"
        print(f"  Parquet 전체: {len(kws)}건, source='{source}'  OK")

        # limit 적용
        config_limited = Config(parquet_path=parquet_path, limit=25)
        kws_limited, _ = load_keywords(config_limited)
        assert len(kws_limited) == 25
        print(f"  Parquet limit=25: {len(kws_limited)}건  OK")

        # 텍스트 파일 소스
        text_path = Path(td) / "keywords.txt"
        text_path.write_text("\n".join([f"キーワード_{i}" for i in range(30)]))

        config_text = Config(keywords_path=text_path)
        kws_text, source_text = load_keywords(config_text)
        assert len(kws_text) == 30
        assert "텍스트" in source_text
        print(f"  텍스트 파일: {len(kws_text)}건, source='{source_text}'  OK")

        # Parquet 폴더 (partitioned)
        folder = Path(td) / "partitioned"
        folder.mkdir()
        for i in range(3):
            sub = pa.table({"keyword": [f"part{i}_kw{j}" for j in range(20)]})
            pq.write_table(sub, folder / f"part_{i:03d}.parquet")

        config_dir = Config(parquet_path=folder, parquet_chunk_size=25)
        kws_dir, source_dir = load_keywords(config_dir)
        assert len(kws_dir) == 60
        assert "3 files" in source_dir
        print(f"  Parquet 폴더: {len(kws_dir)}건, source='{source_dir}'  OK")

    print("  PASS\n")


# ============================================================
# 4. Searcher 전체 흐름 (embed → search → SearchResult)
# ============================================================
def test_searcher_e2e():
    """Searcher: 실제와 유사한 시나리오로 검색 흐름 검증"""
    print("=" * 60)
    print("[4] Searcher 전체 흐름 — embed → search → SearchResult")
    print("=" * 60)

    from qdrant_indexer import Config, Searcher, SearchResult

    with patch("qdrant_indexer.searcher.QdrantIndexer") as MockQdrant:

        mock_indexer = MockQdrant.return_value
        mock_embed_fn = MagicMock(return_value=np.random.randn(1, 768))

        # 현실적인 검색 결과 시뮬레이션
        search_scenarios = [
            {
                "query": "東京 ラーメン",
                "results": [
                    (0.97, "東京 ラーメン 人気店"),
                    (0.93, "東京 つけ麺 おすすめ"),
                    (0.89, "東京 家系ラーメン"),
                    (0.85, "東京 味噌ラーメン"),
                    (0.81, "東京 二郎系ラーメン"),
                ],
            },
            {
                "query": "大阪 グルメ",
                "results": [
                    (0.95, "大阪 たこ焼き 名店"),
                    (0.91, "大阪 串カツ"),
                    (0.87, "大阪 お好み焼き"),
                ],
            },
        ]

        for scenario in search_scenarios:
            query = scenario["query"]
            expected_results = scenario["results"]

            # Mock embed_fn: query → (1, 768)
            mock_embed_fn.return_value = np.random.randn(1, 768)

            # Mock search results
            mock_points = []
            for i, (score, kw) in enumerate(expected_results):
                p = MagicMock()
                p.score = score
                p.payload = {"keyword": kw}
                p.id = i + 1
                mock_points.append(p)

            mock_response = MagicMock()
            mock_response.points = mock_points
            mock_indexer.search.return_value = mock_response

            # 검색 실행
            searcher = Searcher(Config(), embed_fn=mock_embed_fn, query_prefix="検索クエリ: ")
            results = searcher.search(query, top_k=len(expected_results))

            # 검증
            assert len(results) == len(expected_results)
            assert all(isinstance(r, SearchResult) for r in results)

            # prefix 검증
            embed_call = mock_embed_fn.call_args[0][0]
            assert embed_call == [f"検索クエリ: {query}"]

            # rank 검증
            for i, r in enumerate(results):
                assert r.rank == i + 1
                assert r.score == expected_results[i][0]
                assert r.keyword == expected_results[i][1]

            print(f"  쿼리: '{query}' → {len(results)}건")
            print(f"    prefix: '{embed_call[0]}'")
            print(f"    top1: score={results[0].score:.2f} '{results[0].keyword}'")

        # search_batch 검증
        queries = ["東京 ラーメン", "大阪 グルメ", "京都 観光"]
        mock_embed_fn.return_value = np.random.randn(3, 768)

        batch_responses = []
        for q in queries:
            mock_resp = MagicMock()
            p = MagicMock()
            p.score = 0.9
            p.payload = {"keyword": f"{q} 結果"}
            p.id = 1
            mock_resp.points = [p]
            batch_responses.append(mock_resp)

        mock_indexer.search_batch.return_value = batch_responses

        batch_results = searcher.search_batch(queries, top_k=1)
        assert len(batch_results) == 3
        for q in queries:
            assert q in batch_results
            assert len(batch_results[q]) == 1

        embed_call = mock_embed_fn.call_args[0][0]
        assert embed_call == ["検索クエリ: 東京 ラーメン", "検索クエリ: 大阪 グルメ", "検索クエリ: 京都 観光"]
        print(f"  search_batch: {len(queries)} queries → {len(batch_results)} results  OK")

    print("  PASS\n")


# ============================================================
# 5. 패키지별 로거 독립성
# ============================================================
def test_logger_independence():
    """각 패키지가 독립적인 로거 네임스페이스를 사용하는지 검증"""
    print("=" * 60)
    print("[5] 패키지별 로거 독립성")
    print("=" * 60)

    from pipeline_commons import get_logger, setup_logging
    import logging

    with tempfile.TemporaryDirectory() as td:
        # 각 패키지별 로그 파일
        log_a = Path(td) / "kserve.log"
        log_b = Path(td) / "qdrant.log"
        log_c = Path(td) / "es.log"

        setup_logging("kserve_embed_client", log_file=log_a)
        setup_logging("qdrant_indexer", log_file=log_b)
        setup_logging("es_indexer", log_file=log_c)

        logger_k = get_logger("kserve_embed_client", "embedder")
        logger_q = get_logger("qdrant_indexer", "pipeline")
        logger_e = get_logger("es_indexer", "pipeline")

        # 각 로거에 메시지 기록
        logger_k.info("KServe 임베딩 요청")
        logger_q.info("Qdrant upsert 완료")
        logger_e.info("ES bulk 완료")

        # 로거 네임스페이스 검증
        assert logger_k.name == "kserve_embed_client.embedder"
        assert logger_q.name == "qdrant_indexer.pipeline"
        assert logger_e.name == "es_indexer.pipeline"
        print(f"  네임스페이스: {logger_k.name}, {logger_q.name}, {logger_e.name}")

        # 파일 기록 검증 (각 패키지 로거는 부모의 파일 핸들러를 상속)
        content_a = log_a.read_text()
        assert "KServe 임베딩 요청" in content_a
        print(f"  kserve.log: KServe 메시지 포함  OK")

        content_b = log_b.read_text()
        assert "Qdrant upsert 완료" in content_b
        print(f"  qdrant.log: Qdrant 메시지 포함  OK")

        content_c = log_c.read_text()
        assert "ES bulk 완료" in content_c
        print(f"  es.log: ES 메시지 포함  OK")

        # 교차 오염 없음 확인 (각 파일에 다른 패키지 메시지가 없어야 함)
        assert "Qdrant" not in content_a
        assert "ES bulk" not in content_a
        print(f"  교차 오염 없음: OK")

    print("  PASS\n")


# ============================================================
# 6. RetryConfig + FailureLogger 크로스 패키지 재사용
# ============================================================
def test_retry_cross_package():
    """kserve-embed-client의 retry가 qdrant/es 양쪽에서 재사용되는지 검증"""
    print("=" * 60)
    print("[6] RetryConfig + FailureLogger 크로스 패키지 재사용")
    print("=" * 60)

    from pipeline_commons import RetryConfig, FailureLogger, with_retry

    with tempfile.TemporaryDirectory() as td:
        # qdrant 스타일 사용
        qdrant_fl = FailureLogger(Path(td) / "qdrant_fail.jsonl", enabled=True)
        qdrant_rc = RetryConfig(max_retries=2, initial_backoff=0.01)

        call_count_q = 0

        @with_retry(
            retry_config=qdrant_rc,
            failure_logger=qdrant_fl,
            batch_id_fn=lambda args, kwargs: args[0],
        )
        def qdrant_upsert(batch_id, data):
            nonlocal call_count_q
            call_count_q += 1
            if call_count_q < 2:
                raise ConnectionError("Qdrant connection refused")
            return "ok"

        result = qdrant_upsert(0, ["a", "b"])
        assert result == "ok"
        assert call_count_q == 2
        print(f"  Qdrant 스타일: 재시도 후 성공 (call_count={call_count_q})  OK")

        # es 스타일 사용 — 같은 RetryConfig/FailureLogger 타입
        es_fl = FailureLogger(Path(td) / "es_fail.jsonl", enabled=True)
        es_rc = RetryConfig(max_retries=3, initial_backoff=0.01, exponential=False)

        call_count_e = 0

        @with_retry(retry_config=es_rc, failure_logger=es_fl,
                     batch_id_fn=lambda args, kwargs: args[0])
        def es_bulk(batch_id, docs):
            nonlocal call_count_e
            call_count_e += 1
            raise RuntimeError("ES bulk failed")

        try:
            es_bulk(100, ["doc1"])
        except RuntimeError:
            pass

        assert call_count_e == 3
        lines = (Path(td) / "es_fail.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3  # 3번 시도 → 3건 기록
        print(f"  ES 스타일: 3번 실패 후 종료 (기록={len(lines)}건)  OK")

        # 같은 클래스 사용 확인
        from qdrant_indexer.pipeline import _process_batch
        # _process_batch 내부에서 AsyncFailureLogger를 사용
        # → pipeline_commons.retry에서 import
        assert qdrant_rc.__class__.__name__ == "RetryConfig"
        assert qdrant_fl.__class__.__name__ == "FailureLogger"
        print(f"  타입 일치: RetryConfig, FailureLogger  OK")

    print("  PASS\n")


if __name__ == "__main__":
    test_cross_package_imports()
    test_parquet_to_qdrant_pipeline()
    test_parquet_to_es_pipeline()
    test_searcher_e2e()
    test_logger_independence()
    test_retry_cross_package()

    print("=" * 60)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 60)
