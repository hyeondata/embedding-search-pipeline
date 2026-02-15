#!/usr/bin/env python3
"""
es-indexer 패키지 예시 — Mock으로 전체 파이프라인 코드 경로 검증

테스트 항목:
  1. Config 기본값 + 클러스터 설정 (BaseConfig 상속)
  2. build_es_client — 단일 노드 / 클러스터 / 인증
  3. ESIndexer — create_index, ensure_index, bulk_index, finalize, CRUD
  4. ESIndexer.from_config — classmethod 팩토리
  5. AsyncFailureLogger — 비동기 실패 기록 (공용 유틸리티)
  6. load_keywords — 텍스트/Parquet 데이터 로드 (공용 유틸리티)
  7. Config BaseConfig 상속 검증
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from es_indexer import Config, DEFAULT_SCHEMA, ESIndexer, build_es_client
from kserve_embed_client import AsyncFailureLogger, BaseConfig, load_keywords


def test_config():
    """Config: 기본값 + 클러스터 설정 + BaseConfig 상속"""
    print("=" * 60)
    print("[1] Config — 기본값 + 클러스터 + BaseConfig 상속")
    print("=" * 60)

    c = Config()
    assert c.es_url == "http://localhost:9200"
    assert c.index_name == "keywords"
    assert c.batch_size == 500
    assert c.workers == 8
    assert c.max_retries == 3
    assert c.es_nodes is None
    assert c.es_fingerprint is None
    print(f"  기본: es_url={c.es_url}, index={c.index_name}")
    print(f"  처리: batch={c.batch_size}, workers={c.workers}")

    # BaseConfig 상속 필드 확인
    assert isinstance(c, BaseConfig)
    assert hasattr(c, "retry_backoff")
    assert hasattr(c, "retry_exponential")
    assert hasattr(c, "retry_max_backoff")
    assert hasattr(c, "log_failures")
    assert c.retry_backoff == 1.0
    assert c.retry_exponential is True
    assert c.log_failures is True
    assert c.bulk_mode is False
    assert c.bulk_replicas == 0
    assert c.bulk_replicas_restore == 0
    assert c.bulk_flush_threshold == "1gb"
    print(f"  BaseConfig 상속: retry_backoff={c.retry_backoff}, exponential={c.retry_exponential}")
    print(f"  벌크: bulk_mode={c.bulk_mode}, replicas={c.bulk_replicas}, flush={c.bulk_flush_threshold}")

    # 클러스터 설정
    c2 = Config(
        es_nodes=["https://es01:9200", "https://es02:9200"],
        es_fingerprint="B1:2A:96:...:CF",
        es_username="elastic",
        es_password="changeme",
        index_name="products",
    )
    assert c2.es_nodes == ["https://es01:9200", "https://es02:9200"]
    assert c2.es_fingerprint is not None
    print(f"  클러스터: {len(c2.es_nodes)}개 노드, fingerprint 설정  OK")

    print("  PASS\n")


def test_default_schema():
    """DEFAULT_SCHEMA: 스키마 구조 검증"""
    print("=" * 60)
    print("[2] DEFAULT_SCHEMA — 구조 검증")
    print("=" * 60)

    assert "settings" in DEFAULT_SCHEMA
    assert "mappings" in DEFAULT_SCHEMA
    assert DEFAULT_SCHEMA["settings"]["number_of_shards"] == 1
    assert DEFAULT_SCHEMA["settings"]["number_of_replicas"] == 0
    assert DEFAULT_SCHEMA["mappings"]["properties"]["keyword"]["type"] == "text"
    assert DEFAULT_SCHEMA["mappings"]["properties"]["keyword"]["analyzer"] == "standard"
    assert DEFAULT_SCHEMA["mappings"]["properties"]["keyword"]["fields"]["raw"]["type"] == "keyword"
    print(f"  shards={DEFAULT_SCHEMA['settings']['number_of_shards']}")
    print(f"  keyword.type=text, analyzer=standard, raw=keyword")
    print("  PASS\n")


def test_build_es_client():
    """build_es_client: 단일 노드 / 클러스터 / 인증 검증"""
    print("=" * 60)
    print("[3] build_es_client — 연결 설정")
    print("=" * 60)

    with patch("es_indexer.indexer.AsyncElasticsearch") as MockES:
        # Case 1: 단일 노드
        config = Config(es_url="http://localhost:9200")
        build_es_client(config)
        call_kwargs = MockES.call_args.kwargs
        assert call_kwargs["hosts"] == ["http://localhost:9200"]
        assert "ssl_assert_fingerprint" not in call_kwargs
        print(f"  단일 노드: hosts={call_kwargs['hosts']}  OK")

        # Case 2: 클러스터 + fingerprint + basic auth
        MockES.reset_mock()
        config2 = Config(
            es_nodes=["https://es01:9200", "https://es02:9200"],
            es_fingerprint="AA:BB:CC",
            es_username="elastic",
            es_password="secret",
        )
        build_es_client(config2)
        call_kwargs = MockES.call_args.kwargs
        assert call_kwargs["hosts"] == ["https://es01:9200", "https://es02:9200"]
        assert call_kwargs["ssl_assert_fingerprint"] == "AA:BB:CC"
        assert call_kwargs["basic_auth"] == ("elastic", "secret")
        assert call_kwargs["verify_certs"] is False
        print(f"  클러스터: nodes={len(call_kwargs['hosts'])}, fingerprint OK, basic_auth OK")

        # Case 3: API Key 우선
        MockES.reset_mock()
        config3 = Config(
            es_nodes=["https://es01:9200"],
            es_fingerprint="AA:BB:CC",
            es_api_key="my-api-key",
            es_username="elastic",  # 무시됨
            es_password="secret",   # 무시됨
        )
        build_es_client(config3)
        call_kwargs = MockES.call_args.kwargs
        assert call_kwargs["api_key"] == "my-api-key"
        assert "basic_auth" not in call_kwargs
        print(f"  API Key 우선: api_key=*** (basic_auth 무시)  OK")

        # Case 4: 클러스터인데 fingerprint 없음 → ValueError
        config4 = Config(es_nodes=["https://es01:9200"])
        try:
            build_es_client(config4)
            assert False, "Should have raised"
        except ValueError as e:
            assert "fingerprint" in str(e).lower()
            print(f"  fingerprint 누락 에러: OK")

        # Case 5: 클러스터인데 인증 없음 → ValueError
        config5 = Config(
            es_nodes=["https://es01:9200"],
            es_fingerprint="AA:BB:CC",
        )
        try:
            build_es_client(config5)
            assert False, "Should have raised"
        except ValueError as e:
            assert "인증" in str(e)
            print(f"  인증 누락 에러: OK")

    print("  PASS\n")


def test_es_indexer_crud():
    """ESIndexer: create_index, ensure_index, bulk_index, finalize, CRUD"""
    print("=" * 60)
    print("[4] ESIndexer — CRUD 메서드")
    print("=" * 60)

    with patch("es_indexer.indexer.AsyncElasticsearch") as MockES:
        mock_es = MockES.return_value
        mock_es.indices = MagicMock()
        mock_es.indices.exists = AsyncMock(return_value=False)
        mock_es.indices.create = AsyncMock()
        mock_es.indices.delete = AsyncMock()
        mock_es.indices.put_settings = AsyncMock()
        mock_es.indices.refresh = AsyncMock()
        mock_es.indices.forcemerge = AsyncMock()

        indexer = ESIndexer("http://localhost:9200", "test_idx")

        # create_index (신규)
        asyncio.run(indexer.create_index())
        mock_es.indices.create.assert_called_once()
        create_kwargs = mock_es.indices.create.call_args.kwargs
        assert create_kwargs["index"] == "test_idx"
        assert create_kwargs["settings"]["refresh_interval"] == "-1"
        print(f"  create_index: refresh=-1, translog=async  OK")

        # create_index (기존 존재 → 삭제 후 재생성)
        mock_es.indices.exists = AsyncMock(return_value=True)
        mock_es.indices.create.reset_mock()
        asyncio.run(indexer.create_index())
        mock_es.indices.delete.assert_called()
        mock_es.indices.create.assert_called()
        print(f"  create_index (기존 삭제): delete + create  OK")

        # ensure_index (이미 존재 → False)
        mock_es.indices.exists = AsyncMock(return_value=True)
        result = asyncio.run(indexer.ensure_index())
        assert result is False
        print(f"  ensure_index (존재): False  OK")

        # ensure_index (없음 → True)
        mock_es.indices.exists = AsyncMock(return_value=False)
        mock_es.indices.create.reset_mock()
        result = asyncio.run(indexer.ensure_index())
        assert result is True
        print(f"  ensure_index (신규): True  OK")

        # finalize (기본 — bulk_config 없음)
        asyncio.run(indexer.finalize())
        put_call = mock_es.indices.put_settings.call_args
        assert "number_of_replicas" not in put_call.kwargs["settings"]
        mock_es.indices.refresh.assert_called()
        mock_es.indices.forcemerge.assert_called()
        print(f"  finalize: put_settings + refresh + forcemerge  OK")

        # create_index(bulk_config) — 벌크 모드 설정 적용
        mock_es.indices.reset_mock()
        mock_es.indices.exists = AsyncMock(return_value=False)
        mock_es.indices.create = AsyncMock()
        bulk_config = {
            "number_of_replicas": 0,
            "flush_threshold_size": "1gb",
            "replicas_restore": 1,
        }
        asyncio.run(indexer.create_index(bulk_config=bulk_config))
        create_kwargs = mock_es.indices.create.call_args.kwargs
        assert create_kwargs["settings"]["number_of_replicas"] == 0
        assert create_kwargs["settings"]["index.translog.flush_threshold_size"] == "1gb"
        print(f"  create_index(bulk_config): replicas=0, flush=1gb  OK")

        # finalize — 벌크 모드 설정 복원
        mock_es.indices.reset_mock()
        mock_es.indices.put_settings = AsyncMock()
        mock_es.indices.refresh = AsyncMock()
        mock_es.indices.forcemerge = AsyncMock()
        asyncio.run(indexer.finalize())
        put_call = mock_es.indices.put_settings.call_args
        restore = put_call.kwargs["settings"]
        assert restore["number_of_replicas"] == 1  # replicas_restore
        assert restore["index.translog.flush_threshold_size"] == "512mb"
        assert restore["index.refresh_interval"] == "1s"
        assert restore["index.translog.durability"] == "request"
        print(f"  finalize(bulk): replicas 복원=1, flush=512mb  OK")

        # 단건 CRUD
        mock_es.index = AsyncMock()
        mock_es.update = AsyncMock()
        mock_es.delete = AsyncMock()
        mock_es.search = AsyncMock(return_value={"hits": {"hits": []}})
        mock_es.get = AsyncMock(return_value={"_source": {"keyword": "テスト"}})
        mock_es.count = AsyncMock(return_value={"count": 42})

        asyncio.run(indexer.index("1", "テスト"))
        mock_es.index.assert_called_with(
            index="test_idx", id="1",
            document={"keyword": "テスト"}, refresh="true"
        )
        print(f"  index('1', 'テスト'): OK")

        asyncio.run(indexer.update("1", {"keyword": "更新"}))
        print(f"  update('1'): OK")

        asyncio.run(indexer.delete("1"))
        print(f"  delete('1'): OK")

        result = asyncio.run(indexer.get("1"))
        assert result == {"keyword": "テスト"}
        print(f"  get('1'): {result}  OK")

        count = asyncio.run(indexer.count())
        assert count == 42
        print(f"  count: {count}  OK")

    print("  PASS\n")


def test_from_config():
    """ESIndexer.from_config: classmethod 팩토리"""
    print("=" * 60)
    print("[5] ESIndexer.from_config — 팩토리")
    print("=" * 60)

    with patch("es_indexer.indexer.AsyncElasticsearch") as MockES:
        config = Config(es_url="http://custom:9200", index_name="my_index")
        indexer = ESIndexer.from_config(config)
        assert indexer.index_name == "my_index"
        print(f"  index_name: {indexer.index_name}  OK")

        # index_name 오버라이드
        indexer2 = ESIndexer.from_config(config, index_name="override")
        assert indexer2.index_name == "override"
        print(f"  index_name override: {indexer2.index_name}  OK")

    print("  PASS\n")


def test_async_failure_logger():
    """AsyncFailureLogger: 비동기 JSONL 실패 기록 (공용 유틸리티)"""
    print("=" * 60)
    print("[6] AsyncFailureLogger — JSONL 실패 기록")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as td:
        fl_path = Path(td) / "failures.jsonl"
        fl = AsyncFailureLogger(fl_path, enabled=True)
        assert fl.count == 0

        async def write_entries():
            await fl.log_failure(100, ConnectionError("Connection refused"),
                                 data_info={"count": 2, "keywords": ["kw1", "kw2"]})
            await fl.log_failure(200, TimeoutError("Timeout"),
                                 data_info={"count": 1, "keywords": ["kw3"]})

        asyncio.run(write_entries())

        assert fl.count == 2
        lines = fl_path.read_text().strip().split("\n")
        assert len(lines) == 2

        r1 = json.loads(lines[0])
        assert r1["batch_id"] == 100
        assert r1["error_type"] == "ConnectionError"
        assert "Connection refused" in r1["error_message"]
        assert r1["count"] == 2
        assert r1["keywords"] == ["kw1", "kw2"]
        print(f"  기록 1: batch_id={r1['batch_id']}, type={r1['error_type']}, count={r1['count']}")

        r2 = json.loads(lines[1])
        assert r2["batch_id"] == 200
        assert r2["error_type"] == "TimeoutError"
        print(f"  기록 2: batch_id={r2['batch_id']}, type={r2['error_type']}")
        print(f"  총 기록: {fl.count}건  OK")

        # disabled 모드 확인
        fl_disabled = AsyncFailureLogger(Path(td) / "none.jsonl", enabled=False)
        asyncio.run(fl_disabled.log_failure(999, RuntimeError("ignored")))
        assert fl_disabled.count == 0
        print(f"  disabled 모드: count={fl_disabled.count}  OK")

    print("  PASS\n")


def test_load_keywords():
    """load_keywords: 텍스트 파일 + Parquet 로드 (공용 유틸리티)"""
    print("=" * 60)
    print("[7] load_keywords — 데이터 로드 (공용 유틸리티)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as td:
        # 텍스트 파일 테스트
        txt_path = Path(td) / "test.txt"
        txt_path.write_text("kw1\nkw2\nkw3\nkw4\nkw5", encoding="utf-8")

        config = Config(keywords_path=txt_path)
        keywords, source = load_keywords(config)
        assert len(keywords) == 5
        assert keywords[0] == "kw1"
        assert "텍스트" in source
        print(f"  텍스트 파일: {len(keywords)}건, source='{source}'  OK")

        # limit 적용
        config_limited = Config(keywords_path=txt_path, limit=3)
        keywords_limited, _ = load_keywords(config_limited)
        assert len(keywords_limited) == 3
        print(f"  텍스트 limit=3: {len(keywords_limited)}건  OK")

        # Parquet 테스트
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq_path = Path(td) / "test.parquet"
        table = pa.table({"keyword": [f"pq_kw{i}" for i in range(50)]})
        pq.write_table(table, pq_path)

        config_pq = Config(parquet_path=pq_path, limit=20)
        keywords_pq, source_pq = load_keywords(config_pq)
        assert len(keywords_pq) == 20
        assert keywords_pq[0] == "pq_kw0"
        assert "Parquet" in source_pq
        print(f"  Parquet limit=20: {len(keywords_pq)}건, source='{source_pq}'  OK")

    print("  PASS\n")


if __name__ == "__main__":
    test_config()
    test_default_schema()
    test_build_es_client()
    test_es_indexer_crud()
    test_from_config()
    test_async_failure_logger()
    test_load_keywords()

    print("=" * 60)
    print("ALL es-indexer EXAMPLES PASSED")
    print("=" * 60)
