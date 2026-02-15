#!/usr/bin/env python3
"""
kserve-embed-client 패키지 예시 — 외부 서비스 없이 전체 코드 경로 검증

테스트 항목:
  1. EmbeddingClient 생성 + Mock HTTP 응답으로 embed() 호출
  2. ParquetReader로 실제 Parquet 파일 읽기
  3. setup_logging + get_logger 파일/콘솔 듀얼 로깅
  4. RetryConfig + with_retry 재시도 동작
  5. FailureLogger JSONL 기록
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from kserve_embed_client import EmbeddingClient
from pipeline_commons import (
    FailureLogger,
    ParquetReader,
    RetryConfig,
    get_logger,
    setup_logging,
    validate_parquet_columns,
    with_retry,
)


def test_embedding_client_text_mode():
    """EmbeddingClient: 텍스트 모드 (tokenizer_name=None) — Mock HTTP로 V2 BYTES 검증"""
    print("=" * 60)
    print("[1] EmbeddingClient — 텍스트 모드 (tokenizer_name=None)")
    print("=" * 60)

    client = EmbeddingClient("http://localhost:8080", "ruri_v3", tokenizer_name=None)
    assert client.tokenizer is None
    print(f"  URL: {client.url}")
    print(f"  tokenizer: None (텍스트 모드)")

    # Mock KServe V2 응답: 3개 텍스트 → (3, 768) 임베딩
    mock_embeddings = np.random.randn(3, 768).tolist()
    flat_data = [val for row in mock_embeddings for val in row]
    mock_response = {
        "outputs": [{
            "name": "embedding",
            "shape": [3, 768],
            "datatype": "FP32",
            "data": flat_data,
        }]
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_response
    mock_resp.raise_for_status = MagicMock()

    with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
        texts = ["東京の天気", "大阪のグルメ", "京都の寺院"]
        result = client.embed(texts)

        # 검증: BYTES payload 전송
        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert payload["inputs"][0]["name"] == "text"
        assert payload["inputs"][0]["shape"] == [3]
        assert payload["inputs"][0]["datatype"] == "BYTES"
        assert payload["inputs"][0]["data"] == texts
        print(f"  요청 payload: shape={payload['inputs'][0]['shape']}, datatype=BYTES")

        # 검증: 올바른 shape 복원
        assert result.shape == (3, 768)
        assert isinstance(result, np.ndarray)
        print(f"  응답 shape: {result.shape}")
        print(f"  dtype: {result.dtype}")
        print(f"  첫 벡터 norm: {np.linalg.norm(result[0]):.4f}")

    print("  PASS\n")


def test_embedding_client_tokenizer_mode():
    """EmbeddingClient: 토크나이저 모드 — input_ids/attention_mask INT64 전송 검증"""
    print("=" * 60)
    print("[1b] EmbeddingClient — 토크나이저 모드")
    print("=" * 60)

    # Mock AutoTokenizer를 사용하여 transformers 의존 없이 테스트
    mock_tokenizer = MagicMock()
    mock_tokens = {
        "input_ids": np.array([[1, 2, 3, 0], [1, 4, 5, 6]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=np.int64),
    }
    mock_tokenizer.return_value = mock_tokens

    client = EmbeddingClient("http://localhost:8080", "ruri_v3", tokenizer_name=None)
    # 수동으로 tokenizer 설정 (mock)
    client.tokenizer = mock_tokenizer

    # Mock KServe V2 응답
    mock_embeddings = np.random.randn(2, 768).tolist()
    flat_data = [val for row in mock_embeddings for val in row]
    mock_response = {
        "outputs": [{
            "name": "sentence_embedding",
            "shape": [2, 768],
            "datatype": "FP32",
            "data": flat_data,
        }]
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_response
    mock_resp.raise_for_status = MagicMock()

    with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
        texts = ["東京の天気", "大阪のグルメ"]
        result = client.embed(texts)

        # 검증: INT64 payload 전송 (input_ids + attention_mask)
        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert len(payload["inputs"]) == 2
        assert payload["inputs"][0]["name"] == "input_ids"
        assert payload["inputs"][0]["datatype"] == "INT64"
        assert payload["inputs"][0]["shape"] == [2, 4]
        assert payload["inputs"][1]["name"] == "attention_mask"
        assert payload["inputs"][1]["datatype"] == "INT64"
        assert payload["inputs"][1]["shape"] == [2, 4]
        print(f"  input_ids: shape={payload['inputs'][0]['shape']}, datatype=INT64")
        print(f"  attention_mask: shape={payload['inputs'][1]['shape']}, datatype=INT64")

        # 검증: 올바른 shape 복원
        assert result.shape == (2, 768)
        print(f"  응답 shape: {result.shape}")

    # V1 + tokenizer → ValueError
    try:
        EmbeddingClient("http://host", "m", protocol="v1", tokenizer_name="some-model")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "V2" in str(e)
        print(f"  V1 + tokenizer → ValueError: OK")

    print("  PASS\n")


def test_parquet_reader():
    """ParquetReader: 임시 Parquet 파일 생성 → 청크 읽기"""
    print("=" * 60)
    print("[2] ParquetReader — Parquet 청크 스트리밍")
    print("=" * 60)

    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as td:
        # 테스트 Parquet 생성 (100행)
        keywords = [f"テスト キーワード {i}" for i in range(100)]
        table = pa.table({"keyword": keywords, "category": [f"cat_{i%5}" for i in range(100)]})
        parquet_path = Path(td) / "test.parquet"
        pq.write_table(table, parquet_path)
        print(f"  생성: {parquet_path.name} (100행)")

        # 전체 읽기 (chunk_size=30)
        reader = ParquetReader(parquet_path, chunk_size=30, text_column="keyword")
        assert reader.total_rows == 100
        print(f"  total_rows: {reader.total_rows}")
        print(f"  num_chunks: {reader.num_chunks}")

        chunks = list(reader.iter_chunks())
        total_read = sum(len(kw) for _, kw in chunks)
        assert total_read == 100
        print(f"  실제 읽은 행: {total_read}")
        print(f"  청크 수: {len(chunks)} (30+30+30+10)")
        print(f"  첫 청크: batch_id={chunks[0][0]}, len={len(chunks[0][1])}")
        print(f"  마지막 청크: batch_id={chunks[-1][0]}, len={len(chunks[-1][1])}")

        # limit 테스트
        reader_limited = ParquetReader(parquet_path, chunk_size=30, text_column="keyword", limit=50)
        assert reader_limited.total_rows == 50
        chunks_limited = list(reader_limited.iter_chunks())
        total_limited = sum(len(kw) for _, kw in chunks_limited)
        assert total_limited == 50
        print(f"  limit=50 테스트: {total_limited}행 읽음  OK")

        # 폴더 읽기 테스트
        folder = Path(td) / "multi"
        folder.mkdir()
        for i in range(3):
            sub = pa.table({"keyword": [f"file{i}_kw{j}" for j in range(20)]})
            pq.write_table(sub, folder / f"part_{i:03d}.parquet")

        reader_dir = ParquetReader(folder, chunk_size=25, text_column="keyword")
        assert reader_dir.total_rows == 60
        assert reader_dir.is_directory is True
        assert len(reader_dir.parquet_files) == 3
        dir_chunks = list(reader_dir.iter_chunks())
        dir_total = sum(len(kw) for _, kw in dir_chunks)
        assert dir_total == 60
        print(f"  폴더 읽기 (3파일, 60행): {dir_total}행  OK")

        # validate_parquet_columns 테스트
        validate_parquet_columns(parquet_path, "keyword")
        print(f"  컬럼 검증 'keyword': OK")

        try:
            validate_parquet_columns(parquet_path, "nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"  컬럼 검증 'nonexistent': ValueError 발생  OK")

    print("  PASS\n")


def test_logging():
    """setup_logging + get_logger: 패키지별 분리 로깅"""
    print("=" * 60)
    print("[3] Logging — 패키지별 분리 + 파일 기록")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as td:
        log_file = Path(td) / "test.log"

        # 패키지별 로거 생성
        setup_logging("test_pkg_a", log_file=log_file)
        logger_a = get_logger("test_pkg_a", "module1")
        logger_b = get_logger("test_pkg_b", "module2")

        assert logger_a.name == "test_pkg_a.module1"
        assert logger_b.name == "test_pkg_b.module2"
        print(f"  logger_a.name: {logger_a.name}")
        print(f"  logger_b.name: {logger_b.name}")

        # Rich markup이 파일에는 plain text로 기록되는지 확인
        logger_a.info("[bold green]볼드 그린 테스트[/bold green]")

        content = log_file.read_text()
        assert "볼드 그린 테스트" in content
        assert "[bold green]" not in content  # Rich markup 제거됨
        assert "test_pkg_a.module1" in content
        print(f"  파일 로그 Rich markup 제거: OK")
        print(f"  파일 로그 패키지명 포함: OK")

        # pkg_name 없이 root logger
        root_logger = get_logger("test_pkg_a")
        assert root_logger.name == "test_pkg_a"
        print(f"  root logger: {root_logger.name}  OK")

    print("  PASS\n")


def test_retry():
    """with_retry + RetryConfig: 재시도 동작 검증"""
    print("=" * 60)
    print("[4] Retry — 재시도 + 지수 백오프")
    print("=" * 60)

    # Case 1: 2번 실패 후 3번째 성공
    call_count = 0

    @with_retry(retry_config=RetryConfig(max_retries=3, initial_backoff=0.01))
    def succeed_on_third():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError(f"attempt {call_count}")
        return "ok"

    result = succeed_on_third()
    assert result == "ok"
    assert call_count == 3
    print(f"  Case 1 (3번째 성공): call_count={call_count}  OK")

    # Case 2: 모든 재시도 실패 → 예외 발생
    fail_count = 0

    @with_retry(retry_config=RetryConfig(max_retries=2, initial_backoff=0.01))
    def always_fail():
        nonlocal fail_count
        fail_count += 1
        raise TimeoutError(f"timeout #{fail_count}")

    try:
        always_fail()
        assert False, "Should have raised"
    except TimeoutError as e:
        assert fail_count == 2
        print(f"  Case 2 (모두 실패): fail_count={fail_count}, error='{e}'  OK")

    # Case 3: 선형 백오프 (exponential=False)
    linear_count = 0

    @with_retry(retry_config=RetryConfig(max_retries=3, initial_backoff=0.01, exponential=False))
    def linear_succeed():
        nonlocal linear_count
        linear_count += 1
        if linear_count < 2:
            raise ValueError("retry me")
        return "linear_ok"

    assert linear_succeed() == "linear_ok"
    print(f"  Case 3 (선형 백오프): call_count={linear_count}  OK")

    # Case 4: with_retry + batch_id_fn + FailureLogger
    with tempfile.TemporaryDirectory() as td:
        fl = FailureLogger(Path(td) / "failures.jsonl", enabled=True)
        batch_call = 0

        @with_retry(
            retry_config=RetryConfig(max_retries=2, initial_backoff=0.01),
            failure_logger=fl,
            batch_id_fn=lambda args, kwargs: args[0],
            data_info_fn=lambda args, kwargs: {"size": len(args[1])},
        )
        def batch_process(batch_id, items):
            nonlocal batch_call
            batch_call += 1
            raise RuntimeError("DB down")

        try:
            batch_process(42, ["a", "b", "c"])
        except RuntimeError:
            pass

        failures = Path(td) / "failures.jsonl"
        lines = failures.read_text().strip().split("\n")
        assert len(lines) == 2  # 2번 재시도 → 2번 기록
        record = json.loads(lines[0])
        assert record["batch_id"] == 42
        assert record["size"] == 3
        print(f"  Case 4 (FailureLogger): {len(lines)}건 기록, batch_id={record['batch_id']}  OK")

    print("  PASS\n")


def test_embedding_client_v1_protocol():
    """EmbeddingClient: V1 프로토콜 (TFServing 호환) 지원"""
    print("=" * 60)
    print("[5] EmbeddingClient — V1 Protocol")
    print("=" * 60)

    # V1 클라이언트 생성 (V1은 텍스트 모드만 지원)
    client = EmbeddingClient("http://localhost:8080", "my_model", protocol="v1", tokenizer_name=None)
    assert client.protocol == "v1"
    assert client.url == "http://localhost:8080/v1/models/my_model:predict"
    print(f"  URL: {client.url}  OK")

    # V1 응답 Mock: predictions는 이미 2D 배열
    mock_embeddings = np.random.randn(3, 768).tolist()
    mock_response = {"predictions": mock_embeddings}

    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_response
    mock_resp.raise_for_status = MagicMock()

    with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
        texts = ["東京の天気", "大阪のグルメ", "京都の寺院"]
        result = client.embed(texts)

        # 검증: V1 payload 형식
        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert "instances" in payload
        assert len(payload["instances"]) == 3
        assert payload["instances"][0] == {"text": "東京の天気"}
        assert payload["instances"][2] == {"text": "京都の寺院"}
        print(f"  V1 payload: instances={len(payload['instances'])}건  OK")

        # 검증: 올바른 shape
        assert result.shape == (3, 768)
        assert isinstance(result, np.ndarray)
        print(f"  응답 shape: {result.shape}  OK")

    # V2와 V1 비교 — 같은 base_url, 다른 프로토콜
    v2 = EmbeddingClient("http://host:8080", "ruri_v3", tokenizer_name=None)
    v1 = EmbeddingClient("http://host:8080", "ruri_v3", protocol="v1", tokenizer_name=None)
    assert v2.url == "http://host:8080/v2/models/ruri_v3/infer"
    assert v1.url == "http://host:8080/v1/models/ruri_v3:predict"
    assert v2.protocol == "v2"
    assert v1.protocol == "v1"
    print(f"  V2 URL: {v2.url}")
    print(f"  V1 URL: {v1.url}")

    # trailing slash 처리 — http:// 이후에 // 없어야 함
    v1_slash = EmbeddingClient("http://host:8080/", "m", protocol="v1", tokenizer_name=None)
    path_part = v1_slash.url.split("://", 1)[1]
    assert "//" not in path_part
    print(f"  trailing slash 정리: {v1_slash.url}  OK")

    # 잘못된 프로토콜
    try:
        EmbeddingClient("http://host", "m", protocol="v3", tokenizer_name=None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "v3" in str(e)
        print(f"  잘못된 프로토콜 거부: ValueError  OK")

    # embed_with_prefix도 V1에서 동작
    with patch.object(client.session, "post", return_value=mock_resp) as mock_post_v1:
        result = client.embed_with_prefix(["テスト"], prefix="検索クエリ: ")

        call_args = mock_post_v1.call_args
        payload = call_args.kwargs["json"]
        assert payload["instances"][0] == {"text": "検索クエリ: テスト"}
        print(f"  embed_with_prefix + V1: OK")

    # timeout 커스텀
    client_fast = EmbeddingClient("http://host:8080", "m", protocol="v2", timeout=30, tokenizer_name=None)
    assert client_fast.timeout == 30
    print(f"  custom timeout: {client_fast.timeout}s  OK")

    print("  PASS\n")


def test_embedding_client_error_handling():
    """EmbeddingClient: HTTP 에러 처리"""
    print("=" * 60)
    print("[6] EmbeddingClient — 에러 핸들링")
    print("=" * 60)

    client = EmbeddingClient("http://localhost:8080", "ruri_v3", tokenizer_name=None)

    # HTTP 500 에러
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("500 Internal Server Error")

    with patch.object(client.session, "post", return_value=mock_resp):
        try:
            client.embed(["test"])
            assert False, "Should have raised"
        except Exception as e:
            assert "500" in str(e)
            print(f"  HTTP 500 에러 전파: OK")

    # 빈 텍스트 리스트 (V2)
    mock_empty_resp = MagicMock()
    mock_empty_resp.json.return_value = {
        "outputs": [{"name": "embedding", "shape": [0, 768], "data": []}]
    }
    mock_empty_resp.raise_for_status = MagicMock()

    with patch.object(client.session, "post", return_value=mock_empty_resp):
        result = client.embed([])
        assert result.shape == (0, 768)
        print(f"  빈 입력 처리 (V2): shape={result.shape}  OK")

    # 빈 텍스트 리스트 (V1)
    client_v1 = EmbeddingClient("http://localhost:8080", "m", protocol="v1", tokenizer_name=None)
    mock_empty_v1 = MagicMock()
    mock_empty_v1.json.return_value = {"predictions": []}
    mock_empty_v1.raise_for_status = MagicMock()

    with patch.object(client_v1.session, "post", return_value=mock_empty_v1):
        result = client_v1.embed([])
        assert result.shape == (0,)  # np.array([]) → (0,)
        print(f"  빈 입력 처리 (V1): shape={result.shape}  OK")

    print("  PASS\n")


if __name__ == "__main__":
    test_embedding_client_text_mode()
    test_embedding_client_tokenizer_mode()
    test_parquet_reader()
    test_logging()
    test_retry()
    test_embedding_client_v1_protocol()
    test_embedding_client_error_handling()

    print("=" * 60)
    print("ALL kserve-embed-client EXAMPLES PASSED")
    print("=" * 60)
