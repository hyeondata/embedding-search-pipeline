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

from kserve_embed_client import (
    EmbeddingClient,
    FailureLogger,
    ParquetReader,
    RetryConfig,
    get_logger,
    setup_logging,
    validate_parquet_columns,
    with_retry,
)


def test_embedding_client():
    """EmbeddingClient: Mock HTTP로 KServe V2 응답 파싱 검증"""
    print("=" * 60)
    print("[1] EmbeddingClient — Mock embed()")
    print("=" * 60)

    client = EmbeddingClient("http://localhost:8080", "ruri_v3")
    print(f"  URL: {client.url}")

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

        # 검증: 올바른 payload 전송
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


def test_embedding_client_error_handling():
    """EmbeddingClient: HTTP 에러 처리"""
    print("=" * 60)
    print("[5] EmbeddingClient — 에러 핸들링")
    print("=" * 60)

    client = EmbeddingClient("http://localhost:8080", "ruri_v3")

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

    # 빈 텍스트 리스트
    mock_empty_resp = MagicMock()
    mock_empty_resp.json.return_value = {
        "outputs": [{"name": "embedding", "shape": [0, 768], "data": []}]
    }
    mock_empty_resp.raise_for_status = MagicMock()

    with patch.object(client.session, "post", return_value=mock_empty_resp):
        result = client.embed([])
        assert result.shape == (0, 768)
        print(f"  빈 입력 처리: shape={result.shape}  OK")

    print("  PASS\n")


if __name__ == "__main__":
    test_embedding_client()
    test_parquet_reader()
    test_logging()
    test_retry()
    test_embedding_client_error_handling()

    print("=" * 60)
    print("ALL kserve-embed-client EXAMPLES PASSED")
    print("=" * 60)
