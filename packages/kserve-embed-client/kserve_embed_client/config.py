"""공용 파이프라인 설정 베이스 클래스"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseConfig:
    """
    파이프라인 공통 설정.

    하위 클래스에서 상속하여 백엔드-specific 필드를 추가:

        @dataclass
        class Config(BaseConfig):
            qdrant_url: str = "http://localhost:6333"
            collection_name: str = "keywords"
            ...

    모든 필드에 기본값이 있으므로 dataclass 상속 시 필드 순서 문제 없음.
    """

    # 데이터 소스
    keywords_path: Path | None = None
    limit: int = 0  # 0 = 전체

    # Parquet 지원
    parquet_path: Path | None = None
    parquet_chunk_size: int = 10000
    parquet_text_column: str = "keyword"

    # 처리
    batch_size: int = 64
    workers: int = 4

    # 재시도
    max_retries: int = 3
    retry_backoff: float = 1.0
    retry_exponential: bool = True
    retry_max_backoff: float = 60.0

    # 실패 로깅
    log_failures: bool = True
    failure_log_path: Path | None = None
