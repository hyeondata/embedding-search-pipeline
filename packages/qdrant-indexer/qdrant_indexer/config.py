"""설정 정의"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # 데이터 소스
    keywords_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "keywords_400k.txt"
    )
    limit: int = 0  # 0 = 전체

    # Parquet 지원
    parquet_path: Path | None = None  # Parquet 파일 경로 (설정 시 keywords_path 무시)
    parquet_chunk_size: int = 10000   # Parquet 청크 크기 (행 수)
    parquet_text_column: str = "keyword"  # 텍스트가 들어있는 컬럼명

    # KServe 임베딩
    kserve_url: str = "http://localhost:8080"
    model_name: str = "ruri_v3"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "keywords"
    vector_dim: int = 768

    # 처리
    batch_size: int = 64
    workers: int = 4

    # 재시도
    max_retries: int = 3              # 배치당 최대 재시도 횟수
    retry_backoff: float = 1.0        # 초기 대기 시간 (초)
    retry_exponential: bool = True    # 지수 백오프 사용 (1s → 2s → 4s → ...)
    retry_max_backoff: float = 60.0   # 최대 대기 시간 (초)

    # 실패 로깅
    log_failures: bool = True         # 실패한 배치를 파일에 기록
    failure_log_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "logs" / "failures.jsonl"
    )
