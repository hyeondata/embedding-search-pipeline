"""설정 정의"""

from dataclasses import dataclass, field
from pathlib import Path

from kserve_embed_client import BaseConfig


@dataclass
class Config(BaseConfig):
    # 데이터 소스 — 기본 경로를 패키지 기준으로 오버라이드
    keywords_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "keywords_400k.txt"
    )

    # KServe 임베딩
    kserve_url: str = "http://localhost:8080"
    model_name: str = "ruri_v3"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "keywords"
    vector_dim: int = 768

    # 벌크 최적화 설정 (bulk_mode=True 일 때만 적용)
    bulk_indexing_threshold: int = 20000  # finalize 시 복원할 indexing_threshold

    # 실패 로깅 — 기본 경로 오버라이드
    failure_log_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "logs" / "failures.jsonl"
    )
