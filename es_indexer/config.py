"""Elasticsearch 인덱서 설정"""

from dataclasses import dataclass, field
from pathlib import Path

# ── 커스텀 스키마가 없을 때 사용되는 기본 스키마 ──
DEFAULT_SCHEMA = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "keyword": {
                "type": "text",
                "analyzer": "standard",
                "fields": {"raw": {"type": "keyword"}},
            },
        }
    },
}


@dataclass
class Config:
    # 데이터 소스
    keywords_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "keywords_400k.txt"
    )
    limit: int = 0  # 0 = 전체

    # Parquet 지원
    parquet_path: Path | None = None  # Parquet 파일 또는 디렉토리 경로 (설정 시 keywords_path 무시)
    parquet_chunk_size: int = 10000   # Parquet 청크 크기 (행 수)
    parquet_text_column: str = "keyword"  # 텍스트가 들어있는 컬럼명

    # Elasticsearch 연결
    es_url: str = "http://localhost:9200"
    es_nodes: list[str] | None = None       # 클러스터 노드 목록 (설정 시 es_url 무시)
    es_fingerprint: str | None = None       # ES 9 TLS 인증서 SHA-256 fingerprint (클러스터 시 필수)
    es_username: str | None = None          # Basic Auth 사용자명
    es_password: str | None = None          # Basic Auth 비밀번호
    es_api_key: str | None = None           # API Key (basic_auth 대신 사용 가능)

    # 인덱스
    index_name: str = "keywords"
    schema: dict | None = None  # None → DEFAULT_SCHEMA 사용

    # 처리
    batch_size: int = 500   # ES bulk 배치 크기
    workers: int = 8        # 동시 코루틴 수

    # 재시도 / 실패 처리
    max_retries: int = 3            # 배치 실패 시 최대 재시도 횟수 (0=재시도 없음)
    retry_delay: float = 1.0        # 첫 재시도 대기 시간 (초, 이후 지수 백오프 ×2)
    dead_letter_path: Path | None = None  # 실패 문서 JSONL 경로 (None=logs/ 에 자동 생성)
