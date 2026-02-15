"""Elasticsearch 인덱서 설정"""

from dataclasses import dataclass, field
from pathlib import Path

from kserve_embed_client import BaseConfig

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
class Config(BaseConfig):
    # BaseConfig 기본값 오버라이드 (ES는 대용량 벌크 처리)
    batch_size: int = 500
    workers: int = 8

    # 데이터 소스 — 기본 경로를 패키지 기준으로 오버라이드
    keywords_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "keywords_400k.txt"
    )

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

    # 실패 로깅 — 기본 경로 오버라이드
    failure_log_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "logs" / "failures.jsonl"
    )
