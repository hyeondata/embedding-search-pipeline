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


def build_default_schema(source_key: str = "keyword") -> dict:
    """source_key에 맞는 기본 스키마를 동적으로 생성.

    source_key="keyword"이면 DEFAULT_SCHEMA와 동일한 결과를 반환.
    """
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                source_key: {
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
    source_key: str = "keyword"  # _source에서 메인 텍스트 필드 이름

    # 벌크 최적화 설정 (bulk_mode=True 일 때만 적용)
    bulk_replicas: int = 0              # 벌크 중 replicas (0=복제 비활성)
    bulk_replicas_restore: int = 0      # finalize 시 복원할 replicas 수
    bulk_flush_threshold: str = "1gb"   # translog.flush_threshold_size (기본 512mb → 1gb)

    # 실패 로깅 — 기본 경로 오버라이드
    failure_log_path: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "logs" / "failures.jsonl"
    )
