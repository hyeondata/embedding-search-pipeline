"""Elasticsearch 인덱스 관리 + 벌크/실시간 인덱싱"""

from __future__ import annotations

from elasticsearch import AsyncElasticsearch, NotFoundError
from elasticsearch.helpers import async_bulk

from .config import Config, DEFAULT_SCHEMA


def build_es_client(config: Config) -> AsyncElasticsearch:
    """Config 기반으로 AsyncElasticsearch 클라이언트를 생성.

    - 단일 노드 (HTTP): es_url 사용 — fingerprint 불필요
    - 클러스터 (HTTPS): es_nodes 사용 — fingerprint 필수

    es_nodes를 지정하면 HTTPS 클러스터로 간주하며,
    es_fingerprint가 반드시 함께 제공되어야 합니다.

    Examples:
        # 로컬 개발 (기존과 동일)
        config = Config(es_url="http://localhost:9200")

        # ES 9 클러스터 + fingerprint (필수)
        config = Config(
            es_nodes=["https://es01:9200", "https://es02:9200"],
            es_fingerprint="B1:2A:...:CF",
            es_username="elastic",
            es_password="changeme",
        )
    """
    hosts = config.es_nodes or [config.es_url]
    is_cluster = config.es_nodes is not None

    # 클러스터 모드: fingerprint 필수 + 인증 필수
    if is_cluster:
        if not config.es_fingerprint:
            raise ValueError(
                "--es_fingerprint 필수: ES 9 클러스터 연결에는 "
                "TLS 인증서 fingerprint가 필요합니다.\n"
                "  확인: docker exec es01 "
                "elasticsearch-reset-password -u elastic --url https://localhost:9200"
            )
        if not config.es_api_key and not (config.es_username and config.es_password):
            raise ValueError(
                "인증 정보 필수: --es_api_key 또는 "
                "--es_username + --es_password를 지정하세요."
            )

    kwargs: dict = {"hosts": hosts}

    # 인증: API Key 우선, 없으면 Basic Auth
    if config.es_api_key:
        kwargs["api_key"] = config.es_api_key
    elif config.es_username and config.es_password:
        kwargs["basic_auth"] = (config.es_username, config.es_password)

    # TLS: fingerprint 적용
    if config.es_fingerprint:
        kwargs["ssl_assert_fingerprint"] = config.es_fingerprint
        kwargs["verify_certs"] = False  # fingerprint가 CA 체인 검증을 대체

    return AsyncElasticsearch(**kwargs)


class ESIndexer:
    """
    Elasticsearch 인덱서.

    두 가지 모드를 지원:
      - Bulk: create_index → bulk_index → finalize (초기 대량 적재)
      - Realtime: ensure_index → index/update/delete (운영 중 CRUD)
    """

    def __init__(self, es_url: str, index_name: str):
        self.es = AsyncElasticsearch(es_url)
        self.index_name = index_name

    @classmethod
    def from_config(cls, config: Config, index_name: str | None = None) -> ESIndexer:
        """Config 객체로 클러스터 연결이 포함된 ESIndexer 생성.

        기존 __init__(es_url, index_name)과 하위 호환을 유지하면서
        fingerprint, basic_auth, 다중 노드 등을 지원.
        """
        instance = cls.__new__(cls)
        instance.es = build_es_client(config)
        instance.index_name = index_name or config.index_name
        return instance

    # ================================================================
    # 인덱스 관리
    # ================================================================

    async def create_index(self, schema: dict | None = None):
        """
        인덱스 생성 (이미 존재하면 삭제 후 재생성).
        벌크 최적화 설정(refresh=-1, translog=async)을 자동 적용.
        """
        if await self.es.indices.exists(index=self.index_name):
            await self.es.indices.delete(index=self.index_name)

        schema = schema or DEFAULT_SCHEMA
        settings = {**schema.get("settings", {})}
        settings.setdefault("refresh_interval", "-1")
        settings.setdefault("index.translog.durability", "async")
        settings.setdefault("index.translog.sync_interval", "30s")

        await self.es.indices.create(
            index=self.index_name,
            settings=settings,
            mappings=schema.get("mappings", {}),
        )

    async def ensure_index(self, schema: dict | None = None) -> bool:
        """
        인덱스가 없을 때만 생성 (기존 데이터 보존).
        벌크 최적화 설정을 적용하지 않음 — 실시간 모드용.

        Returns: True면 새로 생성됨, False면 이미 존재.
        """
        if await self.es.indices.exists(index=self.index_name):
            return False

        schema = schema or DEFAULT_SCHEMA
        await self.es.indices.create(
            index=self.index_name,
            settings=schema.get("settings", {}),
            mappings=schema.get("mappings", {}),
        )
        return True

    # ================================================================
    # Bulk 인덱싱 — Bulk / Realtime 공통
    # ================================================================

    async def bulk_index(self, start_id: int, keywords: list[str]):
        """키워드 리스트 → ES 도큐먼트로 벌크 저장"""
        actions = [
            {
                "_index": self.index_name,
                "_id": str(start_id + i),
                "_source": {"keyword": keywords[i]},
            }
            for i in range(len(keywords))
        ]
        success, errors = await async_bulk(
            self.es, actions, chunk_size=len(actions), raise_on_error=False
        )
        if errors:
            raise RuntimeError(f"Bulk index errors: {len(errors)} failures")
        return success

    async def finalize(self):
        """Bulk 완료 후: refresh 복원 + translog 동기화 + force merge"""
        await self.es.indices.put_settings(
            index=self.index_name,
            settings={
                "index.refresh_interval": "1s",
                "index.translog.durability": "request",
            },
        )
        await self.es.indices.refresh(index=self.index_name)
        await self.es.indices.forcemerge(
            index=self.index_name, max_num_segments=1
        )

    # ================================================================
    # Realtime CRUD — 단건 조작
    # ================================================================

    async def index(self, doc_id: str, keyword: str, refresh: bool = True):
        """단일 문서 인덱싱 (upsert). refresh=True면 즉시 검색 가능."""
        await self.es.index(
            index=self.index_name,
            id=doc_id,
            document={"keyword": keyword},
            refresh="true" if refresh else "false",
        )

    async def update(self, doc_id: str, fields: dict, refresh: bool = True):
        """문서 부분 업데이트. fields의 키-값만 변경."""
        await self.es.update(
            index=self.index_name,
            id=doc_id,
            doc=fields,
            refresh="true" if refresh else "false",
        )

    async def delete(self, doc_id: str, refresh: bool = True):
        """문서 삭제. 존재하지 않으면 무시."""
        try:
            await self.es.delete(
                index=self.index_name,
                id=doc_id,
                refresh="true" if refresh else "false",
            )
        except NotFoundError:
            pass

    # ================================================================
    # 검색
    # ================================================================

    async def search(self, query: str, size: int = 10):
        """keyword 필드에 대한 전문 검색"""
        return await self.es.search(
            index=self.index_name,
            query={"match": {"keyword": query}},
            size=size,
        )

    async def get(self, doc_id: str) -> dict | None:
        """ID로 문서 조회. 없으면 None."""
        try:
            result = await self.es.get(index=self.index_name, id=doc_id)
            return result["_source"]
        except NotFoundError:
            return None

    # ================================================================
    # 상태 확인
    # ================================================================

    async def count(self) -> int:
        result = await self.es.count(index=self.index_name)
        return result["count"]

    async def refresh(self):
        """수동 리프레시 — 모든 pending 문서를 검색 가능하게"""
        await self.es.indices.refresh(index=self.index_name)

    async def close(self):
        await self.es.close()
