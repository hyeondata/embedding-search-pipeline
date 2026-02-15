"""Qdrant 컬렉션 관리 + 벡터 저장 + 검색 (Sync + Async)"""

import asyncio
import logging
import time

import numpy as np
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance,
    OptimizersConfigDiff,
    PointStruct,
    QueryRequest,
    VectorParams,
)


logger = logging.getLogger(__name__)


class QdrantIndexer:
    """
    Qdrant 벡터 DB 인덱서 (Sync).

    컬렉션 생성 → 벡터 upsert → 유사도 검색 흐름을 캡슐화.
    """

    def __init__(self, url: str, collection_name: str, vector_dim: int):
        self.url = url
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_dim = vector_dim

    # ── 컬렉션 관리 ─────────────────────────────────

    def create_collection(self, bulk_mode: bool = False):
        """
        컬렉션 생성 (이미 존재하면 삭제 후 재생성).

        Args:
            bulk_mode: True면 indexing_threshold=0으로 HNSW 인덱싱 비활성화.
                       대량 업로드 후 finalize()로 인덱싱 활성화 필요.
        """
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)

        kwargs: dict = {
            "collection_name": self.collection_name,
            "vectors_config": VectorParams(
                size=self.vector_dim,
                distance=Distance.COSINE,
            ),
        }

        if bulk_mode:
            kwargs["optimizers_config"] = OptimizersConfigDiff(
                indexing_threshold=0,
            )

        self.client.create_collection(**kwargs)

    def finalize(self, indexing_threshold: int = 20000):
        """
        벌크 업로드 완료 후 HNSW 인덱싱 활성화 + 완료 대기.

        Args:
            indexing_threshold: 복원할 indexing_threshold (기본: 20000, Qdrant 기본값)
        """
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=indexing_threshold,
            ),
        )

        logger.info(
            f"인덱싱 활성화 (indexing_threshold={indexing_threshold}), "
            "완료 대기 중..."
        )

        last_log = time.monotonic()
        while True:
            info = self.client.get_collection(self.collection_name)
            status = info.optimizer_status.status
            if status == "ok":
                break
            # 30초마다 상태 로그
            now = time.monotonic()
            if now - last_log >= 30:
                logger.info(
                    f"인덱싱 진행 중... "
                    f"(vectors={info.points_count:,}, status={status})"
                )
                last_log = now
            time.sleep(2.0)

        logger.info("인덱싱 완료")

    # ── 데이터 저장 ─────────────────────────────────

    def upsert_batch(self, start_id: int, keywords: list[str], embeddings: np.ndarray):
        """키워드 + 임베딩 → Qdrant 포인트로 저장"""
        points = [
            PointStruct(
                id=start_id + i,
                vector=embeddings[i].tolist(),
                payload={"keyword": keywords[i]},
            )
            for i in range(len(keywords))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    # ── 검색 (단건) ───────────────────────────────────

    def search(self, query_vector: list[float], top_k: int = 5):
        """쿼리 벡터로 유사 키워드 검색"""
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )

    # ── 검색 (배치) ───────────────────────────────────

    def search_batch(self, query_vectors: list[list[float]], top_k: int = 5):
        """
        여러 쿼리 벡터를 한 번의 API 호출로 검색 (query_batch_points).

        Args:
            query_vectors: 쿼리 벡터 리스트 (각 768-dim)
            top_k: 각 쿼리당 반환 결과 수

        Returns:
            list[QueryResponse] — 각 쿼리에 대한 검색 결과 리스트
        """
        requests = [
            QueryRequest(query=vec, limit=top_k, with_payload=True)
            for vec in query_vectors
        ]
        return self.client.query_batch_points(
            collection_name=self.collection_name,
            requests=requests,
        )

    # ── 상태 확인 ───────────────────────────────────

    @property
    def count(self) -> int:
        return self.client.get_collection(self.collection_name).points_count


class AsyncQdrantIndexer:
    """
    Qdrant 벡터 DB 인덱서 (Async).

    asyncio 이벤트 루프에서 직접 I/O를 다중화하여
    ThreadPoolExecutor 없이 동시 검색을 수행.
    """

    def __init__(self, url: str, collection_name: str, vector_dim: int):
        self.client = AsyncQdrantClient(url=url)
        self.collection_name = collection_name
        self.vector_dim = vector_dim

    async def search(self, query_vector: list[float], top_k: int = 5):
        """단건 비동기 검색"""
        return await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )

    async def search_batch(self, query_vectors: list[list[float]], top_k: int = 5):
        """
        배치 비동기 검색 (query_batch_points).

        1번의 HTTP 호출로 N개 쿼리를 동시에 검색.
        """
        requests = [
            QueryRequest(query=vec, limit=top_k, with_payload=True)
            for vec in query_vectors
        ]
        return await self.client.query_batch_points(
            collection_name=self.collection_name,
            requests=requests,
        )

    async def close(self):
        await self.client.close()

    @property
    def count(self) -> int:
        """동기적으로 카운트 조회 (초기화 시 사용)"""
        sync_client = QdrantClient(url=self.client._client.rest_uri)
        c = sync_client.get_collection(self.collection_name).points_count
        sync_client.close()
        return c
