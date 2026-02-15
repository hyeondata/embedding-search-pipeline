"""
qdrant-indexer — Qdrant 벡터 인덱싱 + 검색 패키지

사용법:
    from kserve_embed_client import EmbeddingClient, RURI_QUERY_PREFIX
    from qdrant_indexer import QdrantIndexer, Searcher, run_indexing, Config

    # 임베딩 클라이언트 생성 (별도 패키지)
    client = EmbeddingClient("http://localhost:8000", "ruri_v3")

    # Bulk 인덱싱 (embed_fn 주입)
    config = Config(limit=100, workers=4)
    run_indexing(config, embed_fn=client.embed, query_prefix=RURI_QUERY_PREFIX)

    # 검색 (embed_fn 주입)
    searcher = Searcher(config, embed_fn=client.embed, query_prefix=RURI_QUERY_PREFIX)
    results = searcher.search("東京 ラーメン", top_k=10)
"""

from .config import Config
from .indexer import AsyncQdrantIndexer, QdrantIndexer
from .pipeline import run_indexing, run_realtime
from .searcher import Searcher, SearchResult, run_batch_search

__all__ = [
    "Config",
    "QdrantIndexer", "AsyncQdrantIndexer",
    "run_indexing", "run_realtime",
    "Searcher", "SearchResult", "run_batch_search",
]
