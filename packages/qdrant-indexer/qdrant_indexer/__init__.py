"""
qdrant-indexer — KServe 임베딩 + Qdrant 벡터 인덱싱 + 검색 패키지

사용법:
    from qdrant_indexer import QdrantIndexer, Searcher, run_indexing, Config

    # Bulk 인덱싱 (컬렉션 재생성)
    config = Config(limit=100, workers=4)
    run_indexing(config)

    # Realtime 인덱싱 (기존 보존 + 증분)
    from qdrant_indexer import run_realtime
    run_realtime(config)

    # 검색 (단건)
    searcher = Searcher(config)
    results = searcher.search("東京 ラーメン", top_k=10)

    # 검색 (대용량 배치)
    from qdrant_indexer import run_batch_search
    run_batch_search(config, Path("data/keywords_400k.txt"), top_k=10)

    # 임베딩 클라이언트 (kserve-embed-client에서 re-export)
    from kserve_embed_client import EmbeddingClient
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
