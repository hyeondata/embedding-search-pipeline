"""
qdrant_indexer — KServe 임베딩 + Qdrant 인덱싱 + 검색 패키지

사용법:
    from qdrant_indexer import EmbeddingClient, QdrantIndexer, run_indexing, Config, Searcher

    # 인덱싱
    config = Config(limit=100, workers=4)
    run_indexing(config)

    # 검색 (단건)
    searcher = Searcher(config)
    results = searcher.search("東京 ラーメン", top_k=10)

    # 검색 (대용량 배치)
    from qdrant_indexer import run_batch_search
    run_batch_search(config, Path("data/keywords_400k.txt"), top_k=10)

    # 개별 컴포넌트 사용
    embedder = EmbeddingClient("http://localhost:8080", "ruri_v3")
    embeddings = embedder.embed(["東京の天気", "大阪のグルメ"])
"""

from .config import Config
from .embedder import EmbeddingClient
from .indexer import AsyncQdrantIndexer, QdrantIndexer
from .log import get_logger, setup_logging
from .parquet_reader import ParquetReader
from .pipeline import run_indexing
from .retry import FailureLogger, RetryConfig
from .searcher import Searcher, SearchResult, run_batch_search

__all__ = [
    "Config", "EmbeddingClient", "QdrantIndexer", "AsyncQdrantIndexer",
    "run_indexing", "Searcher", "SearchResult", "run_batch_search",
    "setup_logging", "get_logger",
    "ParquetReader", "RetryConfig", "FailureLogger",
]
