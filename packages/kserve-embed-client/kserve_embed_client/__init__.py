"""
kserve-embed-client — KServe V2 임베딩 클라이언트 + 공용 유틸리티

사용법:
    from kserve_embed_client import EmbeddingClient

    client = EmbeddingClient("http://localhost:8080", "ruri_v3")
    embeddings = client.embed(["東京の天気", "大阪のグルメ"])
    # → (2, 768) numpy array

    # 공용 유틸리티
    from kserve_embed_client import (
        BaseConfig, load_keywords, batch_iter,
        PipelineStats, timer,
        RURI_QUERY_PREFIX, RURI_DOCUMENT_PREFIX,
        RetryConfig, FailureLogger, AsyncFailureLogger,
        with_retry, async_with_retry,
    )
"""

from .config import BaseConfig
from .data import batch_iter, load_keywords
from .embedder import (
    RURI_DOCUMENT_PREFIX,
    RURI_ENCODE_PREFIX,
    RURI_QUERY_PREFIX,
    SUPPORTED_PROTOCOLS,
    EmbeddingClient,
)
from .log import get_logger, setup_logging
from .parquet_reader import ParquetReader, validate_parquet_columns
from .retry import (
    AsyncFailureLogger,
    FailureLogger,
    RetryConfig,
    async_with_retry,
    with_retry,
)
from .stats import PipelineStats
from .timer import timer

__all__ = [
    # Config
    "BaseConfig",
    # Data
    "load_keywords",
    "batch_iter",
    # Embedder
    "EmbeddingClient",
    "RURI_QUERY_PREFIX",
    "RURI_DOCUMENT_PREFIX",
    "RURI_ENCODE_PREFIX",
    "SUPPORTED_PROTOCOLS",
    # Logging
    "setup_logging",
    "get_logger",
    # Parquet
    "ParquetReader",
    "validate_parquet_columns",
    # Retry
    "RetryConfig",
    "FailureLogger",
    "AsyncFailureLogger",
    "with_retry",
    "async_with_retry",
    # Stats
    "PipelineStats",
    # Timer
    "timer",
]
