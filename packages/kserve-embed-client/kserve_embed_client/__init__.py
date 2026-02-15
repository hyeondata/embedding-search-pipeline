"""
kserve-embed-client — KServe V2 임베딩 클라이언트 + 공용 유틸리티

사용법:
    from kserve_embed_client import EmbeddingClient

    client = EmbeddingClient("http://localhost:8080", "ruri_v3")
    embeddings = client.embed(["東京の天気", "大阪のグルメ"])
    # → (2, 768) numpy array

    # 공용 유틸리티
    from kserve_embed_client import ParquetReader, setup_logging, get_logger
    from kserve_embed_client import RetryConfig, FailureLogger, with_retry
"""

from .embedder import EmbeddingClient
from .log import get_logger, setup_logging
from .parquet_reader import ParquetReader, validate_parquet_columns
from .retry import FailureLogger, RetryConfig, with_retry

__all__ = [
    "EmbeddingClient",
    "setup_logging", "get_logger",
    "ParquetReader", "validate_parquet_columns",
    "RetryConfig", "FailureLogger", "with_retry",
]
