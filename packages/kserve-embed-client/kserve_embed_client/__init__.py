"""
kserve-embed-client — KServe V2 임베딩 클라이언트 (클라이언트 사이드 토크나이저)

사용법:
    from kserve_embed_client import EmbeddingClient

    client = EmbeddingClient("http://localhost:8000", "ruri_v3")
    embeddings = client.embed(["東京の天気", "大阪のグルメ"])
    # → (2, 768) numpy array
"""

from .embedder import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_TOKENIZER,
    RURI_DOCUMENT_PREFIX,
    RURI_ENCODE_PREFIX,
    RURI_QUERY_PREFIX,
    SUPPORTED_PROTOCOLS,
    EmbeddingClient,
)

__all__ = [
    "EmbeddingClient",
    "RURI_QUERY_PREFIX",
    "RURI_DOCUMENT_PREFIX",
    "RURI_ENCODE_PREFIX",
    "SUPPORTED_PROTOCOLS",
    "DEFAULT_TOKENIZER",
    "DEFAULT_MAX_LENGTH",
]
