"""
es_indexer — Elasticsearch 벌크/실시간 인덱싱 패키지

Bulk 모드 (초기 대량 적재):
    from es_indexer import Config, run_indexing
    run_indexing(Config(limit=100, workers=8))

Realtime 모드 (운영 중 추가):
    from es_indexer import Config, run_realtime
    run_realtime(Config(keywords_path="new_data.txt", workers=4))

개별 CRUD (async):
    from es_indexer import ESIndexer
    indexer = ESIndexer("http://localhost:9200", "products")
    await indexer.ensure_index()
    await indexer.index("1", "北海道産 ラーメン")
    await indexer.search("ラーメン")
"""

from .config import Config, DEFAULT_SCHEMA
from .indexer import ESIndexer, build_es_client
from .pipeline import run_indexing, run_realtime

__all__ = [
    "Config", "DEFAULT_SCHEMA", "ESIndexer", "build_es_client",
    "run_indexing", "run_realtime",
]
