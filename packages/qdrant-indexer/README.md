# qdrant-indexer

KServe embedding + Qdrant vector indexing and search pipeline.

## Install

```bash
pip install qdrant-indexer
```

## Usage

```python
from qdrant_indexer import Config, Searcher, run_indexing

# Indexing
config = Config(limit=100, workers=4)
run_indexing(config)

# Search
searcher = Searcher(config)
results = searcher.search("東京 ラーメン", top_k=10)
```
