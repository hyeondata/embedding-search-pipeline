# es-indexer

Elasticsearch bulk and realtime indexing pipeline.

## Install

```bash
pip install es-indexer
```

## Usage

```python
from es_indexer import Config, run_indexing, run_realtime

# Bulk indexing
config = Config(limit=100, workers=8)
run_indexing(config)

# Realtime mode
run_realtime(config)
```
