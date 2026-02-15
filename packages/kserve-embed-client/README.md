# kserve-embed-client

KServe V2 Inference Protocol embedding client with shared data utilities.

## Install

```bash
pip install kserve-embed-client
```

## Usage

```python
from kserve_embed_client import EmbeddingClient

client = EmbeddingClient("http://localhost:8080", "ruri_v3")
embeddings = client.embed(["東京の天気", "大阪のグルメ"])
# → (2, 768) numpy array
```
