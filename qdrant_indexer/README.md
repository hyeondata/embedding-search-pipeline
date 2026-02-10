# qdrant_indexer 패키지

KServe V2 프로토콜로 텍스트를 임베딩하고, Qdrant 벡터 DB에 **인덱싱 + 검색**하는 파이프라인 패키지.

## 목차

- [아키텍처](#아키텍처)
- [사전 조건](#사전-조건)
- [Quick Start](#quick-start)
- [CLI 사용법](#cli-사용법)
  - [인덱싱 CLI](#인덱싱-cli-index_to_qdrantpy)
  - [검색 CLI](#검색-cli-search_qdrantpy)
- [API Reference](#api-reference)
  - [Config](#config)
  - [EmbeddingClient](#embeddingclient)
  - [QdrantIndexer](#qdrantindexer)
  - [Searcher](#searcher)
  - [run_indexing()](#run_indexing)
  - [run_batch_search()](#run_batch_search)
  - [Logging (setup_logging / get_logger)](#logging-setup_logging--get_logger)
- [파일 구조](#파일-구조)
- [성능 튜닝](#성능-튜닝)
- [개발에 사용된 핵심 기술](#개발에-사용된-핵심-기술)

---

## 아키텍처

### 인덱싱 파이프라인

```
keywords_400k.txt  OR  keywords.parquet  OR  keywords_partitioned/*.parquet
       │                     │                        │
       └─────────────────────┴────────────────────────┘
                             │
                             ▼
                      ParquetReader (청크 스트리밍)
                             │
                             ▼
  ┌─────────────────────────────────────────────┐
  │           run_indexing (pipeline.py)          │
  │                                               │
  │  ThreadPoolExecutor(max_workers=N)            │
  │  ┌─────────┐ ┌─────────┐     ┌─────────┐    │
  │  │ Worker 1 │ │ Worker 2 │ ... │ Worker N │    │
  │  └────┬─────┘ └────┬─────┘     └────┬─────┘    │
  │       │            │                │          │
  │       ▼            ▼                ▼          │
  │  ┌──────────────────────────────────────┐     │
  │  │  EmbeddingClient (embedder.py)       │     │
  │  │  POST /v2/models/ruri_v3/infer       │     │
  │  │  requests.Session (커넥션 풀링)       │     │
  │  └──────────────────────────────────────┘     │
  │       │                                        │
  │       ▼                                        │
  │  ┌──────────────────────────────────────┐     │
  │  │  KServe Server (kserve_server.py)    │     │
  │  │  ONNX Runtime + CoreMLExecutionProvider│     │
  │  │  ruri-v3-310m → 768-dim embedding    │     │
  │  └──────────────────────────────────────┘     │
  │       │                                        │
  │       ▼                                        │
  │  ┌──────────────────────────────────────┐     │
  │  │  QdrantIndexer (indexer.py)          │     │
  │  │  upsert() → Qdrant :6333            │     │
  │  └──────────────────────────────────────┘     │
  │                                               │
  │  _Stats: thread-safe 진행률 + RPS 측정        │
  └─────────────────────────────────────────────┘
```

### 검색 파이프라인

```
  ┌─ 단건 검색 ────────────────────────────────┐
  │  Searcher.search("東京 ラーメン", top_k=10)  │
  │    ↓ "検索クエリ: " prefix 추가              │
  │    ↓ EmbeddingClient.embed([1건])            │
  │    ↓ QdrantIndexer.search(top_k)             │
  │    → list[SearchResult]                      │
  └──────────────────────────────────────────────┘

  ┌─ 대용량 배치 검색 ────────────────────────────────┐
  │  run_batch_search(config, queries_path, top_k=10)  │
  │                                                     │
  │  ThreadPoolExecutor(max_workers=N)                  │
  │  ┌─────────┐ ┌─────────┐     ┌─────────┐          │
  │  │ Worker 1 │ │ Worker 2 │ ... │ Worker N │          │
  │  └────┬─────┘ └────┬─────┘     └────┬─────┘          │
  │       │            │                │                │
  │       ▼            ▼                ▼                │
  │  embed(batch) → search(쿼리별) → JSONL 기록          │
  │                                                     │
  │  _SearchStats: thread-safe 진행률 + QPS 측정        │
  │  out_lock: thread-safe JSONL 파일 쓰기              │
  └─────────────────────────────────────────────────────┘
       │
       ▼
  search_results_*.jsonl
```

**인덱싱 vs 검색 데이터 흐름 비교:**

| 단계 | 인덱싱 | 검색 |
|------|--------|------|
| 1 | 키워드 로드 | 쿼리 로드 |
| 2 | embed(batch) | embed(batch) + `"検索クエリ: "` prefix |
| 3 | Qdrant upsert (배치 1회) | Qdrant search (쿼리별 N회) |
| 4 | 검증 (벡터 수 확인) | JSONL 결과 저장 |
| 단위 | t/s (texts/sec) | q/s (queries/sec) |

---

## 사전 조건

### 1. KServe 임베딩 서버 실행

```bash
cd kserve/m1-inference
python kserve_server.py --http_port 8080
```

서버가 `http://localhost:8080`에서 V2 프로토콜로 임베딩 API를 제공합니다.

### 2. Qdrant 실행

```bash
cd kserve/m1-inference
docker compose up -d qdrant
```

Qdrant가 `http://localhost:6333`에서 REST API를 제공합니다.

### 3. 의존 패키지

```
requests        # HTTP 클라이언트
numpy           # 임베딩 배열 처리
qdrant-client   # Qdrant Python SDK
rich            # 콘솔 Rich 로깅 + CLI 출력
pyarrow         # Parquet 파일 읽기 (선택적)
```

---

## Quick Start

### 인덱싱 (텍스트 파일)

```python
from qdrant_indexer import Config, run_indexing

config = Config(workers=8, batch_size=64)
run_indexing(config)
```

### 인덱싱 (Parquet 파일)

```python
from pathlib import Path
from qdrant_indexer import Config, run_indexing

config = Config(
    parquet_path=Path("data/keywords.parquet"),
    parquet_chunk_size=10000,
    parquet_text_column="keyword",
    workers=8,
    batch_size=64,
)
run_indexing(config)
```

### 검색 (단건)

```python
from qdrant_indexer import Config, Searcher

searcher = Searcher(Config())
results = searcher.search("東京 ラーメン", top_k=10)
for r in results:
    print(f"#{r.rank} [{r.score:.4f}] {r.keyword}")
```

### 검색 (대용량 배치)

```python
from pathlib import Path
from qdrant_indexer import Config, run_batch_search

config = Config(workers=8, batch_size=64)
run_batch_search(config, Path("data/keywords_400k.txt"), top_k=10)
# → logs/search_results_YYYYMMDD_HHMMSS.jsonl
```

### 개별 컴포넌트 사용

```python
from qdrant_indexer import EmbeddingClient, QdrantIndexer

# 임베딩만 사용
embedder = EmbeddingClient("http://localhost:8080", "ruri_v3")
embeddings = embedder.embed(["東京 ラーメン おすすめ", "大阪 観光"])
print(embeddings.shape)  # (2, 768)

# Qdrant만 사용
indexer = QdrantIndexer("http://localhost:6333", "keywords", 768)
indexer.create_collection()
indexer.upsert_batch(start_id=0, keywords=["東京"], embeddings=embeddings[:1])
print(indexer.count)  # 1

# 유사도 검색
results = indexer.search(embeddings[0].tolist(), top_k=5)
for point in results.points:
    print(f"score={point.score:.4f}  {point.payload['keyword']}")
```

---

## CLI 사용법

### 인덱싱 CLI (`index_to_qdrant.py`)

```bash
# 소규모 테스트 (처음 100건)
python index_to_qdrant.py --limit 100

# 기본 설정 (workers=4, batch=64)
python index_to_qdrant.py

# 최적 성능 (M1 벤치마크 기준)
python index_to_qdrant.py --workers 8 --batch_size 64

# 커스텀 키워드 파일 + 컬렉션명
python index_to_qdrant.py --keywords data/custom.txt --collection my_keywords

# Parquet 파일 읽기
python index_to_qdrant.py --parquet data/keywords.parquet --limit 10000

# Parquet 디렉토리 읽기 (여러 파일)
python index_to_qdrant.py --parquet data/keywords_partitioned/ --workers 8
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--keywords` | `data/keywords_400k.txt` | 키워드 파일 경로 (줄바꿈 구분) |
| `--parquet` | `None` | Parquet 파일 또는 디렉토리 경로 (설정 시 --keywords 무시) |
| `--parquet_chunk_size` | `10000` | Parquet 청크 크기 (행 수) |
| `--parquet_column` | `keyword` | 텍스트가 들어있는 컬럼명 |
| `--limit` | `0` | 처리할 키워드 수 (0=전체) |
| `--batch_size` | `64` | 배치당 키워드 수 |
| `--workers` | `4` | 동시 처리 워커 수 |
| `--collection` | `keywords` | Qdrant 컬렉션 이름 |

### 검색 CLI (`search_qdrant.py`)

#### 단건 검색

```bash
python search_qdrant.py "東京 ラーメン"
python search_qdrant.py "東京 ラーメン" --top_k 20
```

#### 복수 쿼리

```bash
python search_qdrant.py "東京 ラーメン" "大阪 観光" "京都 寺院"
```

#### 대화형 모드

```bash
python search_qdrant.py --interactive
python search_qdrant.py -i --top_k 5
```

```
  검색> 東京 ラーメン
  Query: "東京 ラーメン"  (0.029초, 10건)
    순위      유사도        ID  키워드
  ────  ───────  ────────  ────────────────
     1   0.9375     61347  東京 有名な ラーメン
     2   0.9342     30830  東京 美味しい ラーメン
     ...
  검색> q
  종료
```

#### 대용량 배치 검색

```bash
# 기본 (전체 쿼리, top_k=10, workers=8, batch=64)
python search_qdrant.py --file data/keywords_400k.txt

# 소규모 테스트 (1000건)
python search_qdrant.py --file data/keywords_400k.txt --limit 1000

# 커스텀 설정
python search_qdrant.py --file data/keywords_400k.txt --top_k 5 --workers 8 --batch_size 64

# 출력 파일 지정
python search_qdrant.py --file data/keywords_400k.txt --output results.jsonl
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--file`, `-f` | — | 배치 검색: 쿼리 파일 경로 (줄바꿈 구분) |
| `--output`, `-o` | `logs/search_results_*.jsonl` | 배치 검색: 결과 JSONL 파일 경로 |
| `--top_k` | `10` | 각 쿼리당 반환 결과 수 |
| `--limit` | `0` | 처리할 쿼리 수 (0=전체) |
| `--batch_size` | `64` | 배치당 쿼리 수 |
| `--workers` | `8` | 동시 워커 수 |
| `--collection` | `keywords` | Qdrant 컬렉션 이름 |
| `--interactive`, `-i` | — | 대화형 검색 모드 |

#### 배치 검색 JSONL 출력 포맷

각 줄에 쿼리 1건의 검색 결과가 JSON으로 저장됩니다:

```json
{
  "query": "東京 ラーメン おすすめ",
  "results": [
    {"rank": 1, "score": 0.9375, "keyword": "東京 有名な ラーメン", "point_id": 61347},
    {"rank": 2, "score": 0.9342, "keyword": "東京 美味しい ラーメン", "point_id": 30830},
    ...
  ]
}
```

> **JSONL 포맷 장점:** 한 줄씩 스트리밍으로 읽고 쓸 수 있어 대용량 결과(400k건)에도 메모리 문제 없음. `jq`, `pandas.read_json(lines=True)` 등으로 후처리 용이.

---

## API Reference

### Config

**파일:** `config.py`

설정을 관리하는 데이터클래스. 모든 필드에 기본값이 있어 `Config()`만으로 사용 가능.

```python
@dataclass
class Config:
    # 데이터 소스
    keywords_path: Path   # 키워드 파일 경로
    limit: int            # 처리할 키워드 수 (0=전체)

    # Parquet 지원
    parquet_path: Path | None       # Parquet 파일/디렉토리 경로
    parquet_chunk_size: int         # Parquet 청크 크기
    parquet_text_column: str        # 텍스트 컬럼명

    # KServe
    kserve_url: str       # KServe 서버 URL
    model_name: str       # KServe 모델 이름

    # Qdrant
    qdrant_url: str       # Qdrant 서버 URL
    collection_name: str  # Qdrant 컬렉션 이름
    vector_dim: int       # 벡터 차원 (768)

    # 처리
    batch_size: int       # 배치 크기
    workers: int          # 동시 워커 수

    # 재시도
    max_retries: int      # 배치당 최대 재시도 횟수
    retry_backoff: float  # 초기 대기 시간
    retry_exponential: bool  # 지수 백오프 사용
    retry_max_backoff: float # 최대 대기 시간

    # 실패 로깅
    log_failures: bool    # 실패한 배치 기록
    failure_log_path: Path # 실패 로그 경로
```

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `keywords_path` | `Path` | `data/keywords_400k.txt` | 줄바꿈으로 구분된 키워드 텍스트 파일 |
| `limit` | `int` | `0` | 처리할 키워드 수. `0`이면 전체 처리 |
| `parquet_path` | `Path \| None` | `None` | Parquet 파일 또는 디렉토리 경로 (설정 시 keywords_path 무시) |
| `parquet_chunk_size` | `int` | `10000` | Parquet 청크 크기 (행 수) |
| `parquet_text_column` | `str` | `keyword` | 텍스트가 들어있는 컬럼명 |
| `kserve_url` | `str` | `http://localhost:8080` | KServe 임베딩 서버 URL |
| `model_name` | `str` | `ruri_v3` | KServe에 등록된 모델 이름 |
| `qdrant_url` | `str` | `http://localhost:6333` | Qdrant REST API URL |
| `collection_name` | `str` | `keywords` | Qdrant 컬렉션 이름 |
| `vector_dim` | `int` | `768` | 임베딩 벡터 차원 (ruri-v3-310m = 768) |
| `batch_size` | `int` | `64` | 한 번에 임베딩/저장할 키워드 수 |
| `workers` | `int` | `4` | ThreadPoolExecutor의 동시 워커 수 |
| `max_retries` | `int` | `3` | 배치 실패 시 최대 재시도 횟수 |
| `retry_backoff` | `float` | `1.0` | 첫 재시도 대기 시간 (초) |
| `retry_exponential` | `bool` | `True` | 지수 백오프 사용 여부 (1s → 2s → 4s) |
| `retry_max_backoff` | `float` | `60.0` | 최대 대기 시간 (초) |
| `log_failures` | `bool` | `True` | 실패한 배치를 파일에 기록 |
| `failure_log_path` | `Path` | `logs/failures.jsonl` | 실패 로그 JSONL 파일 경로 |

---

### EmbeddingClient

**파일:** `embedder.py`

KServe V2 Inference Protocol을 사용하여 텍스트를 임베딩 벡터로 변환하는 HTTP 클라이언트.

#### `__init__(base_url: str, model_name: str)`

KServe 서버와의 HTTP 세션을 초기화합니다.

- `base_url`: KServe 서버 URL (예: `http://localhost:8080`)
- `model_name`: 모델 이름 (예: `ruri_v3`)
- 내부적으로 `requests.Session()`을 생성하여 **HTTP 커넥션 풀링**을 활용

```python
embedder = EmbeddingClient("http://localhost:8080", "ruri_v3")
```

#### `embed(texts: list[str]) -> np.ndarray`

텍스트 리스트를 임베딩 벡터 배열로 변환합니다.

- **입력:** `texts` — 임베딩할 텍스트 리스트
- **출력:** `np.ndarray` — shape `(N, 768)`, N = 입력 텍스트 수
- **프로토콜:** KServe V2 Inference Protocol
- **타임아웃:** 120초

```python
embeddings = embedder.embed(["東京の天気", "大阪のグルメ", "京都の観光"])
print(embeddings.shape)  # (3, 768)
print(type(embeddings))  # <class 'numpy.ndarray'>
```

**V2 프로토콜 요청 포맷:**

```json
{
  "inputs": [{
    "name": "text",
    "shape": [3],
    "datatype": "BYTES",
    "data": ["東京の天気", "大阪のグルメ", "京都の観光"]
  }]
}
```

**V2 프로토콜 응답 포맷:**

```json
{
  "outputs": [{
    "name": "sentence_embedding",
    "shape": [3, 768],
    "datatype": "FP32",
    "data": [0.0123, -0.0456, ..., 0.0789]
  }]
}
```

> **Note:** V2 프로토콜은 `data`를 항상 **flat 1D 배열**로 전달합니다. `shape` 필드를 사용하여 `np.array(data).reshape(shape)`으로 원래 차원을 복원합니다.

---

### QdrantIndexer

**파일:** `indexer.py`

Qdrant 벡터 DB의 컬렉션 관리, 벡터 저장, 유사도 검색을 캡슐화한 클래스.

#### `__init__(url: str, collection_name: str, vector_dim: int)`

Qdrant 클라이언트를 초기화합니다.

```python
indexer = QdrantIndexer("http://localhost:6333", "keywords", 768)
```

#### `create_collection()`

Qdrant 컬렉션을 생성합니다. **이미 존재하면 삭제 후 재생성**합니다.

- Distance 메트릭: `COSINE`
- 벡터 크기: `vector_dim` (768)

```python
indexer.create_collection()
```

> **주의:** 기존 데이터가 모두 삭제됩니다.

#### `upsert_batch(start_id: int, keywords: list[str], embeddings: np.ndarray)`

키워드와 임베딩을 Qdrant에 배치로 저장합니다.

```python
indexer.upsert_batch(
    start_id=100,
    keywords=["東京 ラーメン", "大阪 たこ焼き", "京都 抹茶"],
    embeddings=embeddings,  # shape (3, 768)
)
```

#### `search(query_vector: list[float], top_k: int = 5)`

쿼리 벡터로 유사 키워드를 검색합니다.

```python
results = indexer.search(query_embedding.tolist(), top_k=5)
for point in results.points:
    print(f"score={point.score:.4f}  keyword={point.payload['keyword']}")
```

#### `count` (property)

현재 컬렉션의 포인트(벡터) 수를 반환합니다.

```python
print(f"저장된 벡터 수: {indexer.count:,}")
```

---

### ParquetReader

**파일:** `parquet_reader.py`

Parquet 파일 또는 디렉토리에서 텍스트 데이터를 청크 단위로 스트리밍 읽기하는 클래스.
대용량 Parquet 파일(수백만 행)도 메모리에 모두 로드하지 않고 청크별로 처리할 수 있습니다.

#### `__init__(parquet_path: Path, chunk_size: int = 10000, text_column: str = "keyword", limit: int = 0)`

Parquet 파일 또는 디렉토리를 초기화합니다.

- `parquet_path`: Parquet 파일 경로 또는 디렉토리 경로
  - **파일**: 단일 .parquet 파일
  - **디렉토리**: 여러 .parquet 파일 (알파벳순 정렬)
- `chunk_size`: 한 번에 읽을 행 수 (기본 10000)
- `text_column`: 텍스트가 들어있는 컬럼명 (기본 "keyword")
- `limit`: 읽을 총 행 수 (0=전체)

```python
from pathlib import Path
from qdrant_indexer import ParquetReader

# 단일 파일
reader = ParquetReader(Path("data/keywords.parquet"), chunk_size=5000)

# 디렉토리 (여러 파일)
reader = ParquetReader(Path("data/keywords_partitioned/"), chunk_size=10000)
```

#### `iter_chunks() -> Iterator[tuple[int, list[str]]]`

Parquet 데이터를 청크 단위로 순회합니다.

- **반환**: `(chunk_id, keywords)` 튜플의 이터레이터
  - `chunk_id`: 청크 번호 (0부터 시작)
  - `keywords`: 텍스트 리스트 (최대 chunk_size개)

```python
reader = ParquetReader(Path("data/keywords.parquet"), chunk_size=1000)

for chunk_id, keywords in reader.iter_chunks():
    print(f"Chunk {chunk_id}: {len(keywords)}건")
    # 각 청크 처리 (예: 임베딩 → Qdrant upsert)
```

#### `total_rows` (property)

전체 행 수를 반환합니다 (limit 적용 전).

```python
reader = ParquetReader(Path("data/keywords.parquet"))
print(f"전체 행 수: {reader.total_rows:,}")
```

**디렉토리 읽기 예시:**

```python
# data/keywords_partitioned/ 디렉토리에 part-00.parquet, part-01.parquet, ... 파일들이 있음
reader = ParquetReader(
    Path("data/keywords_partitioned/"),
    chunk_size=10000,
    text_column="keyword",
)

print(f"파일 수: {len(reader.parquet_files)}")  # 예: 10
print(f"전체 행 수: {reader.total_rows:,}")     # 예: 1,000,000

for chunk_id, keywords in reader.iter_chunks():
    # 스트리밍 처리: 메모리에 한 번에 10000행만 로드
    embeddings = embedder.embed(keywords)
    indexer.upsert_batch(chunk_id * 10000, keywords, embeddings)
```

> **Note:** Parquet 디렉토리는 Spark, Dask 등에서 파티션된 데이터를 저장할 때 사용하는 일반적인 패턴입니다. 각 part-*.parquet 파일을 순서대로 읽어 전체 데이터를 처리합니다.

---

### Searcher

**파일:** `searcher.py`

텍스트 쿼리를 입력하면 임베딩 + 검색을 자동으로 수행하는 고수준 클래스.
ruri-v3의 비대칭 검색 prefix(`"検索クエリ: "`)를 자동으로 추가합니다.

#### `__init__(config: Config = None)`

EmbeddingClient와 QdrantIndexer를 내부적으로 생성합니다.

```python
from qdrant_indexer import Config, Searcher

searcher = Searcher(Config())
# 또는
searcher = Searcher()  # Config() 기본값 사용
```

#### `search(query: str, top_k: int = 5, prefix: str = None) -> list[SearchResult]`

**단건 검색.** 텍스트 쿼리를 임베딩한 뒤 Qdrant에서 top-k 유사 키워드를 반환합니다.

- `query`: 검색할 텍스트
- `top_k`: 반환할 결과 수 (기본 5)
- `prefix`: 쿼리 prefix (기본: `"検索クエリ: "`)

```python
results = searcher.search("東京 ラーメン", top_k=10)
for r in results:
    print(f"#{r.rank} [{r.score:.4f}] {r.keyword} (id={r.point_id})")
```

**출력 예시:**

```
#1 [0.9375] 東京 有名な ラーメン (id=61347)
#2 [0.9342] 東京 美味しい ラーメン (id=30830)
#3 [0.9311] 最新の ラーメン 東京 (id=85924)
```

#### `search_batch(queries: list[str], top_k: int = 5, prefix: str = None) -> dict[str, list[SearchResult]]`

**소규모 배치 검색.** 여러 쿼리를 한 번에 임베딩(HTTP 1회)한 뒤 각각 Qdrant 검색합니다.

- `queries`: 검색할 텍스트 리스트
- **반환:** `{쿼리 텍스트: [SearchResult, ...]}` 딕셔너리

```python
batch_results = searcher.search_batch(
    ["東京 ラーメン", "大阪 観光", "京都 寺院"],
    top_k=5,
)
for query, results in batch_results.items():
    print(f"\n{query}:")
    for r in results:
        print(f"  #{r.rank} [{r.score:.4f}] {r.keyword}")
```

> **`search()` vs `search_batch()`**: `search()`는 쿼리 1건당 HTTP 요청 1회. `search_batch()`는 N건의 쿼리를 임베딩 1회로 처리하므로, 소규모(~수백건) 다중 쿼리에 효율적.

#### SearchResult

검색 결과 1건을 나타내는 데이터클래스.

```python
@dataclass
class SearchResult:
    rank: int           # 순위 (1부터)
    score: float        # 코사인 유사도 (0~1)
    keyword: str        # 원본 키워드 텍스트
    point_id: int       # Qdrant 포인트 ID
```

---

### run_indexing()

**파일:** `pipeline.py`

전체 인덱싱 파이프라인을 실행하는 메인 함수.

#### `run_indexing(config: Config, log_path: Path = None)`

**4단계 파이프라인:**

1. **키워드 로드** — 텍스트 파일에서 키워드 읽기
2. **컬렉션 생성** — Qdrant 컬렉션 초기화
3. **동시 인덱싱** — ThreadPoolExecutor로 embed + upsert 병렬 처리
4. **검증** — 벡터 수 확인 + 샘플 유사도 검색

**콘솔 출력 (Rich 포맷):**

```
──────────────── 키워드 임베딩 → Qdrant 인덱싱 (Concurrent) ────────────────
[05:54:54] INFO  Log: logs/indexing_20260209_055454.log
           INFO  [1/4] 키워드 로드
           INFO  400,000건
           INFO  [2/4] Qdrant 컬렉션: keywords
           INFO  [3/4] 동시 인덱싱 (workers=8, batch=64)
           INFO  [  0.0%]      64/400,000  embed=5316ms  upsert=57ms  avg=12 t/s  interval=12 t/s
           INFO  [ 25.0%] 100,000/400,000  embed=523ms   upsert=32ms  avg=89 t/s  interval=95 t/s
           ...
           INFO  [4/4] 검증
           INFO  처리량: 95 texts/sec
           INFO  완료!
```

**파일 로그 (plain text):**

```
2026-02-09 05:54:54  qdrant_indexer.pipeline  [1/4] 키워드 로드
2026-02-09 05:54:55  qdrant_indexer.pipeline  400,000건
2026-02-09 05:55:01  qdrant_indexer.pipeline  [  0.0%]  64/400,000  embed=5316ms  ...
```

> 콘솔은 RichHandler가 컬러+타임스탬프로 출력하고, 파일은 `_PlainFormatter`가 Rich markup을 제거한 plain text로 기록합니다.

| 항목 | 설명 |
|------|------|
| `pct%` | 전체 진행률 |
| `embed` | 해당 배치의 임베딩 소요 시간 (ms) |
| `upsert` | 해당 배치의 Qdrant 저장 소요 시간 (ms) |
| `avg` | 전체 평균 처리량 (texts/sec) |
| `interval` | 최근 구간의 처리량 (texts/sec) |

```python
from qdrant_indexer import Config, run_indexing

config = Config(workers=8, batch_size=64)
run_indexing(config)
# 콘솔: Rich colored 출력
# 파일: logs/indexing_20260208_222401.log (plain text)
```

---

### run_batch_search()

**파일:** `searcher.py`

대용량 쿼리 파일을 읽어 동시(concurrent) 임베딩 + 검색 후 결과를 JSONL 파일로 저장하는 배치 검색 파이프라인.

#### `run_batch_search(config, queries_path, output_path=None, top_k=10, limit=0, log_path=None)`

**3단계 파이프라인:**

1. **쿼리 로드** — 텍스트 파일에서 쿼리 읽기 (줄바꿈 구분)
2. **동시 검색** — ThreadPoolExecutor로 embed + search 병렬 처리, JSONL 스트리밍 기록
3. **통계 출력** — 총 쿼리 수, wall time, QPS

**파라미터:**

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `config` | `Config` | — | 설정 인스턴스 (kserve_url, batch_size, workers 등) |
| `queries_path` | `Path` | — | 쿼리 파일 경로 (한 줄에 쿼리 1개) |
| `output_path` | `Path` | `logs/search_results_*.jsonl` | 결과 JSONL 파일 경로 |
| `top_k` | `int` | `10` | 각 쿼리당 반환 결과 수 |
| `limit` | `int` | `0` | 처리할 쿼리 수 (0=전체) |
| `log_path` | `Path` | `logs/` | 로그 디렉토리 |

**반환:** `Path` — 생성된 결과 JSONL 파일 경로

**로그 출력 포맷:**

```
[  6.4%]      64/1,000  embed=6532ms  search=688ms  avg=9 q/s  interval=9 q/s
[100.0%]   1,000/1,000  embed=5620ms  search=394ms  avg=71 q/s  interval=135 q/s
```

| 항목 | 설명 |
|------|------|
| `embed` | 해당 배치의 임베딩 소요 시간 (ms) |
| `search` | 해당 배치의 Qdrant 검색 소요 시간 (ms, 쿼리별 N회 합계) |
| `avg` | 전체 평균 처리량 (queries/sec) |
| `interval` | 최근 구간의 처리량 (queries/sec) |

```python
from pathlib import Path
from qdrant_indexer import Config, run_batch_search

config = Config(workers=8, batch_size=64)
output = run_batch_search(config, Path("data/keywords_400k.txt"), top_k=10, limit=1000)
# → logs/search_results_20260208_225449.jsonl
```

---

### Logging (setup_logging / get_logger)

**파일:** `log.py`

`logging` + `rich.logging.RichHandler`를 조합한 통합 로깅 시스템.
`logger.info()` 한 번 호출로 콘솔(Rich colored)과 파일(plain text)에 동시 출력합니다.

#### 아키텍처

```
logger.info("[bold green]완료![/bold green]")
       │
       ├── RichHandler (콘솔)
       │   → 컬러 + 볼드 + 타임스탬프: [05:54:55] INFO 완료!
       │
       └── FileHandler + _PlainFormatter (파일)
           → 2026-02-09 05:54:55  qdrant_indexer.pipeline  완료!
                                                           ↑ markup 자동 제거
```

#### `setup_logging(log_file: Path = None, level: int = logging.INFO) -> Logger`

패키지 루트 로거(`qdrant_indexer`)에 핸들러를 설정합니다.

- **RichHandler**: 첫 호출 시 1회만 추가 (중복 방지)
- **FileHandler**: `log_file` 인자가 있을 때마다 추가 (인덱싱/검색별 별도 파일)
- `_PlainFormatter`가 `rich.text.Text.from_markup().plain`으로 Rich markup 태그를 자동 제거

```python
from qdrant_indexer import setup_logging

# 콘솔만
setup_logging()

# 콘솔 + 파일
setup_logging(log_file=Path("logs/my_task.log"))
```

#### `get_logger(name: str) -> Logger`

패키지 하위 로거를 반환합니다. 부모 로거의 핸들러를 자동 상속합니다 (`propagate=True`).

```python
from qdrant_indexer import get_logger

logger = get_logger("my_module")  # → logging.getLogger("qdrant_indexer.my_module")
logger.info("[bold cyan]처리 중...[/bold cyan]")
# Console: cyan 볼드로 "처리 중..." 출력
# File:    "처리 중..." (plain text)
```

#### Rich markup 사용법

`logger.info()` 메시지에 Rich markup을 사용할 수 있습니다. 콘솔에서는 스타일이 적용되고, 파일에는 plain text로 기록됩니다.

| Markup | 콘솔 효과 | 파일 출력 |
|--------|-----------|----------|
| `[bold]텍스트[/bold]` | **볼드** | 텍스트 |
| `[green]텍스트[/green]` | 초록색 | 텍스트 |
| `[cyan]123ms[/cyan]` | 시안색 | 123ms |
| `[bold green]완료![/bold green]` | **초록 볼드** | 완료! |

> **Note:** 대괄호 `[`를 literal로 출력하려면 `\\[` 로 이스케이프해야 합니다. 예: `logger.info("[bold]\\[1/4] 시작[/bold]")` → **[1/4] 시작**

---

**JSONL 결과를 pandas로 읽기:**

```python
import pandas as pd

df = pd.read_json("logs/search_results_20260208_225449.jsonl", lines=True)
print(df.head())
#                    query                                            results
# 0  東京 ラーメン おすすめ  [{'rank': 1, 'score': 0.9375, 'keyword': '東京...
# 1  大阪 観光スポット      [{'rank': 1, 'score': 0.9130, 'keyword': '今日...

# 각 쿼리의 top-1 점수만 추출
df["top1_score"] = df["results"].apply(lambda r: r[0]["score"] if r else 0)
print(f"평균 top-1 유사도: {df['top1_score'].mean():.4f}")
```

---

## 파일 구조

```
m1-inference/
├── index_to_qdrant.py          # 인덱싱 CLI
├── search_qdrant.py            # 검색 CLI (단건/배치/대화형)
├── qdrant_indexer/
│   ├── __init__.py             # 패키지 초기화 + public API export
│   ├── config.py               # Config 데이터클래스
│   ├── log.py                  # 통합 로깅 (RichHandler + FileHandler)
│   ├── embedder.py             # EmbeddingClient (KServe V2 HTTP)
│   ├── indexer.py              # QdrantIndexer (Qdrant SDK wrapper)
│   ├── parquet_reader.py       # ParquetReader (Parquet 청크 스트리밍)
│   ├── retry.py                # RetryConfig + FailureLogger (재시도 + 실패 로깅)
│   ├── pipeline.py             # run_indexing() 인덱싱 파이프라인
│   ├── searcher.py             # Searcher + run_batch_search() 검색 파이프라인
│   └── README.md               # 이 문서
├── data/
│   └── keywords_400k.txt       # 생성된 키워드 파일 (400,000건)
├── logs/
│   ├── indexing_*.log          # 인덱싱 로그
│   ├── batch_search_*.log      # 배치 검색 로그
│   └── search_results_*.jsonl  # 배치 검색 결과 (JSONL)
├── kserve_server.py            # KServe 임베딩 서버 (ONNX Runtime)
├── docker-compose.yml          # Qdrant + Elasticsearch
└── generate_keywords.py        # 키워드 생성기
```

### 각 파일의 역할

| 파일 | 역할 | 핵심 클래스/함수 |
|------|------|------------------|
| `__init__.py` | 패키지 진입점, public API 노출 | `Config`, `EmbeddingClient`, `QdrantIndexer`, `AsyncQdrantIndexer`, `ParquetReader`, `RetryConfig`, `FailureLogger`, `run_indexing`, `Searcher`, `SearchResult`, `run_batch_search`, `setup_logging`, `get_logger` |
| `config.py` | 설정값 정의 | `Config` (dataclass) |
| `log.py` | 통합 로깅 설정 (Rich + File) | `setup_logging()`, `get_logger()`, `_PlainFormatter` |
| `embedder.py` | KServe 임베딩 HTTP 클라이언트 | `EmbeddingClient.embed()` |
| `indexer.py` | Qdrant 벡터 DB 조작 | `QdrantIndexer`, `AsyncQdrantIndexer` |
| `parquet_reader.py` | Parquet 청크 스트리밍 | `ParquetReader.iter_chunks()`, `.total_rows` |
| `retry.py` | 재시도 + 실패 로깅 | `RetryConfig`, `FailureLogger`, `with_retry()` |
| `pipeline.py` | 인덱싱 파이프라인 | `run_indexing()`, `_Stats`, `_create_batch_processor()` |
| `searcher.py` | 검색 (단건 + 배치) | `Searcher.search()`, `.search_batch()`, `run_batch_search()`, `_SearchStats` |

---

## 성능 튜닝

### 인덱싱 벤치마크 (M1 Mac, ruri-v3-310m FP16)

| batch_size | workers | 처리량 (t/s) | 400k 예상 시간 |
|:----------:|:-------:|:-----------:|:-------------:|
| 32 | 4 | 54 | 124분 |
| 64 | 4 | 71 | 94분 |
| 64 | 8 | **95** | **71분** |
| 128 | 8 | 91 | 74분 |
| 256 | 8 | 89 | 75분 |

**인덱싱 최적 설정: `batch_size=64, workers=8` (95 t/s)**

### 검색 벤치마크 (M1 Mac, 96k 벡터)

| 규모 | batch_size | workers | top_k | 처리량 (q/s) | 소요 시간 |
|:----:|:----------:|:-------:|:-----:|:-----------:|:---------:|
| 100건 | 32 | 4 | 5 | 64 | 1.6초 |
| 1,000건 | 64 | 8 | 10 | **71** | **14.2초** |

> **인덱싱 vs 검색 속도 차이**: 검색이 인덱싱보다 약간 느린 이유는, 인덱싱에서 Qdrant upsert는 배치 1회이지만, 검색은 각 쿼리마다 Qdrant search를 개별 호출(N회)하기 때문입니다.

### 튜닝 가이드

#### batch_size

- **작은 값 (16-32):** 각 HTTP 요청이 가벼워 응답이 빠르지만, 요청 오버헤드가 큼
- **최적 값 (64):** KServe 서버의 GPU 활용율과 HTTP 오버헤드의 균형점
- **큰 값 (128-256):** GPU 포화로 추가 이점 없이 메모리만 증가

#### workers

- **작은 값 (1-2):** I/O 대기 시간 동안 GPU가 유휴 상태
- **최적 값 (8):** 한 워커가 응답 대기 중 다른 워커가 요청 전송 → GPU 연속 활용
- **큰 값 (16+):** 컨텍스트 스위칭 오버헤드로 역효과

#### top_k (검색 전용)

- **작은 값 (1-5):** Qdrant HNSW 탐색이 빨라 search_ms 감소
- **큰 값 (50-100):** HNSW 탐색 범위가 넓어져 search_ms 증가, 결과 파일 크기도 증가

---

## 개발에 사용된 핵심 기술

### 1. requests.Session — HTTP 커넥션 풀링

`embedder.py`에서 사용. `requests.Session()`은 TCP 커넥션을 재사용하여
매 요청마다 TCP handshake + TLS negotiation을 반복하지 않습니다.

```python
self.session = requests.Session()
resp = self.session.post(self.url, json=payload, timeout=120)
```

멀티스레드 환경에서도 `Session`은 내부적으로 `urllib3.HTTPConnectionPool`을 사용하여
스레드 간 커넥션을 안전하게 공유합니다.

### 2. ThreadPoolExecutor — 동시 배치 처리

`pipeline.py`와 `searcher.py` 모두에서 사용. I/O 바운드 작업(HTTP 임베딩 요청 + Qdrant 작업)을
여러 스레드에서 동시에 처리하여 처리량을 극대화합니다.

```python
with ThreadPoolExecutor(max_workers=config.workers) as pool:
    futures = [pool.submit(worker_fn, batch) for batch in batches]
    for future in as_completed(futures):
        future.result()  # 에러 전파
```

**왜 Thread인가 (Process가 아닌)?**
- I/O 바운드 작업이므로 GIL의 영향을 거의 받지 않음
- 프로세스 간 데이터 복사 오버헤드 없음
- `requests.Session` 커넥션 풀 공유 가능

### 3. threading.Lock — Thread-safe 통계/파일 쓰기

두 가지 용도로 사용됩니다:

**(a) 통계 집계 (`_Stats`, `_SearchStats`):**

여러 워커 스레드가 동시에 통계를 업데이트하므로 Lock으로 race condition을 방지합니다.

```python
class _SearchStats:
    def update(self, count, embed_ms, search_ms):
        with self._lock:
            self.searched += count
            self._interval_count += count
```

**(b) JSONL 파일 쓰기 (`out_lock`):**

배치 검색에서 여러 워커가 동시에 결과 파일에 쓸 때 줄이 섞이는 것을 방지합니다.

```python
with out_lock:
    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
```

### 4. KServe V2 Inference Protocol

`embedder.py`에서 사용하는 HTTP 프로토콜. 표준화된 텐서 입출력 포맷을 제공합니다.

**핵심 특징:**
- `data`는 항상 **flat 1D 배열** (protobuf의 repeated 필드 제약)
- `shape`으로 원래 차원 복원: `np.array(data).reshape(shape)`
- `datatype`으로 타입 지정: `BYTES`(문자열), `FP32`(float), `INT64` 등

**엔드포인트:** `POST /v2/models/{model_name}/infer`

### 5. ruri-v3 비대칭 검색 prefix

ruri-v3 모델은 비대칭 검색(Asymmetric Search)을 지원합니다:

| 용도 | Prefix | 예시 |
|------|--------|------|
| 검색 쿼리 | `"検索クエリ: "` | `"検索クエリ: 東京 ラーメン"` |
| 검색 문서 | `"検索文書: "` | `"検索文書: 東京の有名なラーメン店"` |
| 의미적 인코딩 | (없음) | `"東京 ラーメン"` |

`Searcher` 클래스가 자동으로 `"検索クエリ: "` prefix를 추가하므로 사용자는 원본 텍스트만 전달하면 됩니다. 인덱싱 시에는 키워드를 prefix 없이 임베딩합니다 (의미적 인코딩).

### 6. Qdrant Client SDK

`indexer.py`에서 사용. Python 네이티브 SDK로 Qdrant REST API를 추상화합니다.

| SDK 메서드 | Qdrant REST API | 용도 |
|------------|-----------------|------|
| `get_collections()` | `GET /collections` | 컬렉션 목록 조회 |
| `delete_collection()` | `DELETE /collections/{name}` | 컬렉션 삭제 |
| `create_collection()` | `PUT /collections/{name}` | 컬렉션 생성 |
| `upsert()` | `PUT /collections/{name}/points` | 벡터 저장 |
| `query_points()` | `POST /collections/{name}/points/query` | 유사도 검색 |
| `get_collection()` | `GET /collections/{name}` | 컬렉션 정보 (점 수 등) |

### 7. JSONL — 대용량 결과 스트리밍

배치 검색 결과를 JSONL(JSON Lines) 포맷으로 저장합니다.

**JSON vs JSONL:**

| | JSON | JSONL |
|-|------|-------|
| 포맷 | `[{}, {}, ...]` | 한 줄에 JSON 1개 |
| 메모리 | 전체 로드 필요 | 한 줄씩 스트리밍 |
| 쓰기 | 완료 후 한번에 | 실시간 append |
| 후처리 | `json.load()` | `jq`, `pandas.read_json(lines=True)` |

400k건의 검색 결과(각 top-10)도 메모리 문제 없이 스트리밍으로 기록할 수 있습니다.

### 8. Rich + logging — 통합 로깅 시스템

`log.py`에서 구현. `logging` 모듈의 Handler 체계를 활용하여 콘솔과 파일에 동시 출력합니다.

**기존 방식의 문제점:**

```python
# Before: print() + logger.info() 이중 호출
print(f"         {msg}")        # 콘솔
logger.info(msg)                # 파일
```

**개선된 방식:**

```python
# After: logger.info() 한 번으로 양쪽 동시 출력
logger.info(f"[bold green]완료![/bold green]")
```

**핵심 구현: `_PlainFormatter`**

Rich markup이 파일 로그에 그대로 들어가는 것을 방지하기 위해, `rich.text.Text.from_markup()`으로 파싱 후 `.plain` 속성으로 plain text를 추출합니다.

```python
class _PlainFormatter(logging.Formatter):
    def format(self, record):
        record.msg = Text.from_markup(str(record.msg)).plain
        return super().format(record)
```

이 방식이 정규식(`re.sub(r'\[.*?\]', '', msg)`)보다 안전한 이유:
- Rich 자체 파서를 사용하므로 중첩 태그(`[bold][green]...[/green][/bold]`)도 정확히 처리
- `[1/4]` 같은 대괄호 리터럴을 실수로 제거하지 않음 (Rich가 markup으로 인식하지 않는 패턴은 그대로 유지)

**핸들러 중복 방지:**

`setup_logging()`은 여러 번 호출될 수 있으므로 (인덱싱 → 검색 순서), RichHandler는 1회만 추가합니다:

```python
has_rich = any(isinstance(h, RichHandler) for h in logger.handlers)
if not has_rich:
    logger.addHandler(RichHandler(...))
```

FileHandler는 호출마다 추가되어 인덱싱/검색별 별도 로그 파일을 생성합니다.

**CLI에서의 Rich 활용:**

| CLI | Rich 기능 | 용도 |
|-----|-----------|------|
| `index_to_qdrant.py` | `console.rule()` | 배너 구분선 |
| `search_qdrant.py` | `rich.table.Table` | 검색 결과 표 형식 출력 |
| 패키지 내부 | `RichHandler` markup | 진행률/RPS 컬러 로깅 |

---

### 9. dataclasses — 설정 및 결과 관리

`Config`와 `SearchResult` 모두 `@dataclass`로 정의되어 타입 안전한 구조체를 간결하게 사용합니다.

```python
@dataclass
class SearchResult:
    rank: int
    score: float
    keyword: str
    point_id: int
```

`SearchResult`를 자체 dataclass로 정의하여 Qdrant SDK의 `ScoredPoint`에 직접 의존하지 않고, 패키지 인터페이스를 깔끔하게 유지합니다.
