# Product → Keyword Linking Pipeline

상품명을 KServe로 임베딩하여 Qdrant 키워드 컬렉션에서 유사 키워드 top-K를 조회한 뒤,
결과를 Elasticsearch에 **nested** 구조로 저장하는 파이프라인.

## 목차

- [전체 시스템 아키텍처](#전체-시스템-아키텍처)
- [파이프라인 흐름](#파이프라인-흐름)
- [사전 조건](#사전-조건)
- [빠른 시작](#빠른-시작)
- [CLI 옵션](#cli-옵션)
- [ES 스키마 — Nested 구조](#es-스키마--nested-구조)
- [동시성 모델](#동시성-모델)
- [사용된 패키지 컴포넌트](#사용된-패키지-컴포넌트)
- [ruri-v3 비대칭 검색](#ruri-v3-비대칭-검색)
- [ES 검색 예제](#es-검색-예제)
- [성능 특성](#성능-특성)
- [파일 구조 요약](#파일-구조-요약)

---

## 전체 시스템 아키텍처

3개의 서비스(KServe, Qdrant, Elasticsearch)를 연결하는 파이프라인입니다.

```
                         link_products_to_keywords.py
                    ┌─────────────────────────────────────┐
                    │                                     │
  products_400k.txt │  ┌────────────────┐                 │
  ─────────────────►│  │ EmbeddingClient │                 │
   100,000 상품명    │  │ (qdrant_indexer) │                 │
                    │  └───────┬────────┘                 │
                    │          │ embed(64 products)        │
                    │          ▼                           │
                    │  ┌────────────────┐                 │
                    │  │  KServe Server  │ :8080           │
                    │  │  ruri-v3-310m   │                 │
                    │  │  768-dim FP32   │                 │
                    │  └───────┬────────┘                 │
                    │          │ (64, 768) embeddings      │
                    │          ▼                           │
                    │  ┌────────────────┐                 │
                    │  │ QdrantIndexer   │                 │
                    │  │ (qdrant_indexer) │                 │
                    │  │ .search() ×64   │                 │
                    │  └───────┬────────┘                 │
                    │          │ 64 × top-30 results       │
                    │          ▼                           │
                    │  ┌────────────────┐   ┌──────────┐ │
                    │  │    Qdrant       │   │ Elastic- │ │
                    │  │  keywords 컬렉션 │   │ search   │ │
                    │  │  96k vectors    │   │ :9200    │ │
                    │  │  :6333          │   │          │ │
                    │  └────────────────┘   └────┬─────┘ │
                    │                            ▲       │
                    │          async_bulk ────────┘       │
                    │          nested {keyword, id, score} │
                    └─────────────────────────────────────┘
```

---

## 파이프라인 흐름

5단계로 구성됩니다.

```
[1/5] 상품 로드
  │   products_400k.txt에서 100,000건 로드
  ▼
[2/5] 클라이언트 초기화
  │   EmbeddingClient (KServe) + QdrantIndexer + ESIndexer
  ▼
[3/5] ES 인덱스 생성
  │   bulk: 재생성 (refresh=-1, translog=async)
  │   realtime: 보존 (없으면 생성)
  ▼
[4/5] 파이프라인 실행 ─── asyncio.gather(*1563_batches)
  │                         │
  │   ┌─────────────────────┼───────────────────────┐
  │   │  Semaphore(4)       │                       │
  │   │                     ▼                       │
  │   │  ┌─ batch (64 products) ──────────────────┐ │
  │   │  │                                        │ │
  │   │  │  run_in_executor (sync thread):        │ │
  │   │  │    1. "検索クエリ: " prefix 추가        │ │
  │   │  │    2. embedder.embed(64건) → KServe    │ │
  │   │  │    3. qdrant.search() ×64 → top-30     │ │
  │   │  │                                        │ │
  │   │  │  async (event loop):                   │ │
  │   │  │    4. async_bulk → ES (64 nested docs) │ │
  │   │  │                                        │ │
  │   │  └────────────────────────────────────────┘ │
  │   └─────────────────────────────────────────────┘
  ▼
[5/5] 최적화 + 검증
      bulk: finalize (refresh + force merge)
      문서 수 확인 + 샘플 결과 출력
```

### 배치 단위 데이터 흐름

```
products (64건)
  │
  ▼ "検索クエリ: " + product_name
["検索クエリ: 贅沢 登山リュック M", "検索クエリ: PURE COTTON ボディバッグ ...", ...]
  │
  ▼ EmbeddingClient.embed()   → KServe HTTP POST
np.ndarray (64, 768)
  │
  ▼ QdrantIndexer.search(vector, top_k=30) ×64회
[
  [ScoredPoint(id=35577, score=0.874, payload={"keyword": "..."}), ...],  # product 1
  [ScoredPoint(id=12345, score=0.891, payload={"keyword": "..."}), ...],  # product 2
  ...
]
  │
  ▼ ES bulk actions 구성
[
  {"_id": "贅沢 登山リュック M", "_source": {
      "product_name": "贅沢 登山リュック M",
      "keywords": [
          {"keyword": "ファッション 便利な 登山", "keyword_id": 35577, "score": 0.8742},
          {"keyword": "一戸建て 快適な 登山",     "keyword_id": 43335, "score": 0.8707},
          ...  // 30건
      ]
  }},
  ...  // 64건
]
  │
  ▼ async_bulk → Elasticsearch
  완료
```

---

## 사전 조건

### 1. 인프라 실행

```bash
# Qdrant + Elasticsearch
docker compose up -d

# KServe 임베딩 서버
python kserve_server.py --http_port 8080
```

### 2. Qdrant keywords 컬렉션 인덱싱 완료

```bash
# 아직 안 했다면:
python index_to_qdrant.py --workers 8 --batch_size 64
```

### 3. 의존 패키지

| 패키지 | 용도 | 설치 |
|--------|------|------|
| `qdrant_indexer` | EmbeddingClient, QdrantIndexer | (로컬 패키지) |
| `es_indexer` | ESIndexer | (로컬 패키지) |
| `elasticsearch[async]` | AsyncElasticsearch + async_bulk | `pip install "elasticsearch[async]"` |
| `requests` | KServe HTTP 클라이언트 | (qdrant_indexer 의존) |
| `qdrant-client` | Qdrant SDK | (qdrant_indexer 의존) |
| `numpy` | 임베딩 배열 처리 | (qdrant_indexer 의존) |

---

## 빠른 시작

```bash
# 테스트 (10건)
python link_products_to_keywords.py --limit 10

# 소규모 테스트 (1000건)
python link_products_to_keywords.py --limit 1000 --workers 4

# 본 실행 (100k건)
python link_products_to_keywords.py --limit 100000 --workers 4

# Realtime 모드 (기존 인덱스 보존)
python link_products_to_keywords.py --mode realtime --limit 1000

# 커스텀 설정
python link_products_to_keywords.py \
  --products data/products_400k.txt \
  --limit 50000 \
  --batch_size 128 \
  --workers 8 \
  --top_k 20 \
  --index my_product_keywords
```

### 실행 출력 예시

```
============================================================
  Product → Keyword Linking Pipeline
============================================================
  Log: logs/link_products_20260208_231109.log

  [1/5] 상품 로드
         10건

  [2/5] 클라이언트 초기화
         KServe: http://localhost:8080/v2/models/ruri_v3
         Qdrant: keywords (96,192 vectors)
         ES:     http://localhost:9200/product_keywords

  [3/5] ES 인덱스: product_keywords (mode=bulk)
         생성 완료 (refresh=-1, translog=async)

  [4/5] 파이프라인 실행 (workers=4, batch=64, top_k=30)
         [100.0%]      10/10  embed+search=289ms  es=17ms  avg=33 p/s  interval=33 p/s

  [5/5] 최적화 + 검증
         도큐먼트 수: 10
         Wall time: 0.3초
         처리량: 33 products/sec

  --- 샘플 결과 ---
  상품: 贅沢 登山リュック M
    id= 35577  score=0.8742  ファッション 便利な 登山
    id= 43335  score=0.8707  一戸建て 快適な 登山
    id= 50816  score=0.8677  ファッション 綺麗な 登山
    id= 55815  score=0.8662  ヘアスタイル 便利な 登山
    id=  6631  score=0.8654  化粧品 本格的な 登山
    ... 외 25건

  완료!
```

---

## CLI 옵션

```
python link_products_to_keywords.py [OPTIONS]
```

### 데이터

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--products` | `data/products_400k.txt` | 상품명 파일 (1줄 = 1상품) |
| `--limit` | `100000` | 처리할 상품 수 (0=전체) |
| `--top_k` | `30` | 상품당 연결할 키워드 수 |

### 처리

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--batch_size` | `64` | KServe 임베딩 배치 크기 |
| `--workers` | `4` | 동시 처리 워커 수 |
| `--mode` | `bulk` | `bulk`=인덱스 재생성, `realtime`=기존 보존 |

### 서비스 연결

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--kserve_url` | `http://localhost:8080` | KServe 서버 URL |
| `--qdrant_url` | `http://localhost:6333` | Qdrant 서버 URL |
| `--es_url` | `http://localhost:9200` | Elasticsearch URL |
| `--model` | `ruri_v3` | KServe 모델 이름 |
| `--collection` | `keywords` | Qdrant 컬렉션 이름 |
| `--index` | `product_keywords` | ES 인덱스 이름 |

---

## ES 스키마 — Nested 구조

### 스키마 정의

```json
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  },
  "mappings": {
    "properties": {
      "product_name": {
        "type": "text",
        "analyzer": "standard",
        "fields": {"raw": {"type": "keyword"}}
      },
      "keywords": {
        "type": "nested",
        "properties": {
          "keyword":    {"type": "text", "fields": {"raw": {"type": "keyword"}}},
          "keyword_id": {"type": "integer"},
          "score":      {"type": "float"}
        }
      }
    }
  }
}
```

### 저장되는 문서 구조

```json
{
  "_index": "product_keywords",
  "_id": "贅沢 登山リュック M",
  "_source": {
    "product_name": "贅沢 登山リュック M",
    "keywords": [
      {"keyword": "ファッション 便利な 登山", "keyword_id": 35577, "score": 0.8742},
      {"keyword": "一戸建て 快適な 登山",     "keyword_id": 43335, "score": 0.8707},
      {"keyword": "ファッション 綺麗な 登山", "keyword_id": 50816, "score": 0.8677},
      ...
    ]
  }
}
```

### 필드 설명

| 필드 | ES 타입 | 출처 | 설명 |
|------|---------|------|------|
| `_id` | — | 상품명 | ES 문서 고유 ID (재실행 시 자동 덮어쓰기) |
| `product_name` | `text` | products_400k.txt | 상품명 전문 검색용 |
| `product_name.raw` | `keyword` | (sub-field) | 상품명 정확 매칭/집계용 |
| `keywords` | `nested` | Qdrant 검색 결과 | 유사 키워드 top-K 배열 |
| `keywords.keyword` | `text` | Qdrant payload | 키워드 텍스트 |
| `keywords.keyword_id` | `integer` | Qdrant point ID | Qdrant 벡터 ID (역추적용) |
| `keywords.score` | `float` | Qdrant cosine | 코사인 유사도 (0~1) |

### Nested vs Object 타입

| | `nested` | `object` |
|--|---------|---------|
| 내부 저장 | 각 항목 = 독립 Lucene 문서 | 필드별 flat 배열 |
| 쿼리 정확성 | `score > 0.9 AND keyword = 登山` 정확 매칭 | cross-object 잘못된 매칭 가능 |
| 인덱스 크기 | 상품당 +30 내부 문서 | 상품당 1 문서 |
| 쿼리 | `nested` 쿼리 필수 | 일반 쿼리 가능 |

이 파이프라인에서는 "score가 0.9 이상인 키워드만 조회" 같은 복합 조건이 필요하므로 `nested`를 사용합니다.

---

## 동시성 모델

### 문제: Sync + Async 혼합

| 컴포넌트 | 동기/비동기 | 라이브러리 |
|----------|-----------|-----------|
| `EmbeddingClient.embed()` | **sync** | `requests.Session` |
| `QdrantIndexer.search()` | **sync** | `qdrant_client` |
| `ESIndexer.es` (async_bulk) | **async** | `AsyncElasticsearch` |

### 해결: asyncio + ThreadPoolExecutor 브릿지

```
asyncio 이벤트 루프 (메인)
  │
  ├── asyncio.Semaphore(workers=4)     ← 동시 배치 수 제한
  │
  ├── batch 1 ─┐
  ├── batch 2 ─┤
  ├── batch 3 ─┤  asyncio.gather()로 동시 실행
  ├── batch 4 ─┤
  │    ...     │
  │            ▼
  │   ┌──────────────────────────────────────────┐
  │   │ async with semaphore:                    │
  │   │                                          │
  │   │   # sync 작업 → ThreadPoolExecutor       │
  │   │   loop.run_in_executor(executor, fn)     │
  │   │     ├── embed(64건)     ~5000ms          │
  │   │     └── search() ×64   ~640ms           │
  │   │                                          │
  │   │   # async 작업 → 이벤트 루프 직접         │
  │   │   await async_bulk(es, 64 docs)  ~50ms  │
  │   └──────────────────────────────────────────┘
  │
  └── 통계 업데이트 (이벤트 루프에서 순차 → Lock 불필요)
```

**embed + qdrant search를 하나의 executor 호출로 묶은 이유:**
- 두 작업 모두 sync이므로 이벤트 루프로 돌아올 필요 없음
- 스케줄링 오버헤드 1회로 감소
- 한 스레드 내에서 embed 결과를 즉시 search에 사용

**Semaphore가 ThreadPoolExecutor와 별도인 이유:**
- ThreadPoolExecutor의 `max_workers`는 OS 스레드 수만 제한
- Semaphore는 동시에 진행 중인 전체 배치 수를 제한
- 메모리 사용량 제어 (N개 배치의 임베딩 + 검색 결과가 동시에 메모리에)

---

## 사용된 패키지 컴포넌트

**기존 패키지를 수정 없이 사용합니다.**

### qdrant_indexer 패키지

| 컴포넌트 | 사용 목적 | 호출 방식 |
|----------|----------|----------|
| `EmbeddingClient(url, model)` | 상품명 → 768-dim 벡터 | `.embed(texts) → np.ndarray` |
| `QdrantIndexer(url, collection, dim)` | 키워드 유사도 검색 | `.search(vector, top_k) → QueryResponse` |

### es_indexer 패키지

| 컴포넌트 | 사용 목적 | 호출 방식 |
|----------|----------|----------|
| `ESIndexer(url, index)` | ES 인덱스 관리 | `.create_index(schema)`, `.ensure_index(schema)` |
| `ESIndexer.es` | 커스텀 문서 bulk 인덱싱 | `async_bulk(indexer.es, actions)` 직접 호출 |
| `ESIndexer.finalize()` | 벌크 후 최적화 | refresh + force merge |
| `ESIndexer.count()` | 문서 수 확인 | 검증용 |

> **왜 `ESIndexer.bulk_index()`를 사용하지 않나?**
> `bulk_index(start_id, keywords: list[str])`는 `{"keyword": str}` 구조로 하드코딩되어 있어
> nested 문서 구조를 인덱싱할 수 없습니다.
> 대신 `ESIndexer.es` (underlying `AsyncElasticsearch` 인스턴스)에 직접
> `elasticsearch.helpers.async_bulk`을 호출하여 커스텀 document 구조를 인덱싱합니다.

---

## ruri-v3 비대칭 검색

ruri-v3 모델은 query와 document에 서로 다른 prefix를 사용하는 비대칭 검색을 지원합니다.

| 용도 | Prefix | 사용 위치 |
|------|--------|----------|
| 검색 쿼리 | `"検索クエリ: "` | **이 파이프라인** (상품명 임베딩 시) |
| 검색 문서 | `"検索文書: "` | (사용하지 않음) |
| 의미적 인코딩 | (없음) | Qdrant 키워드 인덱싱 시 |

**이 파이프라인에서의 역할:**
- 상품명 = 검색 **쿼리** → `"検索クエリ: "` prefix 추가
- Qdrant의 키워드 = 검색 **대상** (prefix 없이 인덱싱됨)
- 비대칭 임베딩으로 query-document 간 유사도 측정 정확도 향상

```python
# 파이프라인 내부에서:
QUERY_PREFIX = "検索クエリ: "
prefixed = [QUERY_PREFIX + p for p in products]
embeddings = embedder.embed(prefixed)  # → KServe 전송
```

---

## ES 검색 예제

인덱싱 후 Elasticsearch에서 다양한 방식으로 조회할 수 있습니다.

### 상품명으로 검색

```bash
curl -s 'localhost:9200/product_keywords/_search?pretty' \
  -H 'Content-Type: application/json' \
  -d '{"query": {"match": {"product_name": "登山 リュック"}}}'
```

### 특정 키워드가 포함된 상품 검색 (nested query)

```bash
curl -s 'localhost:9200/product_keywords/_search?pretty' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "nested": {
        "path": "keywords",
        "query": {
          "match": {"keywords.keyword": "登山"}
        }
      }
    }
  }'
```

### score 기준 필터링 (nested + range)

```bash
curl -s 'localhost:9200/product_keywords/_search?pretty' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "nested": {
        "path": "keywords",
        "query": {
          "bool": {
            "must": [
              {"match": {"keywords.keyword": "登山"}},
              {"range": {"keywords.score": {"gte": 0.85}}}
            ]
          }
        }
      }
    }
  }'
```

### 상품 ID로 직접 조회

```bash
# _id = 상품명
curl -s 'localhost:9200/product_keywords/_doc/贅沢 登山リュック M?pretty'
```

### 문서 수 확인

```bash
curl -s 'localhost:9200/product_keywords/_count?pretty'
```

### Python에서 조회

```python
import asyncio
from es_indexer import ESIndexer

async def main():
    es = ESIndexer("http://localhost:9200", "product_keywords")

    # nested query: score >= 0.85인 키워드가 있는 상품
    result = await es.es.search(
        index="product_keywords",
        query={
            "nested": {
                "path": "keywords",
                "query": {
                    "range": {"keywords.score": {"gte": 0.85}}
                },
                "inner_hits": {"size": 3}  # 매칭된 키워드도 반환
            }
        },
        size=5,
    )

    for hit in result["hits"]["hits"]:
        print(f"\n상품: {hit['_source']['product_name']}")
        for inner in hit["inner_hits"]["keywords"]["hits"]["hits"]:
            kw = inner["_source"]
            print(f"  score={kw['score']:.4f}  {kw['keyword']}")

    await es.close()

asyncio.run(main())
```

---

## 성능 특성

### 병목 분석

```
┌──────────────────┬─────────┬───────────────────────────┐
│ 단계              │ 소요시간 │ 특성                       │
├──────────────────┼─────────┼───────────────────────────┤
│ embed (64건)     │ ~5000ms │ GPU 추론, 최대 병목         │
│ search ×64       │ ~640ms  │ HNSW 검색, I/O 바운드      │
│ async_bulk (64건) │ ~50ms   │ ES HTTP, 매우 빠름         │
└──────────────────┴─────────┴───────────────────────────┘

1 batch 총 소요: ~5.7초 (embed 지배적)
```

### 예상 처리량

| 규모 | workers | batch | 예상 시간 | 비고 |
|:----:|:-------:|:-----:|:---------:|------|
| 10건 | 4 | 64 | ~0.3초 | 테스트용 |
| 1,000건 | 4 | 64 | ~1.5분 | 소규모 검증 |
| 10,000건 | 4 | 64 | ~15분 | 중규모 |
| 100,000건 | 4 | 64 | ~25분 | 기본 설정 |
| 100,000건 | 8 | 64 | ~18분 | workers 증가 |

> 실제 처리량은 KServe 서버의 GPU 성능에 크게 의존합니다.

### 튜닝 가이드

| 파라미터 | 권장값 | 주의사항 |
|---------|--------|---------|
| `batch_size` | **64** | KServe 모델의 최적 배치 크기. 128 이상은 이점 미미 |
| `workers` | **4-8** | GPU 유휴 시간을 줄이는 I/O 병렬화. 너무 크면 KServe 과부하 |
| `top_k` | **30** | 값이 클수록 Qdrant 검색+ES 문서 크기 증가 |
| `mode` | `bulk` | 초기 적재 시. `realtime`은 매 배치 refresh로 느림 |

### 메모리 사용량

```
동시 배치 수 (workers) × 배치당 메모리:
  임베딩: 64 × 768 × 4 bytes = 196 KB
  Qdrant 결과: 64 × 30 결과 × ~100 bytes = 192 KB
  ES actions: 64 문서 × ~3 KB = 192 KB
  ────────────────────────────
  배치당 ~580 KB × 4 workers ≈ 2.3 MB (매우 적음)
```

---

## 파일 구조 요약

```
m1-inference/
├── link_products_to_keywords.py  ★ 이 파이프라인
├── PIPELINE.md                   ★ 이 문서
│
├── qdrant_indexer/                 (수정 없이 사용)
│   ├── embedder.py                 EmbeddingClient — KServe V2 HTTP
│   ├── indexer.py                  QdrantIndexer — .search(vector, top_k)
│   └── ...
│
├── es_indexer/                     (수정 없이 사용)
│   ├── indexer.py                  ESIndexer — .es, .create_index(), .finalize()
│   └── ...
│
├── data/
│   ├── keywords_400k.txt           Qdrant에 인덱싱된 키워드 원본
│   └── products_400k.txt           상품명 (이 파이프라인의 입력)
│
├── logs/
│   └── link_products_*.log         파이프라인 실행 로그
│
├── docker-compose.yml              Qdrant(:6333) + Elasticsearch(:9200)
└── kserve_server.py                KServe 임베딩 서버(:8080)
```
