# es_indexer — Elasticsearch 벌크/실시간 인덱싱 패키지

Elasticsearch에 대량의 텍스트 데이터를 **asyncio 기반 동시성**으로 고속 인덱싱하는 Python 패키지.
**Bulk 모드**(초기 대량 적재)와 **Realtime 모드**(운영 중 실시간 CRUD)를 모두 지원.

## 목차

- [두 가지 모드 비교](#두-가지-모드-비교)
- [아키텍처](#아키텍처)
- [사전 조건](#사전-조건)
- [빠른 시작](#빠른-시작)
- [CLI 사용법](#cli-사용법)
- [Parquet 데이터 소스](#parquet-데이터-소스)
- [ES 9 클러스터 연결](#es-9-클러스터-연결)
- [재시도 / Dead Letter](#재시도--dead-letter)
- [Python API](#python-api)
  - [Config](#config)
  - [DEFAULT_SCHEMA](#default_schema)
  - [ESIndexer](#esindexer)
  - [build_es_client](#build_es_client)
  - [run_indexing / run_realtime](#run_indexing--run_realtime)
- [Realtime CRUD 예제](#realtime-crud-예제)
- [커스텀 스키마](#커스텀-스키마)
- [성능 최적화 상세](#성능-최적화-상세)
- [파이프라인 내부 동작](#파이프라인-내부-동작)
- [성능 벤치마크](#성능-벤치마크)

---

## 두 가지 모드 비교

| | Bulk 모드 | Realtime 모드 |
|--|----------|--------------|
| **용도** | 초기 대량 적재, 인덱스 재구축 | 운영 중 추가/수정/삭제 |
| **인덱스** | 삭제 후 재생성 (`create_index`) | 없으면 생성, 있으면 보존 (`ensure_index`) |
| **refresh** | `-1` (비활성) → 끝에 복원 | `1s` 유지 + 배치마다 즉시 refresh |
| **translog** | `async` (유실 가능) | `request` (안전) |
| **검색 가능 시점** | `finalize()` 이후 | 각 배치 즉시 |
| **처리량** | ~50,000 docs/sec | ~2,000 docs/sec |
| **CRUD** | bulk_index만 | index / update / delete / search / get |

---

## 아키텍처

```
index_to_es.py (CLI)
    │
    ▼
es_indexer/
├── config.py      ← Config 데이터클래스 + DEFAULT_SCHEMA
├── indexer.py      ← ESIndexer (AsyncElasticsearch 래퍼)
├── pipeline.py     ← run_indexing() 비동기 파이프라인
└── __init__.py     ← 패키지 public API
```

**데이터 흐름:**

```
텍스트 파일 (1줄=1문서)  또는  Parquet 파일/디렉토리
    │                              │
    ▼  keywords_path               ▼  parquet_path (우선)
[키워드 리스트]  ←──────────  ParquetReader.iter_chunks()
    │
    ▼  batch_size 단위로 분할
[배치 1] [배치 2] [배치 3] ... [배치 N]
    │       │       │              │
    ▼       ▼       ▼              ▼     ← asyncio.Semaphore(workers)
  bulk    bulk    bulk    ...    bulk    ← ES _bulk API
    │       │       │              │
    ▼       ▼       ▼              ▼     ← 실패 시 재시도 (지수 백오프)
         Elasticsearch 인덱스           ← 최종 실패 → Dead Letter JSONL
    │
    ▼  finalize()
  refresh → force_merge → 완료
```

---

## 사전 조건

```bash
# 1. Elasticsearch 실행 (docker-compose.yml 기준)
docker compose up -d

# 2. Python 의존성 설치
pip install "elasticsearch[async]"
```

| 의존성 | 버전 | 용도 |
|--------|------|------|
| `elasticsearch[async]` | >=8.0 | AsyncElasticsearch + async_bulk |
| `aiohttp` | (자동 설치) | elasticsearch[async]의 내부 비동기 HTTP 전송 |
| `rich` | >=13.0 | 콘솔 Progress bar + 로깅 |
| `pyarrow` | (선택) | Parquet 파일 읽기 (`--parquet` 사용 시 필요) |

> **ES 9 클러스터**: TLS가 기본 활성화되어 있으므로 `--es_fingerprint`와 인증 정보가 필수입니다. [ES 9 클러스터 연결](#es-9-클러스터-연결) 섹션을 참고하세요.

---

## 빠른 시작

### CLI

```bash
# Bulk 모드 — 100건 테스트
python index_to_es.py --limit 100

# Bulk 모드 — 전체 400k 인덱싱
python index_to_es.py --keywords data/products_400k.txt --workers 8

# Realtime 모드 — 기존 인덱스에 추가
python index_to_es.py --mode realtime --limit 100

# Realtime 모드 — 새 데이터 파일 추가 적재
python index_to_es.py --mode realtime --keywords data/products_400k.txt
```

### CLI — Parquet

```bash
# Parquet 파일 읽기
python index_to_es.py --parquet data/keywords.parquet --limit 10000

# Parquet 디렉토리 (여러 파일)
python index_to_es.py --parquet data/keywords_partitioned/ --workers 16

# 컬럼명 지정
python index_to_es.py --parquet data/products.parquet --parquet_column product_name
```

### CLI — ES 9 클러스터

```bash
# ES 9 클러스터 + fingerprint 인증
python index_to_es.py --limit 100 \
    --es_nodes https://es01:9200 https://es02:9200 https://es03:9200 \
    --es_fingerprint "B1:2A:96:D3:6E:..." \
    --es_username elastic --es_password changeme

# API Key 방식
python index_to_es.py --limit 100 \
    --es_nodes https://es01:9200 \
    --es_fingerprint "B1:2A:96:D3:6E:..." \
    --es_api_key "VnVhQ2ZHY0JDZGJrU..."
```

### Python

```python
from es_indexer import Config, run_indexing, run_realtime

# Bulk 모드 — 로컬
config = Config(keywords_path="data/products_400k.txt", workers=8)
run_indexing(config)

# Bulk 모드 — ES 9 클러스터
config = Config(
    keywords_path="data/products_400k.txt",
    es_nodes=["https://es01:9200", "https://es02:9200"],
    es_fingerprint="B1:2A:96:D3:6E:...",
    es_username="elastic",
    es_password="changeme",
    workers=16,
)
run_indexing(config)

# Realtime 모드
config = Config(keywords_path="data/new_products.txt", workers=4)
run_realtime(config)
```

---

## CLI 사용법

```
python index_to_es.py [OPTIONS]
```

**기본 옵션:**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--mode` | `bulk` | `bulk`=인덱스 재생성+대량적재, `realtime`=인덱스 보존+즉시리프레시 |
| `--keywords` | `data/keywords_400k.txt` | 입력 텍스트 파일 경로 (1줄 = 1문서) |
| `--limit` | `0` | 처리할 최대 건수 (0 = 전체) |
| `--batch_size` | `500` | ES _bulk API 1회 호출당 문서 수 |
| `--workers` | `8` | 동시 실행 코루틴 수 |
| `--index` | `keywords` | ES 인덱스 이름 |
| `--es_url` | `http://localhost:9200` | ES 접속 URL (단일 노드) |
| `--schema` | `None` | 커스텀 스키마 JSON 파일 경로 |

**데이터 소스 옵션:**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--parquet` | `None` | Parquet 파일 또는 디렉토리 경로 (설정 시 `--keywords` 무시) |
| `--parquet_chunk_size` | `10000` | Parquet 청크 크기 (행 수) |
| `--parquet_column` | `keyword` | 텍스트가 들어있는 컬럼명 |

**재시도 / 실패 처리 옵션:**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--max_retries` | `3` | 배치 실패 시 최대 재시도 횟수 (0=비활성) |
| `--retry_delay` | `1.0` | 첫 재시도 대기 시간 (초, 이후 ×2 지수 백오프) |
| `--dead_letter_path` | `None` | 실패 문서 JSONL 파일 경로 (미지정 시 `logs/`에 자동 생성) |

**ES 클러스터 연결 옵션 (ES 9+):**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--es_nodes` | `None` | 클러스터 노드 URL 목록 (설정 시 `--es_url` 무시) |
| `--es_fingerprint` | `None` | TLS 인증서 SHA-256 fingerprint (`--es_nodes` 사용 시 **필수**) |
| `--es_username` | `None` | Basic Auth 사용자명 |
| `--es_password` | `None` | Basic Auth 비밀번호 |
| `--es_api_key` | `None` | API Key (`--es_username`/`--es_password` 대신 사용) |

> `--es_nodes` 지정 시 `--es_fingerprint`와 인증 정보(`--es_username`+`--es_password` 또는 `--es_api_key`)가 **필수**입니다. 누락 시 `ValueError`가 발생합니다.

---

## Parquet 데이터 소스

`--parquet` 옵션으로 **Parquet 파일** 또는 **Parquet 디렉토리**(파티셔닝)를 데이터 소스로 사용할 수 있습니다.
Parquet이 지정되면 `--keywords` 텍스트 파일은 무시됩니다.

### 데이터 소스 우선순위

```
--parquet 지정?
    │
    ├─ Yes → ParquetReader.iter_chunks()로 청크 단위 읽기
    │         └─ 디렉토리면 내부 .parquet 파일 전체 스캔
    │
    └─ No  → --keywords 텍스트 파일 (1줄 = 1문서)
```

### 지원 형식

| 입력 | 예시 | 동작 |
|------|------|------|
| 단일 Parquet 파일 | `data/keywords.parquet` | 파일 하나 청크 읽기 |
| Parquet 디렉토리 | `data/keywords_partitioned/` | 디렉토리 내 모든 `.parquet` 파일 순차 읽기 |
| 텍스트 파일 (기본) | `data/keywords_400k.txt` | 1줄 = 1문서, 전체 메모리 로드 |

### 의존성

Parquet 사용 시 `qdrant_indexer` 패키지의 `ParquetReader`와 `pyarrow`가 필요합니다:

```bash
pip install pyarrow
```

> `ParquetReader`가 import 되지 않으면 `--parquet` 사용 시 `ImportError`가 발생합니다.

### Python 예제

```python
from pathlib import Path
from es_indexer import Config, run_indexing

# Parquet 파일
config = Config(
    parquet_path=Path("data/keywords.parquet"),
    parquet_text_column="keyword",
    parquet_chunk_size=10000,
    workers=8,
)
run_indexing(config)

# Parquet 디렉토리 (여러 파일)
config = Config(
    parquet_path=Path("data/keywords_partitioned/"),
    parquet_text_column="product_name",
    workers=16,
)
run_indexing(config)
```

---

## ES 9 클러스터 연결

ES 9는 TLS가 기본 활성화되어 있어, 클러스터 연결 시 **fingerprint + 인증이 필수**입니다.

### Fingerprint 확인 방법

```bash
# 방법 1: ES 최초 기동 시 콘솔 출력 확인
docker logs es01 2>&1 | grep "fingerprint"

# 방법 2: openssl로 직접 추출
openssl s_client -connect es01:9200 -servername es01 < /dev/null 2>/dev/null \
  | openssl x509 -fingerprint -sha256 -noout

# 방법 3: elasticsearch-certutil (컨테이너 내부)
docker exec es01 bin/elasticsearch-certutil fingerprint --fingerprint-type sha256
```

### 연결 방식

| 방식 | 필요 정보 | 적합 환경 |
|------|----------|-----------|
| `--es_url` (단일 노드) | URL만 | 로컬 개발 (HTTP) |
| `--es_nodes` + `--es_fingerprint` + `--es_username`/`--es_password` | 노드 URL + fingerprint + Basic Auth | 프로덕션 클러스터 |
| `--es_nodes` + `--es_fingerprint` + `--es_api_key` | 노드 URL + fingerprint + API Key | 프로덕션 클러스터 (권장) |

### 검증 규칙

`--es_nodes` 지정 시 다음이 **자동으로 검증**됩니다:

1. `--es_fingerprint` 필수 — 누락 시 `ValueError`
2. 인증 정보 필수 — `--es_api_key` 또는 `--es_username` + `--es_password` 중 하나

```
# fingerprint 없이 클러스터 연결 시도 → 에러
$ python index_to_es.py --es_nodes https://es01:9200
ValueError: --es_fingerprint 필수: ES 9 클러스터 연결에는 TLS 인증서 fingerprint가 필요합니다.
```

### 연결 흐름

```
--es_nodes 지정?
    │
    ├─ No  → AsyncElasticsearch(es_url)  [HTTP, 인증 없음]
    │
    └─ Yes → fingerprint 있는가?
              │
              ├─ No  → ValueError 발생
              │
              └─ Yes → 인증 정보 있는가?
                        │
                        ├─ No  → ValueError 발생
                        │
                        └─ Yes → AsyncElasticsearch(
                                   hosts=[node1, node2, ...],
                                   ssl_assert_fingerprint="...",
                                   basic_auth=(...) 또는 api_key="..."
                                 )
```

---

## 재시도 / Dead Letter

대용량 인덱싱 시 **네트워크 장애, ES 노드 과부하** 등으로 배치가 실패할 수 있습니다.
데이터 유실 방지를 위해 **지수 백오프 재시도** + **Dead Letter Queue(DLQ)** 패턴을 지원합니다.

### 재시도 흐름

```
배치 _bulk 호출
    │
    ├─ 성공 → stats 업데이트, 다음 배치
    │
    └─ 실패 → 재시도 횟수 초과?
               │
               ├─ No  → delay 대기 (지수 백오프) → 재시도
               │         1.0s → 2.0s → 4.0s → ...
               │
               └─ Yes → Dead Letter JSONL에 기록
                         stats.failed_docs += count
```

- **첫 시도 + max_retries 재시도** = 총 `max_retries + 1`회 시도
- 재시도 간격: `retry_delay × 2^(attempt-1)` (지수 백오프)
- ES `_bulk` API는 같은 `_id`를 재전송하면 덮어쓰므로 **배치 전체 재시도가 안전** (멱등성)

### Dead Letter 파일 형식

JSONL (1줄 = 1 실패 배치):

```json
{"batch_id": 1000, "count": 500, "error": "ConnectionError(...)", "ts": "2026-02-11T06:10:45", "keywords": ["kw1", "kw2", ...]}
{"batch_id": 1500, "count": 500, "error": "TimeoutError(...)", "ts": "2026-02-11T06:11:02", "keywords": ["kw501", "kw502", ...]}
```

| 필드 | 설명 |
|------|------|
| `batch_id` | 배치 시작 인덱스 |
| `count` | 실패 문서 수 |
| `error` | 마지막 예외 메시지 |
| `ts` | 기록 시각 (ISO 8601) |
| `keywords` | 원본 키워드 리스트 (재처리용) |

### Dead Letter 파일 위치

| 설정 | 경로 |
|------|------|
| `--dead_letter_path` 지정 | 지정한 경로 그대로 사용 |
| 미지정 (기본) | 로그 파일과 같은 디렉토리에 `*.dead_letter.jsonl`로 자동 생성 |

### Dead Letter 수동 재처리

```python
import json

# Dead Letter 파일 읽기
with open("logs/es_bulk_20260211.dead_letter.jsonl") as f:
    for line in f:
        record = json.loads(line)
        keywords = record["keywords"]
        # keywords를 다시 인덱싱 (예: run_indexing 또는 직접 bulk_index)
```

### 결과 요약 테이블

파이프라인 완료 시 Rich 테이블에 재시도/실패 정보가 조건부로 표시됩니다:

```
┌─────────────┬──────────────────────────────┐
│ 항목        │                           값 │
├─────────────┼──────────────────────────────┤
│ 도큐먼트 수 │                      400,000 │
│ Wall time   │                        7.8초 │
│ 처리량      │             51,282 docs/sec  │
│ 재시도 횟수 │                          5회 │  ← retries > 0 일 때만
│ 실패 문서   │        1,000건 (2 배치)      │  ← failed_docs > 0 일 때만
│ Dead Letter │ logs/es_bulk_*.dead_letter…  │
└─────────────┴──────────────────────────────┘
```

---

## Python API

### `Config`

**모듈:** `es_indexer.config`

인덱싱 파이프라인의 모든 설정을 담는 데이터클래스.

```python
from pathlib import Path
from es_indexer import Config

# 로컬 단일 노드
config = Config(
    keywords_path="data/products_400k.txt",
    es_url="http://localhost:9200",
    index_name="products",
    workers=8,
)

# ES 9 클러스터
config = Config(
    keywords_path="data/products_400k.txt",
    es_nodes=["https://es01:9200", "https://es02:9200", "https://es03:9200"],
    es_fingerprint="B1:2A:96:D3:6E:...",
    es_username="elastic",
    es_password="changeme",
    index_name="products",
    workers=16,
)

# Parquet 파일 읽기
config = Config(
    parquet_path=Path("data/keywords.parquet"),
    parquet_chunk_size=10000,
    parquet_text_column="keyword",
    workers=8,
)
```

**데이터 필드:**

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `keywords_path` | `Path` | `data/keywords_400k.txt` | 1줄 1문서 텍스트 파일 |
| `limit` | `int` | `0` | 처리 제한 (0=전체) |
| `parquet_path` | `Path \| None` | `None` | Parquet 파일 또는 디렉토리 경로 (설정 시 `keywords_path` 무시) |
| `parquet_chunk_size` | `int` | `10000` | Parquet 청크 크기 (행 수) |
| `parquet_text_column` | `str` | `keyword` | 텍스트가 들어있는 컬럼명 |

**ES 연결 필드:**

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `es_url` | `str` | `http://localhost:9200` | 단일 노드 URL |
| `es_nodes` | `list[str] \| None` | `None` | 클러스터 노드 URL 목록 (설정 시 `es_url` 무시) |
| `es_fingerprint` | `str \| None` | `None` | TLS 인증서 SHA-256 fingerprint (`es_nodes` 시 **필수**) |
| `es_username` | `str \| None` | `None` | Basic Auth 사용자명 |
| `es_password` | `str \| None` | `None` | Basic Auth 비밀번호 |
| `es_api_key` | `str \| None` | `None` | API Key (basic_auth 대신 사용) |

**인덱스/처리 필드:**

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `index_name` | `str` | `keywords` | 인덱스 이름 |
| `schema` | `dict \| None` | `None` | ES 인덱스 스키마 (None이면 DEFAULT_SCHEMA) |
| `batch_size` | `int` | `500` | _bulk API 1회당 문서 수 |
| `workers` | `int` | `8` | asyncio.Semaphore 동시성 한도 |

**재시도 / 실패 처리 필드:**

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `max_retries` | `int` | `3` | 배치 실패 시 최대 재시도 횟수 (0=비활성) |
| `retry_delay` | `float` | `1.0` | 첫 재시도 대기 시간 (초, 이후 ×2 지수 백오프) |
| `dead_letter_path` | `Path \| None` | `None` | 실패 문서 JSONL 경로 (None=logs/에 자동 생성) |

---

### `DEFAULT_SCHEMA`

**모듈:** `es_indexer.config`

`Config.schema`가 `None`일 때 자동 적용되는 기본 인덱스 스키마.

```python
from es_indexer import DEFAULT_SCHEMA

# 구조:
DEFAULT_SCHEMA = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "keyword": {
                "type": "text",           # 전문 검색용
                "analyzer": "standard",
                "fields": {
                    "raw": {"type": "keyword"}  # 정확 매칭 / 집계용
                },
            },
        }
    },
}
```

**필드 설명:**

| 필드 | ES 타입 | 용도 |
|------|---------|------|
| `keyword` | `text` | 형태소 분석 후 전문 검색 (`match` 쿼리) |
| `keyword.raw` | `keyword` | 정확 매칭 (`term` 쿼리), 집계 (`aggs`), 정렬 |

---

### `ESIndexer`

**모듈:** `es_indexer.indexer`

Elasticsearch 인덱스 생명주기 관리 + 벌크/실시간 인덱싱을 담당하는 비동기 클래스.

```python
from es_indexer import ESIndexer, Config

# 방법 1: 단일 노드 직접 생성 (하위 호환)
indexer = ESIndexer(es_url="http://localhost:9200", index_name="products")

# 방법 2: Config 기반 생성 (클러스터 지원)
config = Config(
    es_nodes=["https://es01:9200", "https://es02:9200"],
    es_fingerprint="B1:2A:96:...",
    es_username="elastic",
    es_password="changeme",
)
indexer = ESIndexer.from_config(config, index_name="products")
```

#### 인덱스 관리

| 메서드 | 모드 | 설명 |
|--------|------|------|
| `create_index(schema=None)` | Bulk | 삭제 후 재생성. 벌크 최적화 자동 적용 |
| `ensure_index(schema=None)` | Realtime | 없으면 생성, 있으면 보존. 최적화 미적용 |

#### Bulk 전용

| 메서드 | 설명 |
|--------|------|
| `bulk_index(start_id, keywords)` | 대량 벌크 저장 (refresh 없음) |
| `finalize()` | refresh 복원 + translog 동기화 + force merge |

#### Realtime CRUD

| 메서드 | 설명 |
|--------|------|
| `index(doc_id, keyword, refresh=True)` | 단일 문서 인덱싱/upsert |
| `index_batch_realtime(start_id, keywords)` | 배치 인덱싱 + 즉시 리프레시 |
| `update(doc_id, fields, refresh=True)` | 문서 부분 업데이트 |
| `delete(doc_id, refresh=True)` | 문서 삭제 (없으면 무시) |

#### 검색 / 조회

| 메서드 | 설명 |
|--------|------|
| `search(query, size=10)` | keyword 필드 전문 검색 |
| `get(doc_id)` | ID로 문서 조회 (없으면 None) |
| `count()` | 문서 수 반환 |
| `refresh()` | 수동 리프레시 |
| `close()` | 클라이언트 정리 |

---

### `build_es_client`

**모듈:** `es_indexer.indexer`

Config 객체로부터 `AsyncElasticsearch` 클라이언트를 직접 생성. `ESIndexer`를 거치지 않고 ES 클라이언트만 필요한 경우 사용.

```python
from es_indexer import Config, build_es_client

config = Config(
    es_nodes=["https://es01:9200", "https://es02:9200"],
    es_fingerprint="B1:2A:96:...",
    es_username="elastic",
    es_password="changeme",
)
es = build_es_client(config)  # AsyncElasticsearch 인스턴스

# 직접 ES API 사용
info = await es.info()
await es.close()
```

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `config` | `Config` | 연결 설정이 담긴 Config 객체 |
| **반환** | `AsyncElasticsearch` | 설정된 ES 클라이언트 |

---

### `run_indexing` / `run_realtime`

**모듈:** `es_indexer.pipeline`

두 가지 파이프라인 함수. 내부적으로 `asyncio.run()`을 호출하므로 동기 코드에서 직접 사용 가능.

```python
from es_indexer import Config, run_indexing, run_realtime

config = Config(keywords_path="data/products_400k.txt", workers=8)

run_indexing(config)   # Bulk 모드
run_realtime(config)   # Realtime 모드
```

| 함수 | 인덱스 처리 | refresh | 후처리 |
|------|-----------|---------|--------|
| `run_indexing(config, log_path=None)` | 삭제 후 재생성 | 끝에 한번 | finalize (force merge) |
| `run_realtime(config, log_path=None)` | 보존 (없으면 생성) | 매 배치 즉시 | 샘플 검색 확인 |

**로그 출력:** `logs/es_bulk_*.log` 또는 `logs/es_realtime_*.log`

---

## Realtime CRUD 예제

`ESIndexer`를 직접 사용한 async CRUD:

```python
import asyncio
from es_indexer import ESIndexer, Config

async def main():
    # 로컬: ESIndexer("http://localhost:9200", "products")
    # 클러스터:
    config = Config(
        es_nodes=["https://es01:9200", "https://es02:9200"],
        es_fingerprint="B1:2A:96:...",
        es_username="elastic", es_password="changeme",
    )
    indexer = ESIndexer.from_config(config, index_name="products")

    # 인덱스 보존 (없으면 생성)
    await indexer.ensure_index()

    # 단일 문서 추가 — 즉시 검색 가능
    await indexer.index("prod_001", "北海道産 プレミアム ラーメン 5食入り")
    await indexer.index("prod_002", "京都 宇治抹茶 ケーキ ギフト用")

    # 배치 추가 — 즉시 리프레시
    await indexer.index_batch_realtime(
        start_id=100,
        keywords=["商品A", "商品B", "商品C"],
    )

    # 검색
    results = await indexer.search("ラーメン")
    for hit in results["hits"]["hits"]:
        print(hit["_source"]["keyword"])

    # ID로 조회
    doc = await indexer.get("prod_001")
    print(doc)  # {"keyword": "北海道産 プレミアム ラーメン 5食入り"}

    # 부분 업데이트
    await indexer.update("prod_001", {"keyword": "北海道産 極上 ラーメン 10食入り"})

    # 삭제
    await indexer.delete("prod_002")

    # 확인
    print(await indexer.count())

    await indexer.close()

asyncio.run(main())
```

---

## 커스텀 스키마

### JSON 파일로 지정 (CLI)

```json
// custom_schema.json
{
  "settings": {
    "number_of_shards": 2,
    "number_of_replicas": 1,
    "analysis": {
      "analyzer": {
        "ja_analyzer": {
          "type": "custom",
          "tokenizer": "kuromoji_tokenizer"
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "keyword": {
        "type": "text",
        "analyzer": "ja_analyzer"
      }
    }
  }
}
```

```bash
python index_to_es.py --schema custom_schema.json
```

### Python에서 dict로 지정

```python
from es_indexer import Config, run_indexing

schema = {
    "settings": {"number_of_shards": 2},
    "mappings": {
        "properties": {
            "keyword": {"type": "text"},
            "category": {"type": "keyword"},
        }
    },
}

config = Config(schema=schema, index_name="products")
run_indexing(config)
```

벌크 최적화 설정(`refresh_interval`, `translog.durability`)은
`setdefault`로 머지되므로 사용자가 명시하지 않아도 자동 적용됨.

---

## 성능 최적화 상세

### 벌크 인덱싱 시 적용되는 ES 최적화

| 설정 | 벌크 중 값 | 완료 후 복원 값 | 효과 |
|------|-----------|----------------|------|
| `refresh_interval` | `"-1"` (비활성) | `"1s"` | 리프레시 생략 → 쓰기 2-5x 가속 |
| `translog.durability` | `"async"` | `"request"` | 비동기 fsync → 디스크 I/O 대기 제거 |
| `translog.sync_interval` | `"30s"` | (기본값) | 트랜스로그 동기화 간격 완화 |
| `number_of_replicas` | `0` | (유지) | 복제 없음 → 쓰기 부하 절감 |
| `forcemerge` | — | 완료 후 실행 | 세그먼트 1개로 병합 → 쿼리 최적화 |

### asyncio 동시성 모델

```
asyncio.gather(*800_tasks)     ← 전체 배치를 태스크로 생성
    │
    ▼
asyncio.Semaphore(workers=8)   ← 동시 실행은 8개로 제한
    │
    ├─ 코루틴 1: bulk 500 docs → ES
    ├─ 코루틴 2: bulk 500 docs → ES     ← 동시 실행
    ├─ ...
    ├─ 코루틴 8: bulk 500 docs → ES
    │
    ├─ 코루틴 9: (대기 중, Semaphore 획득 대기)
    └─ ...
```

- Semaphore가 해제되면 다음 코루틴이 즉시 시작
- 코루틴 간 전환은 `await` 지점에서 발생 (OS 스레드 전환 없음)
- AsyncElasticsearch 내부의 aiohttp 커넥션 풀이 HTTP 연결 재사용

### 튜닝 가이드

| workers | batch_size | 적합한 환경 |
|---------|-----------|-------------|
| 4 | 500 | 단일 노드 ES, 로컬 개발 |
| 8 | 500 | 단일 노드 ES, 프로덕션 |
| 16 | 1000 | 다중 노드 ES 클러스터 |
| 32 | 2000 | 대규모 클러스터 + 고성능 네트워크 |

`workers` 를 ES 노드 수 × 2 정도로 설정하면 최적.
`batch_size`가 너무 크면 ES의 `http.max_content_length` (기본 100MB)에 걸릴 수 있음.

---

## 파이프라인 내부 동작

### _Stats 클래스 (pipeline.py)

실시간 진행 통계를 수집하고 로그에 출력하는 내부 클래스.

```
[  5.0%]  20,000/400,000  bulk=131ms  avg=27,606 t/s  interval=27,362 t/s
```

| 필드 | 의미 |
|------|------|
| `bulk=131ms` | 해당 배치의 _bulk API 응답 시간 |
| `avg=27,606 t/s` | 시작부터 현재까지의 평균 처리량 |
| `interval=27,362 t/s` | 직전 구간의 순간 처리량 |

로그 출력 조건: 처음 500건, 이후 5000건마다, 마지막 배치.

### 파일 구조 상세

```
es_indexer/
├── __init__.py     # Public API: Config, DEFAULT_SCHEMA, ESIndexer, build_es_client,
│                   #   run_indexing, run_realtime
├── config.py       # Config 데이터클래스
│                   #   데이터 소스 (텍스트/Parquet), ES 연결/클러스터, 재시도 설정
│                   #   DEFAULT_SCHEMA
├── indexer.py      # build_es_client() — Config → AsyncElasticsearch 팩토리
│                   # ESIndexer — AsyncElasticsearch 래퍼
│                   #   .from_config() 클러스터 연결, create_index(), bulk_index(), ...
├── pipeline.py     # run_indexing() / run_realtime() — Rich 로깅 비동기 파이프라인
│                   #   _Stats (Progress bar + 재시도/실패 카운터)
│                   #   _DeadLetterWriter (실패 배치 JSONL 기록)
│                   #   _process_batch (재시도 + 지수 백오프 + Dead Letter)
│                   #   Parquet 로드: ParquetReader (qdrant_indexer에서 import)
└── README.md       # 이 문서
```

---

## 성능 벤치마크

docker-compose.yml 기반 ES 9.0 단일 노드 (macOS, 1GB heap):

### Bulk 모드

| 데이터 | workers | batch_size | Wall Time | 처리량 |
|--------|---------|-----------|-----------|--------|
| 100건 | 4 | 500 | 0.02초 | ~4,500 docs/sec |
| 1,000건 | 8 | 500 | 0.2초 | ~5,000 docs/sec |
| 400,000건 | 8 | 500 | 7.8초 | **~51,500 docs/sec** |

### Realtime 모드

| 데이터 | workers | batch_size | Wall Time | 처리량 |
|--------|---------|-----------|-----------|--------|
| 50건 (신규) | 8 | 500 | 0.02초 | ~2,100 docs/sec |
| 100건 (추가) | 8 | 500 | 0.1초 | ~760 docs/sec |

Realtime은 매 배치마다 `indices.refresh()` 호출로 인해 Bulk 대비 처리량이 낮지만,
각 배치 완료 즉시 모든 문서가 검색 가능해지는 트레이드오프.
