# Embedding Search Pipeline

HuggingFace 임베딩 모델을 ONNX로 변환하여 **KServe**(Triton) 서빙하고,
**Qdrant** 벡터 DB + **Elasticsearch** 키워드 인덱싱을 통해
대규모 유사 검색 파이프라인을 구축하는 프로젝트.

## Architecture

```
HuggingFace Model → ONNX Export → KServe (Triton Inference Server)
                                       │
                                       ▼ embedding API
                           ┌───────────┴───────────┐
                           ▼                       ▼
                    Qdrant (Vector DB)      Elasticsearch
                    키워드 벡터 인덱싱       키워드 텍스트 인덱싱
                           │                       │
                           └───────────┬───────────┘
                                       ▼
                          Product → Keyword Linking
                          (상품-키워드 유사도 매칭)
```

## Quick Start

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 인프라 실행 (ES + Qdrant)
docker compose up -d

# 3. ONNX 모델 변환
python export_onnx.py --model cl-nagoya/ruri-v3-310m --output-dir output_onnx

# 4. 임베딩 서버 실행
python kserve_server.py --model-name ruri_v3 --onnx-path output_onnx/model.onnx

# 5. 키워드 인덱싱 (Qdrant + ES)
python index_to_qdrant.py --limit 1000
python index_to_es.py --limit 1000

# 6. 상품-키워드 연결
python link_products_to_keywords.py --limit 100
```

## Supported Models

`export_onnx.py`로 ONNX 변환 가능한 HuggingFace 임베딩 모델이면 사용 가능합니다.

| 모델 | 차원 | 언어 | 비고 |
|------|------|------|------|
| `cl-nagoya/ruri-v3-310m` | 768 | 일본어 | 기본 모델, 비대칭 검색 프리픽스 지원 |
| `intfloat/multilingual-e5-large` | 1024 | 다국어 | `"query: "` / `"passage: "` 프리픽스 |
| `BAAI/bge-m3` | 1024 | 다국어 | Dense + Sparse 하이브리드 |
| 기타 HuggingFace 모델 | - | - | ONNX export 가능한 Encoder 모델 |

## Project Structure

```
embedding-search-pipeline/
├── docker-compose.yml           # ES + Qdrant 인프라
├── requirements.txt             # Python 의존성
│
├── export_onnx.py               # HuggingFace → ONNX 변환
├── kserve_server.py             # KServe 임베딩 서버
├── test_client.py               # 서버 테스트 클라이언트
│
├── generate_keywords.py         # 키워드 데이터 생성
├── generate_products.py         # 상품 데이터 생성
│
├── index_to_qdrant.py           # Qdrant 벡터 인덱싱 (CLI)
├── index_to_es.py               # ES 텍스트 인덱싱 (CLI)
├── search_qdrant.py             # Qdrant 검색 (CLI)
├── link_products_to_keywords.py # 상품-키워드 연결 파이프라인
├── bench_pipeline.py            # 벤치마크
│
├── qdrant_indexer/              # Qdrant 인덱싱 패키지
│   ├── embedder.py              #   KServe 임베딩 클라이언트
│   ├── indexer.py               #   Qdrant upsert/search
│   ├── searcher.py              #   배치 검색
│   ├── parquet_reader.py        #   Parquet 데이터 로더
│   ├── pipeline.py              #   비동기 인덱싱 파이프라인
│   └── README.md
│
├── es_indexer/                  # ES 인덱싱 패키지
│   ├── indexer.py               #   AsyncElasticsearch 래퍼
│   ├── pipeline.py              #   비동기 파이프라인 (재시도 + Dead Letter)
│   └── README.md
│
├── data/                        # 데이터 (git 제외, .gitkeep만 포함)
├── logs/                        # 로그 (git 제외)
└── PIPELINE.md                  # 파이프라인 상세 문서
```

## Key Features

- **ONNX 변환**: HuggingFace 모델 → ONNX (FP16, Operator Fusion 최적화)
- **KServe 서빙**: Triton 기반 고성능 임베딩 API
- **Qdrant 인덱싱**: 비동기 배치 벡터 인덱싱 (~3,700 vectors/sec)
- **ES 인덱싱**: asyncio 동시성 벌크 인덱싱 (~51,000 docs/sec)
- **상품-키워드 연결**: 임베딩 유사도 기반 top-K 매칭 + ES nested 저장
- **Parquet 지원**: 대용량 데이터 청크 읽기
- **ES 9 클러스터**: TLS fingerprint 인증 지원
- **재시도 + Dead Letter**: 지수 백오프 재시도, 실패 문서 JSONL 기록

## Data Flow

```
텍스트/Parquet 데이터
    ↓
KServe 임베딩 (batch)
    ↓
Qdrant 벡터 인덱싱  ──→  유사 키워드 검색 (top-K)
    │                         ↓
    │                    ES nested 문서 저장
    ↓                    (상품 ↔ 키워드 매핑)
ES 텍스트 인덱싱
```

## Documentation

- [es_indexer README](es_indexer/README.md) — ES 인덱싱 패키지 상세 (CLI, API, 성능 튜닝)
- [qdrant_indexer README](qdrant_indexer/README.md) — Qdrant 인덱싱 패키지 상세
- [PIPELINE.md](PIPELINE.md) — 전체 파이프라인 아키텍처 문서
