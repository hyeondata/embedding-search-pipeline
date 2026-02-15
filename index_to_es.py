#!/usr/bin/env python3
# index_to_es.py
"""
키워드 → Elasticsearch 인덱싱 (CLI 엔트리포인트)

사전 조건:
  Elasticsearch:  docker compose up -d

실행:
  # 로컬 단일 노드 (기존과 동일)
  python index_to_es.py --limit 100
  python index_to_es.py --workers 16 --batch_size 1000

  # Parquet 파일 읽기
  python index_to_es.py --parquet data/keywords.parquet --limit 10000

  # Parquet 디렉토리 읽기 (여러 파일)
  python index_to_es.py --parquet data/keywords_partitioned/ --workers 16

  # Realtime 모드
  python index_to_es.py --mode realtime --limit 100

  # ES 9 클러스터 + fingerprint 인증
  python index_to_es.py --limit 100 \\
      --es_nodes https://es01:9200 https://es02:9200 https://es03:9200 \\
      --es_fingerprint "B1:2A:96:..." \\
      --es_username elastic --es_password changeme
"""

import argparse
import json
from pathlib import Path

from es_indexer import Config, run_indexing, run_realtime


def main():
    parser = argparse.ArgumentParser(
        description="키워드 → Elasticsearch (async concurrent)"
    )
    parser.add_argument(
        "--mode", choices=["bulk", "realtime"], default="bulk",
        help="bulk=인덱스 재생성+대량적재, realtime=인덱스 보존+즉시리프레시",
    )

    # ── 데이터 소스 ──
    data = parser.add_argument_group("데이터 소스")
    data.add_argument("--keywords", type=Path, default=Config().keywords_path)
    data.add_argument("--parquet", type=Path, default=None, help="Parquet 파일 또는 디렉토리 경로 (설정 시 --keywords 무시)")
    data.add_argument("--parquet_chunk_size", type=int, default=10000, help="Parquet 청크 크기 (행 수)")
    data.add_argument("--parquet_column", default="keyword", help="텍스트가 들어있는 컬럼명")

    parser.add_argument("--limit", type=int, default=0, help="0=전체")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--index", default="keywords")
    parser.add_argument("--es_url", default="http://localhost:9200")
    parser.add_argument(
        "--schema", type=Path, default=None,
        help="커스텀 스키마 JSON 파일 경로 (미지정 시 기본 스키마)",
    )

    # ── ES 클러스터 연결 ──
    cluster = parser.add_argument_group("ES 클러스터 연결 (ES 9+)")
    cluster.add_argument(
        "--es_nodes", nargs="+", default=None,
        help="클러스터 노드 URL 목록 (설정 시 --es_url 무시)",
    )
    cluster.add_argument(
        "--es_fingerprint", default=None,
        help="TLS 인증서 SHA-256 fingerprint (--es_nodes 사용 시 필수)",
    )
    cluster.add_argument("--es_username", default=None, help="Basic Auth 사용자명")
    cluster.add_argument("--es_password", default=None, help="Basic Auth 비밀번호")
    cluster.add_argument(
        "--es_api_key", default=None,
        help="API Key (--es_username/--es_password 대신 사용)",
    )

    # ── 재시도 / 실패 처리 ──
    retry = parser.add_argument_group("재시도 / 실패 처리")
    retry.add_argument(
        "--max_retries", type=int, default=3,
        help="배치 실패 시 최대 재시도 횟수 (0=재시도 없음, default: 3)",
    )
    retry.add_argument(
        "--retry_backoff", type=float, default=1.0,
        help="첫 재시도 대기 시간 (초, 이후 ×2 지수 백오프, default: 1.0)",
    )
    retry.add_argument(
        "--failure_log", type=Path, default=None,
        help="실패 문서 JSONL 파일 경로 (미지정 시 logs/ 에 자동 생성)",
    )

    args = parser.parse_args()

    schema = None
    if args.schema:
        schema = json.loads(args.schema.read_text(encoding="utf-8"))

    config_kwargs = {
        "keywords_path": args.keywords,
        "parquet_path": args.parquet,
        "parquet_chunk_size": args.parquet_chunk_size,
        "parquet_text_column": args.parquet_column,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "index_name": args.index,
        "es_url": args.es_url,
        "es_nodes": args.es_nodes,
        "es_fingerprint": args.es_fingerprint,
        "es_username": args.es_username,
        "es_password": args.es_password,
        "es_api_key": args.es_api_key,
        "schema": schema,
        "max_retries": args.max_retries,
        "retry_backoff": args.retry_backoff,
    }
    if args.failure_log:
        config_kwargs["failure_log_path"] = args.failure_log

    config = Config(**config_kwargs)

    if args.mode == "bulk":
        run_indexing(config)
    else:
        run_realtime(config)


if __name__ == "__main__":
    main()
