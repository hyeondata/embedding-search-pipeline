#!/usr/bin/env python3
# index_to_qdrant.py
"""
키워드 → KServe 임베딩 → Qdrant 인덱싱 (CLI 엔트리포인트)

사전 조건:
  1. KServe 서버:  python kserve_server.py --http_port 8080
  2. Qdrant:       docker compose up -d

실행:
  # 텍스트 파일
  python index_to_qdrant.py --limit 10                     # 소규모 테스트
  python index_to_qdrant.py --workers 8 --batch_size 64    # 최적 설정

  # Parquet 파일
  python index_to_qdrant.py --parquet data/keywords.parquet --parquet_column keyword
  python index_to_qdrant.py --parquet data/large.parquet --parquet_chunk_size 50000 --limit 100000
  python index_to_qdrant.py --parquet data/keywords.parquet --max_retries 5 --no_log_failures
"""

import argparse
from pathlib import Path

from rich.console import Console

from qdrant_indexer import Config, run_indexing

console = Console()


def main():
    parser = argparse.ArgumentParser(description="키워드 → 임베딩 → Qdrant (concurrent + retry)")

    # 데이터 소스
    parser.add_argument("--keywords", type=Path, help="텍스트 파일 경로 (기본값: data/keywords_400k.txt)")
    parser.add_argument("--parquet", type=Path, help="Parquet 파일 경로 (설정 시 --keywords 무시)")
    parser.add_argument("--parquet_chunk_size", type=int, default=10000, help="Parquet 청크 크기 (기본: 10000)")
    parser.add_argument("--parquet_column", default="keyword", help="Parquet 텍스트 컬럼명 (기본: keyword)")
    parser.add_argument("--limit", type=int, default=0, help="처리할 행 수 (0=전체)")

    # 처리
    parser.add_argument("--batch_size", type=int, default=64, help="임베딩 배치 크기 (기본: 64)")
    parser.add_argument("--workers", type=int, default=4, help="동시 워커 수 (기본: 4)")
    parser.add_argument("--collection", default="keywords", help="Qdrant 컬렉션 이름")

    # 재시도
    parser.add_argument("--max_retries", type=int, default=3, help="배치당 최대 재시도 횟수 (기본: 3)")
    parser.add_argument("--retry_backoff", type=float, default=1.0, help="초기 대기 시간 (기본: 1.0초)")
    parser.add_argument("--no_exponential_backoff", action="store_true", help="지수 백오프 비활성화")

    # 실패 로깅
    parser.add_argument("--no_log_failures", action="store_true", help="실패 로그 비활성화")
    parser.add_argument("--failure_log", type=Path, help="실패 로그 파일 경로")

    args = parser.parse_args()

    # Config 생성
    config_kwargs = {
        "limit": args.limit,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "collection_name": args.collection,
        "max_retries": args.max_retries,
        "retry_backoff": args.retry_backoff,
        "retry_exponential": not args.no_exponential_backoff,
        "log_failures": not args.no_log_failures,
    }

    if args.parquet:
        config_kwargs.update({
            "parquet_path": args.parquet,
            "parquet_chunk_size": args.parquet_chunk_size,
            "parquet_text_column": args.parquet_column,
        })
    elif args.keywords:
        config_kwargs["keywords_path"] = args.keywords

    if args.failure_log:
        config_kwargs["failure_log_path"] = args.failure_log

    config = Config(**config_kwargs)

    console.rule("[bold]키워드 임베딩 → Qdrant 인덱싱 (Concurrent + Retry)[/bold]")

    run_indexing(config)


if __name__ == "__main__":
    main()
