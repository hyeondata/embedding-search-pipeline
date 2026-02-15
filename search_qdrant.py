#!/usr/bin/env python3
# search_qdrant.py
"""
Qdrant 유사 키워드 검색 (CLI)

사전 조건:
  1. KServe 서버:  Triton 또는 KServe 추론 서버 실행 중
  2. Qdrant:       docker compose up -d

실행:
  # 단건 검색
  python search_qdrant.py "東京 ラーメン"
  python search_qdrant.py "東京 ラーメン" --top_k 20

  # 복수 쿼리
  python search_qdrant.py "東京 ラーメン" "大阪 観光" "京都 寺院"

  # 대화형 모드
  python search_qdrant.py --interactive

  # 대용량 배치 검색 (파일 입력 → JSONL 출력)
  python search_qdrant.py --file data/keywords_400k.txt --top_k 10
"""

import argparse
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from kserve_embed_client import EmbeddingClient, RURI_QUERY_PREFIX
from qdrant_indexer import Config, Searcher, run_batch_search

console = Console()


def print_results(query: str, results, elapsed: float):
    """검색 결과를 Rich Table로 출력"""
    console.print(f"\n  Query: [bold]\"{query}\"[/bold]  ({elapsed:.3f}초, {len(results)}건)")

    table = Table(show_header=True, header_style="bold", padding=(0, 1))
    table.add_column("순위", justify="right", width=4)
    table.add_column("유사도", justify="right", width=7)
    table.add_column("ID", justify="right", width=8)
    table.add_column("키워드")

    for r in results:
        table.add_row(
            str(r.rank),
            f"{r.score:.4f}",
            str(r.point_id),
            r.keyword,
        )

    console.print(table)


def run_single(searcher: Searcher, queries: list[str], top_k: int):
    """단건/복수 쿼리 검색"""
    if len(queries) == 1:
        start = time.perf_counter()
        results = searcher.search(queries[0], top_k=top_k)
        elapsed = time.perf_counter() - start
        print_results(queries[0], results, elapsed)
    else:
        start = time.perf_counter()
        batch_results = searcher.search_batch(queries, top_k=top_k)
        elapsed = time.perf_counter() - start
        console.print(f"\n  배치 임베딩 + 검색 총 시간: [cyan]{elapsed:.3f}초[/cyan]")
        for query, results in batch_results.items():
            print_results(query, results, elapsed / len(queries))


def run_interactive(searcher: Searcher, top_k: int):
    """대화형 검색 모드"""
    console.print("\n  [bold]대화형 검색 모드[/bold] (종료: q 또는 Ctrl+C)")
    console.print(f"  top_k={top_k}\n")

    while True:
        try:
            query = input("  검색> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n  종료")
            break

        if not query or query.lower() in ("q", "quit", "exit"):
            console.print("  종료")
            break

        start = time.perf_counter()
        results = searcher.search(query, top_k=top_k)
        elapsed = time.perf_counter() - start
        print_results(query, results, elapsed)
        console.print()


def main():
    parser = argparse.ArgumentParser(description="Qdrant 유사 키워드 검색")
    parser.add_argument("queries", nargs="*", help="검색할 텍스트 (복수 가능)")
    parser.add_argument("--top_k", type=int, default=10, help="반환할 결과 수 (기본: 10)")
    parser.add_argument("--collection", default="keywords", help="Qdrant 컬렉션 이름")
    parser.add_argument("--interactive", "-i", action="store_true", help="대화형 검색 모드")

    # KServe
    parser.add_argument("--kserve_url", default="http://localhost:8000", help="KServe 서비스 URL")
    parser.add_argument("--model_name", default="ruri_v3", help="모델 이름")

    # 배치 검색 옵션
    parser.add_argument("--file", "-f", type=Path, help="배치 검색: 쿼리 파일 경로 (줄바꿈 구분)")
    parser.add_argument("--output", "-o", type=Path, help="배치 검색: 결과 JSONL 파일 경로")
    parser.add_argument("--limit", type=int, default=0, help="배치 검색: 처리할 쿼리 수 (0=전체)")
    parser.add_argument("--batch_size", type=int, default=64, help="배치 검색: 배치 크기 (기본: 64)")
    parser.add_argument("--workers", type=int, default=8, help="배치 검색: 동시 워커 수 (기본: 8)")

    args = parser.parse_args()

    # 임베딩 클라이언트 생성
    client = EmbeddingClient(args.kserve_url, args.model_name)

    config = Config(
        collection_name=args.collection,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    # 배치 검색 모드
    if args.file:
        console.rule("[bold]대용량 배치 검색 (Concurrent)[/bold]")
        run_batch_search(
            config=config,
            queries_path=args.file,
            embed_fn=client.embed,
            query_prefix=RURI_QUERY_PREFIX,
            output_path=args.output,
            top_k=args.top_k,
            limit=args.limit,
        )
        return

    # 단건/대화형 검색
    searcher = Searcher(config, embed_fn=client.embed, query_prefix=RURI_QUERY_PREFIX)
    count = searcher.indexer.count
    console.rule(f"[bold]Qdrant 검색  (컬렉션: {args.collection}, 벡터: {count:,}건)[/bold]")

    if args.interactive:
        run_interactive(searcher, args.top_k)
    elif args.queries:
        run_single(searcher, args.queries, args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
