"""텍스트 쿼리 → 임베딩 → Qdrant top-k 검색 (단건 + 대용량 배치)

embed_fn (Callable[[list[str]], np.ndarray])을 외부에서 주입받아
EmbeddingClient에 대한 직접 의존을 제거.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from pipeline_commons import (
    PipelineStats,
    batch_iter,
    get_logger,
    setup_logging,
    timer,
)

from .config import Config
from .indexer import AsyncQdrantIndexer, QdrantIndexer

_PKG = "qdrant_indexer"
logger = get_logger(_PKG, "searcher")

# ── 타입 별칭 ────────────────────────────────────
EmbedFn = Callable[[list[str]], np.ndarray]


@dataclass
class SearchResult:
    """검색 결과 1건."""
    rank: int
    score: float
    keyword: str
    point_id: int
    payload: dict = field(default_factory=dict)


class Searcher:
    """
    텍스트 쿼리로 Qdrant에서 유사 키워드를 검색.

    사용법:
        searcher = Searcher(config, embed_fn=client.embed, query_prefix="検索クエリ: ")
        results = searcher.search("東京 ラーメン", top_k=10)
    """

    def __init__(
        self,
        config: Config = None,
        *,
        embed_fn: EmbedFn,
        query_prefix: str = "",
        payload_key: str = None,
    ):
        config = config or Config()
        self.embed_fn = embed_fn
        self.query_prefix = query_prefix
        self.indexer = QdrantIndexer(config.qdrant_url, config.collection_name, config.vector_dim)
        self.payload_key = payload_key or config.payload_key

    def search(self, query: str, top_k: int = 5, prefix: str = None) -> list[SearchResult]:
        """텍스트 쿼리 → 임베딩 → top-k 유사도 검색."""
        if prefix is None:
            prefix = self.query_prefix
        prefixed = f"{prefix}{query}"

        embedding = self.embed_fn([prefixed])
        response = self.indexer.search(embedding[0].tolist(), top_k=top_k)

        return [
            SearchResult(
                rank=i + 1,
                score=point.score,
                keyword=point.payload.get(self.payload_key, ""),
                point_id=point.id,
                payload=point.payload,
            )
            for i, point in enumerate(response.points)
        ]

    def search_batch(self, queries: list[str], top_k: int = 5, prefix: str = None) -> dict[str, list[SearchResult]]:
        """여러 쿼리를 한 번에 검색."""
        if prefix is None:
            prefix = self.query_prefix
        prefixed = [f"{prefix}{q}" for q in queries]

        embeddings = self.embed_fn(prefixed)

        vectors = [embeddings[i].tolist() for i in range(len(queries))]
        responses = self.indexer.search_batch(vectors, top_k=top_k)

        results = {}
        for idx, query in enumerate(queries):
            results[query] = [
                SearchResult(
                    rank=i + 1,
                    score=point.score,
                    keyword=point.payload.get(self.payload_key, ""),
                    point_id=point.id,
                    payload=point.payload,
                )
                for i, point in enumerate(responses[idx].points)
            ]
        return results


# ============================================================
# 대용량 배치 검색 파이프라인 (asyncio + query_batch_points)
# ============================================================

async def _async_search_batch(
    queries: list[str],
    embed_fn: EmbedFn,
    async_indexer: AsyncQdrantIndexer,
    top_k: int,
    prefix: str,
    payload_key: str,
    stats: PipelineStats,
    out_file,
    loop: asyncio.AbstractEventLoop,
):
    """단일 배치: 임베딩 → query_batch_points(1회) → JSONL 기록"""
    prefixed = [f"{prefix}{q}" for q in queries]

    with timer() as t_embed:
        embeddings = await loop.run_in_executor(None, embed_fn, prefixed)

    with timer() as t_search:
        vectors = [embeddings[i].tolist() for i in range(len(queries))]
        responses = await async_indexer.search_batch(vectors, top_k=top_k)

    lines = []
    for idx, query in enumerate(queries):
        record = {
            "query": query,
            "results": [
                {
                    "rank": i + 1,
                    "score": round(point.score, 6),
                    payload_key: point.payload.get(payload_key, ""),
                    "point_id": point.id,
                    "payload": point.payload,
                }
                for i, point in enumerate(responses[idx].points)
            ],
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    out_file.write("\n".join(lines) + "\n")

    stats.update(len(queries), embed_ms=t_embed.ms, search_ms=t_search.ms)


async def _run_async_batch_search(
    config: Config,
    queries: list[str],
    embed_fn: EmbedFn,
    top_k: int,
    output_path: Path,
    stats: PipelineStats,
    query_prefix: str,
):
    """asyncio 이벤트 루프에서 배치 검색 실행"""
    async_indexer = AsyncQdrantIndexer(config.qdrant_url, config.collection_name, config.vector_dim)
    loop = asyncio.get_running_loop()

    sem = asyncio.Semaphore(config.workers)
    payload_key = config.payload_key

    batches = [batch for _, batch in batch_iter(queries, config.batch_size)]

    with open(output_path, "w", encoding="utf-8") as out_file:
        async def bounded_search(batch):
            async with sem:
                await _async_search_batch(
                    batch, embed_fn, async_indexer, top_k,
                    query_prefix, payload_key, stats, out_file, loop,
                )

        await asyncio.gather(*[bounded_search(batch) for batch in batches])

    await async_indexer.close()


def run_batch_search(
    config: Config,
    queries_path: Path,
    embed_fn: EmbedFn,
    *,
    query_prefix: str = "",
    output_path: Path = None,
    top_k: int = 10,
    limit: int = 0,
    log_path: Path = None,
):
    """
    대용량 배치 검색 파이프라인 (asyncio + query_batch_points):
      1. 쿼리 로드 (텍스트 파일, 줄바꿈 구분)
      2. asyncio로 동시 embed + query_batch_points
      3. 결과를 JSONL 파일로 저장
      4. 통계 출력
    """
    # 로그 설정
    if log_path is None:
        log_path = queries_path.parent.parent / "logs" if queries_path.parent.name == "data" else queries_path.parent / "logs"
    log_path.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"batch_search_{ts}.log"
    setup_logging(_PKG, log_file=log_file)

    if output_path is None:
        output_path = log_path / f"search_results_{ts}.jsonl"

    logger.info(f"Log: {log_file}")
    logger.info(f"Output: {output_path}")

    # 1. 쿼리 로드
    logger.info("[bold]\\[1/3] 쿼리 로드[/bold]")
    queries = queries_path.read_text(encoding="utf-8").strip().split("\n")
    if limit > 0:
        queries = queries[:limit]
    total = len(queries)
    logger.info(f"[cyan]{total:,}[/cyan]건")
    logger.info(f"queries={total}, top_k={top_k}, batch={config.batch_size}, workers={config.workers}")

    # 2. Qdrant 벡터 수 확인
    sync_indexer = QdrantIndexer(config.qdrant_url, config.collection_name, config.vector_dim)
    vec_count = sync_indexer.count
    logger.info(
        f"[bold]\\[2/3] 동시 검색[/bold] "
        f"(asyncio, workers=[cyan]{config.workers}[/cyan], "
        f"batch=[cyan]{config.batch_size}[/cyan], top_k=[cyan]{top_k}[/cyan])"
    )
    logger.info(f"Qdrant 벡터: [cyan]{vec_count:,}[/cyan]건")

    stats = PipelineStats(total, log_fn=logger.info, unit="q/s")

    asyncio.run(_run_async_batch_search(
        config, queries, embed_fn, top_k, output_path, stats, query_prefix,
    ))

    wall = stats.wall_sec

    # 3. 통계
    logger.info("[bold]\\[3/3] 완료[/bold]")
    logger.info(f"검색 쿼리: [cyan]{total:,}[/cyan]건")
    logger.info(f"결과 파일: {output_path}")
    logger.info(f"Wall time: [cyan]{wall:.1f}초[/cyan]")
    logger.info(f"처리량: [bold green]{total / wall:.0f} queries/sec[/bold green]")

    # 샘플 출력
    pk = config.payload_key
    logger.info("[bold]--- 샘플 결과 (처음 3건) ---[/bold]")
    with open(output_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            record = json.loads(line)
            q = record["query"]
            top1 = record["results"][0] if record["results"] else {}
            logger.info(
                f'"{q}" → top1: [green][{top1.get("score", 0):.4f}][/green] '
                f'{top1.get(pk, "N/A")}'
            )

    logger.info("[bold green]완료![/bold green]")
    return output_path
