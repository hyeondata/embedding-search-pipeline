"""텍스트 쿼리 → 임베딩 → Qdrant top-k 검색 (단건 + 대용량 배치)"""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .embedder import EmbeddingClient
from .indexer import AsyncQdrantIndexer, QdrantIndexer
from .log import get_logger, setup_logging

logger = get_logger("searcher")


@dataclass
class SearchResult:
    """검색 결과 1건"""
    rank: int
    score: float
    keyword: str
    point_id: int


class Searcher:
    """
    텍스트 쿼리로 Qdrant에서 유사 키워드를 검색.

    ruri-v3 비대칭 임베딩:
      - 검색 쿼리: "検索クエリ: " prefix
      - 검색 문서: "検索文書: " prefix
      - 의미적 인코딩: prefix 없음

    사용법:
        searcher = Searcher(config)
        results = searcher.search("東京 ラーメン", top_k=10)
        for r in results:
            print(f"#{r.rank} score={r.score:.4f} {r.keyword}")
    """

    QUERY_PREFIX = "検索クエリ: "

    def __init__(self, config: Config = None):
        config = config or Config()
        self.embedder = EmbeddingClient(config.kserve_url, config.model_name)
        self.indexer = QdrantIndexer(config.qdrant_url, config.collection_name, config.vector_dim)

    def search(self, query: str, top_k: int = 5, prefix: str = None) -> list[SearchResult]:
        """
        텍스트 쿼리 → 임베딩 → top-k 유사도 검색

        Args:
            query:  검색할 텍스트
            top_k:  반환할 결과 수
            prefix: 쿼리 prefix (기본: "検索クエリ: ")

        Returns:
            SearchResult 리스트 (score 내림차순)
        """
        if prefix is None:
            prefix = self.QUERY_PREFIX
        prefixed = f"{prefix}{query}"

        embedding = self.embedder.embed([prefixed])
        response = self.indexer.search(embedding[0].tolist(), top_k=top_k)

        return [
            SearchResult(
                rank=i + 1,
                score=point.score,
                keyword=point.payload["keyword"],
                point_id=point.id,
            )
            for i, point in enumerate(response.points)
        ]

    def search_batch(self, queries: list[str], top_k: int = 5, prefix: str = None) -> dict[str, list[SearchResult]]:
        """
        여러 쿼리를 한 번에 검색.
        임베딩 1회 + query_batch_points 1회로 처리.

        Args:
            queries: 검색할 텍스트 리스트
            top_k:   각 쿼리당 반환할 결과 수
            prefix:  쿼리 prefix (기본: "検索クエリ: ")

        Returns:
            {쿼리 텍스트: [SearchResult, ...]} 딕셔너리
        """
        if prefix is None:
            prefix = self.QUERY_PREFIX
        prefixed = [f"{prefix}{q}" for q in queries]

        embeddings = self.embedder.embed(prefixed)

        # query_batch_points: N번 HTTP → 1번 HTTP
        vectors = [embeddings[i].tolist() for i in range(len(queries))]
        responses = self.indexer.search_batch(vectors, top_k=top_k)

        results = {}
        for idx, query in enumerate(queries):
            results[query] = [
                SearchResult(
                    rank=i + 1,
                    score=point.score,
                    keyword=point.payload["keyword"],
                    point_id=point.id,
                )
                for i, point in enumerate(responses[idx].points)
            ]
        return results


# ============================================================
# 대용량 배치 검색 파이프라인 (asyncio + query_batch_points)
# ============================================================

class _SearchStats:
    """배치 검색 진행 통계"""

    LOG_INTERVAL = 1000

    def __init__(self, total: int):
        self.total = total
        self.searched = 0
        self._start = time.perf_counter()
        self._interval_start = time.perf_counter()
        self._interval_count = 0

    def update(self, count: int, embed_ms: float, search_ms: float):
        self.searched += count
        self._interval_count += count
        n = self.searched

        elapsed = time.perf_counter() - self._start
        avg_qps = n / elapsed if elapsed > 0 else 0

        should_log = (
            n <= 100
            or n % self.LOG_INTERVAL == 0
            or n == self.total
        )
        if should_log:
            now = time.perf_counter()
            interval_sec = now - self._interval_start
            interval_count = self._interval_count
            self._interval_start = now
            self._interval_count = 0
            interval_qps = interval_count / interval_sec if interval_sec > 0 else 0
            pct = n / self.total * 100

            logger.info(
                f"[bold]\\[{pct:5.1f}%][/bold] {n:>7,}/{self.total:,}  "
                f"embed=[cyan]{embed_ms:.0f}ms[/cyan]  "
                f"search=[cyan]{search_ms:.0f}ms[/cyan]  "
                f"avg=[green]{avg_qps:.0f} q/s[/green]  "
                f"interval=[green]{interval_qps:.0f} q/s[/green]"
            )

    @property
    def wall_sec(self) -> float:
        return time.perf_counter() - self._start


async def _async_search_batch(
    queries: list[str],
    embedder: EmbeddingClient,
    async_indexer: AsyncQdrantIndexer,
    top_k: int,
    prefix: str,
    stats: _SearchStats,
    out_file,
    loop: asyncio.AbstractEventLoop,
):
    """단일 배치: 임베딩 → query_batch_points(1회) → JSONL 기록"""
    prefixed = [f"{prefix}{q}" for q in queries]

    # 임베딩 (sync HTTP → run_in_executor로 비동기화)
    t0 = time.perf_counter()
    embeddings = await loop.run_in_executor(None, embedder.embed, prefixed)
    embed_ms = (time.perf_counter() - t0) * 1000

    # query_batch_points: N개 쿼리를 1번의 API 호출로
    t0 = time.perf_counter()
    vectors = [embeddings[i].tolist() for i in range(len(queries))]
    responses = await async_indexer.search_batch(vectors, top_k=top_k)
    search_ms = (time.perf_counter() - t0) * 1000

    # JSONL 기록
    lines = []
    for idx, query in enumerate(queries):
        record = {
            "query": query,
            "results": [
                {
                    "rank": i + 1,
                    "score": round(point.score, 6),
                    "keyword": point.payload["keyword"],
                    "point_id": point.id,
                }
                for i, point in enumerate(responses[idx].points)
            ],
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    # 파일 쓰기 (한번에 모아서)
    out_file.write("\n".join(lines) + "\n")

    stats.update(len(queries), embed_ms, search_ms)


async def _run_async_batch_search(
    config: Config,
    queries: list[str],
    top_k: int,
    output_path: Path,
    stats: _SearchStats,
):
    """asyncio 이벤트 루프에서 배치 검색 실행"""
    embedder = EmbeddingClient(config.kserve_url, config.model_name)
    async_indexer = AsyncQdrantIndexer(config.qdrant_url, config.collection_name, config.vector_dim)
    loop = asyncio.get_event_loop()

    # Semaphore로 동시 실행 배치 수 제한 (workers)
    sem = asyncio.Semaphore(config.workers)

    batches = [
        queries[i : i + config.batch_size]
        for i in range(0, len(queries), config.batch_size)
    ]

    with open(output_path, "w", encoding="utf-8") as out_file:
        async def bounded_search(batch):
            async with sem:
                await _async_search_batch(
                    batch, embedder, async_indexer, top_k,
                    Searcher.QUERY_PREFIX, stats, out_file, loop,
                )

        # 모든 배치를 동시에 스케줄링 (Semaphore가 동시 실행 수를 제한)
        await asyncio.gather(*[bounded_search(batch) for batch in batches])

    await async_indexer.close()


def run_batch_search(
    config: Config,
    queries_path: Path,
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

    성능 개선:
      - query_batch_points: 배치 N개 쿼리를 1번 HTTP 호출 (기존 N번 → 1번)
      - AsyncQdrantClient: asyncio 네이티브 I/O 다중화 (ThreadPool 오버헤드 제거)
      - Semaphore: 동시 실행 배치 수를 workers로 제한

    Args:
        config:       Config 인스턴스 (kserve_url, qdrant_url, batch_size, workers)
        queries_path: 쿼리 파일 경로 (한 줄에 쿼리 1개)
        output_path:  결과 JSONL 파일 (기본: logs/search_results_YYYYMMDD_HHMMSS.jsonl)
        top_k:        각 쿼리당 반환 결과 수
        limit:        처리할 쿼리 수 (0=전체)
        log_path:     로그 디렉토리
    """
    # 로그 설정
    if log_path is None:
        log_path = queries_path.parent.parent / "logs" if queries_path.parent.name == "data" else queries_path.parent / "logs"
    log_path.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"batch_search_{ts}.log"
    setup_logging(log_file=log_file)

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
    logger.info("query_batch_points + AsyncQdrantClient")

    stats = _SearchStats(total)

    # asyncio 이벤트 루프 실행
    asyncio.run(_run_async_batch_search(config, queries, top_k, output_path, stats))

    wall = stats.wall_sec

    # 3. 통계
    logger.info("[bold]\\[3/3] 완료[/bold]")
    logger.info(f"검색 쿼리: [cyan]{total:,}[/cyan]건")
    logger.info(f"결과 파일: {output_path}")
    logger.info(f"Wall time: [cyan]{wall:.1f}초[/cyan]")
    logger.info(f"처리량: [bold green]{total / wall:.0f} queries/sec[/bold green]")

    # 샘플 출력
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
                f'{top1.get("keyword", "N/A")}'
            )

    logger.info("[bold green]완료![/bold green]")
    return output_path
