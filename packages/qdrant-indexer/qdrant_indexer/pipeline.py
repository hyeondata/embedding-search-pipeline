"""동시 인덱싱 파이프라인 (Parquet 지원 + 재시도 + 실패 로깅)"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kserve_embed_client import (
    EmbeddingClient,
    FailureLogger,
    ParquetReader,
    RetryConfig,
    get_logger,
    setup_logging,
    with_retry,
)

from .config import Config
from .indexer import QdrantIndexer

_PKG = "qdrant_indexer"
logger = get_logger(_PKG, "pipeline")


# ============================================================
# Thread-safe 통계
# ============================================================
class _Stats:
    """인덱싱 진행 통계 + 구간별 RPS 측정 (thread-safe)"""

    LOG_INTERVAL = 1000  # N건마다 로그 출력

    def __init__(self, total: int):
        self.total = total
        self.indexed = 0
        self.embed_ms = 0.0
        self.upsert_ms = 0.0
        self._lock = threading.Lock()
        self._start = time.perf_counter()
        # 구간별 RPS 측정
        self._interval_start = time.perf_counter()
        self._interval_count = 0

    def update(self, count: int, embed_ms: float, upsert_ms: float):
        with self._lock:
            self.indexed += count
            self.embed_ms += embed_ms
            self.upsert_ms += upsert_ms
            self._interval_count += count
            n = self.indexed

        elapsed = time.perf_counter() - self._start
        avg_rps = n / elapsed if elapsed > 0 else 0

        should_log = (
            n <= 100
            or n % self.LOG_INTERVAL == 0
            or n == self.total
        )
        if should_log:
            # 구간 RPS 계산
            now = time.perf_counter()
            with self._lock:
                interval_sec = now - self._interval_start
                interval_count = self._interval_count
                self._interval_start = now
                self._interval_count = 0
            interval_rps = interval_count / interval_sec if interval_sec > 0 else 0
            pct = n / self.total * 100

            logger.info(
                f"[bold]\\[{pct:5.1f}%][/bold] {n:>7,}/{self.total:,}  "
                f"embed=[cyan]{embed_ms:.0f}ms[/cyan]  "
                f"upsert=[cyan]{upsert_ms:.0f}ms[/cyan]  "
                f"avg=[green]{avg_rps:.0f} t/s[/green]  "
                f"interval=[green]{interval_rps:.0f} t/s[/green]"
            )

    @property
    def wall_sec(self) -> float:
        return time.perf_counter() - self._start


# ============================================================
# 배치 처리 함수
# ============================================================
def _create_batch_processor(embedder, indexer, stats, failure_logger, retry_config):
    """
    재시도 데코레이터가 적용된 배치 처리 함수를 반환합니다.

    ThreadPoolExecutor에서 함수를 pickle할 때 데코레이터 적용이 안 되는 문제를 우회하기 위해
    팩토리 패턴을 사용합니다.
    """
    @with_retry(
        retry_config=retry_config,
        failure_logger=failure_logger,
        batch_id_fn=lambda args, kwargs: args[0],
        data_info_fn=lambda args, kwargs: {"keywords_count": len(args[1])},
    )
    def process_batch(batch_id: int, keywords: list[str]):
        """단일 배치: 임베딩 → Qdrant upsert (워커 스레드에서 실행)"""
        t0 = time.perf_counter()
        embeddings = embedder.embed(keywords)
        embed_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        indexer.upsert_batch(start_id=batch_id, keywords=keywords, embeddings=embeddings)
        upsert_ms = (time.perf_counter() - t0) * 1000

        stats.update(len(keywords), embed_ms, upsert_ms)

    return process_batch


# ============================================================
# 메인 파이프라인
# ============================================================
def run_indexing(config: Config, log_path: Path = None):
    """
    전체 인덱싱 파이프라인:
      1. 키워드 로드 (텍스트 파일 또는 Parquet)
      2. Qdrant 컬렉션 생성
      3. 동시 배치 처리 (embed + upsert + 재시도)
      4. 검증 + 샘플 검색
    """
    # 로그 파일 설정
    if log_path is None:
        log_path = (
            config.parquet_path.parent / "logs"
            if config.parquet_path
            else config.keywords_path.parent.parent / "logs"
        )
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"indexing_{time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(_PKG, log_file=log_file)

    logger.info(f"Log: {log_file}")
    logger.info(f"config: {config}")

    # 1. 키워드 로드
    logger.info("[bold]\\[1/4] 데이터 로드[/bold]")

    if config.parquet_path:
        reader = ParquetReader(
            config.parquet_path,
            chunk_size=config.parquet_chunk_size,
            text_column=config.parquet_text_column,
            limit=config.limit,
        )
        total = reader.total_rows
        logger.info(
            f"[cyan]Parquet: {config.parquet_path.name}[/cyan] "
            f"({total:,}행, 청크={config.parquet_chunk_size:,})"
        )
        data_source = "parquet"

    else:
        keywords = config.keywords_path.read_text(encoding="utf-8").strip().split("\n")
        if config.limit > 0:
            keywords = keywords[: config.limit]
        total = len(keywords)
        logger.info(f"[cyan]텍스트 파일: {config.keywords_path.name}[/cyan] ({total:,}건)")
        data_source = "text"

    # 2. Qdrant 컬렉션 생성
    logger.info(f"[bold]\\[2/4] Qdrant 컬렉션: {config.collection_name}[/bold]")
    indexer = QdrantIndexer(config.qdrant_url, config.collection_name, config.vector_dim)
    indexer.create_collection()
    logger.info(f"생성 완료 (dim={config.vector_dim}, COSINE)")

    # 3. 재시도 + 실패 로깅 설정
    retry_config = RetryConfig(
        max_retries=config.max_retries,
        initial_backoff=config.retry_backoff,
        exponential=config.retry_exponential,
        max_backoff=config.retry_max_backoff,
    )
    failure_logger = FailureLogger(config.failure_log_path, enabled=config.log_failures)

    if config.log_failures:
        logger.info(f"실패 로그: [cyan]{config.failure_log_path}[/cyan]")
    logger.info(
        f"재시도: [cyan]{config.max_retries}회[/cyan] "
        f"(백오프: {config.retry_backoff}s{'×2^n' if config.retry_exponential else ''})"
    )

    # 4. 동시 배치 처리
    logger.info(
        f"[bold]\\[3/4] 동시 인덱싱[/bold] "
        f"(workers=[cyan]{config.workers}[/cyan], batch=[cyan]{config.batch_size}[/cyan])"
    )
    embedder = EmbeddingClient(config.kserve_url, config.model_name)
    stats = _Stats(total)

    process_batch = _create_batch_processor(embedder, indexer, stats, failure_logger, retry_config)

    if data_source == "parquet":
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            futures = []

            for chunk_id, chunk_keywords in reader.iter_chunks():
                for i in range(0, len(chunk_keywords), config.batch_size):
                    batch = chunk_keywords[i : i + config.batch_size]
                    batch_id = chunk_id * config.parquet_chunk_size + i
                    futures.append(pool.submit(process_batch, batch_id, batch))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"[bold red]배치 처리 실패 (최종)[/bold red]: {e}")

    else:
        batches = [
            (i, keywords[i : i + config.batch_size])
            for i in range(0, total, config.batch_size)
        ]

        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            futures = [
                pool.submit(process_batch, bid, batch)
                for bid, batch in batches
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"[bold red]배치 처리 실패 (최종)[/bold red]: {e}")

    wall = stats.wall_sec

    # 5. 검증
    logger.info("[bold]\\[4/4] 검증[/bold]")
    logger.info(f"벡터 수: [cyan]{indexer.count:,}[/cyan]")
    logger.info(f"Wall time: [cyan]{wall:.1f}초[/cyan]")
    logger.info(f"처리량: [bold green]{total / wall:.0f} texts/sec[/bold green]")
    logger.info(f"임베딩 합계: {stats.embed_ms / 1000:.1f}초 (스레드 CPU time)")
    logger.info(f"upsert 합계: {stats.upsert_ms / 1000:.1f}초")

    # 샘플 검색
    logger.info("[bold]--- 샘플 검색 ---[/bold]")
    query = "東京 ラーメン おすすめ"
    query_emb = embedder.embed([f"検索クエリ: {query}"])
    results = indexer.search(query_emb[0].tolist(), top_k=5)

    logger.info(f"쿼리: {query}")
    for r in results.points:
        logger.info(f"  score=[green]{r.score:.4f}[/green]  {r.payload['keyword']}")

    logger.info("[bold green]완료![/bold green]")
