"""비동기 인덱싱 파이프라인 — Rich 로깅 + Progress Bar + 재시도 + 실패 로깅

콘솔: RichHandler (색상, 포맷)  +  Rich Progress (프로그레스 바)
파일: FileHandler (plain text + timestamp)

Bulk 모드:
    인덱스 재생성 → refresh 비활성 → 대량 적재 → finalize
Realtime 모드:
    인덱스 보존 → 배치 단위로 인덱싱 + 즉시 리프레시

재시도: 배치 실패 시 지수 백오프로 max_retries 회 재시도
실패 로깅: 최종 실패 배치를 JSONL 파일에 기록 (수동 재처리용)
"""

import asyncio
import logging
import time
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from kserve_embed_client import (
    AsyncFailureLogger,
    PipelineStats,
    batch_iter,
    load_keywords,
    timer,
)

from .config import Config
from .indexer import ESIndexer

console = Console()
logger = logging.getLogger("es_indexer")


# ============================================================
# 로깅 설정
# ============================================================
def _setup_logger(config: Config, log_path: Path | None, prefix: str) -> Path:
    """
    RichHandler (콘솔) + FileHandler (파일) 듀얼 로깅.

    하나의 logger.info()로 콘솔(Rich 포맷)과 파일(plain text)에 동시 기록.
    """
    logger.handlers.clear()

    if log_path is None:
        log_path = config.keywords_path.parent.parent / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.log"

    # File handler — plain text + timestamp
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    logger.addHandler(fh)

    # Rich handler — formatted console
    rh = RichHandler(
        console=console,
        show_path=False,
        show_time=True,
        show_level=True,
        rich_tracebacks=True,
    )
    rh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rh)

    logger.setLevel(logging.INFO)
    logger.info(f"config: {config}")
    return log_file


# ============================================================
# Rich Progress bar
# ============================================================
def _create_progress() -> Progress:
    """파이프라인용 Rich Progress bar 생성"""
    return Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("\u2022"),
        TextColumn("[green]{task.fields[throughput]}[/]"),
        TextColumn("\u2022"),
        TextColumn("[yellow]bulk={task.fields[last_bulk]}[/]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def _make_progress_callback(progress: Progress, task_id):
    """PipelineStats.on_update 콜백 — Rich Progress bar 연동"""

    def callback(stats: PipelineStats, count: int, timing_ms: dict[str, float]):
        rps = stats.avg_rps
        bulk_ms = timing_ms.get("bulk_ms", 0)
        progress.update(
            task_id,
            advance=count,
            throughput=f"{rps:,.0f} docs/s",
            last_bulk=f"{bulk_ms:.0f}ms",
        )

    return callback


# ============================================================
# 배치 처리 코루틴
# ============================================================
async def _process_batch(
    batch_id: int,
    keywords: list[str],
    indexer: ESIndexer,
    semaphore: asyncio.Semaphore,
    stats: PipelineStats,
    config: Config,
    failure_logger: AsyncFailureLogger,
    progress: Progress,
    progress_task_id,
    refresh: bool = False,
):
    """bulk_index + 재시도(지수 백오프) + 최종 실패 시 실패 로깅."""
    async with semaphore:
        last_error: Exception | None = None

        for attempt in range(1, config.max_retries + 1):
            try:
                with timer() as t:
                    await indexer.bulk_index(batch_id, keywords)
                    if refresh:
                        await indexer.refresh()
                stats.update(len(keywords), bulk_ms=t.ms)
                return
            except Exception as e:
                last_error = e
                if attempt >= config.max_retries:
                    break
                backoff = config.retry_backoff * (2 ** (attempt - 1))
                if config.retry_exponential:
                    backoff = min(backoff, config.retry_max_backoff)
                stats.record_retry()
                logger.warning(
                    f"Batch {batch_id} 실패 (시도 {attempt}/{config.max_retries}), "
                    f"{backoff:.1f}초 후 재시도: {e}"
                )
                await asyncio.sleep(backoff)

        logger.error(
            f"Batch {batch_id} 최종 실패 ({len(keywords)}건, "
            f"{config.max_retries}회 시도): {last_error}"
        )
        stats.record_failure(len(keywords))
        progress.update(progress_task_id, advance=len(keywords))
        await failure_logger.log_failure(
            batch_id, last_error,
            data_info={"count": len(keywords), "keywords": keywords},
        )


def _summary_table(title: str, rows: list[tuple[str, str]]) -> Table:
    """결과 요약 Rich Table"""
    table = Table(title=title, show_header=False, border_style="dim")
    table.add_column("항목", style="bold")
    table.add_column("값", justify="right", style="cyan")
    for label, value in rows:
        table.add_row(label, value)
    return table


def _resolve_failure_log_path(config: Config, log_file: Path) -> Path:
    """실패 로그 파일 경로 결정."""
    if config.failure_log_path:
        return config.failure_log_path
    return log_file.with_suffix(".failures.jsonl")


def _build_summary_rows(
    count: int, total: int, wall: float,
    stats: PipelineStats, failure_logger: AsyncFailureLogger,
) -> list[tuple[str, str]]:
    """요약 테이블 행 생성."""
    rows = [
        ("도큐먼트 수", f"{count:,}"),
        ("Wall time", f"{wall:.1f}초"),
        ("처리량", f"{total / wall:,.0f} docs/sec"),
        ("bulk 합계", f"{stats.get_timing('bulk_ms') / 1000:.1f}초 (코루틴 time)"),
    ]
    if stats.retries > 0:
        rows.append(("재시도 횟수", f"{stats.retries}회"))
    if stats.failed_count > 0:
        rows.append(("실패 문서", f"[red]{stats.failed_count:,}건 ({stats.failed_batches} 배치)[/]"))
        rows.append(("실패 로그", str(failure_logger.log_path)))
    return rows


# ============================================================
# Bulk 파이프라인
# ============================================================
async def _run_bulk(config: Config, log_path: Path | None = None):
    log_file = _setup_logger(config, log_path, "es_bulk")
    logger.info(f"Log \u2192 {log_file}")

    # [1/4] 키워드 로드
    logger.info("[1/4] 키워드 로드")
    keywords, source = load_keywords(config)
    total = len(keywords)
    logger.info(f"{source} \u2192 {total:,}건 로드 완료")

    # [2/4] ES 인덱스 재생성
    schema_label = "custom" if config.schema else "default"
    logger.info(f"[2/4] ES 인덱스: {config.index_name} (schema={schema_label})")
    indexer = ESIndexer.from_config(config)
    await indexer.create_index(config.schema)
    logger.info("생성 완료 (refresh=-1, translog=async)")

    # [3/4] 동시 배치 처리
    logger.info(
        f"[3/4] 비동기 인덱싱 "
        f"(workers={config.workers}, batch={config.batch_size}, "
        f"retries={config.max_retries})"
    )
    batches = list(batch_iter(keywords, config.batch_size))

    fl_path = _resolve_failure_log_path(config, log_file)
    failure_logger = AsyncFailureLogger(fl_path, enabled=config.log_failures)

    progress = _create_progress()
    with progress:
        task_id = progress.add_task("Indexing", total=total, throughput="--", last_bulk="--")
        stats = PipelineStats(
            total,
            on_update=_make_progress_callback(progress, task_id),
            log_fn=logger.info,
            log_interval=5000,
            unit="docs/s",
        )
        semaphore = asyncio.Semaphore(config.workers)

        await asyncio.gather(*[
            _process_batch(
                bid, batch, indexer, semaphore, stats,
                config, failure_logger, progress, task_id,
                refresh=False,
            )
            for bid, batch in batches
        ])

    wall = stats.wall_sec

    # [4/4] 최적화 + 검증
    logger.info("[4/4] 인덱스 최적화 (refresh + force merge)")
    await indexer.finalize()

    count = await indexer.count()
    rows = _build_summary_rows(count, total, wall, stats, failure_logger)
    console.print(_summary_table("결과 요약", rows))

    for label, value in rows:
        logger.info(f"{label}: {value}")

    await indexer.close()
    logger.info("완료")


# ============================================================
# Realtime 파이프라인
# ============================================================
async def _run_realtime(config: Config, log_path: Path | None = None):
    log_file = _setup_logger(config, log_path, "es_realtime")
    logger.info(f"Log \u2192 {log_file}")

    # [1/4] 키워드 로드
    logger.info("[1/4] 키워드 로드")
    keywords, source = load_keywords(config)
    total = len(keywords)
    logger.info(f"{source} \u2192 {total:,}건 로드 완료")

    # [2/4] ES 인덱스 보존
    schema_label = "custom" if config.schema else "default"
    logger.info(f"[2/4] ES 인덱스: {config.index_name} (schema={schema_label})")
    indexer = ESIndexer.from_config(config)
    created = await indexer.ensure_index(config.schema)
    if created:
        logger.info("신규 생성 (refresh=1s, translog=request)")
    else:
        existing = await indexer.count()
        logger.info(f"기존 유지 (현재 {existing:,}건)")

    # [3/4] 실시간 인덱싱
    logger.info(
        f"[3/4] 실시간 인덱싱 "
        f"(workers={config.workers}, batch={config.batch_size}, "
        f"retries={config.max_retries})"
    )
    batches = list(batch_iter(keywords, config.batch_size))

    fl_path = _resolve_failure_log_path(config, log_file)
    failure_logger = AsyncFailureLogger(fl_path, enabled=config.log_failures)

    progress = _create_progress()
    with progress:
        task_id = progress.add_task("Realtime", total=total, throughput="--", last_bulk="--")
        stats = PipelineStats(
            total,
            on_update=_make_progress_callback(progress, task_id),
            log_fn=logger.info,
            log_interval=5000,
            unit="docs/s",
        )
        semaphore = asyncio.Semaphore(config.workers)

        await asyncio.gather(*[
            _process_batch(
                bid, batch, indexer, semaphore, stats,
                config, failure_logger, progress, task_id,
                refresh=True,
            )
            for bid, batch in batches
        ])

    wall = stats.wall_sec

    # [4/4] 검증 + 샘플 검색
    count = await indexer.count()
    logger.info("[4/4] 검증")

    rows = _build_summary_rows(count, total, wall, stats, failure_logger)
    console.print(_summary_table("결과 요약", rows))

    for label, value in rows:
        logger.info(f"{label}: {value}")

    # 샘플 검색
    sample = keywords[0] if keywords else ""
    if sample:
        results = await indexer.search(sample, size=3)
        hits = results["hits"]["hits"]
        logger.info(f"실시간 검색 확인 \u2014 쿼리: {sample}")
        for hit in hits:
            logger.info(f"  score={hit['_score']:.4f}  {hit['_source']['keyword']}")

    await indexer.close()
    logger.info("완료")


# ============================================================
# Public API — 동기 래퍼
# ============================================================
def run_indexing(config: Config, log_path: Path | None = None):
    """Bulk 모드 — 인덱스 재생성 후 대량 적재"""
    console.print(
        Panel.fit(
            "[bold]Bulk 모드[/] \u2014 인덱스 재생성 + 대량 적재",
            border_style="green",
        )
    )
    asyncio.run(_run_bulk(config, log_path))


def run_realtime(config: Config, log_path: Path | None = None):
    """Realtime 모드 — 기존 인덱스 보존 + 즉시 리프레시"""
    console.print(
        Panel.fit(
            "[bold]Realtime 모드[/] \u2014 인덱스 보존 + 즉시 리프레시",
            border_style="blue",
        )
    )
    asyncio.run(_run_realtime(config, log_path))
