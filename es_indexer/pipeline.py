"""비동기 인덱싱 파이프라인 — Rich 로깅 + Progress Bar + 재시도 + Dead Letter

콘솔: RichHandler (색상, 포맷)  +  Rich Progress (프로그레스 바)
파일: FileHandler (plain text + timestamp)

Bulk 모드:
    인덱스 재생성 → refresh 비활성 → 대량 적재 → finalize
Realtime 모드:
    인덱스 보존 → 배치 단위로 인덱싱 + 즉시 리프레시

재시도: 배치 실패 시 지수 백오프로 max_retries 회 재시도
Dead Letter: 최종 실패 배치를 JSONL 파일에 기록 (수동 재처리용)
"""

import asyncio
import json
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

from .config import Config
from .indexer import ESIndexer

# qdrant_indexer에서 ParquetReader 가져오기
try:
    from qdrant_indexer import ParquetReader
except ImportError:
    ParquetReader = None  # ParquetReader 없으면 텍스트 파일만 지원

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
# 진행 통계 (Progress 연동)
# ============================================================
class _Stats:
    """Progress bar 업데이트 + 파일 로깅 + 재시도/실패 통계"""

    LOG_INTERVAL = 5000

    def __init__(self, total: int, progress: Progress, task_id):
        self.total = total
        self.indexed = 0
        self.bulk_ms = 0.0
        self.retries = 0        # 재시도 발생 횟수
        self.failed_docs = 0    # 최종 실패 문서 수
        self.failed_batches = 0 # 최종 실패 배치 수
        self._start = time.perf_counter()
        self._progress = progress
        self._task_id = task_id

    def update(self, count: int, bulk_ms: float):
        self.indexed += count
        self.bulk_ms += bulk_ms
        elapsed = time.perf_counter() - self._start
        rps = self.indexed / elapsed if elapsed > 0 else 0

        # Rich Progress bar 갱신
        self._progress.update(
            self._task_id,
            advance=count,
            throughput=f"{rps:,.0f} docs/s",
            last_bulk=f"{bulk_ms:.0f}ms",
        )

        # 파일 로그 (주기적)
        n = self.indexed
        if n <= 500 or n % self.LOG_INTERVAL == 0 or n == self.total:
            pct = n / self.total * 100
            logger.info(
                f"[{pct:5.1f}%] {n:>7,}/{self.total:,}  "
                f"bulk={bulk_ms:.0f}ms  avg={rps:,.0f} t/s"
            )

    def record_retry(self):
        self.retries += 1

    def record_failure(self, doc_count: int):
        self.failed_docs += doc_count
        self.failed_batches += 1
        # Progress bar도 advance (실패 문서를 건너뛰어 100%에 도달)
        self._progress.update(self._task_id, advance=doc_count)

    @property
    def wall_sec(self) -> float:
        return time.perf_counter() - self._start


def _create_progress() -> Progress:
    """파이프라인용 Rich Progress bar 생성"""
    return Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[green]{task.fields[throughput]}[/]"),
        TextColumn("•"),
        TextColumn("[yellow]bulk={task.fields[last_bulk]}[/]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


# ============================================================
# Dead Letter Writer — 최종 실패 배치를 JSONL로 기록
# ============================================================
class _DeadLetterWriter:
    """실패 배치를 JSONL 파일에 비동기 안전하게 기록.

    파일 형식 (1줄 = 1 실패 배치):
        {"batch_id": 1000, "count": 500, "error": "...", "ts": "...", "keywords": [...]}
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()
        self._count = 0

    async def write(self, batch_id: int, keywords: list[str], error: str):
        record = {
            "batch_id": batch_id,
            "count": len(keywords),
            "error": error,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "keywords": keywords,
        }
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._count += 1

    @property
    def count(self) -> int:
        return self._count


# ============================================================
# 배치 처리 코루틴 — 재시도 + Dead Letter
# ============================================================
async def _process_batch(
    batch_id: int,
    keywords: list[str],
    indexer: ESIndexer,
    semaphore: asyncio.Semaphore,
    stats: _Stats,
    config: Config,
    dead_letter: _DeadLetterWriter,
    refresh: bool = False,
):
    """bulk_index + 재시도(지수 백오프) + 최종 실패 시 Dead Letter 기록."""
    async with semaphore:
        last_error: Exception | None = None

        for attempt in range(1, config.max_retries + 2):  # 1 = 최초, 2~N+1 = 재시도
            try:
                t0 = time.perf_counter()
                await indexer.bulk_index(batch_id, keywords)
                if refresh:
                    await indexer.refresh()
                stats.update(len(keywords), (time.perf_counter() - t0) * 1000)
                return  # 성공
            except Exception as e:
                last_error = e
                is_last = attempt > config.max_retries
                if is_last:
                    break
                # 지수 백오프: 1s → 2s → 4s → ...
                delay = config.retry_delay * (2 ** (attempt - 1))
                stats.record_retry()
                logger.warning(
                    f"Batch {batch_id} 실패 (시도 {attempt}/{config.max_retries + 1}), "
                    f"{delay:.1f}초 후 재시도: {e}"
                )
                await asyncio.sleep(delay)

        # 모든 재시도 소진 → Dead Letter에 기록
        logger.error(
            f"Batch {batch_id} 최종 실패 ({len(keywords)}건, "
            f"{config.max_retries + 1}회 시도): {last_error}"
        )
        stats.record_failure(len(keywords))
        await dead_letter.write(batch_id, keywords, str(last_error))


def _summary_table(title: str, rows: list[tuple[str, str]]) -> Table:
    """결과 요약 Rich Table"""
    table = Table(title=title, show_header=False, border_style="dim")
    table.add_column("항목", style="bold")
    table.add_column("값", justify="right", style="cyan")
    for label, value in rows:
        table.add_row(label, value)
    return table


def _resolve_dead_letter_path(config: Config, log_file: Path) -> Path:
    """Dead letter 파일 경로 결정. None이면 로그 파일과 같은 디렉토리에 자동 생성."""
    if config.dead_letter_path:
        config.dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
        return config.dead_letter_path
    return log_file.with_suffix(".dead_letter.jsonl")


def _build_summary_rows(
    count: int, total: int, wall: float,
    stats: _Stats, dead_letter: _DeadLetterWriter,
) -> list[tuple[str, str]]:
    """요약 테이블 행 생성. 재시도/실패가 있으면 추가 행 포함."""
    rows = [
        ("도큐먼트 수", f"{count:,}"),
        ("Wall time", f"{wall:.1f}초"),
        ("처리량", f"{total / wall:,.0f} docs/sec"),
        ("bulk 합계", f"{stats.bulk_ms / 1000:.1f}초 (코루틴 time)"),
    ]
    if stats.retries > 0:
        rows.append(("재시도 횟수", f"{stats.retries}회"))
    if stats.failed_docs > 0:
        rows.append(("실패 문서", f"[red]{stats.failed_docs:,}건 ({stats.failed_batches} 배치)[/]"))
        rows.append(("Dead Letter", str(dead_letter.path)))
    return rows


# ============================================================
# Bulk 파이프라인
# ============================================================
async def _run_bulk(config: Config, log_path: Path | None = None):
    log_file = _setup_logger(config, log_path, "es_bulk")
    logger.info(f"Log → {log_file}")

    # [1/4] 키워드 로드
    logger.info("[1/4] 키워드 로드")

    if config.parquet_path:
        # Parquet 파일 또는 디렉토리 읽기
        if ParquetReader is None:
            raise ImportError("Parquet 지원을 위해 qdrant_indexer 패키지가 필요합니다")
        reader = ParquetReader(
            config.parquet_path,
            chunk_size=config.parquet_chunk_size,
            text_column=config.parquet_text_column,
            limit=config.limit,
        )
        keywords = []
        for _, chunk_keywords in reader.iter_chunks():
            keywords.extend(chunk_keywords)
        total = len(keywords)
        source = f"Parquet: {config.parquet_path.name}"
        if config.parquet_path.is_dir():
            source += f" ({len(reader.parquet_files)} files)"
        logger.info(f"{source} → {total:,}건 로드 완료")
    else:
        # 텍스트 파일 읽기 (기존 방식)
        keywords = config.keywords_path.read_text(encoding="utf-8").strip().split("\n")
        if config.limit > 0:
            keywords = keywords[: config.limit]
        total = len(keywords)
        logger.info(f"텍스트 파일: {config.keywords_path.name} → {total:,}건 로드 완료")

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
    batches = [
        (i, keywords[i : i + config.batch_size])
        for i in range(0, total, config.batch_size)
    ]

    dl_path = _resolve_dead_letter_path(config, log_file)
    dead_letter = _DeadLetterWriter(dl_path)

    progress = _create_progress()
    with progress:
        task_id = progress.add_task("Indexing", total=total, throughput="--", last_bulk="--")
        stats = _Stats(total, progress, task_id)
        semaphore = asyncio.Semaphore(config.workers)

        await asyncio.gather(*[
            _process_batch(
                bid, batch, indexer, semaphore, stats,
                config, dead_letter, refresh=False,
            )
            for bid, batch in batches
        ])

    wall = stats.wall_sec

    # [4/4] 최적화 + 검증
    logger.info("[4/4] 인덱스 최적화 (refresh + force merge)")
    await indexer.finalize()

    count = await indexer.count()
    rows = _build_summary_rows(count, total, wall, stats, dead_letter)
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
    logger.info(f"Log → {log_file}")

    # [1/4] 키워드 로드
    logger.info("[1/4] 키워드 로드")

    if config.parquet_path:
        # Parquet 파일 또는 디렉토리 읽기
        if ParquetReader is None:
            raise ImportError("Parquet 지원을 위해 qdrant_indexer 패키지가 필요합니다")
        reader = ParquetReader(
            config.parquet_path,
            chunk_size=config.parquet_chunk_size,
            text_column=config.parquet_text_column,
            limit=config.limit,
        )
        keywords = []
        for _, chunk_keywords in reader.iter_chunks():
            keywords.extend(chunk_keywords)
        total = len(keywords)
        source = f"Parquet: {config.parquet_path.name}"
        if config.parquet_path.is_dir():
            source += f" ({len(reader.parquet_files)} files)"
        logger.info(f"{source} → {total:,}건 로드 완료")
    else:
        # 텍스트 파일 읽기 (기존 방식)
        keywords = config.keywords_path.read_text(encoding="utf-8").strip().split("\n")
        if config.limit > 0:
            keywords = keywords[: config.limit]
        total = len(keywords)
        logger.info(f"텍스트 파일: {config.keywords_path.name} → {total:,}건 로드 완료")

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
    batches = [
        (i, keywords[i : i + config.batch_size])
        for i in range(0, total, config.batch_size)
    ]

    dl_path = _resolve_dead_letter_path(config, log_file)
    dead_letter = _DeadLetterWriter(dl_path)

    progress = _create_progress()
    with progress:
        task_id = progress.add_task("Realtime", total=total, throughput="--", last_bulk="--")
        stats = _Stats(total, progress, task_id)
        semaphore = asyncio.Semaphore(config.workers)

        await asyncio.gather(*[
            _process_batch(
                bid, batch, indexer, semaphore, stats,
                config, dead_letter, refresh=True,
            )
            for bid, batch in batches
        ])

    wall = stats.wall_sec

    # [4/4] 검증 + 샘플 검색
    count = await indexer.count()
    logger.info("[4/4] 검증")

    rows = _build_summary_rows(count, total, wall, stats, dead_letter)
    console.print(_summary_table("결과 요약", rows))

    for label, value in rows:
        logger.info(f"{label}: {value}")

    # 샘플 검색
    sample = keywords[0] if keywords else ""
    if sample:
        results = await indexer.search(sample, size=3)
        hits = results["hits"]["hits"]
        logger.info(f"실시간 검색 확인 — 쿼리: {sample}")
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
            "[bold]Bulk 모드[/] — 인덱스 재생성 + 대량 적재",
            border_style="green",
        )
    )
    asyncio.run(_run_bulk(config, log_path))


def run_realtime(config: Config, log_path: Path | None = None):
    """Realtime 모드 — 기존 인덱스 보존 + 즉시 리프레시"""
    console.print(
        Panel.fit(
            "[bold]Realtime 모드[/] — 인덱스 보존 + 즉시 리프레시",
            border_style="blue",
        )
    )
    asyncio.run(_run_realtime(config, log_path))
