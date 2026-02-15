"""파이프라인 진행 통계"""

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


class PipelineStats:
    """
    Thread-safe 파이프라인 진행 통계.

    Generic timing via **timing_ms kwargs:
        stats.update(count=64, embed_ms=120.5, upsert_ms=45.2)
        stats.update(count=64, bulk_ms=200.0)

    옵션:
        on_update: 업데이트마다 호출될 콜백 (예: ES Rich Progress bar)
        log_fn:    로그 함수 (기본: logger.info)
        log_interval: 로그 출력 간격 (기본: 1000)
        unit:      처리량 단위 (기본: "t/s")

    사용 예:
        stats = PipelineStats(total=10000, unit="docs/s")
        stats.update(64, embed_ms=15.0, upsert_ms=8.0)
        print(f"처리량: {stats.avg_rps:.0f} docs/s")
    """

    def __init__(
        self,
        total: int,
        *,
        on_update: Callable[["PipelineStats", int, dict[str, float]], None] | None = None,
        log_fn: Callable[..., None] | None = None,
        log_interval: int = 1000,
        unit: str = "t/s",
    ):
        self.total = total
        self.processed = 0
        self.retries = 0
        self.failed_count = 0
        self.failed_batches = 0

        self._timings: dict[str, float] = {}
        self._lock = threading.Lock()
        self._start = time.perf_counter()
        self._interval_start = time.perf_counter()
        self._interval_count = 0

        self._on_update = on_update
        self._log_fn = log_fn or logger.info
        self._log_interval = log_interval
        self._unit = unit

    def update(self, count: int, **timing_ms: float):
        """
        처리 완료 기록.

        Args:
            count: 처리된 항목 수
            **timing_ms: 이름별 소요 시간 (ms)
                예: embed_ms=120.5, upsert_ms=45.2
        """
        with self._lock:
            self.processed += count
            self._interval_count += count
            for key, val in timing_ms.items():
                self._timings[key] = self._timings.get(key, 0.0) + val
            n = self.processed

        if self._on_update:
            self._on_update(self, count, timing_ms)

        self._maybe_log(n, timing_ms)

    def record_retry(self):
        """재시도 횟수 증가."""
        with self._lock:
            self.retries += 1

    def record_failure(self, doc_count: int):
        """실패 기록."""
        with self._lock:
            self.failed_count += doc_count
            self.failed_batches += 1

    def get_timing(self, key: str) -> float:
        """특정 타이밍의 누적값 (ms) 반환."""
        with self._lock:
            return self._timings.get(key, 0.0)

    @property
    def wall_sec(self) -> float:
        return time.perf_counter() - self._start

    @property
    def avg_rps(self) -> float:
        w = self.wall_sec
        return self.processed / w if w > 0 else 0.0

    def _maybe_log(self, n: int, last_timing: dict[str, float]):
        should_log = (
            n <= 100
            or n % self._log_interval == 0
            or n == self.total
        )
        if not should_log:
            return

        now = time.perf_counter()
        with self._lock:
            interval_sec = now - self._interval_start
            interval_count = self._interval_count
            self._interval_start = now
            self._interval_count = 0

        elapsed = time.perf_counter() - self._start
        avg_rps = n / elapsed if elapsed > 0 else 0
        interval_rps = interval_count / interval_sec if interval_sec > 0 else 0
        pct = n / self.total * 100 if self.total > 0 else 0

        timing_parts = "  ".join(
            f"{k}=[cyan]{v:.0f}ms[/cyan]" for k, v in last_timing.items()
        )

        self._log_fn(
            f"[bold]\\[{pct:5.1f}%][/bold] {n:>7,}/{self.total:,}  "
            f"{timing_parts}  "
            f"avg=[green]{avg_rps:.0f} {self._unit}[/green]  "
            f"interval=[green]{interval_rps:.0f} {self._unit}[/green]"
        )
