"""간단한 타이머 컨텍스트 매니저"""

import time
from contextlib import contextmanager
from typing import Generator


class TimerResult:
    """타이머 결과. with 블록 종료 후 .ms, .sec 참조."""

    __slots__ = ("ms", "sec", "_start")

    def __init__(self):
        self.ms: float = 0.0
        self.sec: float = 0.0
        self._start: float = 0.0


@contextmanager
def timer() -> Generator[TimerResult, None, None]:
    """
    소요 시간 측정 컨텍스트 매니저.

    사용 예:
        with timer() as t:
            embeddings = client.embed(texts)
        print(f"임베딩: {t.ms:.0f}ms")

    기존 패턴 대체:
        t0 = time.perf_counter()
        embeddings = client.embed(texts)
        ms = (time.perf_counter() - t0) * 1000
    """
    result = TimerResult()
    result._start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - result._start
        result.sec = elapsed
        result.ms = elapsed * 1000
