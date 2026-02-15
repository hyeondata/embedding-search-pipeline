"""재시도 로직 + 실패 로깅 (sync + async)"""

import asyncio
import json
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RetryConfig:
    """재시도 설정"""

    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        exponential: bool = True,
        max_backoff: float = 60.0,
    ):
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.exponential = exponential
        self.max_backoff = max_backoff


class FailureLogger:
    """실패한 작업을 JSONL 파일에 기록"""

    def __init__(self, log_path: Path, enabled: bool = True):
        self.log_path = log_path
        self.enabled = enabled
        if enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_failure(
        self,
        batch_id: int,
        error: Exception,
        data_info: dict[str, Any],
        attempt: int,
    ):
        """실패 기록"""
        if not self.enabled:
            return

        record = {
            "batch_id": batch_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "attempt": attempt,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **data_info,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.warning(
            f"[red]실패 기록[/red] batch_id={batch_id} attempt={attempt} "
            f"error={type(error).__name__}: {error}"
        )


def with_retry(
    retry_config: RetryConfig,
    failure_logger: FailureLogger | None = None,
    batch_id_fn: Callable = None,
    data_info_fn: Callable = None,
):
    """
    재시도 데코레이터.

    Args:
        retry_config:    재시도 설정
        failure_logger:  실패 로거 (None이면 로깅 안 함)
        batch_id_fn:     함수 인자에서 batch_id 추출하는 함수
        data_info_fn:    함수 인자에서 추가 정보 추출하는 함수

    사용 예:
        @with_retry(
            retry_config=RetryConfig(max_retries=3),
            failure_logger=FailureLogger(Path("failures.jsonl")),
            batch_id_fn=lambda args, kwargs: args[0],
            data_info_fn=lambda args, kwargs: {"keywords_count": len(args[1])},
        )
        def process_batch(batch_id, keywords):
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, retry_config.max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if failure_logger and batch_id_fn:
                        batch_id = batch_id_fn(args, kwargs)
                        data_info = data_info_fn(args, kwargs) if data_info_fn else {}
                        failure_logger.log_failure(batch_id, e, data_info, attempt)

                    if attempt >= retry_config.max_retries:
                        logger.error(
                            f"[bold red]최종 실패[/bold red] "
                            f"(시도 {attempt}/{retry_config.max_retries}): {e}"
                        )
                        raise

                    if retry_config.exponential:
                        backoff = min(
                            retry_config.initial_backoff * (2 ** (attempt - 1)),
                            retry_config.max_backoff,
                        )
                    else:
                        backoff = retry_config.initial_backoff

                    logger.warning(
                        f"[yellow]재시도 대기[/yellow] "
                        f"({attempt}/{retry_config.max_retries}) "
                        f"{backoff:.1f}초 후 재시도... error: {e}"
                    )
                    time.sleep(backoff)

            raise last_exception

        return wrapper

    return decorator


# ============================================================
# Async 버전
# ============================================================


class AsyncFailureLogger:
    """
    비동기 안전 실패 로거 (JSONL).

    asyncio.Lock으로 동시 쓰기 안전성 보장.
    FailureLogger의 async 버전.

    사용 예:
        dl = AsyncFailureLogger(Path("failures.jsonl"))
        await dl.log_failure(batch_id=0, error=e, data_info={"count": 10})
    """

    def __init__(self, log_path: Path, enabled: bool = True):
        self.log_path = log_path
        self.enabled = enabled
        self._lock = asyncio.Lock()
        self._count = 0
        if enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    async def log_failure(
        self,
        batch_id: int,
        error: Exception | str,
        data_info: dict[str, Any] | None = None,
    ):
        """실패 배치를 JSONL에 기록."""
        if not self.enabled:
            return

        error_type = type(error).__name__ if isinstance(error, Exception) else "str"
        error_msg = str(error)

        record = {
            "batch_id": batch_id,
            "error_type": error_type,
            "error_message": error_msg,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **(data_info or {}),
        }
        async with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._count += 1

        logger.warning(f"[red]실패 기록[/red] batch_id={batch_id}: {error_msg}")

    @property
    def count(self) -> int:
        return self._count


def async_with_retry(
    retry_config: RetryConfig,
    failure_logger: AsyncFailureLogger | None = None,
    batch_id_fn: Callable | None = None,
    data_info_fn: Callable | None = None,
):
    """
    비동기 재시도 데코레이터.

    with_retry의 async 버전. await asyncio.sleep()을 사용.

    사용 예:
        @async_with_retry(
            retry_config=RetryConfig(max_retries=3),
            failure_logger=AsyncFailureLogger(Path("f.jsonl")),
            batch_id_fn=lambda args, kwargs: args[0],
        )
        async def process_batch(batch_id, keywords):
            await indexer.bulk_index(batch_id, keywords)
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, retry_config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if failure_logger and batch_id_fn:
                        batch_id = batch_id_fn(args, kwargs)
                        data_info = data_info_fn(args, kwargs) if data_info_fn else {}
                        await failure_logger.log_failure(batch_id, e, data_info)

                    if attempt >= retry_config.max_retries:
                        logger.error(
                            f"[bold red]최종 실패[/bold red] "
                            f"(시도 {attempt}/{retry_config.max_retries}): {e}"
                        )
                        raise

                    if retry_config.exponential:
                        backoff = min(
                            retry_config.initial_backoff * (2 ** (attempt - 1)),
                            retry_config.max_backoff,
                        )
                    else:
                        backoff = retry_config.initial_backoff

                    logger.warning(
                        f"[yellow]재시도 대기[/yellow] "
                        f"({attempt}/{retry_config.max_retries}) "
                        f"{backoff:.1f}초 후 재시도... error: {e}"
                    )
                    await asyncio.sleep(backoff)

            raise last_exception

        return wrapper

    return decorator
