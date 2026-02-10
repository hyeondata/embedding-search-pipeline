"""재시도 로직 + 실패 로깅"""

import json
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from .log import get_logger

logger = get_logger("retry")


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
        batch_id_fn:     함수 인자에서 batch_id 추출하는 함수 (예: lambda args, kwargs: args[0])
        data_info_fn:    함수 인자에서 추가 정보 추출하는 함수 (예: lambda args, kwargs: {"size": len(args[1])})

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

                    # 실패 로깅
                    if failure_logger and batch_id_fn:
                        batch_id = batch_id_fn(args, kwargs)
                        data_info = data_info_fn(args, kwargs) if data_info_fn else {}
                        failure_logger.log_failure(batch_id, e, data_info, attempt)

                    # 마지막 시도면 예외 발생
                    if attempt >= retry_config.max_retries:
                        logger.error(
                            f"[bold red]최종 실패[/bold red] "
                            f"(시도 {attempt}/{retry_config.max_retries}): {e}"
                        )
                        raise

                    # 백오프 대기
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

            # 여기 도달하면 안 되지만, 안전장치
            raise last_exception

        return wrapper

    return decorator
