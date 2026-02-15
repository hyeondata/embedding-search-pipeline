"""
패키지 통합 로깅 설정 (Rich console + plain-text file)

설계:
  - Console: RichHandler (colored, timestamps, markup 지원)
  - File:    FileHandler (plain text, Rich markup 자동 제거)
  - logger.info() 한 번 호출로 양쪽에 동시 출력

사용법:
    from kserve_embed_client.log import setup_logging, get_logger

    logger = get_logger("qdrant_indexer", "pipeline")
    setup_logging("qdrant_indexer", log_file=Path("x.log"))
    logger.info("[bold green]완료![/bold green]")
"""

import logging
from pathlib import Path

from rich.logging import RichHandler
from rich.text import Text

DEFAULT_PKG = "kserve_embed_client"


class _PlainFormatter(logging.Formatter):
    """
    Rich markup 태그를 제거하는 FileHandler용 Formatter.

    rich.text.Text.from_markup()으로 안전하게 파싱 후 .plain으로 plain text 추출.
    예: "[bold green]완료![/bold green]" → "완료!"
    """

    def format(self, record: logging.LogRecord) -> str:
        original_msg = record.msg
        try:
            record.msg = Text.from_markup(str(record.msg)).plain
        except (ValueError, KeyError, AttributeError):
            pass
        result = super().format(record)
        record.msg = original_msg
        return result


def setup_logging(
    pkg_name: str = DEFAULT_PKG,
    log_file: Path = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    패키지 루트 로거에 핸들러를 설정.

    - RichHandler: 첫 호출 시 1회만 추가 (콘솔 출력)
    - FileHandler: log_file 인자가 있을 때마다 추가 (파일 로그)

    Args:
        pkg_name: 패키지 이름 (로거 이름)
        log_file: 로그 파일 경로 (None이면 콘솔만)
        level:    로그 레벨 (기본: INFO)

    Returns:
        패키지 루트 로거
    """
    logger = logging.getLogger(pkg_name)
    logger.setLevel(level)

    # RichHandler는 1회만 추가
    has_rich = any(isinstance(h, RichHandler) for h in logger.handlers)
    if not has_rich:
        console = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=True,
            log_time_format="[%H:%M:%S]",
        )
        console.setLevel(level)
        logger.addHandler(console)

    # FileHandler: 호출 시마다 추가 (인덱싱/검색별 별도 파일)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(_PlainFormatter("%(asctime)s  %(name)s  %(message)s"))
        fh.setLevel(level)
        logger.addHandler(fh)

    return logger


def get_logger(pkg_name: str, name: str = None) -> logging.Logger:
    """
    패키지 하위 로거 반환.

    예: get_logger("qdrant_indexer", "pipeline")
        → logging.getLogger("qdrant_indexer.pipeline")

    name을 생략하면 패키지 루트 로거를 반환.

    Args:
        pkg_name: 패키지 이름
        name:     모듈 이름 (선택)

    Returns:
        logging.Logger
    """
    if name:
        return logging.getLogger(f"{pkg_name}.{name}")
    return logging.getLogger(pkg_name)
