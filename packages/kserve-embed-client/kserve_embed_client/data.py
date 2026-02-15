"""데이터 로딩 + 배치 유틸리티"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Protocol, Sequence, TypeVar

from .parquet_reader import ParquetReader

logger = logging.getLogger(__name__)

T = TypeVar("T")


class _HasDataFields(Protocol):
    """load_keywords가 기대하는 최소 인터페이스 (duck typing)."""

    keywords_path: Path | None
    limit: int
    parquet_path: Path | None
    parquet_chunk_size: int
    parquet_text_column: str


def load_keywords(config: _HasDataFields) -> tuple[list[str], str]:
    """
    Config에서 키워드를 로드.

    parquet_path가 설정되어 있으면 Parquet에서, 아니면 keywords_path 텍스트 파일에서 읽음.
    Parquet은 단일 파일과 디렉토리 모두 지원.

    Args:
        config: keywords_path, limit, parquet_path, parquet_chunk_size,
                parquet_text_column 속성을 가진 객체 (BaseConfig 또는 하위 클래스)

    Returns:
        (keywords, source_description)

    Raises:
        ValueError: keywords_path와 parquet_path 모두 None일 때
        FileNotFoundError: 파일/폴더가 없을 때
    """
    if config.parquet_path:
        reader = ParquetReader(
            config.parquet_path,
            chunk_size=config.parquet_chunk_size,
            text_column=config.parquet_text_column,
            limit=config.limit,
        )
        keywords: list[str] = []
        for _, chunk_keywords in reader.iter_chunks():
            keywords.extend(chunk_keywords)
        source = f"Parquet: {config.parquet_path.name}"
        if config.parquet_path.is_dir():
            source += f" ({len(reader.parquet_files)} files)"
    else:
        if config.keywords_path is None:
            raise ValueError(
                "keywords_path 또는 parquet_path 중 하나를 지정해야 합니다."
            )
        keywords = [
            line for line in
            config.keywords_path.read_text(encoding="utf-8").strip().split("\n")
            if line  # 빈 줄 제거
        ]
        if config.limit > 0:
            keywords = keywords[: config.limit]
        source = f"텍스트 파일: {config.keywords_path.name}"

    return keywords, source


def batch_iter(
    items: Sequence[T], batch_size: int
) -> Iterator[tuple[int, Sequence[T]]]:
    """
    시퀀스를 배치 단위로 분할하는 이터레이터.

    Yields:
        (start_index, batch) — start_index는 원본 시퀀스에서의 시작 위치

    사용 예:
        for start, batch in batch_iter(keywords, 64):
            process(start, batch)
    """
    for i in range(0, len(items), batch_size):
        yield i, items[i : i + batch_size]
