"""Parquet 파일/폴더 청크 단위 읽기"""

from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from .log import get_logger

logger = get_logger("parquet_reader")


class ParquetReader:
    """
    대용량 Parquet 파일(들)을 청크 단위로 읽는 리더.

    단일 파일 또는 폴더 경로 모두 지원:
      - 단일 파일: 해당 파일을 청크 단위로 읽음
      - 폴더: 폴더 내 모든 .parquet 파일을 순서대로 읽음 (알파벳 순)

    pyarrow.parquet의 iter_batches()를 사용하여 메모리 효율적으로 스트리밍합니다.

    사용 예:
        # 단일 파일
        reader = ParquetReader("data/keywords.parquet", chunk_size=10000, text_column="keyword")

        # 폴더 (모든 .parquet 파일)
        reader = ParquetReader("data/partitioned/", chunk_size=10000, text_column="keyword")

        for batch_id, keywords in reader.iter_chunks():
            print(f"Batch {batch_id}: {len(keywords)} keywords")
    """

    def __init__(
        self,
        parquet_path: Path,
        chunk_size: int = 10000,
        text_column: str = "keyword",
        limit: int = 0,
    ):
        """
        Args:
            parquet_path: Parquet 파일 또는 폴더 경로
            chunk_size:   한 번에 읽을 행 수
            text_column:  텍스트가 들어있는 컬럼명
            limit:        읽을 최대 행 수 (0=전체)
        """
        self.parquet_path = Path(parquet_path)
        self.chunk_size = chunk_size
        self.text_column = text_column
        self.limit = limit

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"경로가 없습니다: {self.parquet_path}")

        # 단일 파일 vs 폴더 구분
        if self.parquet_path.is_file():
            self.parquet_files = [self.parquet_path]
            self.is_directory = False
        elif self.parquet_path.is_dir():
            # 폴더 내 모든 .parquet 파일 찾기 (알파벳 순)
            self.parquet_files = sorted(self.parquet_path.glob("*.parquet"))
            if not self.parquet_files:
                raise FileNotFoundError(
                    f"폴더에 .parquet 파일이 없습니다: {self.parquet_path}"
                )
            self.is_directory = True
        else:
            raise ValueError(f"유효하지 않은 경로: {self.parquet_path}")

        # 전체 행 수 계산
        self.total_rows = 0
        for pf in self.parquet_files:
            parquet_file = pq.ParquetFile(pf)
            self.total_rows += parquet_file.metadata.num_rows

        if limit > 0:
            self.total_rows = min(self.total_rows, limit)

        if self.is_directory:
            logger.info(
                f"Parquet 폴더 로드: [cyan]{self.parquet_path.name}[/cyan] "
                f"({len(self.parquet_files)}개 파일, {self.total_rows:,}행, 컬럼: {text_column})"
            )
            for i, pf in enumerate(self.parquet_files[:3]):  # 처음 3개만 표시
                logger.info(f"  [{i+1}] {pf.name}")
            if len(self.parquet_files) > 3:
                logger.info(f"  ... 외 {len(self.parquet_files) - 3}개")
        else:
            logger.info(
                f"Parquet 파일 로드: [cyan]{self.parquet_path.name}[/cyan] "
                f"({self.total_rows:,}행, 컬럼: {text_column})"
            )

    def iter_chunks(self) -> Iterator[tuple[int, list[str]]]:
        """
        청크 단위로 텍스트 데이터를 반환하는 제너레이터.

        여러 파일일 경우 파일 순서대로 연속해서 읽습니다.

        Yields:
            (batch_id, keywords) — batch_id는 0부터 시작, keywords는 문자열 리스트
        """
        batch_id = 0
        total_read = 0

        # 각 파일을 순서대로 처리
        for file_idx, parquet_file_path in enumerate(self.parquet_files):
            if self.is_directory:
                logger.info(
                    f"파일 [{file_idx + 1}/{len(self.parquet_files)}] 읽는 중: "
                    f"[cyan]{parquet_file_path.name}[/cyan]"
                )

            parquet_file = pq.ParquetFile(parquet_file_path)

            # pyarrow의 iter_batches로 청크 단위 스트리밍
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                # PyArrow Table → pandas DataFrame
                df = batch.to_pandas()

                # 텍스트 컬럼 추출
                if self.text_column not in df.columns:
                    available = ", ".join(df.columns)
                    raise ValueError(
                        f"컬럼 '{self.text_column}'이 파일 '{parquet_file_path.name}'에 없습니다. "
                        f"사용 가능한 컬럼: {available}"
                    )

                keywords = df[self.text_column].astype(str).tolist()

                # limit 적용
                if self.limit > 0:
                    remaining = self.limit - total_read
                    if remaining <= 0:
                        logger.info(
                            f"Parquet 읽기 완료 (limit 도달): [cyan]{total_read:,}[/cyan]행 "
                            f"([cyan]{batch_id}[/cyan]개 청크)"
                        )
                        return
                    if len(keywords) > remaining:
                        keywords = keywords[:remaining]

                total_read += len(keywords)

                yield (batch_id, keywords)
                batch_id += 1

                if self.limit > 0 and total_read >= self.limit:
                    logger.info(
                        f"Parquet 읽기 완료 (limit 도달): [cyan]{total_read:,}[/cyan]행 "
                        f"([cyan]{batch_id}[/cyan]개 청크)"
                    )
                    return

        logger.info(
            f"Parquet 읽기 완료: [cyan]{total_read:,}[/cyan]행 "
            f"([cyan]{batch_id}[/cyan]개 청크)"
        )

    @property
    def num_chunks(self) -> int:
        """예상 청크 수"""
        return (self.total_rows + self.chunk_size - 1) // self.chunk_size


def validate_parquet_columns(parquet_path: Path, text_column: str):
    """
    Parquet 파일의 컬럼을 검증하고, 사용 가능한 컬럼 목록을 출력합니다.

    Args:
        parquet_path: Parquet 파일 경로
        text_column:  텍스트 컬럼명

    Raises:
        ValueError: 컬럼이 존재하지 않을 때
    """
    parquet_file = pq.ParquetFile(parquet_path)
    schema = parquet_file.schema_arrow
    columns = [field.name for field in schema]

    if text_column not in columns:
        raise ValueError(
            f"컬럼 '{text_column}'이 Parquet 파일에 없습니다.\n"
            f"사용 가능한 컬럼: {', '.join(columns)}"
        )

    logger.info(f"Parquet 컬럼 검증 완료: '{text_column}' ✓")
    logger.info(f"전체 컬럼: {', '.join(columns)}")
