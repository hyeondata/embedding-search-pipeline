"""동시 인덱싱 파이프라인 (Parquet 지원 + 재시도 + 실패 로깅)"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kserve_embed_client import (
    EmbeddingClient,
    FailureLogger,
    PipelineStats,
    ParquetReader,
    RURI_QUERY_PREFIX,
    RetryConfig,
    batch_iter,
    get_logger,
    load_keywords,
    setup_logging,
    timer,
    with_retry,
)

from .config import Config
from .indexer import QdrantIndexer

_PKG = "qdrant_indexer"
logger = get_logger(_PKG, "pipeline")


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
        with timer() as t_embed:
            embeddings = embedder.embed(keywords)

        with timer() as t_upsert:
            indexer.upsert_batch(start_id=batch_id, keywords=keywords, embeddings=embeddings)

        stats.update(len(keywords), embed_ms=t_embed.ms, upsert_ms=t_upsert.ms)

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
        keywords, source = load_keywords(config)
        total = len(keywords)
        logger.info(f"[cyan]{source}[/cyan] ({total:,}건)")
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
    stats = PipelineStats(total, log_fn=logger.info)

    process_batch = _create_batch_processor(embedder, indexer, stats, failure_logger, retry_config)

    if data_source == "parquet":
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            futures = []

            for chunk_id, chunk_keywords in reader.iter_chunks():
                for start, batch in batch_iter(chunk_keywords, config.batch_size):
                    batch_id = chunk_id * config.parquet_chunk_size + start
                    futures.append(pool.submit(process_batch, batch_id, batch))

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"[bold red]배치 처리 실패 (최종)[/bold red]: {e}")

    else:
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            futures = [
                pool.submit(process_batch, start, batch)
                for start, batch in batch_iter(keywords, config.batch_size)
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
    logger.info(f"임베딩 합계: {stats.get_timing('embed_ms') / 1000:.1f}초 (스레드 CPU time)")
    logger.info(f"upsert 합계: {stats.get_timing('upsert_ms') / 1000:.1f}초")

    # 샘플 검색
    logger.info("[bold]--- 샘플 검색 ---[/bold]")
    query = "東京 ラーメン おすすめ"
    query_emb = embedder.embed([f"{RURI_QUERY_PREFIX}{query}"])
    results = indexer.search(query_emb[0].tolist(), top_k=5)

    logger.info(f"쿼리: {query}")
    for r in results.points:
        logger.info(f"  score=[green]{r.score:.4f}[/green]  {r.payload['keyword']}")

    logger.info("[bold green]완료![/bold green]")
