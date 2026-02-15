"""비동기 인덱싱 파이프라인 (Parquet 지원 + 재시도 + 실패 로깅)

EmbeddingClient.embed() — sync (requests) → run_in_executor로 비동기화
AsyncQdrantIndexer     — 네이티브 async (httpx 기반 AsyncQdrantClient)
동시성 제어           — asyncio.Semaphore (config.workers)
"""

import asyncio
import time
from pathlib import Path

from kserve_embed_client import (
    AsyncFailureLogger,
    EmbeddingClient,
    PipelineStats,
    ParquetReader,
    RURI_QUERY_PREFIX,
    batch_iter,
    get_logger,
    load_keywords,
    setup_logging,
    timer,
)

from .config import Config
from .indexer import AsyncQdrantIndexer

_PKG = "qdrant_indexer"
logger = get_logger(_PKG, "pipeline")


# ============================================================
# 배치 처리 코루틴
# ============================================================
async def _process_batch(
    batch_id: int,
    keywords: list[str],
    embedder: EmbeddingClient,
    indexer: AsyncQdrantIndexer,
    semaphore: asyncio.Semaphore,
    stats: PipelineStats,
    config: Config,
    failure_logger: AsyncFailureLogger,
):
    """
    단일 배치: 임베딩(executor) → Qdrant upsert(async) + 재시도.

    EmbeddingClient.embed()은 sync(requests)이므로 run_in_executor로 실행.
    AsyncQdrantIndexer.upsert_batch()는 네이티브 async.
    """
    async with semaphore:
        last_error: Exception | None = None

        for attempt in range(1, config.max_retries + 1):
            try:
                loop = asyncio.get_running_loop()

                with timer() as t_embed:
                    embeddings = await loop.run_in_executor(
                        None, embedder.embed, keywords
                    )

                with timer() as t_upsert:
                    await indexer.upsert_batch(
                        start_id=batch_id,
                        keywords=keywords,
                        embeddings=embeddings,
                    )

                stats.update(len(keywords), embed_ms=t_embed.ms, upsert_ms=t_upsert.ms)
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
        await failure_logger.log_failure(
            batch_id, last_error,
            data_info={"count": len(keywords), "keywords": keywords},
        )


# ============================================================
# 메인 파이프라인 (비동기)
# ============================================================
async def _run_indexing(config: Config, log_path: Path = None):
    """
    전체 인덱싱 파이프라인 (비동기):
      1. 키워드 로드 (텍스트 파일 또는 Parquet)
      2. Qdrant 컬렉션 생성
      3. 비동기 배치 처리 (embed via executor + upsert async + 재시도)
      4. (bulk_mode) HNSW 인덱싱 복원
      5. 검증 + 샘플 검색
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
    total_steps = 5 if config.bulk_mode else 4
    logger.info(f"[bold]\\[1/{total_steps}] 데이터 로드[/bold]")

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
    mode_label = " bulk_mode" if config.bulk_mode else ""
    logger.info(f"[bold]\\[2/{total_steps}] Qdrant 컬렉션: {config.collection_name}{mode_label}[/bold]")
    indexer = AsyncQdrantIndexer(config.qdrant_url, config.collection_name, config.vector_dim)
    await indexer.create_collection(bulk_mode=config.bulk_mode)
    if config.bulk_mode:
        logger.info(f"생성 완료 (dim={config.vector_dim}, COSINE, indexing_threshold=0)")
    else:
        logger.info(f"생성 완료 (dim={config.vector_dim}, COSINE)")

    # 3. 실패 로깅 설정
    failure_logger = AsyncFailureLogger(config.failure_log_path, enabled=config.log_failures)

    if config.log_failures:
        logger.info(f"실패 로그: [cyan]{config.failure_log_path}[/cyan]")
    logger.info(
        f"재시도: [cyan]{config.max_retries}회[/cyan] "
        f"(백오프: {config.retry_backoff}s{'×2^n' if config.retry_exponential else ''})"
    )

    # 4. 비동기 배치 처리
    logger.info(
        f"[bold]\\[3/{total_steps}] 비동기 인덱싱[/bold] "
        f"(workers=[cyan]{config.workers}[/cyan], batch=[cyan]{config.batch_size}[/cyan])"
    )
    embedder = EmbeddingClient(config.kserve_url, config.model_name)
    stats = PipelineStats(total, log_fn=logger.info)
    semaphore = asyncio.Semaphore(config.workers)

    if data_source == "parquet":
        batches = []
        for chunk_id, chunk_keywords in reader.iter_chunks():
            for start, batch in batch_iter(chunk_keywords, config.batch_size):
                batch_id = chunk_id * config.parquet_chunk_size + start
                batches.append((batch_id, batch))
    else:
        batches = list(batch_iter(keywords, config.batch_size))

    await asyncio.gather(*[
        _process_batch(
            bid, batch, embedder, indexer, semaphore,
            stats, config, failure_logger,
        )
        for bid, batch in batches
    ])

    wall = stats.wall_sec

    # 4a. 벌크 모드 인덱싱 복원
    if config.bulk_mode:
        step = total_steps - 1
        logger.info(
            f"[bold]\\[{step}/{total_steps}] 인덱싱 복원[/bold] "
            f"(indexing_threshold={config.bulk_indexing_threshold})"
        )
        await indexer.finalize(indexing_threshold=config.bulk_indexing_threshold)

    # 5. 검증
    logger.info(f"[bold]\\[{total_steps}/{total_steps}] 검증[/bold]")
    count = await indexer.get_count()
    logger.info(f"벡터 수: [cyan]{count:,}[/cyan]")
    logger.info(f"Wall time: [cyan]{wall:.1f}초[/cyan]")
    logger.info(f"처리량: [bold green]{total / wall:.0f} texts/sec[/bold green]")
    logger.info(f"임베딩 합계: {stats.get_timing('embed_ms') / 1000:.1f}초 (코루틴 time)")
    logger.info(f"upsert 합계: {stats.get_timing('upsert_ms') / 1000:.1f}초")

    # 샘플 검색
    logger.info("[bold]--- 샘플 검색 ---[/bold]")
    query = "東京 ラーメン おすすめ"
    loop = asyncio.get_running_loop()
    query_emb = await loop.run_in_executor(
        None, embedder.embed, [f"{RURI_QUERY_PREFIX}{query}"]
    )
    results = await indexer.search(query_emb[0].tolist(), top_k=5)

    logger.info(f"쿼리: {query}")
    for r in results.points:
        logger.info(f"  score=[green]{r.score:.4f}[/green]  {r.payload.get(config.payload_key, r.payload)}")

    await indexer.close()
    logger.info("[bold green]완료![/bold green]")


# ============================================================
# Realtime 파이프라인 (비동기)
# ============================================================
async def _run_realtime(config: Config, log_path: Path = None):
    """
    실시간 인덱싱 파이프라인 (비동기) — 기존 컬렉션 보존 + 증분 적재:
      1. 키워드 로드 (텍스트 파일 또는 Parquet)
      2. Qdrant 컬렉션 보존 (없으면 생성)
      3. 비동기 배치 처리 (embed via executor + upsert async + 재시도)
      4. 검증 + 샘플 검색

    run_indexing()과의 차이:
      - 컬렉션을 삭제하지 않고 기존 데이터 보존
      - ID를 기존 포인트 수부터 시작하여 충돌 방지
      - bulk_mode 최적화 비적용 (HNSW 인덱싱 활성 상태 유지)
    """
    # 로그 파일 설정
    if log_path is None:
        log_path = (
            config.parquet_path.parent / "logs"
            if config.parquet_path
            else config.keywords_path.parent.parent / "logs"
        )
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"realtime_{time.strftime('%Y%m%d_%H%M%S')}.log"
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

    # 2. Qdrant 컬렉션 보존
    logger.info(f"[bold]\\[2/4] Qdrant 컬렉션: {config.collection_name}[/bold]")
    indexer = AsyncQdrantIndexer(config.qdrant_url, config.collection_name, config.vector_dim)
    created = await indexer.ensure_collection()
    if created:
        id_offset = 0
        logger.info(f"신규 생성 (dim={config.vector_dim}, COSINE)")
    else:
        id_offset = await indexer.get_count()
        logger.info(f"기존 유지 (현재 [cyan]{id_offset:,}[/cyan]건)")

    # 3. 실패 로깅 설정
    failure_logger = AsyncFailureLogger(config.failure_log_path, enabled=config.log_failures)

    if config.log_failures:
        logger.info(f"실패 로그: [cyan]{config.failure_log_path}[/cyan]")

    # 4. 비동기 배치 처리
    logger.info(
        f"[bold]\\[3/4] 실시간 인덱싱[/bold] "
        f"(workers=[cyan]{config.workers}[/cyan], batch=[cyan]{config.batch_size}[/cyan], "
        f"id_offset=[cyan]{id_offset:,}[/cyan])"
    )
    embedder = EmbeddingClient(config.kserve_url, config.model_name)
    stats = PipelineStats(total, log_fn=logger.info)
    semaphore = asyncio.Semaphore(config.workers)

    if data_source == "parquet":
        batches = []
        for chunk_id, chunk_keywords in reader.iter_chunks():
            for start, batch in batch_iter(chunk_keywords, config.batch_size):
                batch_id = id_offset + chunk_id * config.parquet_chunk_size + start
                batches.append((batch_id, batch))
    else:
        batches = [(id_offset + start, batch) for start, batch in batch_iter(keywords, config.batch_size)]

    await asyncio.gather(*[
        _process_batch(
            bid, batch, embedder, indexer, semaphore,
            stats, config, failure_logger,
        )
        for bid, batch in batches
    ])

    wall = stats.wall_sec

    # 5. 검증
    logger.info("[bold]\\[4/4] 검증[/bold]")
    count = await indexer.get_count()
    logger.info(f"벡터 수: [cyan]{count:,}[/cyan] (추가: {total:,}건)")
    logger.info(f"Wall time: [cyan]{wall:.1f}초[/cyan]")
    logger.info(f"처리량: [bold green]{total / wall:.0f} texts/sec[/bold green]")
    logger.info(f"임베딩 합계: {stats.get_timing('embed_ms') / 1000:.1f}초 (코루틴 time)")
    logger.info(f"upsert 합계: {stats.get_timing('upsert_ms') / 1000:.1f}초")

    # 샘플 검색
    logger.info("[bold]--- 샘플 검색 ---[/bold]")
    query = "東京 ラーメン おすすめ"
    loop = asyncio.get_running_loop()
    query_emb = await loop.run_in_executor(
        None, embedder.embed, [f"{RURI_QUERY_PREFIX}{query}"]
    )
    results = await indexer.search(query_emb[0].tolist(), top_k=5)

    logger.info(f"쿼리: {query}")
    for r in results.points:
        logger.info(f"  score=[green]{r.score:.4f}[/green]  {r.payload.get(config.payload_key, r.payload)}")

    await indexer.close()
    logger.info("[bold green]완료![/bold green]")


# ============================================================
# Public API — 동기 래퍼
# ============================================================
def run_indexing(config: Config, log_path: Path | None = None):
    """Bulk 모드 — 컬렉션 재생성 후 대량 적재 (동기 래퍼)"""
    asyncio.run(_run_indexing(config, log_path))


def run_realtime(config: Config, log_path: Path | None = None):
    """Realtime 모드 — 기존 컬렉션 보존 + 증분 적재 (동기 래퍼)"""
    asyncio.run(_run_realtime(config, log_path))
