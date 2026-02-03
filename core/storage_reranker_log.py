"""
Storage module for reranker logging and analytics.

This module provides a client for logging reranker performance with initial RRF
rankings and final Cohere reranked results. The reranker_logs and reranker_results
tables use BIGSERIAL as primary key (not UUID) and are append-only for analytics.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg

from core.schemas import RerankerLog, RerankerResult
from core.storage_base import BaseStorageClient
from utils.logger import get_logger

logger = get_logger(__name__)


class RerankerLogClient(BaseStorageClient):
    """
    Storage client for reranker logging and analytics.

    Handles logging of reranking operations with two tables:
    1. reranker_logs: Aggregate metadata about reranking operation
    2. reranker_results: Individual result rankings (before/after)

    Key differences from other storage clients:
    - Uses BIGSERIAL primary keys (not UUID)
    - Dual-table structure (logs + results)
    - Computes derived metrics (rank_change, avg_rank_change)
    - Links 1:1 with query_logs table
    """

    def __init__(self, connection_pool=None):
        """
        Initialize the RerankerLogClient for reranker_logs and reranker_results tables.

        Args:
            connection_pool: Optional DatabaseConnectionPool instance for API mode
        """
        super().__init__(connection_pool=connection_pool)
        logger.info("RerankerLogClient initialized")

    def log_reranking(
        self,
        query_log_id: int,
        rrf_results: List[Dict[str, Any]],
        reranked_results: List[Dict[str, Any]],
        model_name: str,
        reranker_latency_ms: int,
        reranker_status: str = "success",
        error_message: Optional[str] = None,
    ) -> int:
        """
        Log reranking operation with initial RRF and final reranked results.

        Args:
            query_log_id: FK to query_logs.id
            rrf_results: Initial RRF results (before reranking)
                Format: [{"rank": 1, "combined_score": 0.045, "chunk": TextChunk}, ...]
            reranked_results: Final reranked results (after reranking)
                Format: [{"rank": 1, "combined_score": 0.95, "chunk": TextChunk,
                          "metadata": {"original_rank": 1, "original_score": 0.045}}, ...]
            model_name: Reranker model ID (e.g., 'cohere.rerank-v3-5:0')
            reranker_latency_ms: Reranking API latency in ms
            reranker_status: Status ('success', 'failed', 'skipped')
            error_message: Error message if failed

        Returns:
            The reranker_logs.id (BIGSERIAL)

        Raises:
            ValueError: If validation fails
            ConnectionError: If database operation fails

        Example:
            >>> client = RerankerLogClient()
            >>> log_id = client.log_reranking(
            ...     query_log_id=12345,
            ...     rrf_results=rrf_list,
            ...     reranked_results=reranked_list,
            ...     model_name="cohere.rerank-v3-5:0",
            ...     reranker_latency_ms=250
            ... )
        """
        # Validation
        if reranker_status not in ("success", "failed", "skipped"):
            raise ValueError(f"Invalid reranker_status: {reranker_status}")

        if not rrf_results:
            raise ValueError("rrf_results cannot be empty")

        logger.debug(
            f"Logging reranking for query_log_id {query_log_id}: "
            f"{len(rrf_results)} RRF → {len(reranked_results)} reranked"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Insert reranker_logs entry
                    cur.execute(
                        """
                        INSERT INTO reranker_logs
                        (query_log_id, reranker_status, model_name, reranker_latency_ms,
                         num_candidates, num_reranked, error_message)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            query_log_id,
                            reranker_status,
                            model_name,
                            reranker_latency_ms,
                            len(rrf_results),
                            len(reranked_results),
                            error_message,
                        ),
                    )

                    result = cur.fetchone()
                    if not result:
                        raise ConnectionError("Failed to retrieve reranker_log ID")
                    reranker_log_id = result[0]

                    # Build mapping: chunk_id → RRF data
                    rrf_map = {
                        str(r["chunk"].chunk_id): {
                            "rank": r["rank"],
                            "score": r["combined_score"],
                        }
                        for r in rrf_results
                    }

                    # Insert reranker_results entries
                    for reranked in reranked_results:
                        chunk = reranked["chunk"]
                        chunk_id = str(chunk.chunk_id)

                        # Get original RRF data
                        rrf_data = rrf_map.get(chunk_id)
                        if not rrf_data:
                            logger.warning(
                                f"Chunk {chunk_id[:8]} in reranked results but not in RRF results"
                            )
                            continue

                        # Compute rank change
                        rrf_rank = rrf_data["rank"]
                        reranked_rank = reranked["rank"]
                        rank_change = rrf_rank - reranked_rank  # Positive = moved up

                        # Insert result
                        cur.execute(
                            """
                            INSERT INTO reranker_results
                            (query_log_id, chunk_id, parent_article_id,
                             rrf_rank, rrf_score, reranked_rank, reranked_score,
                             rank_change, model_name)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                query_log_id,
                                chunk.chunk_id,
                                chunk.parent_article_id,
                                rrf_rank,
                                rrf_data["score"],
                                reranked_rank,
                                reranked["combined_score"],
                                rank_change,
                                model_name,
                            ),
                        )

                    # Compute and update aggregate metrics
                    cur.execute(
                        """
                        UPDATE reranker_logs
                        SET avg_rank_change = (
                            SELECT AVG(rank_change)
                            FROM reranker_results
                            WHERE query_log_id = %s
                        )
                        WHERE id = %s
                        """,
                        (query_log_id, reranker_log_id),
                    )

                    logger.info(
                        f"Reranking logged: reranker_log_id={reranker_log_id}, "
                        f"{len(reranked_results)} results"
                    )

                    return reranker_log_id

        except psycopg.IntegrityError as e:
            logger.error(f"Integrity error logging reranking: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to log reranking: {str(e)}")
            raise ConnectionError(f"Database error: {e}") from e

    def get_reranking_by_query_log_id(
        self, query_log_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve reranking metadata and results for a query.

        Args:
            query_log_id: The query log ID

        Returns:
            Dict with 'log' and 'results' keys, or None if not found
        """
        logger.debug(f"Fetching reranking for query_log_id {query_log_id}")

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Fetch reranker_logs entry
                    cur.execute(
                        """
                        SELECT id, reranker_status, model_name, reranker_latency_ms,
                               num_candidates, num_reranked, error_message,
                               avg_rank_change, created_at
                        FROM reranker_logs
                        WHERE query_log_id = %s
                        """,
                        (query_log_id,),
                    )
                    log_row = cur.fetchone()

                    if not log_row:
                        return None

                    # Fetch reranker_results entries
                    cur.execute(
                        """
                        SELECT chunk_id, parent_article_id, rrf_rank, rrf_score,
                               reranked_rank, reranked_score, rank_change
                        FROM reranker_results
                        WHERE query_log_id = %s
                        ORDER BY reranked_rank
                        """,
                        (query_log_id,),
                    )
                    result_rows = cur.fetchall()

                    return {
                        "log": {
                            "id": log_row[0],
                            "reranker_status": log_row[1],
                            "model_name": log_row[2],
                            "reranker_latency_ms": log_row[3],
                            "num_candidates": log_row[4],
                            "num_reranked": log_row[5],
                            "error_message": log_row[6],
                            "avg_rank_change": log_row[7],
                            "created_at": log_row[8],
                        },
                        "results": [
                            {
                                "chunk_id": r[0],
                                "parent_article_id": r[1],
                                "rrf_rank": r[2],
                                "rrf_score": r[3],
                                "reranked_rank": r[4],
                                "reranked_score": r[5],
                                "rank_change": r[6],
                            }
                            for r in result_rows
                        ],
                    }

        except Exception as e:
            logger.error(
                f"Failed to fetch reranking for query_log_id {query_log_id}: {str(e)}"
            )
            raise ConnectionError(f"Database error: {e}") from e

    def get_reranking_effectiveness_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Calculate reranking effectiveness statistics.

        Returns metrics like:
        - Success rate
        - Average latency
        - Average rank change
        - Top-k stability

        Args:
            start_time: Optional start of time range
            end_time: Optional end of time range

        Returns:
            Dict with aggregated statistics
        """
        logger.debug("Calculating reranking effectiveness stats")

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT
                            COUNT(*) as total_rerankings,
                            COUNT(*) FILTER (WHERE reranker_status = 'success') as successful_rerankings,
                            AVG(reranker_latency_ms) as avg_latency_ms,
                            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY reranker_latency_ms) as p95_latency_ms,
                            AVG(num_candidates) as avg_num_candidates,
                            AVG(avg_rank_change) as avg_rank_change_overall
                        FROM reranker_logs
                        WHERE 1=1
                    """
                    params = []

                    if start_time:
                        query += " AND created_at >= %s"
                        params.append(start_time)

                    if end_time:
                        query += " AND created_at <= %s"
                        params.append(end_time)

                    cur.execute(query, tuple(params))
                    row = cur.fetchone()

                    if not row or row[0] == 0:
                        return {
                            "total_rerankings": 0,
                            "success_rate": 0.0,
                            "avg_latency_ms": 0.0,
                            "p95_latency_ms": 0.0,
                            "avg_num_candidates": 0.0,
                            "avg_rank_change": 0.0,
                        }

                    total = row[0]
                    successful = row[1]

                    return {
                        "total_rerankings": total,
                        "successful_rerankings": successful,
                        "success_rate": (successful / total * 100)
                        if total > 0
                        else 0.0,
                        "avg_latency_ms": float(row[2]) if row[2] else 0.0,
                        "p95_latency_ms": float(row[3]) if row[3] else 0.0,
                        "avg_num_candidates": float(row[4]) if row[4] else 0.0,
                        "avg_rank_change": float(row[5]) if row[5] else 0.0,
                        "time_period_start": start_time,
                        "time_period_end": end_time,
                    }

        except Exception as e:
            logger.error(f"Failed to calculate reranking stats: {str(e)}")
            raise ConnectionError(f"Database error: {e}") from e
