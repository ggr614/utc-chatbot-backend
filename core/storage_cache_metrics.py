"""
Storage module for cache metrics tracking and analytics.

This module provides a client for logging cache performance metrics
and querying aggregated analytics data. The cache_metrics table uses
BIGSERIAL as primary key (not UUID) and is append-only for analytics.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

import psycopg

from core.schemas import CacheHitRateStats, CacheLatencyStats, CacheMetric
from core.storage_base import BaseStorageClient
from utils.logger import PerformanceLogger, get_logger

logger = get_logger(__name__)


class CacheMetricsClient(BaseStorageClient):
    """
    Storage client for cache metrics tracking and analytics.

    Handles logging of cache events (hits/misses) and provides
    aggregated analytics queries for performance monitoring.

    Key differences from other storage clients:
    - Uses BIGSERIAL primary key (not UUID)
    - Append-only design (no updates/deletes)
    - Analytics-focused with aggregation queries
    - Optional foreign key to warm_cache_entries
    """

    def __init__(self):
        """Initialize the CacheMetricsClient for metrics storage."""
        super().__init__()
        logger.info("CacheMetricsClient initialized for cache_metrics table")

    def log_cache_event(
        self,
        cache_type: str,
        cache_entry_id: Optional[UUID] = None,
        latency_ms: Optional[int] = None,
        user_id: Optional[str] = None,
        request_timestamp: Optional[datetime] = None,
    ) -> int:
        """
        Log a single cache event to the metrics table.

        Args:
            cache_type: Type of cache event ('hit', 'miss', 'warm_hit', etc.)
            cache_entry_id: UUID of warm_cache_entries record (None for misses)
            latency_ms: Response latency in milliseconds
            user_id: User identifier for analytics
            request_timestamp: Event timestamp (defaults to NOW())

        Returns:
            The auto-generated metric ID (BIGSERIAL)

        Raises:
            ConnectionError: If database operation fails
            ValueError: If cache_type is invalid

        Example:
            >>> client = CacheMetricsClient()
            >>> metric_id = client.log_cache_event(
            ...     cache_type='hit',
            ...     cache_entry_id=uuid.uuid4(),
            ...     latency_ms=45,
            ...     user_id='user123'
            ... )
        """
        if not cache_type or not cache_type.strip():
            raise ValueError("cache_type cannot be empty")

        logger.debug(
            f"Logging cache event: type={cache_type}, "
            f"entry_id={cache_entry_id}, latency={latency_ms}ms"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Let database handle request_timestamp default if not provided
                    if request_timestamp:
                        cur.execute(
                            """
                            INSERT INTO cache_metrics
                            (cache_entry_id, request_timestamp, cache_type, latency_ms, user_id)
                            VALUES (%s, %s, %s, %s, %s)
                            RETURNING id
                            """,
                            (
                                cache_entry_id,
                                request_timestamp,
                                cache_type,
                                latency_ms,
                                user_id,
                            ),
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO cache_metrics
                            (cache_entry_id, cache_type, latency_ms, user_id)
                            VALUES (%s, %s, %s, %s)
                            RETURNING id
                            """,
                            (
                                cache_entry_id,
                                cache_type,
                                latency_ms,
                                user_id,
                            ),
                        )

                    result = cur.fetchone()
                    if not result:
                        raise ConnectionError(
                            "Failed to retrieve metric ID after insert"
                        )
                    metric_id = result[0]

                    logger.debug(f"Cache event logged successfully with ID {metric_id}")
                    return metric_id

        except psycopg.IntegrityError as e:
            # Foreign key violation (cache_entry_id doesn't exist)
            logger.error(f"Integrity error logging cache event: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to log cache event: {str(e)}")
            raise

    def insert_metrics(self, metrics: List[CacheMetric]) -> None:
        """
        Bulk insert cache metrics into database.

        Args:
            metrics: List of validated CacheMetric objects

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> metrics = [
            ...     CacheMetric(cache_type='hit', latency_ms=50),
            ...     CacheMetric(cache_type='miss', latency_ms=200),
            ... ]
            >>> client.insert_metrics(metrics)
        """
        if not metrics:
            logger.warning("No metrics to insert")
            return

        logger.info(f"Inserting {len(metrics)} cache metrics into database")

        try:
            with PerformanceLogger(logger, f"Insert {len(metrics)} metrics"):
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        for idx, metric in enumerate(metrics, 1):
                            try:
                                cur.execute(
                                    """
                                    INSERT INTO cache_metrics
                                    (cache_entry_id, request_timestamp, cache_type,
                                     latency_ms, user_id)
                                    VALUES (%s, %s, %s, %s, %s)
                                    RETURNING id
                                    """,
                                    (
                                        metric.cache_entry_id,
                                        metric.request_timestamp,
                                        metric.cache_type,
                                        metric.latency_ms,
                                        metric.user_id,
                                    ),
                                )
                                result = cur.fetchone()
                                if result:
                                    metric.id = result[0]

                                logger.debug(
                                    f"Inserted metric {idx}/{len(metrics)} (ID: {metric.id})"
                                )

                            except psycopg.IntegrityError as e:
                                logger.error(
                                    f"Integrity error inserting metric {idx}: {str(e)}"
                                )
                                raise
                            except psycopg.Error as e:
                                logger.error(
                                    f"Database error inserting metric {idx}: {str(e)}"
                                )
                                raise

                logger.info(f"Successfully inserted {len(metrics)} metrics")

        except Exception as e:
            logger.error(f"Failed to insert metrics: {str(e)}")
            raise

    def get_metrics_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        cache_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[CacheMetric]:
        """
        Retrieve cache metrics for a specific time range.

        Args:
            start_time: Beginning of time range (inclusive)
            end_time: End of time range (inclusive)
            cache_type: Filter by cache type (optional)
            user_id: Filter by user ID (optional)
            limit: Maximum number of records to return (optional)

        Returns:
            List of CacheMetric objects

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> from datetime import datetime, timedelta, timezone
            >>> end = datetime.now(timezone.utc)
            >>> start = end - timedelta(hours=24)
            >>> metrics = client.get_metrics_by_time_range(start, end, cache_type='hit')
        """
        logger.debug(
            f"Fetching metrics for time range: {start_time} to {end_time} "
            f"(type={cache_type}, user={user_id})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Build dynamic query based on filters
                    query = """
                        SELECT id, cache_entry_id, request_timestamp, cache_type,
                               latency_ms, user_id
                        FROM cache_metrics
                        WHERE request_timestamp >= %s AND request_timestamp <= %s
                    """
                    params: List = [start_time, end_time]

                    if cache_type:
                        query += " AND cache_type = %s"
                        params.append(cache_type)

                    if user_id:
                        query += " AND user_id = %s"
                        params.append(user_id)

                    query += " ORDER BY request_timestamp DESC"

                    if limit:
                        query += " LIMIT %s"
                        params.append(limit)

                    cur.execute(query, tuple(params))
                    rows = cur.fetchall()

                    metrics = []
                    for row in rows:
                        metrics.append(
                            CacheMetric(
                                id=row[0],
                                cache_entry_id=row[1],
                                request_timestamp=row[2],
                                cache_type=row[3],
                                latency_ms=row[4],
                                user_id=row[5],
                            )
                        )

                    logger.info(f"Retrieved {len(metrics)} metrics for time range")
                    return metrics

        except Exception as e:
            logger.error(f"Failed to fetch metrics by time range: {str(e)}")
            raise

    def get_metrics_by_cache_entry(
        self, cache_entry_id: UUID, limit: Optional[int] = None
    ) -> List[CacheMetric]:
        """
        Retrieve all metrics for a specific cache entry.

        Args:
            cache_entry_id: UUID of the warm_cache_entries record
            limit: Maximum number of records to return (optional)

        Returns:
            List of CacheMetric objects

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> entry_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
            >>> metrics = client.get_metrics_by_cache_entry(entry_id, limit=100)
        """
        logger.debug(
            f"Fetching metrics for cache entry {cache_entry_id} (limit={limit})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if limit:
                        cur.execute(
                            """
                            SELECT id, cache_entry_id, request_timestamp, cache_type,
                                   latency_ms, user_id
                            FROM cache_metrics
                            WHERE cache_entry_id = %s
                            ORDER BY request_timestamp DESC
                            LIMIT %s
                            """,
                            (cache_entry_id, limit),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, cache_entry_id, request_timestamp, cache_type,
                                   latency_ms, user_id
                            FROM cache_metrics
                            WHERE cache_entry_id = %s
                            ORDER BY request_timestamp DESC
                            """,
                            (cache_entry_id,),
                        )

                    rows = cur.fetchall()
                    metrics = []
                    for row in rows:
                        metrics.append(
                            CacheMetric(
                                id=row[0],
                                cache_entry_id=row[1],
                                request_timestamp=row[2],
                                cache_type=row[3],
                                latency_ms=row[4],
                                user_id=row[5],
                            )
                        )

                    logger.info(f"Retrieved {len(metrics)} metrics for cache entry")
                    return metrics

        except Exception as e:
            logger.error(
                f"Failed to fetch metrics for cache entry {cache_entry_id}: {str(e)}"
            )
            raise

    def get_cache_hit_rate(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str] = None,
    ) -> CacheHitRateStats:
        """
        Calculate cache hit rate statistics for a time range.

        Args:
            start_time: Beginning of time range (inclusive)
            end_time: End of time range (inclusive)
            user_id: Filter by user ID (optional)

        Returns:
            CacheHitRateStats object with aggregated statistics

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> from datetime import datetime, timedelta, timezone
            >>> end = datetime.now(timezone.utc)
            >>> start = end - timedelta(days=7)
            >>> stats = client.get_cache_hit_rate(start, end)
            >>> print(f"Hit rate: {stats.hit_rate:.2f}%")
        """
        logger.debug(
            f"Calculating cache hit rate for {start_time} to {end_time} (user={user_id})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if user_id:
                        cur.execute(
                            """
                            SELECT
                                COUNT(*) as total_requests,
                                COUNT(CASE WHEN cache_type IN ('hit', 'warm_hit') THEN 1 END) as cache_hits,
                                COUNT(CASE WHEN cache_type = 'miss' THEN 1 END) as cache_misses
                            FROM cache_metrics
                            WHERE request_timestamp >= %s
                              AND request_timestamp <= %s
                              AND user_id = %s
                            """,
                            (start_time, end_time, user_id),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT
                                COUNT(*) as total_requests,
                                COUNT(CASE WHEN cache_type IN ('hit', 'warm_hit') THEN 1 END) as cache_hits,
                                COUNT(CASE WHEN cache_type = 'miss' THEN 1 END) as cache_misses
                            FROM cache_metrics
                            WHERE request_timestamp >= %s
                              AND request_timestamp <= %s
                            """,
                            (start_time, end_time),
                        )

                    row = cur.fetchone()

                    if not row or row[0] == 0:
                        logger.warning("No metrics found for time range")
                        return CacheHitRateStats(
                            total_requests=0,
                            cache_hits=0,
                            cache_misses=0,
                            hit_rate=0.0,
                            time_period_start=start_time,
                            time_period_end=end_time,
                        )

                    total_requests = row[0]
                    cache_hits = row[1]
                    cache_misses = row[2]
                    hit_rate = (
                        (cache_hits / total_requests * 100)
                        if total_requests > 0
                        else 0.0
                    )

                    stats = CacheHitRateStats(
                        total_requests=total_requests,
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        hit_rate=hit_rate,
                        time_period_start=start_time,
                        time_period_end=end_time,
                    )

                    logger.info(
                        f"Cache hit rate calculated: {hit_rate:.2f}% "
                        f"({cache_hits}/{total_requests})"
                    )
                    return stats

        except Exception as e:
            logger.error(f"Failed to calculate cache hit rate: {str(e)}")
            raise

    def get_latency_stats(
        self,
        start_time: datetime,
        end_time: datetime,
        cache_type: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> CacheLatencyStats:
        """
        Calculate latency statistics for cache operations.

        Uses PostgreSQL percentile functions to compute p50, p95, p99.

        Args:
            start_time: Beginning of time range (inclusive)
            end_time: End of time range (inclusive)
            cache_type: Filter by cache type (optional)
            user_id: Filter by user ID (optional)

        Returns:
            CacheLatencyStats object with aggregated statistics

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> stats = client.get_latency_stats(start, end, cache_type='hit')
            >>> print(f"Avg latency: {stats.avg_latency_ms:.2f}ms")
            >>> print(f"P95 latency: {stats.p95_latency_ms:.2f}ms")
        """
        logger.debug(
            f"Calculating latency stats for {start_time} to {end_time} "
            f"(type={cache_type}, user={user_id})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Build dynamic query based on filters
                    query = """
                        SELECT
                            AVG(latency_ms) as avg_latency,
                            MIN(latency_ms) as min_latency,
                            MAX(latency_ms) as max_latency,
                            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) as p50_latency,
                            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency,
                            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_latency,
                            COUNT(*) as total_requests
                        FROM cache_metrics
                        WHERE request_timestamp >= %s
                          AND request_timestamp <= %s
                          AND latency_ms IS NOT NULL
                    """
                    params: List = [start_time, end_time]

                    if cache_type:
                        query += " AND cache_type = %s"
                        params.append(cache_type)

                    if user_id:
                        query += " AND user_id = %s"
                        params.append(user_id)

                    cur.execute(query, tuple(params))
                    row = cur.fetchone()

                    if not row or row[6] == 0:
                        logger.warning("No latency data found for time range")
                        return CacheLatencyStats(
                            avg_latency_ms=0.0,
                            min_latency_ms=0,
                            max_latency_ms=0,
                            p50_latency_ms=0.0,
                            p95_latency_ms=0.0,
                            p99_latency_ms=0.0,
                            total_requests=0,
                            time_period_start=start_time,
                            time_period_end=end_time,
                        )

                    stats = CacheLatencyStats(
                        avg_latency_ms=float(row[0]) if row[0] else 0.0,
                        min_latency_ms=row[1] if row[1] else 0,
                        max_latency_ms=row[2] if row[2] else 0,
                        p50_latency_ms=float(row[3]) if row[3] else 0.0,
                        p95_latency_ms=float(row[4]) if row[4] else 0.0,
                        p99_latency_ms=float(row[5]) if row[5] else 0.0,
                        total_requests=row[6],
                        time_period_start=start_time,
                        time_period_end=end_time,
                    )

                    logger.info(
                        f"Latency stats calculated: avg={stats.avg_latency_ms:.2f}ms, "
                        f"p95={stats.p95_latency_ms:.2f}ms, "
                        f"p99={stats.p99_latency_ms:.2f}ms"
                    )
                    return stats

        except Exception as e:
            logger.error(f"Failed to calculate latency stats: {str(e)}")
            raise

    def get_total_metrics_count(self) -> int:
        """
        Get the total number of metrics records in the database.

        Returns:
            Count of metrics records

        Raises:
            ConnectionError: If database operation fails
        """
        logger.debug("Getting total metrics count")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM cache_metrics")
                    result = cur.fetchone()
                    count = result[0] if result else 0
                    logger.debug(f"Total metrics count: {count}")
                    return count
        except Exception as e:
            logger.error(f"Failed to get metrics count: {str(e)}")
            raise

    def get_metrics_count_by_type(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """
        Get count of metrics grouped by cache_type.

        Args:
            start_time: Beginning of time range (optional)
            end_time: End of time range (optional)

        Returns:
            Dictionary mapping cache_type -> count

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> counts = client.get_metrics_count_by_type(start, end)
            >>> print(f"Hits: {counts.get('hit', 0)}, Misses: {counts.get('miss', 0)}")
        """
        logger.debug(
            f"Getting metrics count by type (time range: {start_time} to {end_time})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if start_time and end_time:
                        cur.execute(
                            """
                            SELECT cache_type, COUNT(*) as count
                            FROM cache_metrics
                            WHERE request_timestamp >= %s AND request_timestamp <= %s
                            GROUP BY cache_type
                            ORDER BY count DESC
                            """,
                            (start_time, end_time),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT cache_type, COUNT(*) as count
                            FROM cache_metrics
                            GROUP BY cache_type
                            ORDER BY count DESC
                            """
                        )

                    rows = cur.fetchall()
                    counts = {row[0]: row[1] for row in rows}

                    logger.info(f"Retrieved counts for {len(counts)} cache types")
                    return counts

        except Exception as e:
            logger.error(f"Failed to get metrics count by type: {str(e)}")
            raise
