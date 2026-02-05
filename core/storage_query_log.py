"""
Storage module for query logging and analytics.

This module provides a client for logging user queries with optional embeddings
and querying aggregated analytics data. The query_logs table uses BIGSERIAL as
primary key (not UUID) and is append-only for analytics.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.types.json import Json

from core.schemas import PopularQuery, QueryCacheStats, QueryLatencyStats, QueryLog
from core.storage_base import BaseStorageClient
from utils.logger import PerformanceLogger, get_logger

logger = get_logger(__name__)


class QueryLogClient(BaseStorageClient):
    """
    Storage client for query logging and analytics.

    Handles logging of user queries with optional embeddings and provides
    aggregated analytics queries for query patterns, latency, and cache performance.

    Key differences from other storage clients:
    - Uses BIGSERIAL primary key (not UUID)
    - Append-only design (no updates/deletes)
    - Analytics-focused with aggregation queries
    - Handles optional vector embeddings (3072 dimensions)
    """

    def __init__(self, connection_pool=None):
        """
        Initialize the QueryLogClient for query_logs table.

        Args:
            connection_pool: Optional DatabaseConnectionPool instance for API mode
        """
        super().__init__(connection_pool=connection_pool)
        logger.info("QueryLogClient initialized for query_logs table")

    def log_query(
        self,
        raw_query: str,
        cache_result: str,
        query_embedding: Optional[List[float]] = None,
        latency_ms: Optional[int] = None,
        email: Optional[str] = None,
        command: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ) -> int:
        """
        Log a single query event to the query_logs table.

        Args:
            raw_query: The original user query text
            cache_result: Cache result type ('hit', 'miss', etc.)
            query_embedding: Optional vector embedding (3072 dimensions)
            latency_ms: Response latency in milliseconds
            email: User email address for analytics
            command: Command type ('bypass', 'q', 'qlong', 'debug', 'debuglong', or None)
            created_at: Query timestamp (defaults to NOW())

        Returns:
            The auto-generated log ID (BIGSERIAL)

        Raises:
            ConnectionError: If database operation fails
            ValueError: If raw_query or cache_result is empty
            ValueError: If query_embedding dimensions don't match (must be 3072)
            ValueError: If command is not a valid value

        Example:
            >>> client = QueryLogClient()
            >>> log_id = client.log_query(
            ...     raw_query="How do I reset my password?",
            ...     cache_result='hit',
            ...     latency_ms=125,
            ...     email='user@example.com',
            ...     command='q'
            ... )
        """
        if not raw_query or not raw_query.strip():
            raise ValueError("raw_query cannot be empty")

        if not cache_result or not cache_result.strip():
            raise ValueError("cache_result cannot be empty")

        if query_embedding is not None and len(query_embedding) != 3072:
            raise ValueError(
                f"query_embedding must have 3072 dimensions, got {len(query_embedding)}"
            )

        if command is not None and command not in (
            "bypass",
            "q",
            "qlong",
            "debug",
            "debuglong",
        ):
            raise ValueError(
                f"command must be 'bypass', 'q', 'qlong', 'debug', 'debuglong', or None, got '{command}'"
            )

        logger.debug(
            f"Logging query: {raw_query[:50]}... "
            f"(cache={cache_result}, latency={latency_ms}ms)"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Let database handle created_at default if not provided
                    if created_at:
                        cur.execute(
                            """
                            INSERT INTO query_logs
                            (raw_query, query_embedding, cache_result, latency_ms, email, command, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            RETURNING id
                            """,
                            (
                                raw_query,
                                query_embedding,
                                cache_result,
                                latency_ms,
                                email,
                                command,
                                created_at,
                            ),
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO query_logs
                            (raw_query, query_embedding, cache_result, latency_ms, email, command)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            RETURNING id
                            """,
                            (
                                raw_query,
                                query_embedding,
                                cache_result,
                                latency_ms,
                                email,
                                command,
                            ),
                        )

                    result = cur.fetchone()
                    if not result:
                        raise ConnectionError("Failed to retrieve log ID after insert")
                    log_id = result[0]

                    logger.debug(f"Query logged successfully with ID {log_id}")
                    return log_id

        except psycopg.IntegrityError as e:
            logger.error(f"Integrity error logging query: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to log query: {str(e)}")
            raise

    def insert_query_logs(self, logs: List[QueryLog]) -> None:
        """
        Bulk insert query logs into database.

        Args:
            logs: List of validated QueryLog objects

        Raises:
            ConnectionError: If database operation fails
            ValueError: If any query_embedding dimensions don't match expected (3072)

        Example:
            >>> logs = [
            ...     QueryLog(raw_query="Query 1", cache_result='hit', latency_ms=50),
            ...     QueryLog(raw_query="Query 2", cache_result='miss', latency_ms=200),
            ... ]
            >>> client.insert_query_logs(logs)
        """
        if not logs:
            logger.warning("No query logs to insert")
            return

        logger.info(f"Inserting {len(logs)} query logs into database")

        try:
            with PerformanceLogger(logger, f"Insert {len(logs)} query logs"):
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        for idx, log in enumerate(logs, 1):
                            try:
                                # Validate embedding dimensions if provided
                                if (
                                    log.query_embedding is not None
                                    and len(log.query_embedding) != 3072
                                ):
                                    raise ValueError(
                                        f"Query log {idx}: embedding must have 3072 dimensions, "
                                        f"got {len(log.query_embedding)}"
                                    )

                                cur.execute(
                                    """
                                    INSERT INTO query_logs
                                    (raw_query, query_embedding, cache_result,
                                     latency_ms, email, created_at)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    RETURNING id
                                    """,
                                    (
                                        log.raw_query,
                                        log.query_embedding,
                                        log.cache_result,
                                        log.latency_ms,
                                        log.email,
                                        log.created_at,
                                    ),
                                )
                                result = cur.fetchone()
                                if result:
                                    log.id = result[0]

                                logger.debug(
                                    f"Inserted query log {idx}/{len(logs)} (ID: {log.id})"
                                )

                            except psycopg.IntegrityError as e:
                                logger.error(
                                    f"Integrity error inserting query log {idx}: {str(e)}"
                                )
                                raise
                            except psycopg.Error as e:
                                logger.error(
                                    f"Database error inserting query log {idx}: {str(e)}"
                                )
                                raise

                logger.info(f"Successfully inserted {len(logs)} query logs")

        except Exception as e:
            logger.error(f"Failed to insert query logs: {str(e)}")
            raise

    def log_query_results(
        self,
        query_log_id: int,
        results: List[Dict[str, Any]],
        search_method: str,
    ) -> None:
        """
        Log retrieval results for a query to the query_results table.

        Args:
            query_log_id: The query log ID (from log_query() return value)
            results: List of result dictionaries with keys:
                - rank (int): 1-indexed position
                - score (float): Relevance score
                - chunk_id (UUID): Chunk identifier
                - parent_article_id (UUID): Article identifier
            search_method: Search method used ('bm25', 'vector', 'hybrid')

        Raises:
            ValueError: If results empty or missing required fields
            ConnectionError: If database operation fails

        Example:
            >>> log_id = client.log_query("password reset", "miss", 125, "user123")
            >>> results = [
            ...     {"rank": 1, "score": 5.2, "chunk_id": uuid1, "parent_article_id": uuid2},
            ...     {"rank": 2, "score": 4.8, "chunk_id": uuid3, "parent_article_id": uuid4},
            ... ]
            >>> client.log_query_results(log_id, results, "bm25")
        """
        if not results:
            raise ValueError("results cannot be empty")

        if search_method not in ("bm25", "vector", "hybrid", "hyde"):
            raise ValueError(
                f"search_method must be 'bm25', 'vector', 'hybrid', or 'hyde', got '{search_method}'"
            )

        logger.debug(
            f"Logging {len(results)} results for query_log_id {query_log_id} "
            f"(method={search_method})"
        )

        try:
            with PerformanceLogger(logger, f"Insert {len(results)} query results"):
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        for idx, result in enumerate(results, 1):
                            try:
                                # Validate required fields
                                if "rank" not in result:
                                    raise ValueError(
                                        f"Result {idx}: missing required field 'rank'"
                                    )
                                if "score" not in result:
                                    raise ValueError(
                                        f"Result {idx}: missing required field 'score'"
                                    )
                                if "chunk_id" not in result:
                                    raise ValueError(
                                        f"Result {idx}: missing required field 'chunk_id'"
                                    )
                                if "parent_article_id" not in result:
                                    raise ValueError(
                                        f"Result {idx}: missing required field 'parent_article_id'"
                                    )

                                # Validate rank >= 1
                                if result["rank"] < 1:
                                    raise ValueError(
                                        f"Result {idx}: rank must be >= 1, got {result['rank']}"
                                    )

                                cur.execute(
                                    """
                                    INSERT INTO query_results
                                    (query_log_id, search_method, rank, score,
                                     chunk_id, parent_article_id)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    """,
                                    (
                                        query_log_id,
                                        search_method,
                                        result["rank"],
                                        result["score"],
                                        result["chunk_id"],
                                        result["parent_article_id"],
                                    ),
                                )

                                logger.debug(
                                    f"Inserted query result {idx}/{len(results)} "
                                    f"(rank={result['rank']}, score={result['score']:.2f})"
                                )

                            except psycopg.IntegrityError as e:
                                logger.error(
                                    f"Integrity error inserting query result {idx}: {str(e)}"
                                )
                                raise
                            except psycopg.Error as e:
                                logger.error(
                                    f"Database error inserting query result {idx}: {str(e)}"
                                )
                                raise

                logger.info(
                    f"Successfully inserted {len(results)} query results for query_log_id {query_log_id}"
                )

        except Exception as e:
            logger.error(f"Failed to insert query results: {str(e)}")
            raise

    def log_query_with_results(
        self,
        raw_query: str,
        cache_result: str,
        search_method: str,
        results: List[Dict[str, Any]],
        latency_ms: Optional[int] = None,
        email: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        command: Optional[str] = None,
    ) -> int:
        """
        Log query and results in a single transaction.

        This is the recommended method for logging queries with their results,
        as it ensures atomicity - both the query and its results are logged
        together, or neither is logged if an error occurs.

        Args:
            raw_query: The original user query text
            cache_result: Cache result type ('hit', 'miss', etc.)
            search_method: Search method used ('bm25', 'vector', 'hybrid')
            results: List of result dictionaries with keys:
                - rank (int): 1-indexed position
                - score (float): Relevance score
                - chunk_id (UUID): Chunk identifier
                - parent_article_id (UUID): Article identifier
            latency_ms: Query response latency in milliseconds
            email: User email address for analytics
            query_embedding: Optional vector embedding (3072 dimensions)
            command: Command type ('bypass', 'q', 'qlong', 'debug', 'debuglong', or None)

        Returns:
            The query log ID

        Raises:
            ValueError: If validation fails
            ConnectionError: If database operation fails

        Example:
            >>> log_id = client.log_query_with_results(
            ...     raw_query="password reset",
            ...     cache_result="miss",
            ...     search_method="bm25",
            ...     results=result_list,
            ...     latency_ms=125,
            ...     email="user@example.com",
            ...     command="q"
            ... )
        """
        logger.debug(
            f"Logging query with {len(results)} results: {raw_query[:50]}... "
            f"(method={search_method})"
        )

        try:
            # First log the query to get the query_log_id
            query_log_id = self.log_query(
                raw_query=raw_query,
                cache_result=cache_result,
                query_embedding=query_embedding,
                latency_ms=latency_ms,
                email=email,
                command=command,
            )

            # Then log the results with the query_log_id
            if results:
                self.log_query_results(
                    query_log_id=query_log_id,
                    results=results,
                    search_method=search_method,
                )

            logger.info(
                f"Successfully logged query and {len(results)} results (query_log_id={query_log_id})"
            )
            return query_log_id

        except Exception as e:
            logger.error(f"Failed to log query with results: {str(e)}")
            raise

    def get_queries_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        cache_result: Optional[str] = None,
        email: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[QueryLog]:
        """
        Retrieve query logs for a specific time range.

        Args:
            start_time: Beginning of time range (inclusive)
            end_time: End of time range (inclusive)
            cache_result: Filter by cache result type (optional)
            email: Filter by user email (optional)
            limit: Maximum number of records to return (optional)

        Returns:
            List of QueryLog objects (without embeddings for performance)

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> from datetime import datetime, timedelta, timezone
            >>> end = datetime.now(timezone.utc)
            >>> start = end - timedelta(hours=24)
            >>> queries = client.get_queries_by_time_range(start, end, cache_result='miss')
        """
        logger.debug(
            f"Fetching queries for time range: {start_time} to {end_time} "
            f"(cache_result={cache_result}, email={email})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Build dynamic query based on filters (exclude embeddings for performance)
                    query = """
                        SELECT id, raw_query, cache_result, latency_ms, email, command, created_at
                        FROM query_logs
                        WHERE created_at >= %s AND created_at <= %s
                    """
                    params: List = [start_time, end_time]

                    if cache_result:
                        query += " AND cache_result = %s"
                        params.append(cache_result)

                    if email:
                        query += " AND email = %s"
                        params.append(email)

                    query += " ORDER BY created_at DESC"

                    if limit:
                        query += " LIMIT %s"
                        params.append(limit)

                    cur.execute(query, tuple(params))
                    rows = cur.fetchall()

                    logs = []
                    for row in rows:
                        logs.append(
                            QueryLog(
                                id=row[0],
                                raw_query=row[1],
                                query_embedding=None,  # Excluded for performance
                                cache_result=row[2],
                                latency_ms=row[3],
                                email=row[4],
                                command=row[5],
                                created_at=row[6],
                            )
                        )

                    logger.info(f"Retrieved {len(logs)} query logs for time range")
                    return logs

        except Exception as e:
            logger.error(f"Failed to fetch queries by time range: {str(e)}")
            raise

    def get_query_by_id(
        self, query_id: int, include_embedding: bool = False
    ) -> Optional[QueryLog]:
        """
        Retrieve a specific query log by ID.

        Args:
            query_id: The query log ID (BIGSERIAL)
            include_embedding: Whether to include the vector embedding (default: False)

        Returns:
            QueryLog object or None if not found

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> query = client.get_query_by_id(12345, include_embedding=True)
        """
        logger.debug(
            f"Fetching query log {query_id} (include_embedding={include_embedding})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if include_embedding:
                        cur.execute(
                            """
                            SELECT id, raw_query, query_embedding, cache_result,
                                   latency_ms, user_id, command, created_at
                            FROM query_logs
                            WHERE id = %s
                            """,
                            (query_id,),
                        )
                        row = cur.fetchone()
                        if row:
                            return QueryLog(
                                id=row[0],
                                raw_query=row[1],
                                query_embedding=row[2],
                                cache_result=row[3],
                                latency_ms=row[4],
                                user_id=row[5],
                                command=row[6],
                                created_at=row[7],
                            )
                    else:
                        cur.execute(
                            """
                            SELECT id, raw_query, cache_result, latency_ms, user_id, command, created_at
                            FROM query_logs
                            WHERE id = %s
                            """,
                            (query_id,),
                        )
                        row = cur.fetchone()
                        if row:
                            return QueryLog(
                                id=row[0],
                                raw_query=row[1],
                                query_embedding=None,
                                cache_result=row[2],
                                latency_ms=row[3],
                                user_id=row[4],
                                command=row[5],
                                created_at=row[6],
                            )

                    logger.debug(f"Query log {query_id} not found")
                    return None

        except Exception as e:
            logger.error(f"Failed to fetch query by ID {query_id}: {str(e)}")
            raise

    def get_query_latency_stats(
        self,
        start_time: datetime,
        end_time: datetime,
        cache_result: Optional[str] = None,
        email: Optional[str] = None,
    ) -> QueryLatencyStats:
        """
        Calculate latency statistics for queries.

        Uses PostgreSQL percentile functions to compute p50, p95, p99.

        Args:
            start_time: Beginning of time range (inclusive)
            end_time: End of time range (inclusive)
            cache_result: Filter by cache result type (optional)
            email: Filter by user email (optional)

        Returns:
            QueryLatencyStats object with aggregated statistics

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> stats = client.get_query_latency_stats(start, end, cache_result='miss')
            >>> print(f"Avg latency: {stats.avg_latency_ms:.2f}ms")
            >>> print(f"P95 latency: {stats.p95_latency_ms:.2f}ms")
        """
        logger.debug(
            f"Calculating query latency stats for {start_time} to {end_time} "
            f"(cache_result={cache_result}, email={email})"
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
                            COUNT(*) as total_queries
                        FROM query_logs
                        WHERE created_at >= %s
                          AND created_at <= %s
                          AND latency_ms IS NOT NULL
                    """
                    params: List = [start_time, end_time]

                    if cache_result:
                        query += " AND cache_result = %s"
                        params.append(cache_result)

                    if email:
                        query += " AND email = %s"
                        params.append(email)

                    cur.execute(query, tuple(params))
                    row = cur.fetchone()

                    if not row or row[6] == 0:
                        logger.warning("No latency data found for time range")
                        return QueryLatencyStats(
                            avg_latency_ms=0.0,
                            min_latency_ms=0,
                            max_latency_ms=0,
                            p50_latency_ms=0.0,
                            p95_latency_ms=0.0,
                            p99_latency_ms=0.0,
                            total_queries=0,
                            time_period_start=start_time,
                            time_period_end=end_time,
                        )

                    stats = QueryLatencyStats(
                        avg_latency_ms=float(row[0]) if row[0] else 0.0,
                        min_latency_ms=row[1] if row[1] else 0,
                        max_latency_ms=row[2] if row[2] else 0,
                        p50_latency_ms=float(row[3]) if row[3] else 0.0,
                        p95_latency_ms=float(row[4]) if row[4] else 0.0,
                        p99_latency_ms=float(row[5]) if row[5] else 0.0,
                        total_queries=row[6],
                        time_period_start=start_time,
                        time_period_end=end_time,
                    )

                    logger.info(
                        f"Query latency stats calculated: avg={stats.avg_latency_ms:.2f}ms, "
                        f"p95={stats.p95_latency_ms:.2f}ms, "
                        f"p99={stats.p99_latency_ms:.2f}ms"
                    )
                    return stats

        except Exception as e:
            logger.error(f"Failed to calculate query latency stats: {str(e)}")
            raise

    def get_popular_queries(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 10,
        min_occurrences: int = 2,
    ) -> List[PopularQuery]:
        """
        Get most frequently asked queries with statistics.

        Args:
            start_time: Beginning of time range (inclusive)
            end_time: End of time range (inclusive)
            limit: Maximum number of popular queries to return
            min_occurrences: Minimum number of times query must appear

        Returns:
            List of PopularQuery objects ordered by frequency

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> popular = client.get_popular_queries(start, end, limit=20)
            >>> for query in popular:
            ...     print(f"{query.raw_query}: {query.query_count} times, "
            ...           f"{query.avg_latency_ms:.1f}ms avg")
        """
        logger.debug(
            f"Fetching popular queries for {start_time} to {end_time} "
            f"(limit={limit}, min_occurrences={min_occurrences})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            raw_query,
                            COUNT(*) as query_count,
                            AVG(latency_ms) as avg_latency_ms,
                            COUNT(CASE WHEN cache_result IN ('hit', 'warm_hit') THEN 1 END)::FLOAT
                                / COUNT(*) * 100 as cache_hit_rate,
                            MAX(created_at) as last_queried_at
                        FROM query_logs
                        WHERE created_at >= %s AND created_at <= %s
                        GROUP BY raw_query
                        HAVING COUNT(*) >= %s
                        ORDER BY query_count DESC
                        LIMIT %s
                        """,
                        (start_time, end_time, min_occurrences, limit),
                    )

                    rows = cur.fetchall()
                    popular_queries = []
                    for row in rows:
                        popular_queries.append(
                            PopularQuery(
                                raw_query=row[0],
                                query_count=row[1],
                                avg_latency_ms=float(row[2]) if row[2] else 0.0,
                                cache_hit_rate=float(row[3]) if row[3] else 0.0,
                                last_queried_at=row[4],
                            )
                        )

                    logger.info(f"Retrieved {len(popular_queries)} popular queries")
                    return popular_queries

        except Exception as e:
            logger.error(f"Failed to fetch popular queries: {str(e)}")
            raise

    def get_query_cache_performance(
        self,
        start_time: datetime,
        end_time: datetime,
        email: Optional[str] = None,
    ) -> QueryCacheStats:
        """
        Calculate cache performance statistics for queries.

        Args:
            start_time: Beginning of time range (inclusive)
            end_time: End of time range (inclusive)
            email: Filter by user email (optional)

        Returns:
            QueryCacheStats object with aggregated statistics

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> stats = client.get_query_cache_performance(start, end)
            >>> print(f"Cache hit rate: {stats.hit_rate:.2f}%")
        """
        logger.debug(
            f"Calculating query cache performance for {start_time} to {end_time} (email={email})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if email:
                        cur.execute(
                            """
                            SELECT
                                COUNT(*) as total_queries,
                                COUNT(CASE WHEN cache_result IN ('hit', 'warm_hit') THEN 1 END) as cache_hits,
                                COUNT(CASE WHEN cache_result = 'miss' THEN 1 END) as cache_misses
                            FROM query_logs
                            WHERE created_at >= %s
                              AND created_at <= %s
                              AND email = %s
                            """,
                            (start_time, end_time, email),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT
                                COUNT(*) as total_queries,
                                COUNT(CASE WHEN cache_result IN ('hit', 'warm_hit') THEN 1 END) as cache_hits,
                                COUNT(CASE WHEN cache_result = 'miss' THEN 1 END) as cache_misses
                            FROM query_logs
                            WHERE created_at >= %s
                              AND created_at <= %s
                            """,
                            (start_time, end_time),
                        )

                    row = cur.fetchone()

                    if not row or row[0] == 0:
                        logger.warning("No query data found for time range")
                        return QueryCacheStats(
                            total_queries=0,
                            cache_hits=0,
                            cache_misses=0,
                            hit_rate=0.0,
                            time_period_start=start_time,
                            time_period_end=end_time,
                        )

                    total_queries = row[0]
                    cache_hits = row[1]
                    cache_misses = row[2]
                    hit_rate = (
                        (cache_hits / total_queries * 100) if total_queries > 0 else 0.0
                    )

                    stats = QueryCacheStats(
                        total_queries=total_queries,
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        hit_rate=hit_rate,
                        time_period_start=start_time,
                        time_period_end=end_time,
                    )

                    logger.info(
                        f"Query cache performance calculated: {hit_rate:.2f}% "
                        f"({cache_hits}/{total_queries})"
                    )
                    return stats

        except Exception as e:
            logger.error(f"Failed to calculate query cache performance: {str(e)}")
            raise

    def get_queries_by_email(
        self,
        email: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[QueryLog]:
        """
        Retrieve all queries for a specific user email.

        Args:
            email: User email address
            start_time: Optional start of time range
            end_time: Optional end of time range
            limit: Maximum number of records to return (optional)

        Returns:
            List of QueryLog objects (without embeddings)

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> queries = client.get_queries_by_email("user@example.com", limit=100)
        """
        logger.debug(
            f"Fetching queries for email {email} "
            f"(time_range={start_time} to {end_time}, limit={limit})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Build dynamic query based on filters
                    query = """
                        SELECT id, raw_query, cache_result, latency_ms, email, command, created_at
                        FROM query_logs
                        WHERE email = %s
                    """
                    params: List = [email]

                    if start_time:
                        query += " AND created_at >= %s"
                        params.append(start_time)

                    if end_time:
                        query += " AND created_at <= %s"
                        params.append(end_time)

                    query += " ORDER BY created_at DESC"

                    if limit:
                        query += " LIMIT %s"
                        params.append(limit)

                    cur.execute(query, tuple(params))
                    rows = cur.fetchall()

                    logs = []
                    for row in rows:
                        logs.append(
                            QueryLog(
                                id=row[0],
                                raw_query=row[1],
                                query_embedding=None,  # Excluded for performance
                                cache_result=row[2],
                                latency_ms=row[3],
                                email=row[4],
                                command=row[5],
                                created_at=row[6],
                            )
                        )

                    logger.info(f"Retrieved {len(logs)} queries for email {email}")
                    return logs

        except Exception as e:
            logger.error(f"Failed to fetch queries for email {email}: {str(e)}")
            raise

    def search_queries_by_text(
        self,
        search_term: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[QueryLog]:
        """
        Search queries by text content (case-insensitive).

        Uses PostgreSQL ILIKE for pattern matching.

        Args:
            search_term: Text to search for in raw_query
            start_time: Optional start of time range
            end_time: Optional end of time range
            limit: Maximum number of records to return

        Returns:
            List of QueryLog objects matching the search term

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> queries = client.search_queries_by_text("password reset")
        """
        logger.debug(
            f"Searching queries for term: '{search_term}' "
            f"(time_range={start_time} to {end_time}, limit={limit})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Build dynamic query based on filters
                    query = """
                        SELECT id, raw_query, cache_result, latency_ms, user_id, command, created_at
                        FROM query_logs
                        WHERE raw_query ILIKE %s
                    """
                    # Add wildcards for ILIKE pattern matching
                    params: List = [f"%{search_term}%"]

                    if start_time:
                        query += " AND created_at >= %s"
                        params.append(start_time)

                    if end_time:
                        query += " AND created_at <= %s"
                        params.append(end_time)

                    query += " ORDER BY created_at DESC LIMIT %s"
                    params.append(limit)

                    cur.execute(query, tuple(params))
                    rows = cur.fetchall()

                    logs = []
                    for row in rows:
                        logs.append(
                            QueryLog(
                                id=row[0],
                                raw_query=row[1],
                                query_embedding=None,  # Excluded for performance
                                cache_result=row[2],
                                latency_ms=row[3],
                                user_id=row[4],
                                command=row[5],
                                created_at=row[6],
                            )
                        )

                    logger.info(
                        f"Found {len(logs)} queries matching search term '{search_term}'"
                    )
                    return logs

        except Exception as e:
            logger.error(f"Failed to search queries by text: {str(e)}")
            raise

    def get_total_query_count(self) -> int:
        """
        Get the total number of query logs in the database.

        Returns:
            Count of query log records

        Raises:
            ConnectionError: If database operation fails
        """
        logger.debug("Getting total query count")
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM query_logs")
                    result = cur.fetchone()
                    count = result[0] if result else 0
                    logger.debug(f"Total query count: {count}")
                    return count
        except Exception as e:
            logger.error(f"Failed to get query count: {str(e)}")
            raise

    def get_query_count_by_cache_result(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """
        Get count of queries grouped by cache_result type.

        Args:
            start_time: Beginning of time range (optional)
            end_time: End of time range (optional)

        Returns:
            Dictionary mapping cache_result -> count

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> counts = client.get_query_count_by_cache_result(start, end)
            >>> print(f"Hits: {counts.get('hit', 0)}, Misses: {counts.get('miss', 0)}")
        """
        logger.debug(
            f"Getting query count by cache_result (time range: {start_time} to {end_time})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if start_time and end_time:
                        cur.execute(
                            """
                            SELECT cache_result, COUNT(*) as count
                            FROM query_logs
                            WHERE created_at >= %s AND created_at <= %s
                            GROUP BY cache_result
                            ORDER BY count DESC
                            """,
                            (start_time, end_time),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT cache_result, COUNT(*) as count
                            FROM query_logs
                            GROUP BY cache_result
                            ORDER BY count DESC
                            """
                        )

                    rows = cur.fetchall()
                    counts = {row[0]: row[1] for row in rows}

                    logger.info(
                        f"Retrieved counts for {len(counts)} cache result types"
                    )
                    return counts

        except Exception as e:
            logger.error(f"Failed to get query count by cache_result: {str(e)}")
            raise

    def get_queries_per_hour(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Get query count aggregated by hour.

        Useful for visualizing query patterns over time.

        Args:
            start_time: Beginning of time range (inclusive)
            end_time: End of time range (inclusive)

        Returns:
            List of dictionaries with 'hour' and 'query_count' keys

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> hourly = client.get_queries_per_hour(start, end)
            >>> for entry in hourly:
            ...     print(f"{entry['hour']}: {entry['query_count']} queries")
        """
        logger.debug(f"Getting queries per hour for {start_time} to {end_time}")

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            date_trunc('hour', created_at) as hour,
                            COUNT(*) as query_count
                        FROM query_logs
                        WHERE created_at >= %s AND created_at <= %s
                        GROUP BY hour
                        ORDER BY hour
                        """,
                        (start_time, end_time),
                    )

                    rows = cur.fetchall()
                    hourly_data = [
                        {"hour": row[0], "query_count": row[1]} for row in rows
                    ]

                    logger.info(f"Retrieved query counts for {len(hourly_data)} hours")
                    return hourly_data

        except Exception as e:
            logger.error(f"Failed to get queries per hour: {str(e)}")
            raise

    def get_results_by_query_log_id(
        self, query_log_id: int, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all results for a specific query log.

        Args:
            query_log_id: The query log ID
            limit: Optional limit on number of results

        Returns:
            List of result dictionaries ordered by rank

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> results = client.get_results_by_query_log_id(12345)
            >>> for r in results:
            ...     print(f"Rank {r['rank']}: {r['score']:.2f}")
        """
        logger.debug(
            f"Fetching results for query_log_id {query_log_id} (limit={limit})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT id, query_log_id, search_method, rank, score,
                               chunk_id, parent_article_id, created_at
                        FROM query_results
                        WHERE query_log_id = %s
                        ORDER BY rank
                    """
                    params = [query_log_id]

                    if limit:
                        query += " LIMIT %s"
                        params.append(limit)

                    cur.execute(query, tuple(params))
                    rows = cur.fetchall()

                    results = []
                    for row in rows:
                        results.append(
                            {
                                "id": row[0],
                                "query_log_id": row[1],
                                "search_method": row[2],
                                "rank": row[3],
                                "score": row[4],
                                "chunk_id": row[5],
                                "parent_article_id": row[6],
                                "created_at": row[7],
                            }
                        )

                    logger.info(
                        f"Retrieved {len(results)} results for query_log_id {query_log_id}"
                    )
                    return results

        except Exception as e:
            logger.error(
                f"Failed to fetch results for query_log_id {query_log_id}: {str(e)}"
            )
            raise

    def get_article_top_k_frequency(
        self,
        article_id,
        top_k: int = 5,
        search_method: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Calculate how often an article appears in top-k results.

        Args:
            article_id: The article UUID to analyze
            top_k: Top-k cutoff (e.g., 5 for "top 5")
            search_method: Filter by method ('bm25', 'vector', 'hybrid')
            start_time: Optional start of time range
            end_time: Optional end of time range

        Returns:
            Dictionary with:
                - article_id: Article UUID
                - search_method: Search method (or None for all)
                - top_k: Top-k cutoff used
                - total_queries: Total queries in time range
                - top_k_appearances: Count of appearances in top-k
                - top_k_rate: Percentage (0-100)
                - avg_rank: Average rank when appearing
                - avg_score: Average score when appearing
                - time_period_start: Start time (or None)
                - time_period_end: End time (or None)

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> stats = client.get_article_top_k_frequency(
            ...     article_id=UUID('...'),
            ...     top_k=5,
            ...     search_method='bm25'
            ... )
            >>> print(f"Article appears in top 5: {stats['top_k_rate']:.1f}%")
        """
        logger.debug(
            f"Calculating top-{top_k} frequency for article {article_id} "
            f"(method={search_method})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Build dynamic query based on filters
                    query = """
                        SELECT
                            COUNT(DISTINCT query_log_id) AS total_queries,
                            COUNT(*) FILTER (WHERE rank <= %s) AS top_k_appearances,
                            COUNT(*) FILTER (WHERE rank <= %s)::FLOAT /
                                NULLIF(COUNT(DISTINCT query_log_id), 0) * 100 AS top_k_rate,
                            AVG(rank) FILTER (WHERE rank <= %s) AS avg_rank,
                            AVG(score) FILTER (WHERE rank <= %s) AS avg_score
                        FROM query_results
                        WHERE parent_article_id = %s
                    """
                    params: List = [top_k, top_k, top_k, top_k, article_id]

                    if search_method:
                        query += " AND search_method = %s"
                        params.append(search_method)

                    if start_time:
                        query += " AND created_at >= %s"
                        params.append(start_time)

                    if end_time:
                        query += " AND created_at <= %s"
                        params.append(end_time)

                    cur.execute(query, tuple(params))
                    row = cur.fetchone()

                    if not row or row[0] == 0:
                        logger.warning(
                            f"No data found for article {article_id} in specified time range"
                        )
                        return {
                            "article_id": article_id,
                            "search_method": search_method,
                            "top_k": top_k,
                            "total_queries": 0,
                            "top_k_appearances": 0,
                            "top_k_rate": 0.0,
                            "avg_rank": 0.0,
                            "avg_score": 0.0,
                            "time_period_start": start_time,
                            "time_period_end": end_time,
                        }

                    stats = {
                        "article_id": article_id,
                        "search_method": search_method,
                        "top_k": top_k,
                        "total_queries": row[0],
                        "top_k_appearances": row[1] if row[1] else 0,
                        "top_k_rate": float(row[2]) if row[2] else 0.0,
                        "avg_rank": float(row[3]) if row[3] else 0.0,
                        "avg_score": float(row[4]) if row[4] else 0.0,
                        "time_period_start": start_time,
                        "time_period_end": end_time,
                    }

                    logger.info(
                        f"Article {article_id} appears in top-{top_k}: "
                        f"{stats['top_k_rate']:.1f}% "
                        f"({stats['top_k_appearances']}/{stats['total_queries']} queries)"
                    )
                    return stats

        except Exception as e:
            logger.error(f"Failed to calculate top-k frequency: {str(e)}")
            raise

    def get_rank_distribution(
        self,
        search_method: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get distribution of ranks for a search method.

        Useful for understanding how well the retrieval method performs.

        Args:
            search_method: Search method ('bm25', 'vector', 'hybrid')
            start_time: Optional start of time range
            end_time: Optional end of time range

        Returns:
            List of dictionaries with 'rank', 'count', 'avg_score' keys

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> dist = client.get_rank_distribution('bm25')
            >>> for entry in dist:
            ...     print(f"Rank {entry['rank']}: {entry['count']} results, "
            ...           f"avg score {entry['avg_score']:.2f}")
        """
        logger.debug(
            f"Getting rank distribution for {search_method} "
            f"(time_range={start_time} to {end_time})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT
                            rank,
                            COUNT(*) AS count,
                            AVG(score) AS avg_score
                        FROM query_results
                        WHERE search_method = %s
                    """
                    params: List = [search_method]

                    if start_time:
                        query += " AND created_at >= %s"
                        params.append(start_time)

                    if end_time:
                        query += " AND created_at <= %s"
                        params.append(end_time)

                    query += " GROUP BY rank ORDER BY rank"

                    cur.execute(query, tuple(params))
                    rows = cur.fetchall()

                    distribution = []
                    for row in rows:
                        distribution.append(
                            {
                                "rank": row[0],
                                "count": row[1],
                                "avg_score": float(row[2]) if row[2] else 0.0,
                            }
                        )

                    logger.info(
                        f"Retrieved rank distribution for {search_method}: "
                        f"{len(distribution)} ranks"
                    )
                    return distribution

        except Exception as e:
            logger.error(f"Failed to get rank distribution: {str(e)}")
            raise

    def get_score_statistics(
        self,
        search_method: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Calculate score statistics for a search method.

        Uses PostgreSQL percentile functions to compute p50, p95, p99.

        Args:
            search_method: Search method ('bm25', 'vector', 'hybrid')
            start_time: Optional start of time range
            end_time: Optional end of time range

        Returns:
            Dictionary with avg, min, max, p50, p95, p99 scores

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> stats = client.get_score_statistics('vector')
            >>> print(f"Avg similarity: {stats['avg_score']:.3f}")
            >>> print(f"P95 similarity: {stats['p95_score']:.3f}")
        """
        logger.debug(
            f"Calculating score statistics for {search_method} "
            f"(time_range={start_time} to {end_time})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT
                            AVG(score) AS avg_score,
                            MIN(score) AS min_score,
                            MAX(score) AS max_score,
                            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) AS p50_score,
                            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY score) AS p95_score,
                            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY score) AS p99_score,
                            COUNT(*) AS total_results
                        FROM query_results
                        WHERE search_method = %s
                    """
                    params: List = [search_method]

                    if start_time:
                        query += " AND created_at >= %s"
                        params.append(start_time)

                    if end_time:
                        query += " AND created_at <= %s"
                        params.append(end_time)

                    cur.execute(query, tuple(params))
                    row = cur.fetchone()

                    if not row or row[6] == 0:
                        logger.warning(
                            f"No score data found for {search_method} in specified time range"
                        )
                        return {
                            "search_method": search_method,
                            "total_results": 0,
                            "avg_score": 0.0,
                            "min_score": 0.0,
                            "max_score": 0.0,
                            "p50_score": 0.0,
                            "p95_score": 0.0,
                            "p99_score": 0.0,
                            "time_period_start": start_time,
                            "time_period_end": end_time,
                        }

                    stats = {
                        "search_method": search_method,
                        "total_results": row[6],
                        "avg_score": float(row[0]) if row[0] else 0.0,
                        "min_score": float(row[1]) if row[1] else 0.0,
                        "max_score": float(row[2]) if row[2] else 0.0,
                        "p50_score": float(row[3]) if row[3] else 0.0,
                        "p95_score": float(row[4]) if row[4] else 0.0,
                        "p99_score": float(row[5]) if row[5] else 0.0,
                        "time_period_start": start_time,
                        "time_period_end": end_time,
                    }

                    logger.info(
                        f"Score statistics for {search_method}: "
                        f"avg={stats['avg_score']:.2f}, "
                        f"p50={stats['p50_score']:.2f}, "
                        f"p95={stats['p95_score']:.2f}, "
                        f"p99={stats['p99_score']:.2f}"
                    )
                    return stats

        except Exception as e:
            logger.error(f"Failed to calculate score statistics: {str(e)}")
            raise

    def get_article_coverage(
        self,
        search_method: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get articles that appear most frequently in results.

        Args:
            search_method: Search method ('bm25', 'vector', 'hybrid')
            start_time: Optional start of time range
            end_time: Optional end of time range
            limit: Maximum number of articles to return

        Returns:
            List of dicts with article_id, num_queries, avg_rank, avg_score, rank_1_count

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> coverage = client.get_article_coverage('bm25', limit=10)
            >>> for article in coverage:
            ...     print(f"Article {article['parent_article_id']}: "
            ...           f"{article['num_queries']} queries, "
            ...           f"avg rank {article['avg_rank']:.1f}")
        """
        logger.debug(
            f"Getting article coverage for {search_method} "
            f"(time_range={start_time} to {end_time}, limit={limit})"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT
                            parent_article_id,
                            COUNT(DISTINCT query_log_id) AS num_queries,
                            AVG(rank) AS avg_rank,
                            AVG(score) AS avg_score,
                            COUNT(*) FILTER (WHERE rank = 1) AS rank_1_count
                        FROM query_results
                        WHERE search_method = %s
                    """
                    params: List = [search_method]

                    if start_time:
                        query += " AND created_at >= %s"
                        params.append(start_time)

                    if end_time:
                        query += " AND created_at <= %s"
                        params.append(end_time)

                    query += """
                        GROUP BY parent_article_id
                        ORDER BY num_queries DESC
                        LIMIT %s
                    """
                    params.append(limit)

                    cur.execute(query, tuple(params))
                    rows = cur.fetchall()

                    coverage = []
                    for row in rows:
                        coverage.append(
                            {
                                "parent_article_id": row[0],
                                "num_queries": row[1],
                                "avg_rank": float(row[2]) if row[2] else 0.0,
                                "avg_score": float(row[3]) if row[3] else 0.0,
                                "rank_1_count": row[4] if row[4] else 0,
                            }
                        )

                    logger.info(
                        f"Retrieved article coverage for {search_method}: "
                        f"{len(coverage)} articles"
                    )
                    return coverage

        except Exception as e:
            logger.error(f"Failed to get article coverage: {str(e)}")
            raise

    def log_llm_response(
        self,
        query_log_id: int,
        response_text: str,
        model_name: Optional[str] = None,
        llm_latency_ms: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        citations: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Log an LLM-generated response for a query.

        Args:
            query_log_id: The query log ID (must exist in query_logs table)
            response_text: Full LLM response text
            model_name: LLM model identifier (e.g., 'gpt-4')
            llm_latency_ms: LLM generation latency in milliseconds
            prompt_tokens: Number of tokens in prompt
            completion_tokens: Number of tokens in completion
            total_tokens: Total tokens (prompt + completion)
            citations: JSONB data with source URLs and chunk IDs
            metadata: JSONB data with model parameters

        Returns:
            The auto-generated llm_response ID (BIGSERIAL)

        Raises:
            ValueError: If response_text is empty or tokens are negative
            psycopg.IntegrityError: If query_log_id doesn't exist or already has response
            ConnectionError: If database operation fails

        Example:
            >>> client = QueryLogClient()
            >>> response_id = client.log_llm_response(
            ...     query_log_id=12345,
            ...     response_text="To reset your password...",
            ...     model_name="gpt-4",
            ...     llm_latency_ms=1250,
            ...     citations={"num_documents_used": 3, "source_urls": [...]}
            ... )
        """
        if not response_text or not response_text.strip():
            raise ValueError("response_text cannot be empty")

        # Validate token counts are non-negative
        if prompt_tokens is not None and prompt_tokens < 0:
            raise ValueError(f"prompt_tokens cannot be negative: {prompt_tokens}")
        if completion_tokens is not None and completion_tokens < 0:
            raise ValueError(
                f"completion_tokens cannot be negative: {completion_tokens}"
            )
        if total_tokens is not None and total_tokens < 0:
            raise ValueError(f"total_tokens cannot be negative: {total_tokens}")
        if llm_latency_ms is not None and llm_latency_ms < 0:
            raise ValueError(f"llm_latency_ms cannot be negative: {llm_latency_ms}")

        logger.debug(
            f"Logging LLM response for query_log_id {query_log_id}: "
            f"model={model_name}, latency={llm_latency_ms}ms, tokens={total_tokens}"
        )

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO llm_responses
                        (query_log_id, response_text, model_name, llm_latency_ms,
                         prompt_tokens, completion_tokens, total_tokens,
                         citations, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            query_log_id,
                            response_text,
                            model_name,
                            llm_latency_ms,
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                            Json(citations) if citations else None,
                            Json(metadata) if metadata else None,
                        ),
                    )

                    result = cur.fetchone()
                    if not result:
                        raise ConnectionError(
                            "Failed to retrieve response ID after insert"
                        )
                    response_id = result[0]

                    logger.debug(
                        f"LLM response logged successfully with ID {response_id}"
                    )
                    return response_id

        except psycopg.IntegrityError as e:
            # This catches:
            # - FK violation (query_log_id doesn't exist)
            # - UNIQUE violation (response already logged for this query_log_id)
            logger.error(f"Integrity error logging LLM response: {str(e)}")
            if "foreign key" in str(e).lower():
                raise ValueError(f"Query log ID {query_log_id} not found") from e
            elif "unique" in str(e).lower():
                raise ValueError(
                    f"LLM response already logged for query_log_id {query_log_id}"
                ) from e
            raise
        except Exception as e:
            logger.error(f"Failed to log LLM response: {str(e)}")
            raise

    def get_llm_response_by_query_log_id(
        self, query_log_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve LLM response for a specific query log.

        Args:
            query_log_id: The query log ID

        Returns:
            Dictionary with LLM response data or None if not found

        Raises:
            ConnectionError: If database operation fails

        Example:
            >>> response = client.get_llm_response_by_query_log_id(12345)
            >>> if response:
            ...     print(f"Model: {response['model_name']}")
            ...     print(f"Response: {response['response_text'][:100]}")
        """
        logger.debug(f"Fetching LLM response for query_log_id {query_log_id}")

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, query_log_id, response_text, model_name,
                               llm_latency_ms, prompt_tokens, completion_tokens,
                               total_tokens, citations, metadata, created_at
                        FROM llm_responses
                        WHERE query_log_id = %s
                        """,
                        (query_log_id,),
                    )
                    row = cur.fetchone()

                    if not row:
                        logger.debug(
                            f"No LLM response found for query_log_id {query_log_id}"
                        )
                        return None

                    return {
                        "id": row[0],
                        "query_log_id": row[1],
                        "response_text": row[2],
                        "model_name": row[3],
                        "llm_latency_ms": row[4],
                        "prompt_tokens": row[5],
                        "completion_tokens": row[6],
                        "total_tokens": row[7],
                        "citations": row[8],  # Already deserialized from JSONB
                        "metadata": row[9],  # Already deserialized from JSONB
                        "created_at": row[10],
                    }

        except Exception as e:
            logger.error(
                f"Failed to fetch LLM response for query_log_id {query_log_id}: {str(e)}"
            )
            raise
