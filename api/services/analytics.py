"""
Analytics service for querying usage statistics.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

import psycopg
from psycopg import Connection

from core.config import get_settings

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for retrieving analytics data from the database."""

    def __init__(self):
        settings = get_settings()
        self._connection_params = {
            "host": settings.DB_HOST,
            "port": settings.DB_PORT,
            "user": settings.DB_USER,
            "password": settings.DB_PASSWORD.get_secret_value(),
            "dbname": settings.DB_NAME,
        }

    @contextmanager
    def get_connection(self):
        """Get a database connection with automatic cleanup."""
        conn = None
        try:
            conn = psycopg.connect(**self._connection_params)
            yield conn
        finally:
            if conn and not conn.closed:
                conn.close()

    def get_conversation_stats(
        self, days: int = 30
    ) -> Dict[str, Any]:
        """
        Get overall conversation statistics.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with total_conversations, unique_users, avg_latency_ms
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            COUNT(*) as total_conversations,
                            COUNT(DISTINCT user_id) as unique_users,
                            AVG(latency_ms) as avg_latency_ms
                        FROM query_logs
                        WHERE created_at >= %s
                        """,
                        (cutoff,),
                    )
                    row = cur.fetchone()

                    return {
                        "total_conversations": row[0] or 0,
                        "unique_users": row[1] or 0,
                        "avg_latency_ms": round(row[2], 2) if row[2] else 0,
                        "period_days": days,
                    }
        except Exception as e:
            logger.error(f"Failed to get conversation stats: {e}")
            raise

    def get_conversations_by_user(
        self, days: int = 30, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get conversation counts grouped by user.

        Args:
            days: Number of days to look back
            limit: Maximum number of users to return

        Returns:
            List of dicts with user_id, conversation_count, last_chat
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            user_id,
                            COUNT(*) as conversation_count,
                            MAX(created_at) as last_chat
                        FROM query_logs
                        WHERE created_at >= %s
                        GROUP BY user_id
                        ORDER BY conversation_count DESC
                        LIMIT %s
                        """,
                        (cutoff, limit),
                    )
                    rows = cur.fetchall()

                    return [
                        {
                            "user_id": row[0] or "anonymous",
                            "conversation_count": row[1],
                            "last_chat": row[2].isoformat() if row[2] else None,
                        }
                        for row in rows
                    ]
        except Exception as e:
            logger.error(f"Failed to get conversations by user: {e}")
            raise

    def get_conversations_over_time(
        self, days: int = 30, group_by: str = "day"
    ) -> List[Dict[str, Any]]:
        """
        Get conversation counts over time.

        Args:
            days: Number of days to look back
            group_by: Grouping interval ('day' or 'hour')

        Returns:
            List of dicts with period and count
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        if group_by == "hour":
            date_trunc = "hour"
        else:
            date_trunc = "day"

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT
                            DATE_TRUNC(%s, created_at) as period,
                            COUNT(*) as count
                        FROM query_logs
                        WHERE created_at >= %s
                        GROUP BY period
                        ORDER BY period ASC
                        """,
                        (date_trunc, cutoff),
                    )
                    rows = cur.fetchall()

                    return [
                        {
                            "period": row[0].isoformat() if row[0] else None,
                            "count": row[1],
                        }
                        for row in rows
                    ]
        except Exception as e:
            logger.error(f"Failed to get conversations over time: {e}")
            raise

    def get_cache_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get cache hit/miss statistics.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with cache statistics
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            cache_result,
                            COUNT(*) as count
                        FROM query_logs
                        WHERE created_at >= %s
                        GROUP BY cache_result
                        """,
                        (cutoff,),
                    )
                    rows = cur.fetchall()

                    results = {row[0]: row[1] for row in rows}
                    total = sum(results.values())

                    return {
                        "total_queries": total,
                        "cache_hits": results.get("hit", 0),
                        "cache_misses": results.get("miss", 0),
                        "hit_rate": round(
                            results.get("hit", 0) / total * 100, 2
                        ) if total > 0 else 0,
                        "period_days": days,
                    }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            raise

    def get_recent_queries(
        self, limit: int = 20, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent queries, optionally filtered by user.

        Args:
            limit: Maximum number of queries to return
            user_id: Optional user ID to filter by

        Returns:
            List of recent query records
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    if user_id:
                        cur.execute(
                            """
                            SELECT id, raw_query, cache_result, latency_ms, user_id, created_at
                            FROM query_logs
                            WHERE user_id = %s
                            ORDER BY created_at DESC
                            LIMIT %s
                            """,
                            (user_id, limit),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, raw_query, cache_result, latency_ms, user_id, created_at
                            FROM query_logs
                            ORDER BY created_at DESC
                            LIMIT %s
                            """,
                            (limit,),
                        )

                    rows = cur.fetchall()

                    return [
                        {
                            "id": row[0],
                            "query": row[1][:200] + "..." if len(row[1]) > 200 else row[1],
                            "cache_result": row[2],
                            "latency_ms": row[3],
                            "user_id": row[4] or "anonymous",
                            "created_at": row[5].isoformat() if row[5] else None,
                        }
                        for row in rows
                    ]
        except Exception as e:
            logger.error(f"Failed to get recent queries: {e}")
            raise
