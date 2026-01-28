"""
Database connection pooling for FastAPI application.

Provides thread-safe connection pooling using psycopg_pool.ConnectionPool
to handle concurrent API requests efficiently without overwhelming the database
or incurring per-request connection overhead.
"""

from psycopg_pool import ConnectionPool
from core.config import get_database_settings
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseConnectionPool:
    """
    Connection pool wrapper for PostgreSQL using psycopg3.

    Provides thread-safe connection pooling to handle concurrent API requests
    without overwhelming the database or incurring connection overhead.

    Args:
        min_size: Minimum number of connections to maintain in the pool
        max_size: Maximum number of connections allowed in the pool
        timeout: Timeout in seconds for getting a connection from the pool

    Example:
        >>> pool = DatabaseConnectionPool(min_size=5, max_size=20)
        >>> with pool.get_connection() as conn:
        ...     cursor = conn.cursor()
        ...     cursor.execute("SELECT * FROM table")
        >>> pool.close()
    """

    def __init__(self, min_size: int = 5, max_size: int = 20, timeout: float = 30.0):
        """
        Initialize connection pool.

        Args:
            min_size: Minimum number of connections to maintain (keep warm)
            max_size: Maximum number of connections allowed
            timeout: Timeout in seconds for getting a connection (fail fast)
        """
        settings = get_database_settings()

        # Build connection string for psycopg3
        conninfo = (
            f"host={settings.HOST} "
            f"port=5432 "
            f"dbname={settings.NAME} "
            f"user={settings.USER} "
            f"password={settings.PASSWORD.get_secret_value()}"
        )

        # Initialize connection pool
        self.pool = ConnectionPool(
            conninfo=conninfo,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
            open=True,  # Open connections immediately at startup
        )

        logger.info(
            f"Connection pool initialized: min={min_size}, max={max_size}, timeout={timeout}s"
        )

    def get_connection(self):
        """
        Get a connection from the pool (context manager).

        Returns:
            Context manager that yields a pooled connection

        Raises:
            PoolTimeout: If no connection is available within timeout period

        Example:
            >>> with pool.get_connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT 1")
        """
        return self.pool.connection()

    def close(self):
        """Close the connection pool and all connections."""
        self.pool.close()
        logger.info("Connection pool closed")

    def get_stats(self) -> dict:
        """
        Get pool statistics for monitoring.

        Returns:
            Dictionary with pool stats (size, available, waiting)
        """
        try:
            stats = self.pool.get_stats()
            return {
                "pool_size": stats.get("pool_size", 0),
                "pool_available": stats.get("pool_available", 0),
                "requests_waiting": stats.get("requests_waiting", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get pool stats: {e}")
            return {"error": str(e)}


# Singleton instance
_pool: Optional[DatabaseConnectionPool] = None


def get_connection_pool(
    min_conn: int = 5, max_conn: int = 20, timeout: float = 30.0
) -> DatabaseConnectionPool:
    """
    Get or create the singleton connection pool.

    This ensures only one pool is created per application instance.

    Args:
        min_conn: Minimum number of connections (only used if creating new pool)
        max_conn: Maximum number of connections (only used if creating new pool)
        timeout: Connection timeout in seconds (only used if creating new pool)

    Returns:
        The singleton DatabaseConnectionPool instance
    """
    global _pool
    if _pool is None:
        _pool = DatabaseConnectionPool(
            min_size=min_conn, max_size=max_conn, timeout=timeout
        )
    return _pool


def close_connection_pool():
    """Close and cleanup the singleton connection pool."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
