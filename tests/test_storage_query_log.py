"""
Tests for the storage_query_log module (QueryLogClient).
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock, patch

from core.schemas import PopularQuery, QueryCacheStats, QueryLatencyStats, QueryLog
from core.storage_query_log import QueryLogClient


class TestQueryLogClient:
    """Test suite for QueryLogClient."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for database connection."""
        with patch("core.storage_base.get_database_settings") as mock:
            settings = Mock()
            settings.HOST = "localhost"
            settings.USER = "test_user"
            settings.PASSWORD.get_secret_value.return_value = "test_password"
            settings.NAME = "test_db"
            mock.return_value = settings
            yield settings

    @pytest.fixture
    def client(self, mock_settings):
        """Create QueryLogClient instance with mocked settings."""
        return QueryLogClient()

    def test_init(self, client):
        """Test client initialization."""
        assert client.db_host == "localhost"
        assert client.db_user == "test_user"
        assert client.db_password == "test_password"
        assert client.db_name == "test_db"
        assert client.db_port == 5432

    def test_log_query_validates_raw_query(self, client):
        """Test that log_query validates raw_query."""
        with pytest.raises(ValueError, match="raw_query cannot be empty"):
            client.log_query("", "hit")

        with pytest.raises(ValueError, match="raw_query cannot be empty"):
            client.log_query("   ", "hit")

    def test_log_query_validates_cache_result(self, client):
        """Test that log_query validates cache_result."""
        with pytest.raises(ValueError, match="cache_result cannot be empty"):
            client.log_query("test query", "")

        with pytest.raises(ValueError, match="cache_result cannot be empty"):
            client.log_query("test query", "   ")

    def test_log_query_validates_embedding_dimensions(self, client):
        """Test that log_query validates embedding dimensions."""
        invalid_embedding = [0.1] * 100  # Wrong size

        with pytest.raises(
            ValueError, match="query_embedding must have 3072 dimensions"
        ):
            client.log_query(
                raw_query="test query",
                cache_result="hit",
                query_embedding=invalid_embedding,
            )

    def test_log_query_success(self, client):
        """Test successful query logging."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (12345,)

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            log_id = client.log_query(
                raw_query="How do I reset my password?",
                cache_result="hit",
                latency_ms=125,
                user_id="user456",
            )

            assert log_id == 12345
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args[0]
            assert "INSERT INTO query_logs" in call_args[0]

    def test_log_query_with_embedding(self, client):
        """Test query logging with embedding."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (12345,)
        valid_embedding = [0.1] * 3072  # Correct size

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            log_id = client.log_query(
                raw_query="What is MFA?",
                cache_result="miss",
                query_embedding=valid_embedding,
                latency_ms=200,
            )

            assert log_id == 12345

    def test_log_query_with_timestamp(self, client):
        """Test query logging with custom timestamp."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (12345,)

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            custom_timestamp = datetime.now(timezone.utc) - timedelta(hours=2)
            log_id = client.log_query(
                raw_query="Test query",
                cache_result="hit",
                latency_ms=100,
                created_at=custom_timestamp,
            )

            assert log_id == 12345
            call_args = mock_cursor.execute.call_args[0]
            assert "created_at" in call_args[0]

    def test_log_query_no_result_raises_error(self, client):
        """Test that missing result raises ConnectionError."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            with pytest.raises(
                ConnectionError, match="Failed to retrieve log ID after insert"
            ):
                client.log_query(raw_query="test", cache_result="hit")

    def test_insert_query_logs_empty_list(self, client):
        """Test inserting empty list."""
        client.insert_query_logs([])
        # Should return early without error

    def test_insert_query_logs_success(self, client):
        """Test successful bulk query logs insertion."""
        logs = [
            QueryLog(
                raw_query="How to reset password?",
                cache_result="hit",
                latency_ms=50,
                user_id="user1",
            ),
            QueryLog(
                raw_query="What is MFA?",
                cache_result="miss",
                latency_ms=200,
                user_id="user2",
            ),
        ]

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(1,), (2,)]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            client.insert_query_logs(logs)

            assert mock_cursor.execute.call_count == 2
            assert logs[0].id == 1
            assert logs[1].id == 2

    def test_insert_query_logs_validates_embedding_dimensions(self, client):
        """Test bulk insert validates embedding dimensions."""
        invalid_embedding = [0.1] * 100
        logs = [
            QueryLog(
                raw_query="Test",
                cache_result="hit",
                query_embedding=invalid_embedding,
            )
        ]

        mock_cursor = MagicMock()

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            with pytest.raises(ValueError, match="embedding must have 3072 dimensions"):
                client.insert_query_logs(logs)

    def test_get_queries_by_time_range(self, client):
        """Test retrieving queries by time range."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, "How to reset password?", "hit", 50, "user1", start_time),
            (2, "What is MFA?", "miss", 200, "user2", end_time),
        ]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            queries = client.get_queries_by_time_range(start_time, end_time)

            assert len(queries) == 2
            assert queries[0].id == 1
            assert queries[0].raw_query == "How to reset password?"
            assert queries[0].cache_result == "hit"
            assert queries[1].cache_result == "miss"

    def test_get_queries_by_time_range_with_filters(self, client):
        """Test time range query with cache_result and user filters."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, "Test query", "hit", 50, "user1", start_time)
        ]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            queries = client.get_queries_by_time_range(
                start_time, end_time, cache_result="hit", user_id="user1", limit=100
            )

            assert len(queries) == 1
            # Verify filters were applied in query
            call_args = mock_cursor.execute.call_args[0]
            assert "cache_result = %s" in call_args[0]
            assert "user_id = %s" in call_args[0]
            assert "LIMIT %s" in call_args[0]

    def test_get_query_by_id_without_embedding(self, client):
        """Test retrieving query by ID without embedding."""
        timestamp = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            1,
            "Test query",
            "hit",
            100,
            "user1",
            timestamp,
        )

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            query = client.get_query_by_id(1, include_embedding=False)

            assert query is not None
            assert query.id == 1
            assert query.raw_query == "Test query"
            assert query.query_embedding is None

    def test_get_query_by_id_with_embedding(self, client):
        """Test retrieving query by ID with embedding."""
        timestamp = datetime.now(timezone.utc)
        embedding = [0.1] * 3072

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (
            1,
            "Test query",
            embedding,
            "hit",
            100,
            "user1",
            timestamp,
        )

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            query = client.get_query_by_id(1, include_embedding=True)

            assert query is not None
            assert query.query_embedding == embedding

    def test_get_query_by_id_not_found(self, client):
        """Test get_query_by_id returns None for non-existent ID."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            query = client.get_query_by_id(999)

            assert query is None

    def test_get_query_latency_stats_success(self, client):
        """Test calculating query latency statistics."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        # avg, min, max, p50, p95, p99, count
        mock_cursor.fetchone.return_value = (
            150.5,
            30,
            600,
            120.0,
            500.0,
            550.0,
            100,
        )

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            stats = client.get_query_latency_stats(start_time, end_time)

            assert isinstance(stats, QueryLatencyStats)
            assert stats.avg_latency_ms == 150.5
            assert stats.min_latency_ms == 30
            assert stats.max_latency_ms == 600
            assert stats.p50_latency_ms == 120.0
            assert stats.p95_latency_ms == 500.0
            assert stats.p99_latency_ms == 550.0
            assert stats.total_queries == 100

    def test_get_query_latency_stats_no_data(self, client):
        """Test latency stats with no data."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (None, None, None, None, None, None, 0)

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            stats = client.get_query_latency_stats(start_time, end_time)

            assert stats.total_queries == 0
            assert stats.avg_latency_ms == 0.0
            assert stats.min_latency_ms == 0
            assert stats.max_latency_ms == 0

    def test_get_query_latency_stats_with_filters(self, client):
        """Test latency stats with cache_result and user filters."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (60.0, 20, 120, 55.0, 110.0, 115.0, 50)

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            stats = client.get_query_latency_stats(
                start_time, end_time, cache_result="hit", user_id="user123"
            )

            assert stats.avg_latency_ms == 60.0
            # Verify filters were applied
            call_args = mock_cursor.execute.call_args[0]
            assert "cache_result = %s" in call_args[0]
            assert "user_id = %s" in call_args[0]

    def test_get_popular_queries(self, client):
        """Test getting popular queries."""
        start_time = datetime.now(timezone.utc) - timedelta(days=7)
        end_time = datetime.now(timezone.utc)
        last_query_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("How to reset password?", 50, 120.5, 75.0, last_query_time),
            ("What is MFA?", 30, 200.0, 50.0, last_query_time),
        ]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            popular = client.get_popular_queries(start_time, end_time, limit=10)

            assert len(popular) == 2
            assert isinstance(popular[0], PopularQuery)
            assert popular[0].raw_query == "How to reset password?"
            assert popular[0].query_count == 50
            assert popular[0].avg_latency_ms == 120.5
            assert popular[0].cache_hit_rate == 75.0

    def test_get_popular_queries_with_min_occurrences(self, client):
        """Test popular queries with minimum occurrence filter."""
        start_time = datetime.now(timezone.utc) - timedelta(days=7)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            popular = client.get_popular_queries(
                start_time, end_time, limit=20, min_occurrences=5
            )

            assert len(popular) == 0
            # Verify min_occurrences was used in HAVING clause
            call_args = mock_cursor.execute.call_args[0]
            assert "HAVING COUNT(*) >=" in call_args[0]

    def test_get_query_cache_performance_success(self, client):
        """Test calculating query cache performance."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (100, 80, 20)  # total, hits, misses

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            stats = client.get_query_cache_performance(start_time, end_time)

            assert isinstance(stats, QueryCacheStats)
            assert stats.total_queries == 100
            assert stats.cache_hits == 80
            assert stats.cache_misses == 20
            assert stats.hit_rate == 80.0
            assert stats.time_period_start == start_time
            assert stats.time_period_end == end_time

    def test_get_query_cache_performance_no_data(self, client):
        """Test cache performance with no data."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0, 0, 0)

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            stats = client.get_query_cache_performance(start_time, end_time)

            assert stats.total_queries == 0
            assert stats.hit_rate == 0.0

    def test_get_query_cache_performance_with_user_filter(self, client):
        """Test cache performance with user filter."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (50, 45, 5)

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            stats = client.get_query_cache_performance(
                start_time, end_time, user_id="user123"
            )

            assert stats.hit_rate == 90.0
            # Verify user filter was applied
            call_args = mock_cursor.execute.call_args[0]
            assert "user_id = %s" in call_args[0]

    def test_get_queries_by_user(self, client):
        """Test retrieving queries for specific user."""
        timestamp = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, "Query 1", "hit", 50, "user123", timestamp),
            (2, "Query 2", "miss", 150, "user123", timestamp),
        ]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            queries = client.get_queries_by_user("user123")

            assert len(queries) == 2
            assert all(q.user_id == "user123" for q in queries)

    def test_get_queries_by_user_with_time_range(self, client):
        """Test user queries with time range filter."""
        start_time = datetime.now(timezone.utc) - timedelta(days=7)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            queries = client.get_queries_by_user(
                "user123", start_time=start_time, end_time=end_time, limit=50
            )

            assert len(queries) == 0
            # Verify time range filters were applied
            call_args = mock_cursor.execute.call_args[0]
            assert "created_at >=" in call_args[0]
            assert "created_at <=" in call_args[0]
            assert "LIMIT %s" in call_args[0]

    def test_search_queries_by_text(self, client):
        """Test searching queries by text."""
        timestamp = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, "How to reset password?", "hit", 100, "user1", timestamp),
            (2, "Reset my password please", "miss", 150, "user2", timestamp),
        ]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            queries = client.search_queries_by_text("password")

            assert len(queries) == 2
            # Verify ILIKE was used
            call_args = mock_cursor.execute.call_args[0]
            assert "ILIKE %s" in call_args[0]

    def test_search_queries_by_text_with_time_range(self, client):
        """Test text search with time range filters."""
        start_time = datetime.now(timezone.utc) - timedelta(days=7)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            queries = client.search_queries_by_text(
                "password", start_time=start_time, end_time=end_time, limit=50
            )

            assert len(queries) == 0
            # Verify time filters and limit were applied
            call_args = mock_cursor.execute.call_args[0]
            assert "created_at >=" in call_args[0]
            assert "created_at <=" in call_args[0]
            assert "LIMIT %s" in call_args[0]

    def test_get_total_query_count(self, client):
        """Test getting total query count."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (5432,)

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            count = client.get_total_query_count()

            assert count == 5432

    def test_get_total_query_count_empty(self, client):
        """Test getting count when table is empty."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            count = client.get_total_query_count()

            assert count == 0

    def test_get_query_count_by_cache_result(self, client):
        """Test getting count grouped by cache_result."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("hit", 200),
            ("miss", 75),
            ("warm_hit", 25),
        ]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            counts = client.get_query_count_by_cache_result()

            assert counts["hit"] == 200
            assert counts["miss"] == 75
            assert counts["warm_hit"] == 25
            assert len(counts) == 3

    def test_get_query_count_by_cache_result_with_time_range(self, client):
        """Test count by cache_result with time range filter."""
        start_time = datetime.now(timezone.utc) - timedelta(days=7)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("hit", 150), ("miss", 50)]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            counts = client.get_query_count_by_cache_result(start_time, end_time)

            assert len(counts) == 2
            # Verify time range was used in query
            call_args = mock_cursor.execute.call_args[0]
            assert "created_at >=" in call_args[0]
            assert "created_at <=" in call_args[0]

    def test_get_queries_per_hour(self, client):
        """Test getting queries aggregated by hour."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=3)
        end_time = datetime.now(timezone.utc)
        hour1 = datetime.now(timezone.utc) - timedelta(hours=2)
        hour2 = datetime.now(timezone.utc) - timedelta(hours=1)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (hour1, 45),
            (hour2, 67),
        ]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            hourly_data = client.get_queries_per_hour(start_time, end_time)

            assert len(hourly_data) == 2
            assert hourly_data[0]["hour"] == hour1
            assert hourly_data[0]["query_count"] == 45
            assert hourly_data[1]["hour"] == hour2
            assert hourly_data[1]["query_count"] == 67

            # Verify date_trunc was used
            call_args = mock_cursor.execute.call_args[0]
            assert "date_trunc('hour'" in call_args[0]
