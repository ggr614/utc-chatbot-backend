"""
Tests for the storage_cache_metrics module (CacheMetricsClient).
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

from core.schemas import CacheHitRateStats, CacheLatencyStats, CacheMetric
from core.storage_cache_metrics import CacheMetricsClient


class TestCacheMetricsClient:
    """Test suite for CacheMetricsClient."""

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
        """Create CacheMetricsClient instance with mocked settings."""
        return CacheMetricsClient()

    def test_init(self, client):
        """Test client initialization."""
        assert client.db_host == "localhost"
        assert client.db_user == "test_user"
        assert client.db_password == "test_password"
        assert client.db_name == "test_db"
        assert client.db_port == 5432

    def test_log_cache_event_validates_cache_type(self, client):
        """Test that log_cache_event validates cache_type."""
        with pytest.raises(ValueError, match="cache_type cannot be empty"):
            client.log_cache_event("")

        with pytest.raises(ValueError, match="cache_type cannot be empty"):
            client.log_cache_event("   ")

    def test_log_cache_event_success(self, client):
        """Test successful cache event logging."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (12345,)

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            cache_entry_id = uuid4()
            metric_id = client.log_cache_event(
                cache_type="hit",
                cache_entry_id=cache_entry_id,
                latency_ms=50,
                user_id="user123",
            )

            assert metric_id == 12345
            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args[0]
            assert "INSERT INTO cache_metrics" in call_args[0]

    def test_log_cache_event_with_timestamp(self, client):
        """Test cache event logging with custom timestamp."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (12345,)

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            custom_timestamp = datetime.now(timezone.utc) - timedelta(hours=2)
            metric_id = client.log_cache_event(
                cache_type="miss",
                latency_ms=200,
                request_timestamp=custom_timestamp,
            )

            assert metric_id == 12345
            call_args = mock_cursor.execute.call_args[0]
            assert "request_timestamp" in call_args[0]

    def test_log_cache_event_no_result_raises_error(self, client):
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
                ConnectionError, match="Failed to retrieve metric ID after insert"
            ):
                client.log_cache_event(cache_type="hit")

    def test_insert_metrics_empty_list(self, client):
        """Test inserting empty list."""
        client.insert_metrics([])
        # Should return early without error

    def test_insert_metrics_success(self, client):
        """Test successful bulk metrics insertion."""
        metrics = [
            CacheMetric(
                cache_type="hit",
                cache_entry_id=uuid4(),
                latency_ms=50,
                user_id="user1",
            ),
            CacheMetric(cache_type="miss", latency_ms=200, user_id="user2"),
        ]

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(1,), (2,)]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            client.insert_metrics(metrics)

            assert mock_cursor.execute.call_count == 2
            assert metrics[0].id == 1
            assert metrics[1].id == 2

    def test_get_metrics_by_time_range(self, client):
        """Test retrieving metrics by time range."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)
        cache_entry_id_1 = uuid4()
        cache_entry_id_2 = uuid4()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, cache_entry_id_1, start_time, "hit", 50, "user1"),
            (2, cache_entry_id_2, end_time, "miss", 200, "user2"),
        ]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            metrics = client.get_metrics_by_time_range(start_time, end_time)

            assert len(metrics) == 2
            assert metrics[0].id == 1
            assert metrics[0].cache_type == "hit"
            assert metrics[1].cache_type == "miss"

    def test_get_metrics_by_time_range_with_filters(self, client):
        """Test time range query with type and user filters."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, uuid4(), start_time, "hit", 50, "user1")
        ]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            metrics = client.get_metrics_by_time_range(
                start_time, end_time, cache_type="hit", user_id="user1", limit=100
            )

            assert len(metrics) == 1
            # Verify filters were applied in query
            call_args = mock_cursor.execute.call_args[0]
            assert "cache_type = %s" in call_args[0]
            assert "user_id = %s" in call_args[0]
            assert "LIMIT %s" in call_args[0]

    def test_get_metrics_by_cache_entry(self, client):
        """Test retrieving metrics for specific cache entry."""
        cache_entry_id = uuid4()
        timestamp = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, cache_entry_id, timestamp, "hit", 45, "user1"),
            (2, cache_entry_id, timestamp, "hit", 52, "user2"),
        ]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            metrics = client.get_metrics_by_cache_entry(cache_entry_id)

            assert len(metrics) == 2
            assert all(m.cache_entry_id == cache_entry_id for m in metrics)

    def test_get_metrics_by_cache_entry_with_limit(self, client):
        """Test cache entry query with limit."""
        cache_entry_id = uuid4()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            client.get_metrics_by_cache_entry(cache_entry_id, limit=50)

            # Verify LIMIT clause was used
            call_args = mock_cursor.execute.call_args[0]
            assert "LIMIT %s" in call_args[0]

    def test_get_cache_hit_rate_success(self, client):
        """Test calculating cache hit rate."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (100, 75, 25)  # total, hits, misses

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            stats = client.get_cache_hit_rate(start_time, end_time)

            assert isinstance(stats, CacheHitRateStats)
            assert stats.total_requests == 100
            assert stats.cache_hits == 75
            assert stats.cache_misses == 25
            assert stats.hit_rate == 75.0
            assert stats.time_period_start == start_time
            assert stats.time_period_end == end_time

    def test_get_cache_hit_rate_no_data(self, client):
        """Test cache hit rate with no data."""
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

            stats = client.get_cache_hit_rate(start_time, end_time)

            assert stats.total_requests == 0
            assert stats.hit_rate == 0.0

    def test_get_cache_hit_rate_with_user_filter(self, client):
        """Test hit rate calculation with user filter."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (50, 40, 10)

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            stats = client.get_cache_hit_rate(start_time, end_time, user_id="user123")

            assert stats.hit_rate == 80.0
            # Verify user filter was applied
            call_args = mock_cursor.execute.call_args[0]
            assert "user_id = %s" in call_args[0]

    def test_get_latency_stats_success(self, client):
        """Test calculating latency statistics."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        # avg, min, max, p50, p95, p99, count
        mock_cursor.fetchone.return_value = (
            100.5,
            20,
            500,
            95.0,
            450.0,
            480.0,
            100,
        )

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            stats = client.get_latency_stats(start_time, end_time)

            assert isinstance(stats, CacheLatencyStats)
            assert stats.avg_latency_ms == 100.5
            assert stats.min_latency_ms == 20
            assert stats.max_latency_ms == 500
            assert stats.p50_latency_ms == 95.0
            assert stats.p95_latency_ms == 450.0
            assert stats.p99_latency_ms == 480.0
            assert stats.total_requests == 100

    def test_get_latency_stats_no_data(self, client):
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

            stats = client.get_latency_stats(start_time, end_time)

            assert stats.total_requests == 0
            assert stats.avg_latency_ms == 0.0
            assert stats.min_latency_ms == 0
            assert stats.max_latency_ms == 0

    def test_get_latency_stats_with_filters(self, client):
        """Test latency stats with cache_type and user filters."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (50.0, 10, 100, 45.0, 90.0, 95.0, 50)

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            stats = client.get_latency_stats(
                start_time, end_time, cache_type="hit", user_id="user123"
            )

            assert stats.avg_latency_ms == 50.0
            # Verify filters were applied
            call_args = mock_cursor.execute.call_args[0]
            assert "cache_type = %s" in call_args[0]
            assert "user_id = %s" in call_args[0]

    def test_get_total_metrics_count(self, client):
        """Test getting total metrics count."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (12345,)

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            count = client.get_total_metrics_count()

            assert count == 12345

    def test_get_total_metrics_count_empty(self, client):
        """Test getting count when table is empty."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            count = client.get_total_metrics_count()

            assert count == 0

    def test_get_metrics_count_by_type(self, client):
        """Test getting count grouped by cache type."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("hit", 150),
            ("miss", 50),
            ("warm_hit", 20),
        ]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            counts = client.get_metrics_count_by_type()

            assert counts["hit"] == 150
            assert counts["miss"] == 50
            assert counts["warm_hit"] == 20
            assert len(counts) == 3

    def test_get_metrics_count_by_type_with_time_range(self, client):
        """Test count by type with time range filter."""
        start_time = datetime.now(timezone.utc) - timedelta(days=7)
        end_time = datetime.now(timezone.utc)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("hit", 100), ("miss", 25)]

        with patch.object(client, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn

            counts = client.get_metrics_count_by_type(start_time, end_time)

            assert len(counts) == 2
            # Verify time range was used in query
            call_args = mock_cursor.execute.call_args[0]
            assert "request_timestamp >=" in call_args[0]
            assert "request_timestamp <=" in call_args[0]
