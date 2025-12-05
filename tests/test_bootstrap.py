"""
Tests for the database bootstrap script.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.bootstrap_db import DatabaseBootstrap


class TestDatabaseBootstrap:
    """Test suite for DatabaseBootstrap class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for database connection."""
        with patch("utils.bootstrap_db.get_settings") as mock:
            settings = Mock()
            settings.DB_HOST = "localhost"
            settings.DB_USER = "test_user"
            settings.DB_PASSWORD.get_secret_value.return_value = "test_password"
            settings.DB_NAME = "test_db"
            mock.return_value = settings
            yield settings

    @pytest.fixture
    def bootstrap(self, mock_settings):
        """Create DatabaseBootstrap instance."""
        return DatabaseBootstrap(dry_run=False)

    @pytest.fixture
    def bootstrap_dry_run(self, mock_settings):
        """Create DatabaseBootstrap instance in dry-run mode."""
        return DatabaseBootstrap(dry_run=True)

    def test_init(self, bootstrap):
        """Test bootstrap initialization."""
        assert bootstrap.db_host == "localhost"
        assert bootstrap.db_user == "test_user"
        assert bootstrap.db_password == "test_password"
        assert bootstrap.db_name == "test_db"
        assert bootstrap.db_port == 5432
        assert bootstrap.dry_run is False

    def test_init_dry_run(self, bootstrap_dry_run):
        """Test bootstrap initialization in dry-run mode."""
        assert bootstrap_dry_run.dry_run is True

    def test_check_table_exists_true(self, bootstrap):
        """Test checking for existing table."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (True,)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        result = bootstrap.check_table_exists(mock_conn, "test_table")

        assert result is True
        mock_cursor.execute.assert_called_once()

    def test_check_table_exists_false(self, bootstrap):
        """Test checking for non-existing table."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (False,)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        result = bootstrap.check_table_exists(mock_conn, "test_table")

        assert result is False

    def test_check_extension_exists_true(self, bootstrap):
        """Test checking for existing extension."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (True,)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        result = bootstrap.check_extension_exists(mock_conn, "vector")

        assert result is True

    def test_check_extension_exists_false(self, bootstrap):
        """Test checking for non-existing extension."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (False,)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        result = bootstrap.check_extension_exists(mock_conn, "vector")

        assert result is False

    def test_get_table_info(self, bootstrap):
        """Test getting table information."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("id", "integer", "NO", None),
            ("title", "text", "NO", None),
        ]
        mock_cursor.fetchone.return_value = (10,)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        info = bootstrap.get_table_info(mock_conn, "test_table")

        assert len(info["columns"]) == 2
        assert info["row_count"] == 10

    def test_create_extensions_dry_run(self, bootstrap_dry_run, capsys):
        """Test creating extensions in dry-run mode."""
        mock_conn = MagicMock()

        with patch.object(
            bootstrap_dry_run, "check_extension_exists", return_value=False
        ):
            bootstrap_dry_run.create_extensions(mock_conn)

        captured = capsys.readouterr()
        assert "[->] Would create extension 'vector'" in captured.out

    def test_create_extensions_already_exists(self, bootstrap_dry_run, capsys):
        """Test creating extensions when already exists."""
        mock_conn = MagicMock()

        with patch.object(
            bootstrap_dry_run, "check_extension_exists", return_value=True
        ):
            bootstrap_dry_run.create_extensions(mock_conn)

        captured = capsys.readouterr()
        assert "[OK] Extension 'vector' already exists" in captured.out

    def test_create_articles_table_dry_run(self, bootstrap_dry_run, capsys):
        """Test creating articles table in dry-run mode."""
        mock_conn = MagicMock()

        with patch.object(bootstrap_dry_run, "check_table_exists", return_value=False):
            bootstrap_dry_run.create_articles_table(mock_conn)

        captured = capsys.readouterr()
        assert "-> Would create table 'articles'" in captured.out

    def test_create_articles_table_already_exists_dry_run(
        self, bootstrap_dry_run, capsys
    ):
        """Test articles table when already exists in dry-run mode."""
        mock_conn = MagicMock()

        with patch.object(bootstrap_dry_run, "check_table_exists", return_value=True):
            with patch.object(
                bootstrap_dry_run,
                "get_table_info",
                return_value={"columns": [], "row_count": 5},
            ):
                bootstrap_dry_run.create_articles_table(mock_conn)

        captured = capsys.readouterr()
        assert "[OK] Table 'articles' already exists (5 rows)" in captured.out

    def test_create_articles_table(self, bootstrap, capsys):
        """Test creating articles table."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(bootstrap, "check_table_exists", return_value=False):
            bootstrap.create_articles_table(mock_conn)

        captured = capsys.readouterr()
        assert "[OK] Created table 'articles' with indexes" in captured.out
        assert mock_cursor.execute.call_count == 2  # CREATE TABLE + CREATE INDEX

    def test_create_embeddings_table_dry_run(self, bootstrap_dry_run, capsys):
        """Test creating embeddings table in dry-run mode."""
        mock_conn = MagicMock()

        with patch.object(bootstrap_dry_run, "check_table_exists", return_value=False):
            bootstrap_dry_run.create_embeddings_table(mock_conn)

        captured = capsys.readouterr()
        assert "-> Would create table 'embeddings'" in captured.out

    def test_create_embeddings_table_without_vector(self, bootstrap, capsys):
        """Test creating embeddings table without vector extension."""
        mock_conn = MagicMock()

        with patch.object(bootstrap, "check_table_exists", return_value=False):
            with patch.object(bootstrap, "check_extension_exists", return_value=False):
                bootstrap.create_embeddings_table(mock_conn)

        captured = capsys.readouterr()
        assert "vector extension not installed" in captured.out

    def test_create_chunks_table_dry_run(self, bootstrap_dry_run, capsys):
        """Test creating chunks table in dry-run mode."""
        mock_conn = MagicMock()

        with patch.object(bootstrap_dry_run, "check_table_exists", return_value=False):
            bootstrap_dry_run.create_chunks_table(mock_conn)

        captured = capsys.readouterr()
        assert "-> Would create table 'article_chunks'" in captured.out

    def test_create_chunks_table_already_exists_dry_run(
        self, bootstrap_dry_run, capsys
    ):
        """Test chunks table when already exists in dry-run mode."""
        mock_conn = MagicMock()

        with patch.object(bootstrap_dry_run, "check_table_exists", return_value=True):
            with patch.object(
                bootstrap_dry_run,
                "get_table_info",
                return_value={"columns": [], "row_count": 5},
            ):
                bootstrap_dry_run.create_chunks_table(mock_conn)

        captured = capsys.readouterr()
        assert "[OK] Table 'article_chunks' already exists (5 rows)" in captured.out

    def test_create_chunks_table(self, bootstrap, capsys):
        """Test creating chunks table."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(bootstrap, "check_table_exists", return_value=False):
            bootstrap.create_chunks_table(mock_conn)

        captured = capsys.readouterr()
        assert "[OK] Created table 'article_chunks' with indexes" in captured.out
        assert mock_cursor.execute.call_count == 2  # CREATE TABLE + CREATE INDEX

    def test_drop_all_tables_dry_run(self, bootstrap_dry_run, capsys):
        """Test dropping tables in dry-run mode."""
        mock_conn = MagicMock()

        def check_exists(conn, table):
            return table in ["articles", "embeddings", "article_chunks"]

        with patch.object(
            bootstrap_dry_run, "check_table_exists", side_effect=check_exists
        ):
            with patch.object(
                bootstrap_dry_run, "get_table_info", return_value={"row_count": 10}
            ):
                bootstrap_dry_run.drop_all_tables(mock_conn)

        captured = capsys.readouterr()
        assert "-> Would drop the following tables:" in captured.out
        assert "articles (10 rows)" in captured.out

    def test_drop_all_tables(self, bootstrap, capsys):
        """Test dropping tables."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        with patch.object(bootstrap, "check_table_exists", return_value=True):
            bootstrap.drop_all_tables(mock_conn)

        captured = capsys.readouterr()
        assert "[OK] Dropped table" in captured.out
        assert mock_cursor.execute.call_count == 3  # Drop articles, embeddings, and article_chunks

    def test_setup_database_dry_run(self, bootstrap_dry_run, capsys):
        """Test full setup in dry-run mode."""
        with patch.object(bootstrap_dry_run, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_get_conn.return_value = mock_conn

            with patch.object(
                bootstrap_dry_run, "check_extension_exists", return_value=False
            ):
                with patch.object(
                    bootstrap_dry_run, "check_table_exists", return_value=False
                ):
                    bootstrap_dry_run.setup_database()

        captured = capsys.readouterr()
        assert "=== DRY RUN MODE - No changes will be made ===" in captured.out
        assert "=== DRY RUN COMPLETE - No changes were made ===" in captured.out

    def test_check_status(self, bootstrap, capsys):
        """Test checking database status."""
        with patch.object(bootstrap, "get_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.__exit__.return_value = None
            mock_get_conn.return_value = mock_conn

            with patch.object(bootstrap, "check_extension_exists", return_value=True):
                with patch.object(bootstrap, "check_table_exists", return_value=True):
                    with patch.object(
                        bootstrap,
                        "get_table_info",
                        return_value={"columns": [], "row_count": 5},
                    ):
                        bootstrap.check_status()

        captured = capsys.readouterr()
        assert "=== Database Status:" in captured.out
        assert "vector: [OK] Installed" in captured.out
        assert "[OK] Exists (5 rows" in captured.out
