"""
Tests for the ingestion module (TDXClient and ArticleProcessor).
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from core.ingestion import ArticleProcessor
from utils.api_client import TDXClient
from core.schemas import TdxArticle


class TestArticleProcessor:
    """Test suite for ArticleProcessor class."""

    @pytest.fixture
    def mock_tdx_client(self):
        """Mock TDX client."""
        with patch("core.ingestion.TDXClient") as mock:
            yield mock.return_value

    @pytest.fixture
    def mock_db_client(self):
        """Mock database client."""
        with patch("core.ingestion.PostgresClient") as mock:
            yield mock.return_value

    @pytest.fixture
    def processor(self, mock_tdx_client, mock_db_client):
        """Create ArticleProcessor instance with mocked dependencies."""
        return ArticleProcessor()

    def test_categorize_articles_new_article(self, processor):
        """Test that new articles are correctly identified."""
        api_articles = [
            {
                "ID": 123,
                "ModifiedDate": "2024-01-01T00:00:00Z",
                "Subject": "Test Article",
                "Body": "Content",
            }
        ]
        db_metadata = {}

        new, updated, unchanged = processor._categorize_articles(
            api_articles, db_metadata
        )

        assert len(new) == 1
        assert len(updated) == 0
        assert len(unchanged) == 0
        assert new[0]["ID"] == 123

    def test_categorize_articles_updated_article(self, processor):
        """Test that updated articles are correctly identified."""
        api_articles = [
            {
                "ID": 123,
                "ModifiedDate": "2024-01-02T00:00:00Z",
                "Subject": "Test Article",
                "Body": "Updated Content",
            }
        ]
        db_metadata = {123: datetime(2024, 1, 1, tzinfo=timezone.utc)}

        new, updated, unchanged = processor._categorize_articles(
            api_articles, db_metadata
        )

        assert len(new) == 0
        assert len(updated) == 1
        assert len(unchanged) == 0
        assert updated[0]["ID"] == 123

    def test_categorize_articles_unchanged_article(self, processor):
        """Test that unchanged articles are correctly identified."""
        api_articles = [
            {
                "ID": 123,
                "ModifiedDate": "2024-01-01T00:00:00Z",
                "Subject": "Test Article",
                "Body": "Content",
            }
        ]
        db_metadata = {123: datetime(2024, 1, 1, tzinfo=timezone.utc)}

        new, updated, unchanged = processor._categorize_articles(
            api_articles, db_metadata
        )

        assert len(new) == 0
        assert len(updated) == 0
        assert len(unchanged) == 1
        assert unchanged[0] == 123

    def test_categorize_articles_handles_none_values(self, processor):
        """Test that articles with None values are skipped."""
        api_articles = [
            {"ID": None, "ModifiedDate": "2024-01-01T00:00:00Z"},
            {"ID": 123, "ModifiedDate": None},
            {"ID": 456, "ModifiedDate": "2024-01-01T00:00:00Z"},
        ]
        db_metadata = {}

        new, updated, unchanged = processor._categorize_articles(
            api_articles, db_metadata
        )

        # Only the article with both ID and ModifiedDate should be processed
        assert len(new) == 1
        assert new[0]["ID"] == 456

    def test_categorize_articles_parses_string_dates(self, processor):
        """Test that string dates are correctly parsed."""
        api_articles = [
            {
                "ID": 123,
                "ModifiedDate": "2024-01-02T00:00:00Z",
                "Subject": "Test",
            }
        ]
        db_metadata = {123: datetime(2024, 1, 1, tzinfo=timezone.utc)}

        new, updated, unchanged = processor._categorize_articles(
            api_articles, db_metadata
        )

        assert len(updated) == 1

    def test_process_articles_filters_phishing(self, processor):
        """Test that phishing category articles are filtered out."""
        articles = [
            {
                "ID": 123,
                "Subject": "Test Article",
                "Body": "Content",
                "ModifiedDate": "2024-01-01T00:00:00Z",
                "CategoryName": "Recent Phishing Emails",
            },
            {
                "ID": 456,
                "Subject": "Normal Article",
                "Body": "Content",
                "ModifiedDate": "2024-01-01T00:00:00Z",
                "CategoryName": "IT Help",
            },
        ]

        processed = processor.process_articles(articles)

        assert len(processed) == 1
        assert processed[0].tdx_article_id == 456

    def test_process_articles_constructs_url(self, processor):
        """Test that article URLs are correctly constructed."""
        articles = [
            {
                "ID": 123,
                "Subject": "Test Article",
                "Body": "Content",
                "ModifiedDate": "2024-01-01T00:00:00Z",
            }
        ]

        processed = processor.process_articles(articles)

        assert len(processed) == 1
        assert (
            str(processed[0].url)
            == "https://utc.teamdynamix.com/TDClient/2717/Portal/KB/ArticleDet?ID=123"
        )

    def test_process_articles_skips_missing_required_fields(self, processor):
        """Test that articles with missing required fields are skipped."""
        articles = [
            {
                "ID": None,
                "Subject": "Test",
                "Body": "Content",
                "ModifiedDate": "2024-01-01T00:00:00Z",
            },
            {
                "ID": 123,
                "Subject": None,
                "Body": "Content",
                "ModifiedDate": "2024-01-01T00:00:00Z",
            },
            {
                "ID": 456,
                "Subject": "Test",
                "Body": None,
                "ModifiedDate": "2024-01-01T00:00:00Z",
            },
            {"ID": 789, "Subject": "Test", "Body": "Content", "ModifiedDate": None},
            {
                "ID": 999,
                "Subject": "Valid",
                "Body": "Valid Content",
                "ModifiedDate": "2024-01-01T00:00:00Z",
            },
        ]

        processed = processor.process_articles(articles)

        # Only the last article with all required fields should be processed
        assert len(processed) == 1
        assert processed[0].tdx_article_id == 999

    def test_process_articles_validates_with_pydantic(self, processor):
        """Test that processed articles are valid Pydantic models."""
        articles = [
            {
                "ID": 123,
                "Subject": "Test Article",
                "Body": "Content",
                "ModifiedDate": "2024-01-01T00:00:00Z",
            }
        ]

        processed = processor.process_articles(articles)

        assert len(processed) == 1
        assert isinstance(processed[0], TdxArticle)
        assert processed[0].tdx_article_id == 123
        assert processed[0].title == "Test Article"
        assert processed[0].content_html == "Content"

    def test_sync_articles(self, processor, mock_tdx_client, mock_db_client):
        """Test the full sync operation."""
        # Setup mocks
        mock_db_client.get_article_metadata.return_value = {
            123: datetime(2024, 1, 1, tzinfo=timezone.utc)
        }
        mock_tdx_client.retrieve_all_articles.return_value = (
            [
                {
                    "ID": 123,
                    "Subject": "Updated Article",
                    "Body": "Content",
                    "ModifiedDate": "2024-01-02T00:00:00Z",
                },
                {
                    "ID": 456,
                    "Subject": "New Article",
                    "Body": "Content",
                    "ModifiedDate": "2024-01-01T00:00:00Z",
                },
            ],
            [],
        )

        result = processor.sync_articles()

        assert "new" in result
        assert "updated" in result
        assert "unchanged" in result
        assert "skipped" in result
        assert len(result["new"]) == 1
        assert len(result["updated"]) == 1

    def test_ingest_and_store(self, processor, mock_tdx_client, mock_db_client):
        """Test the complete ingestion workflow."""
        # Setup mocks
        mock_db_client.get_article_metadata.return_value = {}
        mock_tdx_client.retrieve_all_articles.return_value = (
            [
                {
                    "ID": 123,
                    "Subject": "Test Article",
                    "Body": "Content",
                    "ModifiedDate": "2024-01-01T00:00:00Z",
                }
            ],
            [],
        )

        stats = processor.ingest_and_store()

        assert stats["new_count"] == 1
        assert stats["updated_count"] == 0
        assert stats["unchanged_count"] == 0
        assert stats["skipped_count"] == 0
        mock_db_client.insert_articles.assert_called_once()

    def test_identify_deleted_articles(
        self, processor, mock_tdx_client, mock_db_client
    ):
        """Test identification of deleted articles."""
        # Setup mocks
        mock_db_client.get_existing_article_ids.return_value = {123, 456, 789}
        mock_tdx_client.list_article_ids.return_value = [123, 789]

        deleted = processor.identify_deleted_articles()

        assert len(deleted) == 1
        assert 456 in deleted


class TestTDXClient:
    """Test suite for TDXClient class."""

    @pytest.fixture
    def client(self):
        """Create TDXClient instance with test credentials."""
        with patch("utils.api_client.get_settings") as mock_settings:
            mock_settings.return_value.BASE_URL = "https://test.teamdynamix.com"
            mock_settings.return_value.APP_ID = 1234
            mock_settings.return_value.WEBSERVICES_KEY.get_secret_value.return_value = (
                "test_key"
            )
            mock_settings.return_value.BEID.get_secret_value.return_value = "test_beid"
            client = TDXClient()
            yield client

    def test_authenticate_success(self, client):
        """Test successful authentication."""
        with patch.object(client, "_request") as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "test_bearer_token"
            mock_request.return_value = mock_response

            token = client.authenticate()

            assert token == "test_bearer_token"
            assert client.bearer_token == "test_bearer_token"
            assert "Bearer test_bearer_token" in str(
                client.session.headers.get("Authorization")
            )

    def test_authenticate_failure(self, client):
        """Test authentication failure."""
        with patch.object(client, "_request") as mock_request:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_request.return_value = mock_response

            with pytest.raises(Exception):
                client.authenticate()

    def test_list_article_ids_returns_integers(self, client):
        """Test that list_article_ids returns list of integers."""
        with patch.object(client, "_request") as mock_request:
            # Mock authentication
            auth_response = Mock()
            auth_response.status_code = 200
            auth_response.text = "token"

            # Mock article list response
            list_response = Mock()
            list_response.status_code = 200
            list_response.json.return_value = [{"ID": 123}, {"ID": 456}]

            mock_request.side_effect = [auth_response, list_response]

            article_ids = client.list_article_ids()

            assert isinstance(article_ids, list)
            assert all(isinstance(id, int) for id in article_ids)
            assert article_ids == [123, 456]

    def test_retrieve_all_articles_success(self, client):
        """Test successful article retrieval."""
        with patch.object(client, "_request") as mock_request:
            with patch.object(client, "list_article_ids") as mock_list:
                # Mock article IDs
                mock_list.return_value = [123, 456]

                # Mock article retrieval
                article_response = Mock()
                article_response.status_code = 200
                article_response.json.return_value = {"ID": 123, "Subject": "Test"}
                mock_request.return_value = article_response

                client.bearer_token = "test_token"

                articles, skipped = client.retrieve_all_articles()

                assert len(articles) == 2
                assert len(skipped) == 0

    def test_retrieve_all_articles_handles_failures(self, client):
        """Test that failed article retrievals are tracked."""
        with patch.object(client, "_request") as mock_request:
            with patch.object(client, "list_article_ids") as mock_list:
                # Mock article IDs
                mock_list.return_value = [123, 456]

                # Mock one success and one failure
                success_response = Mock()
                success_response.status_code = 200
                success_response.json.return_value = {"ID": 123, "Subject": "Test"}

                failure_response = Mock()
                failure_response.status_code = 404
                failure_response.text = "Not found"

                mock_request.side_effect = [success_response, failure_response]

                client.bearer_token = "test_token"

                articles, skipped = client.retrieve_all_articles()

                assert len(articles) == 1
                assert len(skipped) == 1
                assert skipped[0][0] == 456
