"""
Tests for reranker module.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone
from uuid import uuid4

from core.reranker import Reranker, RerankerResult
from core.schemas import TextChunk
from pydantic import HttpUrl
from litellm.exceptions import (
    RateLimitError,
    Timeout,
    AuthenticationError,
    APIError,
)


@pytest.fixture
def mock_litellm_settings():
    """Mock LiteLLM settings."""
    settings = Mock()
    settings.RERANKER_MODEL = "cohere-rerank-v3-5"
    settings.PROXY_BASE_URL = "http://localhost:4000"
    settings.PROXY_API_KEY = Mock()
    settings.PROXY_API_KEY.get_secret_value = Mock(return_value="test-api-key")
    return settings


@pytest.fixture
def mock_rerank_response():
    """Create a mock litellm.rerank response."""
    response = Mock()

    result0 = Mock()
    result0.index = 2
    result0.relevance_score = 0.95

    result1 = Mock()
    result1.index = 0
    result1.relevance_score = 0.87

    result2 = Mock()
    result2.index = 1
    result2.relevance_score = 0.72

    response.results = [result0, result1, result2]
    return response


@pytest.fixture
def sample_fused_results():
    """Create sample fused results from hybrid search."""
    base_time = datetime.now(timezone.utc)

    chunks = [
        TextChunk(
            chunk_id=uuid4(),
            parent_article_id=uuid4(),
            chunk_sequence=0,
            text_content="Password reset instructions for system users.",
            token_count=10,
            source_url=HttpUrl("https://example.com/1"),
            last_modified_date=base_time,
        ),
        TextChunk(
            chunk_id=uuid4(),
            parent_article_id=uuid4(),
            chunk_sequence=0,
            text_content="VPN troubleshooting guide for remote access.",
            token_count=10,
            source_url=HttpUrl("https://example.com/2"),
            last_modified_date=base_time,
        ),
        TextChunk(
            chunk_id=uuid4(),
            parent_article_id=uuid4(),
            chunk_sequence=0,
            text_content="Email configuration steps for new employees.",
            token_count=10,
            source_url=HttpUrl("https://example.com/3"),
            last_modified_date=base_time,
        ),
    ]

    # Simulate RRF fusion output format
    return [
        {"rank": 1, "combined_score": 0.045, "chunk": chunks[0]},
        {"rank": 2, "combined_score": 0.038, "chunk": chunks[1]},
        {"rank": 3, "combined_score": 0.032, "chunk": chunks[2]},
    ]


class TestReranker:
    """Test suite for Reranker."""

    @patch("core.reranker.get_litellm_settings")
    def test_initialization_success(
        self,
        mock_get_settings,
        mock_litellm_settings,
    ):
        """Test successful reranker initialization."""
        mock_get_settings.return_value = mock_litellm_settings

        reranker = Reranker()

        assert reranker.model == "cohere-rerank-v3-5"
        assert reranker.api_base == "http://localhost:4000"
        assert reranker.max_retries == 3
        assert reranker.timeout == 30.0

    @patch("core.reranker.get_litellm_settings")
    def test_initialization_empty_api_key(self, mock_get_settings):
        """Test initialization fails with empty API key."""
        settings = Mock()
        settings.RERANKER_MODEL = "cohere-rerank-v3-5"
        settings.PROXY_BASE_URL = "http://localhost:4000"
        settings.PROXY_API_KEY = Mock()
        settings.PROXY_API_KEY.get_secret_value = Mock(return_value="")
        mock_get_settings.return_value = settings

        with pytest.raises(ValueError, match="LITELLM_PROXY_API_KEY is empty"):
            Reranker()

    @patch("core.reranker.litellm.rerank")
    @patch("core.reranker.get_litellm_settings")
    def test_rerank_basic(
        self,
        mock_get_settings,
        mock_litellm_rerank,
        mock_litellm_settings,
        mock_rerank_response,
        sample_fused_results,
    ):
        """Test basic reranking functionality."""
        mock_get_settings.return_value = mock_litellm_settings
        mock_litellm_rerank.return_value = mock_rerank_response

        reranker = Reranker()
        results = reranker.rerank(
            query="password reset help", results=sample_fused_results
        )

        # Verify results structure
        assert len(results) == 3
        assert all("rank" in r for r in results)
        assert all("combined_score" in r for r in results)
        assert all("chunk" in r for r in results)

        # Verify reranking changed order (based on mock response)
        assert results[0]["rank"] == 1
        assert results[0]["combined_score"] == 0.95
        assert results[1]["rank"] == 2
        assert results[1]["combined_score"] == 0.87
        assert results[2]["rank"] == 3
        assert results[2]["combined_score"] == 0.72

        # Verify metadata was added
        assert "metadata" in results[0]
        assert "original_rank" in results[0]["metadata"]
        assert "original_score" in results[0]["metadata"]

        # Verify litellm.rerank was called correctly
        mock_litellm_rerank.assert_called_once()
        call_kwargs = mock_litellm_rerank.call_args[1]
        assert call_kwargs["model"] == "cohere-rerank-v3-5"
        assert call_kwargs["query"] == "password reset help"
        assert len(call_kwargs["documents"]) == 3

    @patch("core.reranker.litellm.rerank")
    @patch("core.reranker.get_litellm_settings")
    def test_rerank_empty_results(
        self,
        mock_get_settings,
        mock_litellm_rerank,
        mock_litellm_settings,
    ):
        """Test reranking with empty results list."""
        mock_get_settings.return_value = mock_litellm_settings

        reranker = Reranker()
        results = reranker.rerank(query="test", results=[])

        assert results == []
        mock_litellm_rerank.assert_not_called()

    @patch("core.reranker.get_litellm_settings")
    def test_rerank_empty_query(
        self,
        mock_get_settings,
        mock_litellm_settings,
        sample_fused_results,
    ):
        """Test reranking with empty query raises ValueError."""
        mock_get_settings.return_value = mock_litellm_settings

        reranker = Reranker()

        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.rerank(query="", results=sample_fused_results)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.rerank(query="   ", results=sample_fused_results)

    @patch("core.reranker.get_litellm_settings")
    def test_rerank_invalid_results_structure(
        self,
        mock_get_settings,
        mock_litellm_settings,
    ):
        """Test reranking with invalid results structure."""
        mock_get_settings.return_value = mock_litellm_settings

        reranker = Reranker()

        # Test with non-dict items
        with pytest.raises(ValueError, match="Results must be a list of dicts"):
            reranker.rerank(query="test", results=["not", "dicts"])

        # Test with missing chunk field
        with pytest.raises(ValueError, match="Each result must have a 'chunk' field"):
            reranker.rerank(query="test", results=[{"rank": 1, "score": 0.5}])

    @patch("core.reranker.litellm.rerank")
    @patch("core.reranker.get_litellm_settings")
    def test_rerank_rate_limit_error(
        self,
        mock_get_settings,
        mock_litellm_rerank,
        mock_litellm_settings,
        sample_fused_results,
    ):
        """Test handling of rate limit errors."""
        mock_get_settings.return_value = mock_litellm_settings
        mock_litellm_rerank.side_effect = RateLimitError(
            "Rate limit exceeded",
            llm_provider="litellm_proxy",
            model="cohere-rerank-v3-5",
        )

        reranker = Reranker()

        with pytest.raises(RuntimeError, match="Reranking rate limit exceeded"):
            reranker.rerank(query="test", results=sample_fused_results)

    @patch("core.reranker.litellm.rerank")
    @patch("core.reranker.get_litellm_settings")
    def test_rerank_authentication_error(
        self,
        mock_get_settings,
        mock_litellm_rerank,
        mock_litellm_settings,
        sample_fused_results,
    ):
        """Test handling of authentication errors."""
        mock_get_settings.return_value = mock_litellm_settings
        mock_litellm_rerank.side_effect = AuthenticationError(
            "Auth failed",
            llm_provider="litellm_proxy",
            model="cohere-rerank-v3-5",
        )

        reranker = Reranker()

        with pytest.raises(RuntimeError, match="Reranking authentication failed"):
            reranker.rerank(query="test", results=sample_fused_results)

    @patch("core.reranker.litellm.rerank")
    @patch("core.reranker.get_litellm_settings")
    def test_rerank_timeout_error(
        self,
        mock_get_settings,
        mock_litellm_rerank,
        mock_litellm_settings,
        sample_fused_results,
    ):
        """Test handling of timeout errors."""
        mock_get_settings.return_value = mock_litellm_settings
        mock_litellm_rerank.side_effect = Timeout(
            "Timeout",
            model="cohere-rerank-v3-5",
            llm_provider="litellm_proxy",
        )

        reranker = Reranker()

        with pytest.raises(RuntimeError, match="Reranking API timeout"):
            reranker.rerank(query="test", results=sample_fused_results)

    @patch("core.reranker.litellm.rerank")
    @patch("core.reranker.get_litellm_settings")
    def test_rerank_api_error(
        self,
        mock_get_settings,
        mock_litellm_rerank,
        mock_litellm_settings,
        sample_fused_results,
    ):
        """Test handling of generic API errors."""
        mock_get_settings.return_value = mock_litellm_settings
        mock_litellm_rerank.side_effect = APIError(
            status_code=500,
            message="Server error",
            llm_provider="litellm_proxy",
            model="cohere-rerank-v3-5",
        )

        reranker = Reranker()

        with pytest.raises(RuntimeError, match="Reranking API error"):
            reranker.rerank(query="test", results=sample_fused_results)

    @patch("core.reranker.litellm.rerank")
    @patch("core.reranker.get_litellm_settings")
    def test_rerank_limits_to_1000_documents(
        self,
        mock_get_settings,
        mock_litellm_rerank,
        mock_litellm_settings,
    ):
        """Test that reranker truncates to 1000 documents."""
        mock_get_settings.return_value = mock_litellm_settings

        # Create 1500 fake results
        large_results = []
        for i in range(1500):
            chunk = Mock()
            chunk.text_content = f"Document {i}"
            large_results.append({"rank": i + 1, "combined_score": 0.5, "chunk": chunk})

        # Mock response with just first result
        mock_response = Mock()
        result0 = Mock()
        result0.index = 0
        result0.relevance_score = 0.9
        mock_response.results = [result0]
        mock_litellm_rerank.return_value = mock_response

        reranker = Reranker()
        results = reranker.rerank(query="test", results=large_results)

        # Verify only 1000 documents were sent
        call_kwargs = mock_litellm_rerank.call_args[1]
        assert len(call_kwargs["documents"]) == 1000

    @patch("core.reranker.litellm.rerank")
    @patch("core.reranker.get_litellm_settings")
    def test_rerank_top_n_parameter(
        self,
        mock_get_settings,
        mock_litellm_rerank,
        mock_litellm_settings,
        sample_fused_results,
    ):
        """Test top_n parameter limits results."""
        mock_get_settings.return_value = mock_litellm_settings

        # Mock response with 2 results
        mock_response = Mock()
        result0 = Mock()
        result0.index = 0
        result0.relevance_score = 0.9
        result1 = Mock()
        result1.index = 1
        result1.relevance_score = 0.8
        mock_response.results = [result0, result1]
        mock_litellm_rerank.return_value = mock_response

        reranker = Reranker()
        results = reranker.rerank(query="test", results=sample_fused_results, top_n=2)

        # Verify API was called with top_n=2
        call_kwargs = mock_litellm_rerank.call_args[1]
        assert call_kwargs["top_n"] == 2

        # Verify only 2 results returned
        assert len(results) == 2

    @patch("core.reranker.litellm.rerank")
    @patch("core.reranker.get_litellm_settings")
    def test_rerank_preserves_system_prompt(
        self,
        mock_get_settings,
        mock_litellm_rerank,
        mock_litellm_settings,
    ):
        """Test that system_prompt from original result is preserved."""
        mock_get_settings.return_value = mock_litellm_settings

        chunk = Mock()
        chunk.text_content = "Some text"
        results_with_prompt = [
            {
                "rank": 1,
                "combined_score": 0.5,
                "chunk": chunk,
                "system_prompt": "You are a helpful assistant.",
            }
        ]

        mock_response = Mock()
        result0 = Mock()
        result0.index = 0
        result0.relevance_score = 0.9
        mock_response.results = [result0]
        mock_litellm_rerank.return_value = mock_response

        reranker = Reranker()
        results = reranker.rerank(query="test", results=results_with_prompt)

        assert results[0]["system_prompt"] == "You are a helpful assistant."

    @patch("core.reranker.litellm.rerank")
    @patch("core.reranker.get_litellm_settings")
    def test_rerank_latency_tracking(
        self,
        mock_get_settings,
        mock_litellm_rerank,
        mock_litellm_settings,
        sample_fused_results,
        mock_rerank_response,
    ):
        """Test that latency is tracked."""
        mock_get_settings.return_value = mock_litellm_settings
        mock_litellm_rerank.return_value = mock_rerank_response

        reranker = Reranker()
        assert reranker.last_rerank_latency_ms == 0

        reranker.rerank(query="test", results=sample_fused_results)

        assert reranker.last_rerank_latency_ms >= 0

    @patch("core.reranker.litellm.rerank")
    @patch("core.reranker.get_litellm_settings")
    def test_rerank_malformed_response(
        self,
        mock_get_settings,
        mock_litellm_rerank,
        mock_litellm_settings,
        sample_fused_results,
    ):
        """Test handling of malformed API response."""
        mock_get_settings.return_value = mock_litellm_settings

        # Response with no results attribute
        mock_response = Mock()
        mock_response.results = None  # Causes _parse_response to handle gracefully
        mock_litellm_rerank.return_value = mock_response

        reranker = Reranker()
        # When results is None/empty, _parse_response returns original results
        results = reranker.rerank(query="test", results=sample_fused_results)
        assert len(results) == 3  # Returns original results unchanged
