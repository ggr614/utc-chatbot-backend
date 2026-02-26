"""
Tests for BM25 search module.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone
from uuid import uuid4

from core.bm25_search import BM25Retriever, BM25SearchResult
from core.schemas import TextChunk


@pytest.fixture
def sample_chunks():
    """Create sample text chunks for testing."""
    base_time = datetime.now(timezone.utc)

    chunks = [
        TextChunk(
            chunk_id=uuid4(),
            parent_article_id=uuid4(),
            chunk_sequence=0,
            text_content="How to reset your password in the system. Visit the login page and click forgot password.",
            token_count=20,
            source_url="https://example.com/article1",
            last_modified_date=base_time,
        ),
        TextChunk(
            chunk_id=uuid4(),
            parent_article_id=uuid4(),
            chunk_sequence=0,
            text_content="VPN connection guide. Install the VPN client and configure your settings.",
            token_count=15,
            source_url="https://example.com/article2",
            last_modified_date=base_time,
        ),
        TextChunk(
            chunk_id=uuid4(),
            parent_article_id=uuid4(),
            chunk_sequence=0,
            text_content="Email configuration for Microsoft Outlook. Set up your email account with these steps.",
            token_count=18,
            source_url="https://example.com/article3",
            last_modified_date=base_time,
        ),
    ]
    return chunks


@pytest.fixture
def mock_db_client(sample_chunks):
    """Create a mock database client."""
    client = Mock()
    client.get_all_chunks.return_value = sample_chunks
    return client


class TestBM25Retriever:
    """Test suite for BM25Retriever."""

    @pytest.fixture(autouse=True)
    def mock_prompt_client(self):
        """Patch PromptStorageClient to prevent real DB connections during search."""
        with patch("core.storage_prompt.PromptStorageClient") as mock_cls:
            mock_cls.return_value.get_prompts_for_article_ids.return_value = {}
            yield mock_cls

    def test_initialization(self, mock_db_client):
        """Test BM25Retriever initialization."""
        retriever = BM25Retriever(
            postgres_client=mock_db_client, k1=1.5, b=0.75, use_cache=True
        )

        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
        assert retriever.use_cache is True

    def test_initialization_invalid_k1(self, mock_db_client):
        """Test initialization with invalid k1."""
        with pytest.raises(ValueError, match="k1 must be non-negative"):
            BM25Retriever(postgres_client=mock_db_client, k1=-1.0)

    def test_initialization_invalid_b(self, mock_db_client):
        """Test initialization with invalid b."""
        with pytest.raises(ValueError, match="b must be between 0 and 1"):
            BM25Retriever(postgres_client=mock_db_client, b=1.5)

    def test_tokenize(self):
        """Test text tokenization."""
        text = "How to reset your password in the system?"
        tokens = BM25Retriever.tokenize(text)

        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)
        assert all(t.islower() for t in tokens)
        # Should contain these tokens
        assert "password" in tokens
        assert "reset" in tokens

    def test_tokenize_empty(self):
        """Test tokenization of empty string."""
        tokens = BM25Retriever.tokenize("")
        assert tokens == []

    def test_search_basic(self, mock_db_client):
        """Test basic search functionality."""
        retriever = BM25Retriever(postgres_client=mock_db_client, use_cache=False)

        results = retriever.search(query="password reset", top_k=2)

        assert isinstance(results, list)
        assert len(results) <= 2
        assert all(isinstance(r, BM25SearchResult) for r in results)

        # Check that results are ranked
        if len(results) > 1:
            assert results[0].score >= results[1].score
            assert results[0].rank == 1
            assert results[1].rank == 2

    def test_search_empty_query(self, mock_db_client):
        """Test search with empty query."""
        retriever = BM25Retriever(postgres_client=mock_db_client)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.search(query="", top_k=5)

    def test_search_invalid_top_k(self, mock_db_client):
        """Test search with invalid top_k."""
        retriever = BM25Retriever(postgres_client=mock_db_client)

        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.search(query="test", top_k=0)

    def test_search_with_min_score(self, mock_db_client):
        """Test search with minimum score filtering."""
        retriever = BM25Retriever(postgres_client=mock_db_client, use_cache=False)

        # Search with high min_score should return fewer results
        all_results = retriever.search(query="password", top_k=10)
        filtered_results = retriever.search(query="password", top_k=10, min_score=5.0)

        assert len(filtered_results) <= len(all_results)
        # All filtered results should have score >= min_score
        assert all(r.score >= 5.0 for r in filtered_results)

    def test_search_no_matches(self, mock_db_client):
        """Test search with query that has no matches."""
        retriever = BM25Retriever(postgres_client=mock_db_client, use_cache=False)

        # Query with terms not in corpus
        results = retriever.search(query="xyzabc123", top_k=5)

        # Should return results but with very low or zero scores
        assert isinstance(results, list)

    def test_batch_search(self, mock_db_client):
        """Test batch search functionality."""
        retriever = BM25Retriever(postgres_client=mock_db_client, use_cache=False)

        queries = ["password reset", "VPN connection", "email setup"]
        results = retriever.batch_search(queries=queries, top_k=2)

        assert isinstance(results, dict)
        assert len(results) == len(queries)
        assert all(query in results for query in queries)
        assert all(isinstance(results[q], list) for q in queries)

    def test_batch_search_empty(self, mock_db_client):
        """Test batch search with empty query list."""
        retriever = BM25Retriever(postgres_client=mock_db_client)

        results = retriever.batch_search(queries=[], top_k=5)

        assert results == {}

    def test_caching(self, mock_db_client):
        """Test that caching works correctly."""
        retriever = BM25Retriever(postgres_client=mock_db_client, use_cache=True)

        # First search should call database
        retriever.search(query="test", top_k=5)
        assert mock_db_client.get_all_chunks.call_count == 1

        # Second search should use cache
        retriever.search(query="another test", top_k=5)
        assert mock_db_client.get_all_chunks.call_count == 1  # Still 1

        # Clear cache
        retriever.clear_cache()

        # Next search should call database again
        retriever.search(query="test again", top_k=5)
        assert mock_db_client.get_all_chunks.call_count == 2

    def test_get_stats(self, mock_db_client, sample_chunks):
        """Test retriever statistics."""
        retriever = BM25Retriever(postgres_client=mock_db_client, use_cache=False)

        stats = retriever.get_stats()

        assert isinstance(stats, dict)
        assert stats["num_chunks"] == len(sample_chunks)
        assert stats["k1"] == retriever.k1
        assert stats["b"] == retriever.b
        assert "avg_doc_length" in stats
        assert "num_unique_terms" in stats

    def test_result_to_dict(self, sample_chunks):
        """Test BM25SearchResult.to_dict()."""
        result = BM25SearchResult(chunk=sample_chunks[0], score=5.5, rank=1)

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["rank"] == 1
        assert result_dict["score"] == 5.5
        assert "chunk_id" in result_dict
        assert "text_content" in result_dict
        assert "source_url" in result_dict
