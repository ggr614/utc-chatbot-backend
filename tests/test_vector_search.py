"""
Tests for Vector Search module.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone
from uuid import uuid4

from core.vector_search import VectorRetriever, VectorSearchResult
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
def mock_embedder():
    """Create a mock embedding generator."""
    embedder = Mock()
    embedder.expected_dim = 3072
    embedder.deployment_name = "text-embedding-3-large"
    # Return a mock embedding vector
    embedder.generate_embedding.return_value = [0.1] * 3072
    return embedder


@pytest.fixture
def mock_vector_store(sample_chunks):
    """Create a mock vector storage client."""
    store = Mock()

    # Mock search_similar_vectors to return dicts with similarities
    def mock_search(query_vector, limit, min_similarity=0.0, status_names=None, category_names=None, is_public=None, tags=None):
        # Return sample chunks as dicts with mock cosine similarities
        results = [
            {
                "chunk_id": sample_chunks[0].chunk_id,
                "parent_article_id": sample_chunks[0].parent_article_id,
                "chunk_sequence": sample_chunks[0].chunk_sequence,
                "text_content": sample_chunks[0].text_content,
                "token_count": sample_chunks[0].token_count,
                "source_url": sample_chunks[0].source_url,
                "created_at": sample_chunks[0].last_modified_date,
                "similarity": 0.9,
            },
            {
                "chunk_id": sample_chunks[1].chunk_id,
                "parent_article_id": sample_chunks[1].parent_article_id,
                "chunk_sequence": sample_chunks[1].chunk_sequence,
                "text_content": sample_chunks[1].text_content,
                "token_count": sample_chunks[1].token_count,
                "source_url": sample_chunks[1].source_url,
                "created_at": sample_chunks[1].last_modified_date,
                "similarity": 0.7,
            },
            {
                "chunk_id": sample_chunks[2].chunk_id,
                "parent_article_id": sample_chunks[2].parent_article_id,
                "chunk_sequence": sample_chunks[2].chunk_sequence,
                "text_content": sample_chunks[2].text_content,
                "token_count": sample_chunks[2].token_count,
                "source_url": sample_chunks[2].source_url,
                "created_at": sample_chunks[2].last_modified_date,
                "similarity": 0.5,
            },
        ]
        # Filter by min_similarity
        filtered = [r for r in results if r["similarity"] >= min_similarity]
        return filtered[:limit]

    store.search_similar_vectors.side_effect = mock_search
    store.get_count.return_value = len(sample_chunks)
    store.get_embedding_by_chunk_id.return_value = [0.1] * 3072
    store.close.return_value = None

    return store


class TestVectorRetriever:
    """Test suite for VectorRetriever."""

    def test_initialization(self, mock_embedder, mock_vector_store):
        """Test VectorRetriever initialization."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        assert retriever.embedder == mock_embedder
        assert retriever.vector_store == mock_vector_store

    def test_initialization_without_args(self):
        """Test initialization creates default components."""
        with (
            patch("core.vector_search.GenerateEmbeddingsOpenAI") as mock_embedder_class,
            patch("core.vector_search.OpenAIVectorStorage") as mock_storage_class,
        ):
            retriever = VectorRetriever()

            # Should create default instances
            mock_embedder_class.assert_called_once()
            mock_storage_class.assert_called_once()

    def test_search_basic(self, mock_embedder, mock_vector_store):
        """Test basic search functionality."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        results = retriever.search(query="password reset", top_k=2)

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, VectorSearchResult) for r in results)

        # Check that results are ranked by similarity
        assert results[0].similarity >= results[1].similarity
        assert results[0].rank == 1
        assert results[1].rank == 2

        # Verify embedding was generated for query
        mock_embedder.generate_embedding.assert_called_once_with("password reset")

    def test_search_empty_query(self, mock_embedder, mock_vector_store):
        """Test search with empty query."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.search(query="", top_k=5)

    def test_search_invalid_top_k(self, mock_embedder, mock_vector_store):
        """Test search with invalid top_k."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.search(query="test", top_k=0)

        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.search(query="test", top_k=-1)

    def test_search_invalid_min_similarity(self, mock_embedder, mock_vector_store):
        """Test search with invalid min_similarity."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        with pytest.raises(ValueError, match="min_similarity must be between 0 and 1"):
            retriever.search(query="test", top_k=5, min_similarity=1.5)

        with pytest.raises(ValueError, match="min_similarity must be between 0 and 1"):
            retriever.search(query="test", top_k=5, min_similarity=-0.1)

    def test_search_with_min_similarity(
        self, mock_embedder, mock_vector_store, sample_chunks
    ):
        """Test search with minimum similarity filtering."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        # Search with high min_similarity should filter out low-similarity results
        results = retriever.search(query="password", top_k=10, min_similarity=0.6)

        # Should only return results with similarity >= 0.6
        # From mock: (0.9, 0.7, 0.5) -> only first two pass filter
        assert len(results) == 2
        assert all(r.similarity >= 0.6 for r in results)

    def test_search_similarity_conversion(self, mock_embedder, mock_vector_store):
        """Test that similarities are correctly retrieved from database."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        results = retriever.search(query="test", top_k=3)

        # Mock returns similarities [0.9, 0.7, 0.5]
        assert results[0].similarity == pytest.approx(0.9, abs=0.01)
        assert results[1].similarity == pytest.approx(0.7, abs=0.01)
        assert results[2].similarity == pytest.approx(0.5, abs=0.01)

    def test_search_embedding_generation_error(self, mock_embedder, mock_vector_store):
        """Test error handling when embedding generation fails."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        # Mock embedding generation to raise an error
        mock_embedder.generate_embedding.side_effect = Exception("API error")

        with pytest.raises(RuntimeError, match="Query embedding generation failed"):
            retriever.search(query="test", top_k=5)

    def test_search_vector_search_error(self, mock_embedder, mock_vector_store):
        """Test error handling when vector search fails."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        # Mock vector search to raise an error
        mock_vector_store.search_similar_vectors.side_effect = Exception(
            "Database error"
        )

        with pytest.raises(RuntimeError, match="Vector search failed"):
            retriever.search(query="test", top_k=5)

    def test_batch_search(self, mock_embedder, mock_vector_store):
        """Test batch search functionality."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        queries = ["password reset", "VPN connection", "email setup"]
        results = retriever.batch_search(queries=queries, top_k=2)

        assert isinstance(results, dict)
        assert len(results) == len(queries)
        assert all(query in results for query in queries)
        assert all(isinstance(results[q], list) for q in queries)
        assert all(len(results[q]) == 2 for q in queries)

    def test_batch_search_empty(self, mock_embedder, mock_vector_store):
        """Test batch search with empty query list."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        results = retriever.batch_search(queries=[], top_k=5)

        assert results == {}

    def test_batch_search_error_handling(self, mock_embedder, mock_vector_store):
        """Test batch search continues on individual query errors."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        # Make embedding generation fail for specific query
        def mock_generate(query):
            if query == "bad_query":
                raise Exception("Error")
            return [0.1] * 3072

        mock_embedder.generate_embedding.side_effect = mock_generate

        queries = ["good_query", "bad_query", "another_good_query"]
        results = retriever.batch_search(queries=queries, top_k=2)

        # Should have results for all queries, bad one returns empty list
        assert len(results) == 3
        assert len(results["good_query"]) == 2
        assert len(results["bad_query"]) == 0
        assert len(results["another_good_query"]) == 2

    def test_get_stats(self, mock_embedder, mock_vector_store):
        """Test retriever statistics."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        stats = retriever.get_stats()

        assert isinstance(stats, dict)
        assert stats["num_embeddings"] == 3
        assert stats["embedding_dimension"] == 3072
        assert stats["model"] == "text-embedding-3-large"
        assert stats["provider"] == "openai"

    def test_get_stats_error_handling(self, mock_embedder, mock_vector_store):
        """Test get_stats handles errors gracefully."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        # Mock error in getting count
        mock_vector_store.get_count.side_effect = Exception("DB error")

        stats = retriever.get_stats()

        # Should return stats with error field
        assert stats["num_embeddings"] == 0
        assert "error" in stats
        assert stats["embedding_dimension"] == 3072

    def test_find_similar_to_chunk(
        self, mock_embedder, mock_vector_store, sample_chunks
    ):
        """Test finding similar chunks to a given chunk."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        chunk_id = str(sample_chunks[0].chunk_id)

        # Mock search_similar_vectors to return chunks including the query chunk
        def mock_search(query_vector, limit, min_similarity=0.0, status_names=None, category_names=None, is_public=None, tags=None):
            results = [
                {
                    "chunk_id": sample_chunks[0].chunk_id,
                    "parent_article_id": sample_chunks[0].parent_article_id,
                    "chunk_sequence": sample_chunks[0].chunk_sequence,
                    "text_content": sample_chunks[0].text_content,
                    "token_count": sample_chunks[0].token_count,
                    "source_url": sample_chunks[0].source_url,
                    "created_at": sample_chunks[0].last_modified_date,
                    "similarity": 1.0,  # The query chunk itself
                },
                {
                    "chunk_id": sample_chunks[1].chunk_id,
                    "parent_article_id": sample_chunks[1].parent_article_id,
                    "chunk_sequence": sample_chunks[1].chunk_sequence,
                    "text_content": sample_chunks[1].text_content,
                    "token_count": sample_chunks[1].token_count,
                    "source_url": sample_chunks[1].source_url,
                    "created_at": sample_chunks[1].last_modified_date,
                    "similarity": 0.8,
                },
                {
                    "chunk_id": sample_chunks[2].chunk_id,
                    "parent_article_id": sample_chunks[2].parent_article_id,
                    "chunk_sequence": sample_chunks[2].chunk_sequence,
                    "text_content": sample_chunks[2].text_content,
                    "token_count": sample_chunks[2].token_count,
                    "source_url": sample_chunks[2].source_url,
                    "created_at": sample_chunks[2].last_modified_date,
                    "similarity": 0.6,
                },
            ]
            return results[:limit]

        mock_vector_store.search_similar_vectors.side_effect = mock_search

        results = retriever.find_similar_to_chunk(chunk_id=chunk_id, top_k=2)

        # Should exclude the query chunk itself
        assert len(results) == 2
        assert all(str(r.chunk.chunk_id) != chunk_id for r in results)

    def test_find_similar_to_chunk_not_found(self, mock_embedder, mock_vector_store):
        """Test find_similar_to_chunk with non-existent chunk ID."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        # Mock get_embedding_by_chunk_id to return None
        mock_vector_store.get_embedding_by_chunk_id.return_value = None

        with pytest.raises(ValueError, match="Chunk ID not found"):
            retriever.find_similar_to_chunk(chunk_id="nonexistent", top_k=5)

    def test_find_similar_to_chunk_with_min_similarity(
        self, mock_embedder, mock_vector_store, sample_chunks
    ):
        """Test find_similar_to_chunk with minimum similarity filtering."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        chunk_id = str(sample_chunks[0].chunk_id)

        # Mock search results
        def mock_search(query_vector, limit, min_similarity=0.0, status_names=None, category_names=None, is_public=None, tags=None):
            results = [
                {
                    "chunk_id": sample_chunks[0].chunk_id,
                    "parent_article_id": sample_chunks[0].parent_article_id,
                    "chunk_sequence": sample_chunks[0].chunk_sequence,
                    "text_content": sample_chunks[0].text_content,
                    "token_count": sample_chunks[0].token_count,
                    "source_url": sample_chunks[0].source_url,
                    "created_at": sample_chunks[0].last_modified_date,
                    "similarity": 1.0,  # Query chunk
                },
                {
                    "chunk_id": sample_chunks[1].chunk_id,
                    "parent_article_id": sample_chunks[1].parent_article_id,
                    "chunk_sequence": sample_chunks[1].chunk_sequence,
                    "text_content": sample_chunks[1].text_content,
                    "token_count": sample_chunks[1].token_count,
                    "source_url": sample_chunks[1].source_url,
                    "created_at": sample_chunks[1].last_modified_date,
                    "similarity": 0.8,
                },
                {
                    "chunk_id": sample_chunks[2].chunk_id,
                    "parent_article_id": sample_chunks[2].parent_article_id,
                    "chunk_sequence": sample_chunks[2].chunk_sequence,
                    "text_content": sample_chunks[2].text_content,
                    "token_count": sample_chunks[2].token_count,
                    "source_url": sample_chunks[2].source_url,
                    "created_at": sample_chunks[2].last_modified_date,
                    "similarity": 0.4,
                },
            ]
            # Filter by min_similarity
            filtered = [r for r in results if r["similarity"] >= min_similarity]
            return filtered[:limit]

        mock_vector_store.search_similar_vectors.side_effect = mock_search

        results = retriever.find_similar_to_chunk(
            chunk_id=chunk_id, top_k=5, min_similarity=0.5
        )

        # Should only return chunks with similarity >= 0.5 and excluding self
        assert len(results) == 1
        assert results[0].similarity >= 0.5

    def test_result_to_dict(self, sample_chunks):
        """Test VectorSearchResult.to_dict()."""
        result = VectorSearchResult(chunk=sample_chunks[0], similarity=0.95, rank=1)

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["rank"] == 1
        assert result_dict["similarity"] == 0.95
        assert "chunk_id" in result_dict
        assert "text_content" in result_dict
        assert "source_url" in result_dict
        assert "parent_article_id" in result_dict

    def test_context_manager(self, mock_embedder, mock_vector_store):
        """Test VectorRetriever as context manager."""
        with VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        ) as retriever:
            assert retriever is not None

        # Should call close on exit
        mock_vector_store.close.assert_called_once()

    def test_close(self, mock_embedder, mock_vector_store):
        """Test explicit close() method."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        retriever.close()

        mock_vector_store.close.assert_called_once()

    def test_close_error_handling(self, mock_embedder, mock_vector_store):
        """Test close() handles errors gracefully."""
        retriever = VectorRetriever(
            embedding_generator=mock_embedder, vector_storage=mock_vector_store
        )

        # Mock close to raise an error
        mock_vector_store.close.side_effect = Exception("Close error")

        # Should not raise, just log the error
        retriever.close()
