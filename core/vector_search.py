"""
Vector Similarity Search Module

Provides dense semantic retrieval using vector embeddings stored in pgvector.
Uses cosine similarity to find semantically similar chunks based on query embeddings.

Key Features:
- Semantic search using OpenAI embeddings
- Efficient similarity search with pgvector's <=> operator
- Configurable similarity thresholds
- Integration with PostgreSQL vector storage
- Returns results with similarity scores and metadata
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from core.storage_vector import OpenAIVectorStorage
from core.embedding import GenerateEmbeddingsOpenAI
from core.schemas import TextChunk
from utils.logger import get_logger, PerformanceLogger

logger = get_logger(__name__)


@dataclass
class VectorSearchResult:
    """
    Represents a single search result from vector similarity search.

    Attributes:
        chunk: The TextChunk object
        similarity: Cosine similarity score (0-1, higher is more similar)
        rank: Position in the ranked results (1-indexed)
        system_prompt: Optional system prompt resolved from article tags
    """

    chunk: TextChunk
    similarity: float
    rank: int
    system_prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary with metadata."""
        return {
            "rank": self.rank,
            "similarity": self.similarity,
            "chunk_id": str(self.chunk.chunk_id),
            "parent_article_id": str(self.chunk.parent_article_id),
            "chunk_sequence": self.chunk.chunk_sequence,
            "text_content": self.chunk.text_content,
            "token_count": self.chunk.token_count,
            "source_url": str(self.chunk.source_url),
            "last_modified_date": self.chunk.last_modified_date.isoformat(),
        }


class VectorRetriever:
    """
    Vector-based semantic search using OpenAI embeddings and pgvector.

    Uses cosine similarity to find semantically related chunks by comparing
    query embeddings with stored chunk embeddings in PostgreSQL.

    Algorithm:
        1. Generate embedding for query using OpenAI API
        2. Query pgvector using cosine distance operator (<=>)
        3. Return top-k most similar chunks with similarity scores

    Similarity Score:
        - Cosine similarity ranges from 0 (dissimilar) to 1 (identical)
        - Calculated as: 1 - cosine_distance
    """

    def __init__(
        self,
        embedding_generator: Optional[GenerateEmbeddingsOpenAI] = None,
        vector_storage: Optional[OpenAIVectorStorage] = None,
    ):
        """
        Initialize vector retriever.

        Args:
            embedding_generator: OpenAI embedding generator (creates new if None)
            vector_storage: Vector storage client (creates new if None)
        """
        logger.info("Initializing VectorRetriever")

        # Initialize embedding generator
        self.embedder = embedding_generator or GenerateEmbeddingsOpenAI()
        logger.debug("Embedding generator initialized")

        # Initialize vector storage
        self.vector_store = vector_storage or OpenAIVectorStorage()
        logger.debug("Vector storage client initialized")

        logger.info("VectorRetriever initialized successfully")

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: Optional[float] = None,
        status_names: Optional[List[str]] = None,
        category_names: Optional[List[str]] = None,
        is_public: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        include_system_prompts: bool = True,
    ) -> List[VectorSearchResult]:
        """
        Search for semantically similar chunks using vector similarity with optional article metadata filtering.

        Filters are applied using AND logic. Within list filters (status_names, category_names, tags),
        OR logic is used (ANY match). For tags, articles must have at least one of the provided tags.

        Args:
            query: Search query string
            top_k: Number of top results to return (default: 10)
            min_similarity: Minimum cosine similarity threshold (0-1, default: None)
            status_names: Filter by article status (e.g., ['Approved', 'Published'])
            category_names: Filter by article category (e.g., ['IT Help', 'Documentation'])
            is_public: Filter by article visibility (True for public, False for private)
            tags: Filter by article tags (ANY match using array overlap operator)
            include_system_prompts: Include system prompts resolved from article tags (default: True)

        Returns:
            List of VectorSearchResult objects, ranked by similarity, with optional system_prompt field

        Raises:
            ValueError: If query is empty or top_k is invalid
            RuntimeError: If embedding generation or search fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if min_similarity is not None and not 0 <= min_similarity <= 1:
            raise ValueError(
                f"min_similarity must be between 0 and 1, got {min_similarity}"
            )

        logger.info(
            f"Searching for: '{query}' (top_k={top_k}, filters: "
            f"status={status_names}, category={category_names}, is_public={is_public}, tags={tags})"
        )

        with PerformanceLogger(logger, f"Vector search for '{query}'"):
            # Generate embedding for query
            logger.debug("Generating query embedding")
            try:
                query_embedding = self.embedder.generate_embedding(query)
            except Exception as e:
                logger.error(f"Failed to generate query embedding: {str(e)}")
                raise RuntimeError(
                    f"Query embedding generation failed: {str(e)}"
                ) from e

            logger.debug(f"Generated embedding with dimension {len(query_embedding)}")

            # Search for similar vectors with filtering
            logger.debug(f"Searching for top {top_k} similar vectors")
            try:
                # Apply similarity threshold if specified
                # Note: vector_store.search_similar_vectors returns dict results
                min_sim_threshold = (
                    min_similarity if min_similarity is not None else 0.0
                )
                db_results = self.vector_store.search_similar_vectors(
                    query_vector=query_embedding,
                    limit=top_k,
                    min_similarity=min_sim_threshold,
                    status_names=status_names,
                    category_names=category_names,
                    is_public=is_public,
                    tags=tags,
                    include_system_prompts=include_system_prompts,
                )
            except Exception as e:
                logger.error(f"Vector search failed: {str(e)}")
                raise RuntimeError(f"Vector search failed: {str(e)}") from e

            logger.debug(f"Retrieved {len(db_results)} results from database")

            # Convert dict results to TextChunk objects with similarities
            results = []
            for result_dict in db_results:
                # Create TextChunk from dict
                chunk = TextChunk(
                    chunk_id=result_dict["chunk_id"],
                    parent_article_id=result_dict["parent_article_id"],
                    chunk_sequence=result_dict["chunk_sequence"],
                    text_content=result_dict["text_content"],
                    token_count=result_dict["token_count"],
                    source_url=result_dict["source_url"],
                    last_modified_date=result_dict.get(
                        "created_at"
                    ),  # Use created_at as last_modified
                )
                similarity = result_dict["similarity"]
                system_prompt = result_dict.get("system_prompt")
                results.append((similarity, chunk, system_prompt))

            # Sort by similarity (descending) and take top_k
            results.sort(reverse=True, key=lambda x: x[0])
            results = results[:top_k]

            # Create result objects with ranks
            search_results = [
                VectorSearchResult(
                    chunk=chunk,
                    similarity=similarity,
                    rank=rank,
                    system_prompt=system_prompt,
                )
                for rank, (similarity, chunk, system_prompt) in enumerate(results, start=1)
            ]

        logger.info(
            f"Found {len(search_results)} results "
            f"(filtered from {len(db_results)} candidates)"
        )

        if search_results:
            logger.debug(
                f"Top result: similarity={search_results[0].similarity:.4f}, "
                f"chunk_id={search_results[0].chunk.chunk_id}"
            )

        return search_results

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
        min_similarity: Optional[float] = None,
        status_names: Optional[List[str]] = None,
        category_names: Optional[List[str]] = None,
        is_public: Optional[bool] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, List[VectorSearchResult]]:
        """
        Perform batch search for multiple queries with optional filtering.

        Args:
            queries: List of query strings
            top_k: Number of top results per query (default: 10)
            min_similarity: Minimum similarity threshold (default: None)
            status_names: Filter by article status (e.g., ['Approved', 'Published'])
            category_names: Filter by article category (e.g., ['IT Help', 'Documentation'])
            is_public: Filter by article visibility (True for public, False for private)
            tags: Filter by article tags (ANY match using array overlap operator)

        Returns:
            Dictionary mapping query -> list of search results
        """
        if not queries:
            logger.warning("No queries provided for batch search")
            return {}

        logger.info(
            f"Performing batch vector search for {len(queries)} queries with filters: "
            f"status={status_names}, category={category_names}, is_public={is_public}, tags={tags}"
        )

        results = {}
        with PerformanceLogger(logger, f"Batch search for {len(queries)} queries"):
            for query in queries:
                try:
                    results[query] = self.search(
                        query=query,
                        top_k=top_k,
                        min_similarity=min_similarity,
                        status_names=status_names,
                        category_names=category_names,
                        is_public=is_public,
                        tags=tags,
                    )
                except Exception as e:
                    logger.error(f"Error searching for '{query}': {str(e)}")
                    results[query] = []

        logger.info(f"Batch search completed for {len(results)} queries")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current retriever statistics.

        Returns:
            Dictionary with statistics about the retriever state
        """
        try:
            # Get count of embeddings in database
            count = self.vector_store.get_count()

            stats = {
                "num_embeddings": count,
                "embedding_dimension": self.embedder.expected_dim,
                "model": self.embedder.deployment_name,
                "provider": "openai",
            }

            logger.debug(f"Retriever stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {
                "num_embeddings": 0,
                "embedding_dimension": self.embedder.expected_dim,
                "model": self.embedder.deployment_name,
                "provider": "openai",
                "error": str(e),
            }

    def find_similar_to_chunk(
        self,
        chunk_id: str,
        top_k: int = 10,
        min_similarity: Optional[float] = None,
        status_names: Optional[List[str]] = None,
        category_names: Optional[List[str]] = None,
        is_public: Optional[bool] = None,
        tags: Optional[List[str]] = None,
    ) -> List[VectorSearchResult]:
        """
        Find chunks similar to a given chunk with optional article metadata filtering.

        Useful for finding related content or recommendations.

        Args:
            chunk_id: ID of the chunk to find similar content for
            top_k: Number of similar chunks to return (default: 10)
            min_similarity: Minimum similarity threshold (default: None)
            status_names: Filter by article status (e.g., ['Approved', 'Published'])
            category_names: Filter by article category (e.g., ['IT Help', 'Documentation'])
            is_public: Filter by article visibility (True for public, False for private)
            tags: Filter by article tags (ANY match using array overlap operator)

        Returns:
            List of VectorSearchResult objects, excluding the query chunk itself

        Raises:
            ValueError: If chunk_id is not found
            RuntimeError: If search fails
        """
        logger.info(
            f"Finding similar chunks to: {chunk_id} with filters: "
            f"status={status_names}, category={category_names}, is_public={is_public}, tags={tags}"
        )

        try:
            # Get the chunk's embedding from database
            chunk_embedding = self.vector_store.get_embedding_by_chunk_id(chunk_id)

            if chunk_embedding is None:
                raise ValueError(f"Chunk ID not found: {chunk_id}")

            # Search for similar vectors (excluding self)
            min_sim_threshold = min_similarity if min_similarity is not None else 0.0
            db_results = self.vector_store.search_similar_vectors(
                query_vector=chunk_embedding,
                limit=top_k + 1,  # +1 because we'll exclude the query chunk
                min_similarity=min_sim_threshold,
                status_names=status_names,
                category_names=category_names,
                is_public=is_public,
                tags=tags,
            )

            # Convert dict results and filter out the query chunk
            results = []
            for result_dict in db_results:
                # Skip the query chunk itself
                if str(result_dict["chunk_id"]) == chunk_id:
                    continue

                # Create TextChunk from dict
                chunk = TextChunk(
                    chunk_id=result_dict["chunk_id"],
                    parent_article_id=result_dict["parent_article_id"],
                    chunk_sequence=result_dict["chunk_sequence"],
                    text_content=result_dict["text_content"],
                    token_count=result_dict["token_count"],
                    source_url=result_dict["source_url"],
                    last_modified_date=result_dict.get("created_at"),
                )
                similarity = result_dict["similarity"]
                results.append((similarity, chunk))

            # Sort and limit
            results.sort(reverse=True, key=lambda x: x[0])
            results = results[:top_k]

            # Create result objects
            search_results = [
                VectorSearchResult(
                    chunk=chunk,
                    similarity=similarity,
                    rank=rank,
                )
                for rank, (similarity, chunk) in enumerate(results, start=1)
            ]

            logger.info(f"Found {len(search_results)} similar chunks")
            return search_results

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to find similar chunks: {str(e)}")
            raise RuntimeError(f"Similar chunk search failed: {str(e)}") from e

    def close(self):
        """Close database connections."""
        logger.info("Closing vector retriever connections")
        try:
            if hasattr(self, "vector_store"):
                self.vector_store.close()
            logger.info("Vector retriever connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit with cleanup."""
        self.close()
