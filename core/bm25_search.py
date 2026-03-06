"""
BM25 Search Module

Provides sparse keyword-based retrieval using the BM25 (Best Matching 25) algorithm.
BM25 is effective for keyword matching and can be used alongside or as an alternative
to dense vector search for hybrid retrieval strategies.

Key Features:
- Fast keyword-based search using BM25 ranking
- Configurable parameters (k1, b)
- Integration with PostgreSQL chunk storage
- Returns results with relevance scores and metadata
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

from rank_bm25 import BM25Okapi

from core.storage_chunk import PostgresClient
from core.schemas import TextChunk
from utils.logger import get_logger, PerformanceLogger

logger = get_logger(__name__)


@dataclass
class BM25SearchResult:
    """
    Represents a single search result from BM25 retrieval.

    Attributes:
        chunk: The TextChunk object
        score: BM25 relevance score
        rank: Position in the ranked results (1-indexed)
        system_prompt: Optional system prompt resolved from article tags
    """

    chunk: TextChunk
    score: float
    rank: int
    system_prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary with metadata."""
        return {
            "rank": self.rank,
            "score": self.score,
            "chunk_id": str(self.chunk.chunk_id),
            "parent_article_id": str(self.chunk.parent_article_id),
            "chunk_sequence": self.chunk.chunk_sequence,
            "text_content": self.chunk.text_content,
            "token_count": self.chunk.token_count,
            "source_url": str(self.chunk.source_url),
            "last_modified_date": self.chunk.last_modified_date.isoformat(),
        }


class BM25Retriever:
    """
    BM25-based text retrieval system for sparse keyword search.

    BM25 (Best Matching 25) is a probabilistic ranking function that ranks
    documents based on query term frequency, document length, and term rarity.

    This implementation uses the rank-bm25 library for efficient BM25 scoring.

    Algorithm:
        score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

    Where:
        - D: document
        - Q: query
        - f(qi, D): frequency of term qi in document D
        - |D|: length of document D
        - avgdl: average document length in collection
        - k1: term frequency saturation parameter (typically 1.2-2.0)
        - b: length normalization parameter (typically 0.75)
        - IDF(qi): inverse document frequency of term qi
    """

    def __init__(
        self,
        postgres_client: Optional[PostgresClient] = None,
        k1: float = 1.5,
        b: float = 0.75,
        use_cache: bool = True,
    ):
        """
        Initialize BM25 retriever.

        Args:
            postgres_client: PostgreSQL client for fetching chunks (creates new if None)
            k1: Term frequency saturation parameter (default: 1.5)
                Higher values give more weight to term frequency
            b: Length normalization parameter (default: 0.75)
               0 = no length normalization, 1 = full normalization
            use_cache: Whether to cache corpus and BM25 model (default: True)

        Raises:
            ValueError: If k1 or b are out of valid ranges
        """
        logger.info("Initializing BM25Retriever")

        # Validate parameters
        if k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b must be between 0 and 1, got {b}")

        self.k1 = k1
        self.b = b
        self.use_cache = use_cache

        # Initialize PostgreSQL client
        self.db_client = postgres_client or PostgresClient()
        logger.debug("Database client initialized")

        # Cache for corpus and BM25 model
        self._chunks_cache: Optional[List[TextChunk]] = None
        self._bm25_model: Optional[BM25Okapi] = None
        self._tokenized_corpus: Optional[List[List[str]]] = None

        logger.info(
            f"BM25Retriever initialized with k1={k1}, b={b}, use_cache={use_cache}"
        )

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text into lowercase terms.

        Uses simple whitespace and punctuation-based tokenization.
        For production, consider more sophisticated tokenization (stemming, lemmatization).

        Args:
            text: Input text to tokenize

        Returns:
            List of lowercase tokens
        """
        # Convert to lowercase and split on whitespace/punctuation
        # Keep alphanumeric characters and common technical symbols
        tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
        return tokens

    def _get_chunks(
        self,
        refresh: bool = False,
        status_names: Optional[List[str]] = None,
        category_names: Optional[List[str]] = None,
        is_public: Optional[bool] = None,
        tags: Optional[List[str]] = None,
    ) -> List[TextChunk]:
        """
        Get chunks from database with optional filtering and caching.

        Caching strategy: Only cache the default corpus (status_names=["Approved"], no other filters).
        Filtered queries bypass the cache to avoid cache explosion.

        Args:
            refresh: Force refresh from database even if cached
            status_names: Filter by article status
            category_names: Filter by article category
            is_public: Filter by article visibility
            tags: Filter by article tags (ANY match)

        Returns:
            List of TextChunk objects with optional article_tags field
        """
        # Check if using non-default filters
        has_filters = (
            (status_names is not None and status_names != ["Approved"])
            or category_names is not None
            or is_public is not None
            or tags is not None
        )

        # Use cache only for default corpus (no filters or default status filter)
        if (
            self.use_cache
            and not refresh
            and not has_filters
            and self._chunks_cache is not None
        ):
            logger.debug(f"Using cached chunks ({len(self._chunks_cache)} chunks)")
            return self._chunks_cache

        # Fetch chunks from database
        logger.debug(
            f"Fetching chunks from database with filters: "
            f"status={status_names}, category={category_names}, is_public={is_public}, tags={tags}"
        )

        with PerformanceLogger(logger, "Fetch chunks"):
            # Use filtered query if any filters are provided
            if has_filters or status_names is not None:
                chunks = self.db_client.get_all_chunks_filtered(
                    status_names=status_names,
                    category_names=category_names,
                    is_public=is_public,
                    tags=tags,
                )
            else:
                # No filters - use standard query (for backward compatibility)
                chunks = self.db_client.get_all_chunks()

        # Only cache default corpus (no custom filters)
        if self.use_cache and not has_filters:
            self._chunks_cache = chunks
            logger.debug(f"Cached {len(chunks)} chunks (default corpus)")
        else:
            logger.debug(f"Not caching filtered corpus ({len(chunks)} chunks)")

        logger.info(
            f"Retrieved {len(chunks)} chunks from database "
            f"(filters: status={status_names}, category={category_names}, "
            f"is_public={is_public}, tags={tags})"
        )
        return chunks

    def _build_bm25_model(
        self, chunks: List[TextChunk], refresh: bool = False, use_cache: bool = True
    ) -> BM25Okapi:
        """
        Build or retrieve cached BM25 model with tag search support.

        For chunks with article_tags, tags are appended to text content for tokenization.
        This makes tags searchable in BM25 queries.

        Args:
            chunks: List of text chunks to index
            refresh: Force rebuild even if cached
            use_cache: Whether to use cached model (False for filtered queries)

        Returns:
            BM25Okapi model
        """
        if (
            use_cache
            and self.use_cache
            and not refresh
            and self._bm25_model is not None
            and self._tokenized_corpus is not None
        ):
            logger.debug("Using cached BM25 model")
            return self._bm25_model

        if not chunks:
            logger.warning("No chunks provided — cannot build BM25 model")
            raise ValueError("Cannot build BM25 model with an empty corpus")

        logger.info(f"Building BM25 model for {len(chunks)} chunks")

        with PerformanceLogger(logger, "Build BM25 model"):
            # Tokenize all chunks, including tags if present
            self._tokenized_corpus = []
            for chunk in chunks:
                # Start with chunk text content
                text = chunk.text_content

                # Append tags if present (makes tags searchable in BM25)
                if chunk.article_tags:
                    tag_text = " ".join(chunk.article_tags)
                    text = f"{text} {tag_text}"
                    logger.debug(
                        f"Appended {len(chunk.article_tags)} tags to chunk {chunk.chunk_id}"
                    )

                # Tokenize combined text
                tokens = self.tokenize(text)
                self._tokenized_corpus.append(tokens)

            # Build BM25 model with custom parameters
            self._bm25_model = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)

        logger.info(
            f"BM25 model built with {len(chunks)} documents, "
            f"avg_doc_length={self._bm25_model.avgdl:.1f} "
            f"(tags included in tokenization)"
        )

        return self._bm25_model

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: Optional[float] = None,
        refresh_corpus: bool = False,
        status_names: Optional[List[str]] = None,
        category_names: Optional[List[str]] = None,
        is_public: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        include_system_prompts: bool = True,
    ) -> List[BM25SearchResult]:
        """
        Search for relevant chunks using BM25 ranking with optional article metadata filtering.

        Tags from articles are included in the searchable corpus, so queries can match on tag keywords.

        Filters are applied using AND logic. Within list filters (status_names, category_names, tags),
        OR logic is used (ANY match). For tags, articles must have at least one of the provided tags.

        Args:
            query: Search query string
            top_k: Number of top results to return (default: 10)
            min_score: Minimum BM25 score threshold (default: None, no filtering)
            refresh_corpus: Force refresh corpus statistics from database
            status_names: Filter by article status (e.g., ['Approved', 'Published'])
            category_names: Filter by article category (e.g., ['IT Help', 'Documentation'])
            is_public: Filter by article visibility (True for public, False for private)
            tags: Filter by article tags (ANY match using array overlap operator)
            include_system_prompts: Include system prompts resolved from article tags (default: True)

        Returns:
            List of BM25SearchResult objects, ranked by relevance, with optional system_prompt field

        Raises:
            ValueError: If query is empty or top_k is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        logger.info(
            f"Searching for: '{query}' (top_k={top_k}, filters: "
            f"status={status_names}, category={category_names}, is_public={is_public}, tags={tags})"
        )

        with PerformanceLogger(logger, f"BM25 search for '{query}'"):
            # Get chunks with filtering
            chunks = self._get_chunks(
                refresh=refresh_corpus,
                status_names=status_names,
                category_names=category_names,
                is_public=is_public,
                tags=tags,
            )

            if not chunks:
                logger.warning("No chunks found in database (after filtering)")
                return []

            # Determine if we should use cache for BM25 model
            # Only cache for default corpus (no custom filters)
            has_filters = (
                (status_names is not None and status_names != ["Approved"])
                or category_names is not None
                or is_public is not None
                or tags is not None
            )
            use_cache = not has_filters

            # Build/retrieve BM25 model (with tags included in tokenization)
            bm25 = self._build_bm25_model(
                chunks, refresh=refresh_corpus, use_cache=use_cache
            )

            # Tokenize query
            query_tokens = self.tokenize(query)
            logger.debug(f"Query tokens: {query_tokens}")

            if not query_tokens:
                logger.warning("Query produced no tokens after tokenization")
                return []

            # Get BM25 scores for all documents
            scores = bm25.get_scores(query_tokens)
            logger.debug(f"Computed scores for {len(scores)} documents")

            # Create (score, index) pairs and filter by min_score
            score_idx_pairs = []
            for idx, score in enumerate(scores):
                if min_score is None or score >= min_score:
                    score_idx_pairs.append((score, idx))

            # Sort by score (descending) and take top_k
            score_idx_pairs.sort(reverse=True, key=lambda x: x[0])
            top_results = score_idx_pairs[:top_k]

            # Create result objects
            results = [
                BM25SearchResult(
                    chunk=chunks[idx],
                    score=score,
                    rank=rank,
                )
                for rank, (score, idx) in enumerate(top_results, start=1)
            ]

            # Fetch system prompts if requested (post-processing batch fetch)
            if include_system_prompts and results:
                from core.storage_prompt import PromptStorageClient

                try:
                    article_ids = list(
                        {str(r.chunk.parent_article_id) for r in results}
                    )
                    logger.debug(
                        f"Fetching system prompts for {len(article_ids)} articles"
                    )

                    prompt_client = PromptStorageClient()
                    prompts = prompt_client.get_prompts_for_article_ids(article_ids)

                    # Attach prompts to results
                    for result in results:
                        article_id = str(result.chunk.parent_article_id)
                        result.system_prompt = prompts.get(article_id)

                    logger.debug(f"Attached system prompts to {len(results)} results")
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch system prompts: {e}. Continuing without prompts."
                    )
                    # Continue without prompts (graceful degradation)

        logger.info(
            f"Found {len(results)} results (filtered from {len(score_idx_pairs)} candidates)"
        )

        if results:
            logger.debug(
                f"Top result: score={results[0].score:.4f}, "
                f"chunk_id={results[0].chunk.chunk_id}"
            )

        return results

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
        min_score: Optional[float] = None,
        status_names: Optional[List[str]] = None,
        category_names: Optional[List[str]] = None,
        is_public: Optional[bool] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, List[BM25SearchResult]]:
        """
        Perform batch search for multiple queries efficiently with optional filtering.

        Builds BM25 model once and reuses for all queries.

        Args:
            queries: List of query strings
            top_k: Number of top results per query (default: 10)
            min_score: Minimum BM25 score threshold (default: None)
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
            f"Performing batch search for {len(queries)} queries with filters: "
            f"status={status_names}, category={category_names}, is_public={is_public}, tags={tags}"
        )

        # Pre-build BM25 model once with filtering
        chunks = self._get_chunks(
            status_names=status_names,
            category_names=category_names,
            is_public=is_public,
            tags=tags,
        )
        if not chunks:
            logger.warning("No chunks found in database (after filtering)")
            return {query: [] for query in queries}

        # Determine cache usage
        has_filters = (
            (status_names is not None and status_names != ["Approved"])
            or category_names is not None
            or is_public is not None
            or tags is not None
        )
        use_cache = not has_filters

        bm25 = self._build_bm25_model(chunks, use_cache=use_cache)

        results = {}
        with PerformanceLogger(logger, f"Batch search for {len(queries)} queries"):
            for query in queries:
                try:
                    results[query] = self.search(
                        query=query,
                        top_k=top_k,
                        min_score=min_score,
                        refresh_corpus=False,  # Reuse cached model
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

    def clear_cache(self):
        """Clear cached corpus and BM25 model."""
        logger.info("Clearing BM25 cache")
        self._chunks_cache = None
        self._bm25_model = None
        self._tokenized_corpus = None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current retriever statistics.

        Returns:
            Dictionary with statistics about the retriever state
        """
        chunks = self._get_chunks()

        # Build model if not cached to get accurate stats
        bm25 = self._build_bm25_model(chunks)

        stats = {
            "num_chunks": len(chunks),
            "k1": self.k1,
            "b": self.b,
            "cache_enabled": self.use_cache,
            "is_cached": self._bm25_model is not None,
            "avg_doc_length": bm25.avgdl,
            "num_unique_terms": len(bm25.idf),
        }

        return stats
