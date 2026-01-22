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

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math
import re
from collections import Counter, defaultdict

from core.storage_raw import PostgresClient
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
    """
    chunk: TextChunk
    score: float
    rank: int

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
            use_cache: Whether to cache corpus statistics (default: True)

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

        # Cache for corpus statistics
        self._corpus_cache: Optional[Dict[str, Any]] = None
        self._chunks_cache: Optional[List[TextChunk]] = None

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
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return tokens

    def _get_chunks(self, refresh: bool = False) -> List[TextChunk]:
        """
        Get all chunks from database with optional caching.

        Args:
            refresh: Force refresh from database even if cached

        Returns:
            List of TextChunk objects
        """
        if self.use_cache and not refresh and self._chunks_cache is not None:
            logger.debug(f"Using cached chunks ({len(self._chunks_cache)} chunks)")
            return self._chunks_cache

        logger.debug("Fetching chunks from database")
        with PerformanceLogger(logger, "Fetch all chunks"):
            chunks = self.db_client.get_all_chunks()

        if self.use_cache:
            self._chunks_cache = chunks
            logger.debug(f"Cached {len(chunks)} chunks")

        logger.info(f"Retrieved {len(chunks)} chunks from database")
        return chunks

    def _compute_corpus_statistics(
        self, chunks: List[TextChunk], refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Compute and cache corpus-level statistics for BM25.

        Statistics include:
        - Document frequencies for each term
        - Average document length
        - Total number of documents

        Args:
            chunks: List of text chunks to analyze
            refresh: Force recomputation even if cached

        Returns:
            Dictionary containing corpus statistics
        """
        if self.use_cache and not refresh and self._corpus_cache is not None:
            logger.debug("Using cached corpus statistics")
            return self._corpus_cache

        logger.info(f"Computing corpus statistics for {len(chunks)} chunks")

        with PerformanceLogger(logger, "Compute corpus statistics"):
            # Document frequency: number of documents containing each term
            doc_freq: Dict[str, int] = defaultdict(int)

            # Document lengths
            doc_lengths: List[int] = []

            # Tokenized documents for later use
            tokenized_docs: List[List[str]] = []

            for chunk in chunks:
                tokens = self.tokenize(chunk.text_content)
                tokenized_docs.append(tokens)
                doc_lengths.append(len(tokens))

                # Count unique terms in this document
                unique_terms = set(tokens)
                for term in unique_terms:
                    doc_freq[term] += 1

            # Compute statistics
            num_docs = len(chunks)
            avg_doc_length = sum(doc_lengths) / num_docs if num_docs > 0 else 0

            # Compute IDF for each term
            # IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5))
            # where N = total docs, n(qi) = docs containing qi
            idf: Dict[str, float] = {}
            for term, freq in doc_freq.items():
                idf[term] = math.log(
                    (num_docs - freq + 0.5) / (freq + 0.5) + 1.0
                )

            stats = {
                "num_docs": num_docs,
                "avg_doc_length": avg_doc_length,
                "doc_freq": doc_freq,
                "idf": idf,
                "doc_lengths": doc_lengths,
                "tokenized_docs": tokenized_docs,
            }

        if self.use_cache:
            self._corpus_cache = stats
            logger.debug("Cached corpus statistics")

        logger.info(
            f"Corpus statistics: {num_docs} docs, "
            f"avg_length={avg_doc_length:.1f}, "
            f"{len(idf)} unique terms"
        )

        return stats

    def _compute_bm25_score(
        self,
        query_terms: List[str],
        doc_tokens: List[str],
        doc_length: int,
        avg_doc_length: float,
        idf: Dict[str, float],
    ) -> float:
        """
        Compute BM25 score for a single document given a query.

        Args:
            query_terms: Tokenized query terms
            doc_tokens: Tokenized document terms
            doc_length: Length of the document (number of tokens)
            avg_doc_length: Average document length in corpus
            idf: IDF scores for all terms

        Returns:
            BM25 relevance score
        """
        # Count term frequencies in document
        term_freq = Counter(doc_tokens)

        score = 0.0
        for term in query_terms:
            if term not in idf:
                # Term not in corpus, skip
                continue

            # Get term frequency in document
            tf = term_freq.get(term, 0)

            # Compute BM25 component for this term
            # score = IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / avg_doc_length)
            )

            score += idf[term] * (numerator / denominator)

        return score

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: Optional[float] = None,
        refresh_corpus: bool = False,
    ) -> List[BM25SearchResult]:
        """
        Search for relevant chunks using BM25 ranking.

        Args:
            query: Search query string
            top_k: Number of top results to return (default: 10)
            min_score: Minimum BM25 score threshold (default: None, no filtering)
            refresh_corpus: Force refresh corpus statistics from database

        Returns:
            List of BM25SearchResult objects, ranked by relevance

        Raises:
            ValueError: If query is empty or top_k is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        logger.info(f"Searching for: '{query}' (top_k={top_k})")

        with PerformanceLogger(logger, f"BM25 search for '{query}'"):
            # Get chunks
            chunks = self._get_chunks(refresh=refresh_corpus)

            if not chunks:
                logger.warning("No chunks found in database")
                return []

            # Compute corpus statistics
            stats = self._compute_corpus_statistics(chunks, refresh=refresh_corpus)

            # Tokenize query
            query_terms = self.tokenize(query)
            logger.debug(f"Query tokens: {query_terms}")

            if not query_terms:
                logger.warning("Query produced no tokens after tokenization")
                return []

            # Score all documents
            scores: List[Tuple[float, int]] = []
            for idx, (chunk, doc_tokens, doc_length) in enumerate(
                zip(chunks, stats["tokenized_docs"], stats["doc_lengths"])
            ):
                score = self._compute_bm25_score(
                    query_terms=query_terms,
                    doc_tokens=doc_tokens,
                    doc_length=doc_length,
                    avg_doc_length=stats["avg_doc_length"],
                    idf=stats["idf"],
                )

                # Apply minimum score filter
                if min_score is None or score >= min_score:
                    scores.append((score, idx))

            # Sort by score (descending) and take top_k
            scores.sort(reverse=True, key=lambda x: x[0])
            top_scores = scores[:top_k]

            # Create result objects
            results = [
                BM25SearchResult(
                    chunk=chunks[idx],
                    score=score,
                    rank=rank,
                )
                for rank, (score, idx) in enumerate(top_scores, start=1)
            ]

        logger.info(
            f"Found {len(results)} results (filtered from {len(scores)} candidates)"
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
    ) -> Dict[str, List[BM25SearchResult]]:
        """
        Perform batch search for multiple queries efficiently.

        Computes corpus statistics once and reuses for all queries.

        Args:
            queries: List of query strings
            top_k: Number of top results per query (default: 10)
            min_score: Minimum BM25 score threshold (default: None)

        Returns:
            Dictionary mapping query -> list of search results
        """
        if not queries:
            logger.warning("No queries provided for batch search")
            return {}

        logger.info(f"Performing batch search for {len(queries)} queries")

        # Pre-compute corpus statistics once
        chunks = self._get_chunks()
        stats = self._compute_corpus_statistics(chunks)

        results = {}
        with PerformanceLogger(logger, f"Batch search for {len(queries)} queries"):
            for query in queries:
                try:
                    results[query] = self.search(
                        query=query,
                        top_k=top_k,
                        min_score=min_score,
                        refresh_corpus=False,  # Reuse cached statistics
                    )
                except Exception as e:
                    logger.error(f"Error searching for '{query}': {str(e)}")
                    results[query] = []

        logger.info(f"Batch search completed for {len(results)} queries")
        return results

    def clear_cache(self):
        """Clear cached corpus statistics and chunks."""
        logger.info("Clearing BM25 cache")
        self._corpus_cache = None
        self._chunks_cache = None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current retriever statistics.

        Returns:
            Dictionary with statistics about the retriever state
        """
        chunks = self._get_chunks()
        stats = self._compute_corpus_statistics(chunks)

        return {
            "num_chunks": stats["num_docs"],
            "avg_doc_length": stats["avg_doc_length"],
            "num_unique_terms": len(stats["idf"]),
            "k1": self.k1,
            "b": self.b,
            "cache_enabled": self.use_cache,
            "is_cached": self._corpus_cache is not None,
        }
