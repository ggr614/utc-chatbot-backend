"""
Search endpoints for BM25, vector, and hybrid retrieval.

Provides three search methods:
1. BM25 - Fast keyword-based sparse retrieval
2. Vector - Semantic dense retrieval via embeddings
3. Hybrid - Combined BM25 + vector with RRF fusion and Cohere reranking

All endpoints include query logging for analytics.
"""

from fastapi import APIRouter, Depends, Request, HTTPException, status
from typing import Annotated, Optional
import time

from api.dependencies import (
    verify_api_key,
    get_bm25_retriever,
    get_vector_retriever,
    get_reranker,
    get_hyde_generator,
    get_query_log_client,
    get_reranker_log_client,
    get_hyde_log_client,
)
from api.models.requests import (
    BM25SearchRequest,
    VectorSearchRequest,
    HybridSearchRequest,
    HyDESearchRequest,
)
from api.models.responses import SearchResponse, SearchResultChunk
from api.utils.hybrid_search import hybrid_search, reciprocal_rank_fusion
from core.bm25_search import BM25Retriever
from core.vector_search import VectorRetriever
from core.reranker import CohereReranker
from core.hyde_generator import HyDEGenerator
from core.storage_query_log import QueryLogClient
from core.storage_reranker_log import RerankerLogClient
from core.storage_hyde_log import HyDELogClient
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/bm25",
    response_model=SearchResponse,
    summary="BM25 Sparse Keyword Search",
    description="Perform BM25 keyword-based sparse retrieval. Best for exact keyword matching, technical terms, and identifiers. Fast (<100ms after corpus cached).",
    tags=["Search"],
)
def search_bm25(
    request: BM25SearchRequest,
    api_key: Annotated[str, Depends(verify_api_key)],
    retriever: Annotated[BM25Retriever, Depends(get_bm25_retriever)],
    query_log_client: Annotated[QueryLogClient, Depends(get_query_log_client)],
) -> SearchResponse:
    """
    Perform BM25 keyword-based sparse retrieval.

    **Best for:**
    - Exact keyword matching
    - Technical terms and acronyms
    - Identifiers (IDs, codes, specific names)

    **Characteristics:**
    - Fast response (~50-100ms after corpus cached)
    - No API calls required (uses in-memory index)
    - Scores are unbounded (typically 0-20 range)

    **Request:**
    - `query`: Search query text (1-1000 chars)
    - `top_k`: Number of results to return (1-100, default 10)
    - `min_score`: Optional minimum BM25 score threshold
    - `user_id`: Optional user identifier for analytics

    **Response:**
    - Ranked results with BM25 scores
    - Latency in milliseconds
    - Metadata (min_score if provided)
    """
    start_time = time.time()

    logger.info(
        f"BM25 search: query='{request.query[:50]}', top_k={request.top_k}, "
        f"min_score={request.min_score}, user_id={request.user_id}"
    )

    try:
        # Perform BM25 search
        results = retriever.search(
            query=request.query, top_k=request.top_k, min_score=request.min_score
        )

        # Convert BM25SearchResult to SearchResultChunk
        result_chunks = [
            SearchResultChunk(
                rank=r.rank,
                score=r.score,
                chunk_id=r.chunk.chunk_id,
                parent_article_id=r.chunk.parent_article_id,
                chunk_sequence=r.chunk.chunk_sequence,
                text_content=r.chunk.text_content,
                token_count=r.chunk.token_count,
                source_url=r.chunk.source_url,
                last_modified_date=r.chunk.last_modified_date,
            )
            for r in results
        ]

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"BM25 search completed: {len(results)} results, {latency_ms}ms latency"
        )

        # Prepare results for logging (extract minimal data)
        results_for_logging = [
            {
                "rank": r.rank,
                "score": r.score,
                "chunk_id": r.chunk_id,
                "parent_article_id": r.parent_article_id,
            }
            for r in result_chunks
        ]

        # Log query and results to database (best-effort, don't fail request if logging fails)
        query_log_id_for_response = None
        try:
            query_log_id_for_response = query_log_client.log_query_with_results(
                raw_query=request.query,
                cache_result="miss",  # No cache implemented yet
                search_method="bm25",
                results=results_for_logging,
                latency_ms=latency_ms,
                user_id=request.user_id,
                query_embedding=None,
                command=request.command,
            )
        except Exception as e:
            logger.error(f"Query and result logging failed: {e}")
            # Don't propagate logging errors to client

        # Build response
        return SearchResponse(
            query=request.query,
            method="bm25",
            results=result_chunks,
            total_results=len(results),
            latency_ms=latency_ms,
            metadata={"min_score": request.min_score}
            if request.min_score is not None
            else {},
            query_log_id=query_log_id_for_response,
        )

    except ValueError as e:
        # Validation errors from retriever
        logger.warning(f"BM25 search validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except Exception as e:
        # Unexpected errors
        logger.error(f"BM25 search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Search failed"
        )


@router.post(
    "/vector",
    response_model=SearchResponse,
    summary="Vector Semantic Search",
    description="Perform vector-based semantic similarity search using embeddings. Best for natural language queries and conceptual similarity. Makes API call (~500ms-1s).",
    tags=["Search"],
)
def search_vector(
    request: VectorSearchRequest,
    api_key: Annotated[str, Depends(verify_api_key)],
    retriever: Annotated[VectorRetriever, Depends(get_vector_retriever)],
    query_log_client: Annotated[QueryLogClient, Depends(get_query_log_client)],
) -> SearchResponse:
    """
    Perform vector-based semantic similarity search.

    **Best for:**
    - Natural language queries
    - Conceptual similarity
    - Synonym and paraphrase matching

    **Characteristics:**
    - Semantic understanding (matches meaning, not just keywords)
    - Slower response (~500ms-1s due to embedding API call)
    - Similarity scores in 0-1 range (cosine similarity)

    **Request:**
    - `query`: Search query text (1-1000 chars)
    - `top_k`: Number of results to return (1-100, default 10)
    - `min_similarity`: Optional minimum similarity threshold (0.0-1.0)
    - `user_id`: Optional user identifier for analytics

    **Response:**
    - Ranked results with similarity scores (0-1)
    - Latency in milliseconds (includes embedding generation)
    - Metadata (min_similarity if provided)

    **Note:** Makes API call to Azure OpenAI for query embedding generation.
    """
    start_time = time.time()

    logger.info(
        f"Vector search: query='{request.query[:50]}', top_k={request.top_k}, "
        f"min_similarity={request.min_similarity}, user_id={request.user_id}"
    )

    try:
        # Perform vector search (includes embedding generation)
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
        )

        # Convert VectorSearchResult to SearchResultChunk
        result_chunks = [
            SearchResultChunk(
                rank=r.rank,
                score=r.similarity,  # Use similarity as score
                chunk_id=r.chunk.chunk_id,
                parent_article_id=r.chunk.parent_article_id,
                chunk_sequence=r.chunk.chunk_sequence,
                text_content=r.chunk.text_content,
                token_count=r.chunk.token_count,
                source_url=r.chunk.source_url,
                last_modified_date=r.chunk.last_modified_date,
            )
            for r in results
        ]

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Vector search completed: {len(results)} results, {latency_ms}ms latency"
        )

        # Prepare results for logging (extract minimal data)
        results_for_logging = [
            {
                "rank": r.rank,
                "score": r.score,
                "chunk_id": r.chunk_id,
                "parent_article_id": r.parent_article_id,
            }
            for r in result_chunks
        ]

        # Log query and results to database (best-effort)
        query_log_id_for_response = None
        try:
            query_log_id_for_response = query_log_client.log_query_with_results(
                raw_query=request.query,
                cache_result="miss",
                search_method="vector",
                results=results_for_logging,
                latency_ms=latency_ms,
                user_id=request.user_id,
                query_embedding=None,  # Could store embedding here if needed
                command=request.command,
            )
        except Exception as e:
            logger.error(f"Query and result logging failed: {e}")

        # Build response
        return SearchResponse(
            query=request.query,
            method="vector",
            results=result_chunks,
            total_results=len(results),
            latency_ms=latency_ms,
            metadata={"min_similarity": request.min_similarity}
            if request.min_similarity is not None
            else {},
            query_log_id=query_log_id_for_response,
        )

    except ValueError as e:
        # Validation errors from retriever
        logger.warning(f"Vector search validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except RuntimeError as e:
        # Embedding API failures
        logger.error(f"Vector search runtime error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service unavailable",
        )

    except Exception as e:
        # Unexpected errors
        logger.error(f"Vector search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Search failed"
        )


@router.post(
    "/hybrid",
    response_model=SearchResponse,
    summary="Hybrid Search with Reranking",
    description="Perform hybrid search combining BM25 and vector retrieval with RRF fusion and Cohere neural reranking. Best overall performance.",
    tags=["Search"],
)
def search_hybrid(
    search_request: HybridSearchRequest,
    api_key: Annotated[str, Depends(verify_api_key)],
    bm25_retriever: Annotated[BM25Retriever, Depends(get_bm25_retriever)],
    vector_retriever: Annotated[VectorRetriever, Depends(get_vector_retriever)],
    reranker: Annotated[Optional[CohereReranker], Depends(get_reranker)],
    query_log_client: Annotated[QueryLogClient, Depends(get_query_log_client)],
    reranker_log_client: Annotated[RerankerLogClient, Depends(get_reranker_log_client)],
) -> SearchResponse:
    """
    Perform hybrid search with neural reranking.

    **Best for:**
    - General-purpose search (best overall performance)
    - Combining keyword precision and semantic understanding
    - Production use cases requiring high relevance

    **Workflow:**
    1. **BM25 Search**: Fast keyword-based retrieval (2×top_k results)
    2. **Vector Search**: Semantic similarity retrieval (2×top_k results)
    3. **RRF Fusion**: Reciprocal Rank Fusion combines and deduplicates results
    4. **Cohere Rerank**: Neural reranking refines relevance using Cohere v3.5

    **Characteristics:**
    - Best relevance quality (neural reranking)
    - Latency: ~800ms-1.5s (vector API + reranking API)
    - Automatic fallback to RRF if reranking fails
    - Scores are Cohere relevance scores (0-1, calibrated)

    **Request:**
    - `query`: Search query text (1-1000 chars)
    - `top_k`: Number of results to return (1-100, default 10)
    - `rrf_k`: RRF constant (default 60, controls rank-based weighting)
    - `min_bm25_score`: Optional BM25 score threshold
    - `min_vector_similarity`: Optional vector similarity threshold
    - `user_id`: Optional user identifier for analytics

    **Response:**
    - Ranked results with Cohere relevance scores (0-1)
    - Latency in milliseconds (includes all API calls)
    - Metadata: reranking status, RRF parameters, diagnostics
    """
    start_time = time.time()

    logger.info(
        f"Hybrid search with reranking: query='{search_request.query[:50]}', top_k={search_request.top_k}, "
        f"rrf_k={search_request.rrf_k}, user_id={search_request.user_id}"
    )

    try:
        # Perform hybrid search with reranking
        results, reranking_metadata = hybrid_search(
            query=search_request.query,
            bm25_retriever=bm25_retriever,
            vector_retriever=vector_retriever,
            reranker=reranker,
            top_k=search_request.top_k,
            rrf_k=search_request.rrf_k,
            min_bm25_score=search_request.min_bm25_score,
            min_vector_similarity=search_request.min_vector_similarity,
        )

        # Convert hybrid results to SearchResultChunk
        result_chunks = [
            SearchResultChunk(
                rank=r["rank"],
                score=r["combined_score"],
                chunk_id=r["chunk"].chunk_id,
                parent_article_id=r["chunk"].parent_article_id,
                chunk_sequence=r["chunk"].chunk_sequence,
                text_content=r["chunk"].text_content,
                token_count=r["chunk"].token_count,
                source_url=r["chunk"].source_url,
                last_modified_date=r["chunk"].last_modified_date,
            )
            for r in results
        ]

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Hybrid search completed: {len(results)} results, {latency_ms}ms latency, "
            f"reranked={reranking_metadata.get('reranked', False)}"
        )

        # Prepare results for logging (extract minimal data)
        results_for_logging = [
            {
                "rank": r.rank,
                "score": r.score,
                "chunk_id": r.chunk_id,
                "parent_article_id": r.parent_article_id,
            }
            for r in result_chunks
        ]

        # Log query and results to database (best-effort)
        query_log_id_for_response = None
        try:
            query_log_id_for_response = query_log_client.log_query_with_results(
                raw_query=search_request.query,
                cache_result="miss",
                search_method="hybrid",
                results=results_for_logging,
                latency_ms=latency_ms,
                user_id=search_request.user_id,
                query_embedding=None,
                command=search_request.command,
            )

            # Log reranking data if query was logged successfully
            if query_log_id_for_response:
                # Extract RRF results from metadata
                rrf_results = reranking_metadata.get("rrf_results_before_reranking", [])

                # Log for both success and failure cases
                if reranking_metadata.get("reranked", False):
                    # Success case - reranking completed successfully
                    reranker_log_client.log_reranking(
                        query_log_id=query_log_id_for_response,
                        rrf_results=rrf_results,
                        reranked_results=results,
                        model_name="cohere.rerank-v3-5:0",
                        reranker_latency_ms=reranking_metadata.get(
                            "reranker_latency_ms", 0
                        ),
                        reranker_status="success",
                    )
                elif reranking_metadata.get("reranking_failed", False):
                    # Failure case - fallback to RRF
                    reranker_log_client.log_reranking(
                        query_log_id=query_log_id_for_response,
                        rrf_results=rrf_results,
                        reranked_results=rrf_results,  # Same as RRF (no change)
                        model_name="cohere.rerank-v3-5:0",
                        reranker_latency_ms=reranking_metadata.get(
                            "reranker_latency_ms", 0
                        ),
                        reranker_status="failed",
                        error_message=reranking_metadata.get("error"),
                    )
        except Exception as e:
            logger.error(f"Query and result logging failed: {e}")

        # Build metadata with RRF and reranking info
        metadata = {
            "rrf_k": search_request.rrf_k,
            **reranking_metadata,
        }

        # Build response
        return SearchResponse(
            query=search_request.query,
            method="hybrid",
            results=result_chunks,
            total_results=len(results),
            latency_ms=latency_ms,
            metadata=metadata,
            query_log_id=query_log_id_for_response,
        )

    except ValueError as e:
        # Validation errors from hybrid_search or retrievers
        logger.warning(f"Hybrid search validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except RuntimeError as e:
        # Search method failures (likely embedding API)
        logger.error(f"Hybrid search runtime error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service degraded",
        )

    except Exception as e:
        # Unexpected errors
        logger.error(f"Hybrid search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Search failed"
        )


@router.post(
    "/hyde",
    response_model=SearchResponse,
    summary="HyDE Search with Hybrid Retrieval",
    description="Generate hypothetical document from query, then perform hybrid search with reranking. Improved semantic matching with higher latency (~1.5-2.5s).",
    tags=["Search"],
)
async def search_hyde(
    search_request: HyDESearchRequest,
    api_key: Annotated[str, Depends(verify_api_key)],
    bm25_retriever: Annotated[BM25Retriever, Depends(get_bm25_retriever)],
    vector_retriever: Annotated[VectorRetriever, Depends(get_vector_retriever)],
    reranker: Annotated[Optional[CohereReranker], Depends(get_reranker)],
    hyde_generator: Annotated[HyDEGenerator, Depends(get_hyde_generator)],
    query_log_client: Annotated[QueryLogClient, Depends(get_query_log_client)],
    reranker_log_client: Annotated[RerankerLogClient, Depends(get_reranker_log_client)],
    hyde_log_client: Annotated[HyDELogClient, Depends(get_hyde_log_client)],
) -> SearchResponse:
    """
    Perform HyDE search with hybrid retrieval and neural reranking.

    **HyDE (Hypothetical Document Embeddings):**
    - Generates hypothetical answer to query using LLM
    - Embeds hypothetical document instead of raw query
    - Better document-to-document semantic matching

    **Best for:**
    - Complex queries requiring domain knowledge
    - Ambiguous or underspecified questions
    - Natural language queries ("My VPN isn't working")
    - Queries where semantic understanding is critical

    **Workflow:**
    1. **HyDE Generation**: LLM generates hypothetical answer (~500-1000ms)
    2. **BM25 Search**: Keyword-based retrieval using original query (2×top_k)
    3. **Vector Search**: Semantic retrieval using hypothetical document (2×top_k)
    4. **RRF Fusion**: Reciprocal Rank Fusion combines results
    5. **Cohere Rerank**: Neural reranking using original query

    **Characteristics:**
    - Best semantic matching quality
    - Latency: ~1.5-2.5s (includes LLM generation)
    - Automatic fallback to standard hybrid on LLM failure
    - Automatic fallback to RRF if reranking fails
    - Scores are Cohere relevance scores (0-1, calibrated)

    **Request:**
    - `query`: Search query text (1-1000 chars)
    - `top_k`: Number of results to return (1-100, default 10)
    - `rrf_k`: RRF constant (default 60)
    - `min_bm25_score`: Optional BM25 score threshold
    - `min_vector_similarity`: Optional vector similarity threshold
    - `user_id`: Optional user identifier for analytics

    **Response:**
    - Ranked results with Cohere relevance scores (0-1)
    - Latency in milliseconds (includes all operations)
    - Metadata: hypothetical document, HyDE latency, reranking status
    """
    start_time = time.time()

    logger.info(
        f"HyDE search: query='{search_request.query[:50]}', top_k={search_request.top_k}, "
        f"rrf_k={search_request.rrf_k}, user_id={search_request.user_id}"
    )

    # Track HyDE-specific metadata
    hyde_metadata = {
        "hyde_failed": False,
        "hypothetical_document": None,
        "hyde_latency_ms": 0,
    }

    try:
        # Step 1: Generate hypothetical document
        hyde_start = time.time()
        hypothetical_doc = None
        token_usage = None
        try:
            logger.debug(f"Generating hypothetical document for query: '{search_request.query[:50]}'")
            hypothetical_doc, token_usage = await hyde_generator.generate_hypothetical_document(
                query=search_request.query,
            )
            hyde_latency_ms = int((time.time() - hyde_start) * 1000)
            hyde_metadata["hyde_latency_ms"] = hyde_latency_ms
            hyde_metadata["hypothetical_document"] = hypothetical_doc
            hyde_metadata["token_usage"] = token_usage
            logger.info(
                f"Generated hypothetical document ({len(hypothetical_doc)} chars) in {hyde_latency_ms}ms: "
                f"'{hypothetical_doc[:100]}...'"
            )
        except Exception as e:
            # Graceful degradation: use original query for vector search
            hyde_latency_ms = int((time.time() - hyde_start) * 1000)
            hyde_metadata["hyde_latency_ms"] = hyde_latency_ms
            hyde_metadata["hyde_failed"] = True
            hyde_metadata["hyde_error"] = str(e)
            hypothetical_doc = search_request.query  # Fallback to original query
            logger.error(
                f"HyDE generation failed ({hyde_latency_ms}ms), falling back to original query: {str(e)}"
            )

        # Step 2: Perform BM25 search with original query (better for keywords)
        fetch_k = search_request.top_k * 2  # Fetch 2x results for fusion
        logger.debug(f"Fetching {fetch_k} BM25 results with original query")
        bm25_results = bm25_retriever.search(
            query=search_request.query,
            top_k=fetch_k,
            min_score=search_request.min_bm25_score,
        )
        logger.debug(f"BM25 search returned {len(bm25_results)} results")

        # Step 3: Perform vector search with hypothetical document
        logger.debug(f"Fetching {fetch_k} vector results with hypothetical document")
        embedding_start = time.time()
        vector_results = vector_retriever.search(
            query=hypothetical_doc,  # Use hypothetical doc for semantic matching
            top_k=fetch_k,
            min_similarity=search_request.min_vector_similarity,
        )
        embedding_latency_ms = int((time.time() - embedding_start) * 1000)
        logger.debug(f"Vector search returned {len(vector_results)} results in {embedding_latency_ms}ms")

        # Step 4: Apply RRF fusion
        logger.debug("Applying RRF fusion")
        fused_results = reciprocal_rank_fusion(
            bm25_results=bm25_results,
            vector_results=vector_results,
            k=search_request.rrf_k,
        )
        logger.debug(f"RRF fusion produced {len(fused_results)} unique results")

        # Step 5: Apply Cohere reranking with original query
        reranking_metadata = {
            "reranked": False,
            "reranking_failed": False,
            "reranker_latency_ms": 0,
            "rrf_results_before_reranking": fused_results.copy(),
        }

        if reranker is None:
            logger.warning("Reranker not available, using RRF results")
            final_results = fused_results[: search_request.top_k]
        else:
            try:
                logger.debug("Applying Cohere reranking with original query")
                reranked_results = reranker.rerank(
                    query=search_request.query,  # Use original query for relevance evaluation
                    results=fused_results,
                )
                reranking_metadata["reranked"] = True
                reranking_metadata["reranker_latency_ms"] = reranker.last_rerank_latency_ms
                final_results = reranked_results[: search_request.top_k]
                logger.info(
                    f"Reranking completed in {reranker.last_rerank_latency_ms}ms, "
                    f"returning top {search_request.top_k} results"
                )
            except Exception as e:
                # Graceful degradation: use RRF results
                logger.error(f"Reranking failed, falling back to RRF: {str(e)}")
                reranking_metadata["reranking_failed"] = True
                reranking_metadata["error"] = str(e)
                final_results = fused_results[: search_request.top_k]

        # Convert results to SearchResultChunk
        result_chunks = [
            SearchResultChunk(
                rank=r["rank"],
                score=r["combined_score"],
                chunk_id=r["chunk"].chunk_id,
                parent_article_id=r["chunk"].parent_article_id,
                chunk_sequence=r["chunk"].chunk_sequence,
                text_content=r["chunk"].text_content,
                token_count=r["chunk"].token_count,
                source_url=r["chunk"].source_url,
                last_modified_date=r["chunk"].last_modified_date,
            )
            for r in final_results
        ]

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"HyDE search completed: {len(result_chunks)} results, {latency_ms}ms total latency, "
            f"hyde_latency={hyde_metadata['hyde_latency_ms']}ms, "
            f"reranked={reranking_metadata.get('reranked', False)}"
        )

        # Prepare results for logging (extract minimal data)
        results_for_logging = [
            {
                "rank": r.rank,
                "score": r.score,
                "chunk_id": r.chunk_id,
                "parent_article_id": r.parent_article_id,
            }
            for r in result_chunks
        ]

        # Log query and results to database (best-effort)
        query_log_id_for_response = None
        try:
            query_log_id_for_response = query_log_client.log_query_with_results(
                raw_query=search_request.query,
                cache_result="miss",
                search_method="hyde",
                results=results_for_logging,
                latency_ms=latency_ms,
                user_id=search_request.user_id,
                query_embedding=None,
                command=search_request.command,
            )

            # Log reranking data if query was logged successfully
            if query_log_id_for_response:
                # Extract RRF results from metadata
                rrf_results = reranking_metadata.get("rrf_results_before_reranking", [])

                # Log for both success and failure cases
                if reranking_metadata.get("reranked", False):
                    # Success case - reranking completed successfully
                    reranker_log_client.log_reranking(
                        query_log_id=query_log_id_for_response,
                        rrf_results=rrf_results,
                        reranked_results=final_results,
                        model_name="cohere.rerank-v3-5:0",
                        reranker_latency_ms=reranking_metadata.get(
                            "reranker_latency_ms", 0
                        ),
                        reranker_status="success",
                    )
                elif reranking_metadata.get("reranking_failed", False):
                    # Failure case - fallback to RRF
                    reranker_log_client.log_reranking(
                        query_log_id=query_log_id_for_response,
                        rrf_results=rrf_results,
                        reranked_results=rrf_results,  # Same as RRF (no change)
                        model_name="cohere.rerank-v3-5:0",
                        reranker_latency_ms=reranking_metadata.get(
                            "reranker_latency_ms", 0
                        ),
                        reranker_status="failed",
                        error_message=reranking_metadata.get("error"),
                    )

                # Log HyDE generation data (best-effort)
                try:
                    hyde_log_client.log_hyde_generation(
                        query_log_id=query_log_id_for_response,
                        hypothetical_document=hypothetical_doc,
                        generation_status="success" if not hyde_metadata["hyde_failed"] else "failed_fallback",
                        model_name=hyde_generator.deployment_name,
                        generation_latency_ms=hyde_metadata["hyde_latency_ms"],
                        embedding_latency_ms=embedding_latency_ms,
                        prompt_tokens=token_usage.get("prompt_tokens") if token_usage else None,
                        completion_tokens=token_usage.get("completion_tokens") if token_usage else None,
                        total_tokens=token_usage.get("total_tokens") if token_usage else None,
                        error_message=hyde_metadata.get("hyde_error"),
                    )
                    logger.debug(f"HyDE generation logged for query_log_id {query_log_id_for_response}")
                except Exception as hyde_log_error:
                    # Don't fail the request if HyDE logging fails
                    logger.error(f"HyDE logging failed: {hyde_log_error}")

        except Exception as e:
            logger.error(f"Query and result logging failed: {e}")

        # Build metadata with HyDE, RRF, and reranking info
        metadata = {
            "rrf_k": search_request.rrf_k,
            **hyde_metadata,
            **reranking_metadata,
        }

        # Build response
        return SearchResponse(
            query=search_request.query,
            method="hyde",
            results=result_chunks,
            total_results=len(result_chunks),
            latency_ms=latency_ms,
            metadata=metadata,
            query_log_id=query_log_id_for_response,
        )

    except ValueError as e:
        # Validation errors from retrievers
        logger.warning(f"HyDE search validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except RuntimeError as e:
        # Service failures (embedding API, LLM API, reranker API)
        logger.error(f"HyDE search runtime error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service degraded",
        )

    except Exception as e:
        # Unexpected errors
        logger.error(f"HyDE search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="HyDE search failed"
        )
