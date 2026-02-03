"""
FastAPI application for RAG helpdesk backend.

Provides REST API endpoints for:
- BM25 keyword search
- Vector semantic search
- Hybrid search (BM25 + vector fusion)
- Query logging and analytics
- Health checks

Architecture:
- Lifespan management for resource initialization/cleanup
- Connection pooling for concurrent requests
- Shared retrievers (initialized once at startup)
- Header-based API key authentication

Run with:
    uvicorn api.main:app --reload  # Development
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4  # Production
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from api.routers import search, health, query_logs
from api.utils.connection_pool import get_connection_pool, close_connection_pool
from core.config import get_api_settings
from core.bm25_search import BM25Retriever
from core.vector_search import VectorRetriever
from core.reranker import CohereReranker
from core.embedding import GenerateEmbeddingsOpenAI
from core.storage_vector import OpenAIVectorStorage
from utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for application startup and shutdown.

    Startup:
    - Initialize database connection pool
    - Initialize BM25 retriever (loads corpus into memory)
    - Initialize vector retriever
    - Initialize Cohere reranker (AWS Bedrock)
    - Pre-warm BM25 corpus cache
    - Store shared resources in app.state

    Shutdown:
    - Close vector retriever
    - Close connection pool
    - Cleanup resources

    This ensures resources are initialized ONCE at startup and properly
    cleaned up at shutdown, rather than per-request.
    """
    logger.info("=" * 80)
    logger.info("Starting FastAPI application...")
    logger.info("=" * 80)

    try:
        # Get API settings
        settings = get_api_settings()
        logger.info(
            f"API Configuration: host={settings.HOST}, port={settings.PORT}, "
            f"workers={settings.WORKERS}, log_level={settings.LOG_LEVEL}"
        )

        # Initialize database connection pool
        logger.info(
            f"Initializing connection pool: min={settings.POOL_MIN_SIZE}, "
            f"max={settings.POOL_MAX_SIZE}, timeout={settings.POOL_TIMEOUT}s"
        )
        connection_pool = get_connection_pool(
            min_conn=settings.POOL_MIN_SIZE,
            max_conn=settings.POOL_MAX_SIZE,
            timeout=settings.POOL_TIMEOUT,
        )
        app.state.connection_pool = connection_pool
        logger.info("✓ Connection pool initialized")

        # Initialize BM25 retriever with connection pool
        logger.info("Initializing BM25 retriever...")
        bm25_retriever = BM25Retriever(
            postgres_client=None,  # Will create its own with pool support
            k1=1.5,
            b=0.75,
            use_cache=True,
        )
        app.state.bm25_retriever = bm25_retriever
        logger.info("✓ BM25 retriever initialized")

        # Pre-warm BM25 corpus cache
        logger.info("Pre-warming BM25 corpus cache...")
        stats = bm25_retriever.get_stats()
        logger.info(
            f"✓ BM25 corpus loaded: {stats.get('num_chunks', 0)} chunks, "
            f"cached={stats.get('is_cached', False)}"
        )

        # Initialize vector retriever with connection pool
        logger.info("Initializing vector retriever...")
        embedding_generator = GenerateEmbeddingsOpenAI()
        vector_storage = OpenAIVectorStorage(connection_pool=connection_pool)
        vector_retriever = VectorRetriever(
            embedding_generator=embedding_generator, vector_storage=vector_storage
        )
        app.state.vector_retriever = vector_retriever
        logger.info("✓ Vector retriever initialized")

        # Get vector stats
        vector_stats = vector_retriever.get_stats()
        logger.info(
            f"✓ Vector storage ready: {vector_stats.get('num_embeddings', 0)} embeddings, "
            f"dim={vector_stats.get('embedding_dimension', 0)}, "
            f"model={vector_stats.get('model', 'unknown')}"
        )

        # Initialize Cohere reranker
        logger.info("Initializing Cohere reranker...")
        try:
            reranker = CohereReranker()
            app.state.reranker = reranker
            logger.info("✓ Cohere reranker initialized")
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}", exc_info=True)
            app.state.reranker = None
            logger.warning(
                "⚠ Reranker unavailable - hybrid search will use RRF fallback"
            )

        # Initialize HyDE generator
        logger.info("Initializing HyDE generator...")
        try:
            from core.hyde_generator import HyDEGenerator

            hyde_generator = HyDEGenerator()
            app.state.hyde_generator = hyde_generator
            logger.info("✓ HyDE generator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize HyDE generator: {e}", exc_info=True)
            app.state.hyde_generator = None
            logger.warning(
                "⚠ HyDE generator unavailable - /search/hyde will return 503"
            )

        logger.info("=" * 80)
        logger.info("FastAPI application startup complete!")
        logger.info("API Documentation: http://localhost:8000/docs")
        logger.info("Health Check: http://localhost:8000/health/")
        logger.info("=" * 80)

        # Yield control to application (app is now serving requests)
        yield

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        raise

    finally:
        # Shutdown: cleanup resources
        logger.info("=" * 80)
        logger.info("Shutting down FastAPI application...")
        logger.info("=" * 80)

        try:
            # Close vector retriever
            if hasattr(app.state, "vector_retriever"):
                logger.info("Closing vector retriever...")
                app.state.vector_retriever.close()
                logger.info("✓ Vector retriever closed")

            # Close connection pool
            if hasattr(app.state, "connection_pool"):
                logger.info("Closing connection pool...")
                close_connection_pool()
                logger.info("✓ Connection pool closed")

            logger.info("=" * 80)
            logger.info("FastAPI application shutdown complete")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)


# Create FastAPI application
app = FastAPI(
    title="RAG Helpdesk API",
    description=(
        "Retrieval-Augmented Generation API for helpdesk knowledge base.\n\n"
        "Provides three search methods:\n"
        "- **BM25**: Fast keyword-based sparse retrieval\n"
        "- **Vector**: Semantic dense retrieval using embeddings\n"
        "- **Hybrid**: Combined BM25 + vector with RRF fusion and Cohere neural reranking\n\n"
        "All endpoints require API key authentication via X-API-Key header.\n\n"
        "**Authentication**: Include header `X-API-Key: your-api-key`"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Include routers
app.include_router(
    health.router,
    prefix="/health",
    tags=["Health"],
)
app.include_router(
    search.router,
    prefix="/api/v1/search",
    tags=["Search"],
)
app.include_router(
    query_logs.router,
    prefix="/api/v1/query-logs",
    tags=["Query Logs"],
)


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


# Optional: Add CORS middleware for development
# Uncomment if frontend needs cross-origin access
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


if __name__ == "__main__":
    # For development: run directly with python
    # Production: use uvicorn command instead
    import uvicorn

    settings = get_api_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,  # Development mode
        log_level=settings.LOG_LEVEL.lower(),
    )
