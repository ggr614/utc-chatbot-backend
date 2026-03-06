"""
Health check endpoints for monitoring and orchestration.

Provides two endpoints:
1. /health/ - Detailed health check with component status
2. /health/ready - Simple readiness probe for Kubernetes/orchestration
"""

from fastapi import APIRouter, Request, Response, status as http_status
from datetime import datetime, timezone

from api.models.responses import HealthResponse
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Health Check",
    description="Detailed health check with component status. No authentication required (for monitoring systems).",
    tags=["Health"],
)
def health_check(request: Request) -> HealthResponse:
    """
    Detailed health check endpoint for monitoring and load balancers.

    Checks the health of individual components:
    - **BM25 retriever**: Verifies corpus is loaded and accessible
    - **Vector retriever**: Verifies embeddings are accessible
    - **Database connection pool**: Verifies pool is healthy

    **Overall Status:**
    - `healthy`: All components operational
    - `degraded`: Some components have issues but system functional
    - `unhealthy`: Critical components failed

    **No authentication required** - this endpoint is intended for
    monitoring systems and load balancers.

    **Response:**
    - `status`: Overall health status
    - `timestamp`: Current server time (UTC)
    - `checks`: Dictionary of individual component checks
    """
    checks = {}
    overall_status = "healthy"

    # Check BM25 retriever
    try:
        retriever = request.app.state.bm25_retriever
        stats = retriever.get_stats()
        checks["bm25"] = {
            "status": "healthy",
            "num_chunks": stats.get("num_chunks", 0),
            "cached": stats.get("is_cached", False),
            "avg_doc_length": stats.get("avg_doc_length"),
        }
        logger.debug(
            f"BM25 health check: {stats.get('num_chunks', 0)} chunks, "
            f"cached={stats.get('is_cached', False)}"
        )
    except AttributeError:
        checks["bm25"] = {
            "status": "unhealthy",
            "error": "BM25 retriever not initialized",
        }
        overall_status = "unhealthy"
        logger.error("BM25 health check failed: retriever not found in app.state")
    except Exception as e:
        checks["bm25"] = {"status": "unhealthy", "error": str(e)}
        overall_status = "unhealthy"
        logger.error(f"BM25 health check failed: {e}")

    # Check vector retriever
    try:
        retriever = request.app.state.vector_retriever
        stats = retriever.get_stats()
        checks["vector"] = {
            "status": "healthy",
            "num_embeddings": stats.get("num_embeddings", 0),
            "embedding_dimension": stats.get("embedding_dimension", 0),
            "model": stats.get("model"),
        }
        logger.debug(
            f"Vector health check: {stats.get('num_embeddings', 0)} embeddings"
        )
    except AttributeError:
        checks["vector"] = {
            "status": "unhealthy",
            "error": "Vector retriever not initialized",
        }
        overall_status = "unhealthy"
        logger.error("Vector health check failed: retriever not found in app.state")
    except Exception as e:
        checks["vector"] = {"status": "unhealthy", "error": str(e)}
        overall_status = "unhealthy"
        logger.error(f"Vector health check failed: {e}")

    # Check connection pool
    try:
        pool = request.app.state.connection_pool
        pool_stats = pool.get_stats()

        # Determine pool health
        pool_size = pool_stats.get("pool_size", 0)
        pool_available = pool_stats.get("pool_available", 0)

        if pool_size == 0:
            pool_status = "unhealthy"
            overall_status = "unhealthy"
        elif pool_available == 0:
            pool_status = "degraded"
            if overall_status == "healthy":
                overall_status = "degraded"
        else:
            pool_status = "healthy"

        checks["database"] = {
            "status": pool_status,
            "pool_size": pool_size,
            "pool_available": pool_available,
            "requests_waiting": pool_stats.get("requests_waiting", 0),
        }
        logger.debug(
            f"Database pool health check: size={pool_size}, available={pool_available}"
        )
    except AttributeError:
        checks["database"] = {
            "status": "degraded",
            "error": "Connection pool not initialized",
        }
        if overall_status == "healthy":
            overall_status = "degraded"
        logger.warning("Database pool health check failed: pool not found in app.state")
    except Exception as e:
        checks["database"] = {"status": "degraded", "error": str(e)}
        if overall_status == "healthy":
            overall_status = "degraded"
        logger.error(f"Database pool health check failed: {e}")

    logger.info(f"Health check completed: overall_status={overall_status}")

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        checks=checks,
    )


@router.get(
    "/ready",
    summary="Readiness Check",
    description="Simple readiness probe for Kubernetes/orchestration. Returns 200 if ready, 503 if not.",
    tags=["Health"],
    status_code=http_status.HTTP_200_OK,
    responses={
        200: {
            "description": "Service is ready to accept traffic",
            "content": {"application/json": {"example": {"status": "ready"}}},
        },
        503: {
            "description": "Service is not ready",
            "content": {"application/json": {"example": {"status": "not ready"}}},
        },
    },
)
def readiness_check(request: Request, response: Response):
    """
    Simple readiness check for Kubernetes/orchestration.

    Returns:
    - **200 OK**: Service is ready to accept traffic (retrievers initialized)
    - **503 Service Unavailable**: Service is not ready

    This endpoint verifies that critical components (BM25 and vector retrievers,
    connection pool) are initialized and accessible.

    **No authentication required** - intended for orchestration systems.

    Used by:
    - Kubernetes readiness probes
    - Load balancers
    - Deployment automation
    """
    try:
        # Check that critical components are initialized
        _ = request.app.state.bm25_retriever
        _ = request.app.state.vector_retriever
        _ = request.app.state.connection_pool

        logger.debug("Readiness check passed: all components initialized")
        return {"status": "ready"}

    except AttributeError as e:
        logger.error(f"Readiness check failed: {e}")
        response.status_code = http_status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not ready", "error": str(e)}

    except Exception as e:
        logger.error(f"Readiness check failed with unexpected error: {e}")
        response.status_code = http_status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not ready", "error": "Internal error"}
