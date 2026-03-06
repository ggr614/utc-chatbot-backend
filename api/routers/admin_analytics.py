"""
Admin router for analytics dashboard.

Provides REST endpoints for querying service analytics data
plus an HTML dashboard page served at /admin/analytics.

No authentication required (internal network only).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse

from api.dependencies import get_query_log_client
from core.storage_query_log import QueryLogClient
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _parse_time_range(range_str: str) -> tuple[datetime, datetime]:
    """Convert range string ('24h', '7d', '30d') to (start, end) datetimes in UTC."""
    end = datetime.now(timezone.utc)
    mapping = {
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
    }
    delta = mapping.get(range_str, timedelta(days=7))
    start = end - delta
    return start, end


@router.get("/admin/analytics", response_class=HTMLResponse, include_in_schema=False)
def analytics_page():
    """Serve the analytics dashboard HTML page."""
    html_path = Path(__file__).parent.parent / "templates" / "admin_analytics.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@router.get(
    "/api/v1/admin/analytics/overview",
    summary="Analytics overview KPIs",
)
def get_overview(
    time_range: str = Query(
        default="7d", alias="range", pattern="^(24h|7d|30d)$"
    ),
    query_log_client: Annotated[
        QueryLogClient, Depends(get_query_log_client)
    ] = ...,
) -> dict:
    """Get high-level KPI summary for the analytics dashboard."""
    start, end = _parse_time_range(time_range)

    try:
        total_all_time = query_log_client.get_total_query_count()
        cache_stats = query_log_client.get_query_cache_performance(start, end)
        latency_stats = query_log_client.get_query_latency_stats(start, end)

        return {
            "total_queries_all_time": total_all_time,
            "queries_in_period": cache_stats.total_queries,
            "cache_hit_rate": round(cache_stats.hit_rate, 1),
            "cache_hits": cache_stats.cache_hits,
            "cache_misses": cache_stats.cache_misses,
            "avg_latency_ms": round(latency_stats.avg_latency_ms, 1),
            "p50_latency_ms": round(latency_stats.p50_latency_ms, 1),
            "p95_latency_ms": round(latency_stats.p95_latency_ms, 1),
            "p99_latency_ms": round(latency_stats.p99_latency_ms, 1),
            "time_range": time_range,
        }
    except Exception as e:
        logger.error(f"Failed to fetch analytics overview: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch analytics overview",
        )


@router.get(
    "/api/v1/admin/analytics/query-volume",
    summary="Query volume over time",
)
def get_query_volume(
    time_range: str = Query(
        default="7d", alias="range", pattern="^(24h|7d|30d)$"
    ),
    query_log_client: Annotated[
        QueryLogClient, Depends(get_query_log_client)
    ] = ...,
) -> dict:
    """Get query count aggregated by hour for time-series chart."""
    start, end = _parse_time_range(time_range)

    try:
        data = query_log_client.get_queries_per_hour(start, end)
        # Convert datetime objects to ISO strings for JSON serialization
        for entry in data:
            entry["hour"] = entry["hour"].isoformat()

        return {"data": data, "time_range": time_range}
    except Exception as e:
        logger.error(f"Failed to fetch query volume: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch query volume data",
        )


@router.get(
    "/api/v1/admin/analytics/cache-performance",
    summary="Cache hit/miss breakdown",
)
def get_cache_performance(
    time_range: str = Query(
        default="7d", alias="range", pattern="^(24h|7d|30d)$"
    ),
    query_log_client: Annotated[
        QueryLogClient, Depends(get_query_log_client)
    ] = ...,
) -> dict:
    """Get cache performance statistics and breakdown by type."""
    start, end = _parse_time_range(time_range)

    try:
        cache_stats = query_log_client.get_query_cache_performance(start, end)
        breakdown = query_log_client.get_query_count_by_cache_result(start, end)

        return {
            "total_queries": cache_stats.total_queries,
            "cache_hits": cache_stats.cache_hits,
            "cache_misses": cache_stats.cache_misses,
            "hit_rate": round(cache_stats.hit_rate, 1),
            "breakdown": breakdown,
            "time_range": time_range,
        }
    except Exception as e:
        logger.error(f"Failed to fetch cache performance: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch cache performance data",
        )


@router.get(
    "/api/v1/admin/analytics/latency",
    summary="Query latency statistics",
)
def get_latency_stats(
    time_range: str = Query(
        default="7d", alias="range", pattern="^(24h|7d|30d)$"
    ),
    query_log_client: Annotated[
        QueryLogClient, Depends(get_query_log_client)
    ] = ...,
) -> dict:
    """Get latency percentile statistics for queries."""
    start, end = _parse_time_range(time_range)

    try:
        stats = query_log_client.get_query_latency_stats(start, end)

        return {
            "avg_latency_ms": round(stats.avg_latency_ms, 1),
            "min_latency_ms": stats.min_latency_ms,
            "max_latency_ms": stats.max_latency_ms,
            "p50_latency_ms": round(stats.p50_latency_ms, 1),
            "p95_latency_ms": round(stats.p95_latency_ms, 1),
            "p99_latency_ms": round(stats.p99_latency_ms, 1),
            "total_queries": stats.total_queries,
            "time_range": time_range,
        }
    except Exception as e:
        logger.error(f"Failed to fetch latency stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch latency statistics",
        )


@router.get(
    "/api/v1/admin/analytics/popular-queries",
    summary="Most frequent queries",
)
def get_popular_queries(
    time_range: str = Query(
        default="7d", alias="range", pattern="^(24h|7d|30d)$"
    ),
    limit: int = Query(default=15, ge=1, le=100),
    min_occurrences: int = Query(default=1, ge=1),
    query_log_client: Annotated[
        QueryLogClient, Depends(get_query_log_client)
    ] = ...,
) -> dict:
    """Get most frequently asked queries with statistics."""
    start, end = _parse_time_range(time_range)

    try:
        popular = query_log_client.get_popular_queries(
            start, end, limit=limit, min_occurrences=min_occurrences
        )

        queries = []
        for q in popular:
            queries.append(
                {
                    "raw_query": q.raw_query,
                    "query_count": q.query_count,
                    "avg_latency_ms": round(q.avg_latency_ms, 1),
                    "cache_hit_rate": round(q.cache_hit_rate, 1),
                    "last_queried_at": (
                        q.last_queried_at.isoformat()
                        if q.last_queried_at
                        else None
                    ),
                }
            )

        return {"queries": queries, "time_range": time_range}
    except Exception as e:
        logger.error(f"Failed to fetch popular queries: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch popular queries",
        )


@router.get(
    "/api/v1/admin/analytics/content-stats",
    summary="Knowledge base content statistics",
)
def get_content_stats(request: Request) -> dict:
    """Get counts of articles, chunks, and embeddings in the knowledge base."""
    try:
        pool = request.app.state.connection_pool
        with pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM articles) AS article_count,
                        (SELECT COUNT(*) FROM article_chunks) AS chunk_count,
                        (SELECT COUNT(*) FROM embeddings_openai) AS embedding_count
                    """
                )
                row = cur.fetchone()

        return {
            "articles": row[0] if row else 0,
            "chunks": row[1] if row else 0,
            "embeddings": row[2] if row else 0,
        }
    except AttributeError:
        logger.error("Connection pool not found in app.state")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Connection pool not initialized",
        )
    except Exception as e:
        logger.error(f"Failed to fetch content stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch content statistics",
        )
