"""
Admin API routes for analytics and management.
"""
from fastapi import APIRouter, Depends, Query
from typing import Optional

from api.middleware.auth import verify_api_key
from api.services.analytics import AnalyticsService

router = APIRouter(prefix="/admin", tags=["admin"])

# Initialize service
analytics_service = AnalyticsService()


@router.get("/health")
async def health_check():
    """Health check endpoint (no auth required)."""
    return {"status": "healthy", "service": "admin-api"}


@router.get("/stats")
async def get_stats(
    days: int = Query(default=30, ge=1, le=365, description="Days to look back"),
    _: str = Depends(verify_api_key),
):
    """
    Get overall conversation statistics.

    Returns total conversations, unique users, and average latency.
    """
    return analytics_service.get_conversation_stats(days=days)


@router.get("/users")
async def get_users(
    days: int = Query(default=30, ge=1, le=365, description="Days to look back"),
    limit: int = Query(default=50, ge=1, le=500, description="Max users to return"),
    _: str = Depends(verify_api_key),
):
    """
    Get conversation counts by user.

    Returns user_id, conversation_count, and last_chat timestamp.
    """
    return analytics_service.get_conversations_by_user(days=days, limit=limit)


@router.get("/conversations/timeline")
async def get_timeline(
    days: int = Query(default=30, ge=1, le=365, description="Days to look back"),
    group_by: str = Query(default="day", pattern="^(day|hour)$", description="Group by day or hour"),
    _: str = Depends(verify_api_key),
):
    """
    Get conversation counts over time.

    Returns time series data for charting.
    """
    return analytics_service.get_conversations_over_time(days=days, group_by=group_by)


@router.get("/cache")
async def get_cache_stats(
    days: int = Query(default=30, ge=1, le=365, description="Days to look back"),
    _: str = Depends(verify_api_key),
):
    """
    Get cache hit/miss statistics.

    Returns cache performance metrics.
    """
    return analytics_service.get_cache_stats(days=days)


@router.get("/queries")
async def get_recent_queries(
    limit: int = Query(default=20, ge=1, le=100, description="Max queries to return"),
    user_id: Optional[str] = Query(default=None, description="Filter by user ID"),
    _: str = Depends(verify_api_key),
):
    """
    Get recent queries.

    Returns recent query logs with optional user filter.
    """
    return analytics_service.get_recent_queries(limit=limit, user_id=user_id)
