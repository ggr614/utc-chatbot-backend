"""
Admin router for managing tag-based system prompts.

Provides REST endpoints for CRUD operations on tag_system_prompts,
plus an HTML admin page served at /admin/prompts.

No authentication required (internal network only).
"""

from pathlib import Path
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse

from api.dependencies import get_prompt_storage_client
from api.models.requests import BulkSavePromptRequest
from api.models.responses import BulkSaveResponse, PromptResponse
from core.storage_prompt import PromptStorageClient
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/admin/prompts", response_class=HTMLResponse, include_in_schema=False)
def admin_page():
    """Serve the system prompt admin HTML page."""
    html_path = Path(__file__).parent.parent / "templates" / "admin_prompts.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@router.get(
    "/api/v1/admin/prompts",
    response_model=List[PromptResponse],
    summary="List all system prompts",
)
def list_prompts(
    client: Annotated[PromptStorageClient, Depends(get_prompt_storage_client)],
) -> List[PromptResponse]:
    """List all tag-based system prompts ordered by priority (descending)."""
    prompts = client.list_all_prompts()
    return [PromptResponse(**p) for p in prompts]


@router.get(
    "/api/v1/admin/categories",
    summary="List distinct article categories",
)
def list_categories(
    client: Annotated[PromptStorageClient, Depends(get_prompt_storage_client)],
) -> dict:
    """Get all distinct category names from the articles table for checkbox display."""
    categories = client.get_distinct_article_categories()
    return {"categories": categories, "total": len(categories)}


@router.post(
    "/api/v1/admin/prompts/bulk-save",
    response_model=BulkSaveResponse,
    summary="Save prompt to multiple tags",
)
def bulk_save_prompt(
    request: BulkSavePromptRequest,
    client: Annotated[PromptStorageClient, Depends(get_prompt_storage_client)],
) -> BulkSaveResponse:
    """
    Save a system prompt assigned to multiple tags at once.

    Upserts rows for each tag in `tags` and deletes rows for tags in
    `remove_tags`. All operations run in a single database transaction.
    """
    result = client.bulk_upsert_prompts(
        system_prompt=request.system_prompt,
        priority=request.priority,
        description=request.description,
        tags_to_upsert=request.tags,
        tags_to_delete=request.remove_tags,
    )
    return BulkSaveResponse(
        created=result["created"],
        updated=result["updated"],
        deleted=result["deleted"],
        message=(
            f"{len(result['created'])} created, "
            f"{len(result['updated'])} updated, "
            f"{len(result['deleted'])} deleted"
        ),
    )


@router.delete(
    "/api/v1/admin/prompts/{tag_name}",
    summary="Delete a prompt by tag",
)
def delete_prompt(
    tag_name: str,
    client: Annotated[PromptStorageClient, Depends(get_prompt_storage_client)],
) -> dict:
    """Delete a single tag's system prompt. Cannot delete __default__."""
    if tag_name == "__default__":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the __default__ prompt",
        )
    deleted = client.delete_prompt(tag_name)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No prompt found for tag '{tag_name}'",
        )
    return {"deleted": True, "tag_name": tag_name}
