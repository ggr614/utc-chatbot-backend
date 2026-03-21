# api/routers/openai_compat.py
"""OpenAI-compatible endpoints for chat completions.

Thin HTTP adapter: parses requests, delegates to ChatService,
wraps output as SSE in OpenAI format.
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from uuid_utils import uuid7

from api.dependencies import get_chat_service, verify_api_key
from api.models.chat import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ModelListResponse,
    ModelObject,
)
from core.config import get_chat_settings
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("/models", response_model=ModelListResponse)
def list_models():
    """List available models (OpenAI-compatible)."""
    settings = get_chat_settings()
    return ModelListResponse(
        data=[
            ModelObject(
                id=settings.MODEL_ID,
                created=int(time.time()),
            )
        ]
    )


@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    chat_service=Depends(get_chat_service),
):
    """OpenAI-compatible chat completions with SSE streaming."""
    settings = get_chat_settings()

    request_id = f"chatcmpl-{uuid7()}"
    created = int(time.time())
    model_id = settings.MODEL_ID

    # Extract email from headers if available
    user_email = request.headers.get("X-User-Email")

    messages = [msg.model_dump() for msg in body.messages]

    async def generate_sse():
        # Role chunk
        role_chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model_id,
            choices=[
                ChatCompletionChunkChoice(
                    delta={"role": "assistant"}, finish_reason=None
                )
            ],
        )
        yield f"data: {role_chunk.model_dump_json()}\n\n"

        usage_data = None

        try:
            async for item in chat_service.handle_chat(messages, user_email):
                if isinstance(item, dict) and "usage" in item:
                    # Usage data yielded after all content
                    usage_data = item["usage"]
                    continue
                if not item:
                    continue
                content_chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=model_id,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta={"content": item}, finish_reason=None
                        )
                    ],
                )
                yield f"data: {content_chunk.model_dump_json()}\n\n"

            # Final chunk with finish_reason and usage
            stop_chunk = ChatCompletionChunk(
                id=request_id,
                created=created,
                model=model_id,
                choices=[ChatCompletionChunkChoice(delta={}, finish_reason="stop")],
                usage=usage_data,
            )
            yield f"data: {stop_chunk.model_dump_json()}\n\n"

        except Exception:
            logger.exception("Error during chat streaming")
            error_data = json.dumps(
                {
                    "error": {
                        "message": "An internal error occurred",
                        "type": "server_error",
                        "code": 503,
                    }
                }
            )
            yield f"data: {error_data}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
