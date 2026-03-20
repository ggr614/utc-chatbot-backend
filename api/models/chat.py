# api/models/chat.py
"""Pydantic models for the OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "assistant", "user"]
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = True
    temperature: float | None = None
    max_tokens: int | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: dict
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: dict | None = None


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "utc"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]
