# tests/test_chat_models.py
"""Tests for OpenAI-compatible Pydantic models."""

from api.models.chat import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ModelObject,
    ModelListResponse,
)


def test_chat_message_with_content():
    msg = ChatMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_chat_message_none_content():
    """Assistant messages may have None content (Open WebUI edge case)."""
    msg = ChatMessage(role="assistant", content=None)
    assert msg.content is None


def test_chat_message_default_content():
    """Content defaults to None when omitted."""
    msg = ChatMessage(role="assistant")
    assert msg.content is None


def test_chat_completion_request_minimal():
    req = ChatCompletionRequest(
        model="utc-helpdesk",
        messages=[ChatMessage(role="user", content="test")],
    )
    assert req.model == "utc-helpdesk"
    assert req.stream is True  # default
    assert len(req.messages) == 1


def test_chat_completion_request_stream_false_accepted():
    """stream=false is accepted (but ignored by router)."""
    req = ChatCompletionRequest(
        model="utc-helpdesk",
        messages=[ChatMessage(role="user", content="test")],
        stream=False,
    )
    assert req.stream is False


def test_chat_completion_chunk_serialization():
    chunk = ChatCompletionChunk(
        id="chatcmpl-abc123",
        created=1700000000,
        model="utc-helpdesk",
        choices=[
            ChatCompletionChunkChoice(delta={"content": "hello"}, finish_reason=None)
        ],
    )
    data = chunk.model_dump()
    assert data["id"] == "chatcmpl-abc123"
    assert data["object"] == "chat.completion.chunk"
    assert data["model"] == "utc-helpdesk"
    assert data["choices"][0]["delta"] == {"content": "hello"}
    assert data["usage"] is None


def test_chat_completion_chunk_with_usage():
    chunk = ChatCompletionChunk(
        id="chatcmpl-abc123",
        created=1700000000,
        model="utc-helpdesk",
        choices=[ChatCompletionChunkChoice(delta={}, finish_reason="stop")],
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )
    assert chunk.usage["total_tokens"] == 150
    assert chunk.choices[0].finish_reason == "stop"


def test_model_list_response():
    resp = ModelListResponse(data=[ModelObject(id="utc-helpdesk", created=1700000000)])
    data = resp.model_dump()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "utc-helpdesk"
    assert data["data"][0]["owned_by"] == "utc"
