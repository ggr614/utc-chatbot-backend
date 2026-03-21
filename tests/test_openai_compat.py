# tests/test_openai_compat.py
"""Integration tests for the OpenAI-compatible router."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.openai_compat import router
from core.chat_service import ChatService


@pytest.fixture
def mock_chat_service():
    """Create a mock ChatService."""
    service = MagicMock(spec=ChatService)
    return service


@pytest.fixture
def app(mock_chat_service):
    """Create a test FastAPI app with the OpenAI compat router."""
    from api.dependencies import verify_api_key

    test_app = FastAPI()
    test_app.state.chat_service = mock_chat_service

    # Mock chat settings
    mock_settings = MagicMock()
    mock_settings.MODEL_ID = "test-model"

    # Use FastAPI dependency_overrides (not patch) because Depends()
    # captures the function reference at import time
    test_app.dependency_overrides[verify_api_key] = lambda: None

    with patch("api.routers.openai_compat.get_chat_settings", return_value=mock_settings):
        test_app.include_router(router, prefix="/v1")
        yield test_app

    test_app.dependency_overrides.clear()


@pytest.fixture
def client(app):
    return TestClient(app)


class TestModelsEndpoint:
    def test_list_models(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "utc"


class TestChatCompletionsEndpoint:
    def test_streams_sse_response(self, client, mock_chat_service):
        """Chat completions should return SSE-formatted streaming response."""
        async def fake_gen(*args, **kwargs):
            yield "Hello "
            yield "world!"

        mock_chat_service.handle_chat = MagicMock(return_value=fake_gen())

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE events
        lines = response.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) >= 3  # role chunk, content chunks, [DONE]
        assert data_lines[-1] == "data: [DONE]"

        # Check first content chunk has required fields
        first_data = json.loads(data_lines[0].removeprefix("data: "))
        assert "id" in first_data
        assert first_data["id"].startswith("chatcmpl-")
        assert "created" in first_data
        assert first_data["model"] == "test-model"
        assert first_data["object"] == "chat.completion.chunk"

    def test_empty_messages_returns_error(self, client, mock_chat_service):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [],
                "stream": True,
            },
        )
        assert response.status_code == 422  # Pydantic validation


class TestAuthentication:
    def test_missing_api_key_returns_401(self):
        """Without a valid API key, auth should fail with 401."""
        test_app = FastAPI()
        test_app.state.chat_service = MagicMock()

        mock_settings = MagicMock()
        mock_settings.MODEL_ID = "test-model"

        # NO dependency_overrides — real verify_api_key runs
        with patch("api.routers.openai_compat.get_chat_settings", return_value=mock_settings), \
             patch.dict("os.environ", {"API_API_KEY": "real-secret-key-min-32-chars-long!!"}):
            test_app.include_router(router, prefix="/v1")
            client = TestClient(test_app, raise_server_exceptions=False)
            # Send request with wrong API key — should get 401
            response = client.get("/v1/models", headers={"X-API-Key": "wrong-key"})
            assert response.status_code == 401
