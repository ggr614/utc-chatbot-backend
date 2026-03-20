# tests/test_config_chat.py
"""Tests for ChatSettings configuration."""
import pytest
from unittest.mock import patch


def test_chat_settings_defaults():
    """ChatSettings should have correct defaults without any env vars."""
    with patch.dict("os.environ", {}, clear=True):
        from core.config import ChatSettings

        settings = ChatSettings()
        assert settings.ENABLE_CONVERSATION_LOGGING is True
        assert settings.MODEL_ID == "utc-helpdesk"
        assert settings.TOP_K == 5
        assert settings.FETCH_TOP_K == 20
        assert settings.RRF_K == 1
        assert settings.MIN_VECTOR_SIMILARITY == 0.0
        assert settings.MAX_CONTEXT_TOKENS == 4000
        assert settings.REQUEST_TIMEOUT == 30.0


def test_chat_settings_from_env():
    """ChatSettings should read from CHAT_ prefixed env vars."""
    env = {
        "CHAT_ENABLE_CONVERSATION_LOGGING": "false",
        "CHAT_MODEL_ID": "test-model",
        "CHAT_TOP_K": "10",
        "CHAT_REQUEST_TIMEOUT": "60.0",
    }
    with patch.dict("os.environ", env, clear=True):
        from core.config import ChatSettings

        settings = ChatSettings()
        assert settings.ENABLE_CONVERSATION_LOGGING is False
        assert settings.MODEL_ID == "test-model"
        assert settings.TOP_K == 10
        assert settings.REQUEST_TIMEOUT == 60.0


def test_get_chat_settings_returns_instance():
    """get_chat_settings() should return a ChatSettings instance."""
    from core.config import get_chat_settings, ChatSettings

    settings = get_chat_settings()
    assert isinstance(settings, ChatSettings)
