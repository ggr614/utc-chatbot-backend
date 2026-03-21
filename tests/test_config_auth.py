"""Tests for AuthSettings configuration."""

import pytest
from unittest.mock import patch

from core.config import AuthSettings, get_auth_settings


class TestAuthSettings:
    """Test suite for AuthSettings."""

    def test_auth_settings_loads_from_env(self):
        """Test AuthSettings loads required and optional fields from env."""
        env_vars = {
            "AUTH_SECRET_KEY": "a" * 64,
            "AUTH_TOKEN_EXPIRE_MINUTES": "60",
            "AUTH_COOKIE_NAME": "test_session",
        }
        with patch.dict("os.environ", env_vars, clear=False):
            settings = AuthSettings()
            assert settings.SECRET_KEY.get_secret_value() == "a" * 64
            assert settings.TOKEN_EXPIRE_MINUTES == 60
            assert settings.COOKIE_NAME == "test_session"

    def test_auth_settings_defaults(self):
        """Test AuthSettings uses correct defaults."""
        env_vars = {
            "AUTH_SECRET_KEY": "b" * 64,
        }
        with patch.dict("os.environ", env_vars, clear=False):
            settings = AuthSettings()
            assert settings.TOKEN_EXPIRE_MINUTES == 120
            assert settings.COOKIE_NAME == "admin_session"

    def test_auth_settings_missing_secret_key_raises(self):
        """Test AuthSettings raises if AUTH_SECRET_KEY not set."""
        with patch.dict("os.environ", {"AUTH_SECRET_KEY": ""}, clear=False):
            with pytest.raises(Exception):
                AuthSettings()

    def test_get_auth_settings_cached(self):
        """Test get_auth_settings returns cached instance."""
        env_vars = {"AUTH_SECRET_KEY": "c" * 64}
        with patch.dict("os.environ", env_vars, clear=False):
            get_auth_settings.cache_clear()
            s1 = get_auth_settings()
            s2 = get_auth_settings()
            assert s1 is s2
            get_auth_settings.cache_clear()
