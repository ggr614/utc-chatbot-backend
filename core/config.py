from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from functools import lru_cache


class TDXSettings(BaseSettings):
    WEBSERVICES_KEY: SecretStr
    BEID: SecretStr
    BASE_URL: str
    APP_ID: int = 2717

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        env_prefix="TDX_",
    )


class DatabaseSettings(BaseSettings):
    HOST: str
    USER: str
    PASSWORD: SecretStr
    NAME: str
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        env_prefix="DB_",
    )


class LiteLLMSettings(BaseSettings):
    """Settings for LiteLLM proxy connection and model configuration."""

    PROXY_BASE_URL: str
    PROXY_API_KEY: SecretStr

    # Model aliases (as defined in litellm_config.yml)
    EMBEDDING_MODEL: str = "text-embedding-large-3"
    CHAT_MODEL: str = "gpt-5.2-chat"
    RERANKER_MODEL: str = "cohere-rerank-v3-5"

    # Embedding config
    EMBED_DIM: int = 3072
    EMBED_MAX_TOKENS: int = 8191

    # Chat config
    CHAT_MAX_TOKENS: int = 8191
    CHAT_COMPLETION_TOKENS: int = 500
    CHAT_TEMPERATURE: float = 0.7

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        env_prefix="LITELLM_",
    )


class APISettings(BaseSettings):
    """Settings for FastAPI application."""

    API_KEY: SecretStr
    ALLOWED_API_KEYS: str = ""
    POOL_MIN_SIZE: int = 5
    POOL_MAX_SIZE: int = 20
    POOL_TIMEOUT: float = 30.0
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    LOG_LEVEL: str = "info"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        env_prefix="API_",
    )


# Cached accessor functions for modules to get their settings


@lru_cache()
def get_tdx_settings() -> TDXSettings:
    """Get cached TDX API settings for ingestion module."""
    return TDXSettings()  # type: ignore[call-arg]


@lru_cache()
def get_database_settings() -> DatabaseSettings:
    """Get cached database settings for storage modules."""
    return DatabaseSettings()  # type: ignore[call-arg]


@lru_cache()
def get_litellm_settings() -> LiteLLMSettings:
    """Get cached LiteLLM proxy settings."""
    return LiteLLMSettings()  # type: ignore[call-arg]


@lru_cache()
def get_api_settings() -> APISettings:
    """Get cached API settings for FastAPI application."""
    return APISettings()  # type: ignore[call-arg]
