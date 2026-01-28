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


class EmbeddingSettings(BaseSettings):
    API_KEY: SecretStr
    ENDPOINT: str
    DEPLOYMENT_NAME: str
    API_VERSION: str
    EMBED_DIM: int
    MAX_TOKENS: int

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        env_prefix="EMBEDDING_",
    )


class ChatSettings(BaseSettings):
    API_KEY: SecretStr
    ENDPOINT: str
    DEPLOYMENT_NAME: str
    API_VERSION: str
    MAX_TOKENS: int
    TEMPERATURE: float
    COMPLETION_TOKENS: int

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        env_prefix="CHAT_",
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
def get_embedding_settings() -> EmbeddingSettings:
    """Get cached embedding settings for embedding module."""
    return EmbeddingSettings()  # type: ignore[call-arg]


@lru_cache()
def get_chat_settings() -> ChatSettings:
    """Get cached chat settings for LLM/chat module."""
    return ChatSettings()  # type: ignore[call-arg]


@lru_cache()
def get_api_settings() -> APISettings:
    """Get cached API settings for FastAPI application."""
    return APISettings()  # type: ignore[call-arg]
