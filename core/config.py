from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, field_validator
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


class ChatSettings(BaseSettings):
    """Settings for the /v1/chat/completions endpoint."""

    ENABLE_CONVERSATION_LOGGING: bool = True
    MODEL_ID: str = "utc-helpdesk"
    TOP_K: int = 5
    FETCH_TOP_K: int = 20
    RRF_K: int = 1  # RRF rank-smoothing constant, tuned low for this corpus
    MIN_VECTOR_SIMILARITY: float = 0.0
    MAX_CONTEXT_TOKENS: int = 4000
    REQUEST_TIMEOUT: float = 30.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        env_prefix="CHAT_",
    )


class AuthSettings(BaseSettings):
    """Settings for admin authentication."""

    SECRET_KEY: SecretStr
    TOKEN_EXPIRE_MINUTES: int = 120
    COOKIE_NAME: str = "admin_session"

    @field_validator("SECRET_KEY")
    @classmethod
    def secret_key_must_not_be_empty(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value():
            raise ValueError("AUTH_SECRET_KEY must not be empty")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        env_prefix="AUTH_",
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


@lru_cache()
def get_chat_settings() -> ChatSettings:
    """Get cached chat endpoint settings."""
    return ChatSettings()  # type: ignore[call-arg]


@lru_cache()
def get_auth_settings() -> AuthSettings:
    """Get cached auth settings for admin authentication."""
    return AuthSettings()  # type: ignore[call-arg]
