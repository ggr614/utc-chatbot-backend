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
