from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from functools import lru_cache


class Settings(BaseSettings):
    # TDX
    WEBSERVICES_KEY: SecretStr
    BEID: SecretStr
    BASE_URL: str
    APP_ID: int = 2717

    # Postgres
    DB_HOST: str
    DB_PORT: int = 5432
    DB_USER: str
    DB_PASSWORD: SecretStr
    DB_NAME: str

    # AWS Bedrock
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: SecretStr
    AWS_SECRET_ACCESS_KEY: SecretStr
    AWS_EMBED_MODEL_ID: str
    AWS_EMBED_DIM: int
    AWS_MAX_TOKENS: int

    # Azure AI Foundry
    AZURE_OPENAI_API_KEY: SecretStr
    AZURE_OPENAI_EMBED_ENDPOINT: str
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_EMBED_DIM: int
    AZURE_MAX_TOKENS: int
    AZURE_OPENAI_CHAT_ENDPOINT: str
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: str
    AZURE_OPENAI_CHAT_API_VERSION: str
    AZURE_OPENAI_CHAT_MAX_TOKENS: int
    AZURE_OPENAI_CHAT_TEMPERATURE: float
    AZURE_OPENAI_CHAT_COMPLETION_TOKENS: int
    AZURE_OPENAI_CHAT_API_KEY: SecretStr

    # API Server
    API_API_KEY: SecretStr
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_LOG_LEVEL: str = "info"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore
