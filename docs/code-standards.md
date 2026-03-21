# Code Standards
# UTC Helpdesk Chatbot RAG Backend

---

## 1. Python Conventions

### Style
- PEP 8 enforced via `ruff`. Run `ruff check .` before committing; `ruff check --fix .` for auto-fixable issues.
- `ruff format .` for formatting (replaces Black).
- Line length: ruff default (88 characters).
- Imports grouped: stdlib, third-party, local — with blank lines between groups.

### Type Hints
All function signatures must have type hints. Use built-in generics (`list[str]`, `dict[str, Any]`) rather than `typing.List`/`typing.Dict` except where backwards compatibility with the existing codebase requires it (some older modules still use `from typing import List, Dict`).

```python
# Correct
def search(self, query: str, top_k: int = 10) -> list[BM25SearchResult]:

# Avoid (legacy style, still present in older modules)
def search(self, query: str, top_k: int = 10) -> List[BM25SearchResult]:
```

### Strings
Use f-strings for all string interpolation. Never use `%` formatting or `.format()` for new code.

```python
logger.info(f"Loaded {len(chunks)} chunks into BM25 corpus")
```

### Docstrings
Module-level docstrings on all `core/` and `api/` modules. Class and method docstrings for public APIs. Keep docstrings factual and brief — describe what, not how. One-line docstrings for simple getters/properties.

---

## 2. Configuration Pattern

### Pydantic BaseSettings with env_prefix
Each concern has its own `BaseSettings` subclass in `core/config.py` with a distinct `env_prefix`. This prevents collisions across the large `.env` file.

```python
class LiteLLMSettings(BaseSettings):
    PROXY_BASE_URL: str
    EMBEDDING_MODEL: str = "text-embedding-large-3"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        env_prefix="LITELLM_",
    )
```

Environment variables use the prefix pattern: `LITELLM_PROXY_BASE_URL`, `DB_HOST`, `TDX_APP_ID`, etc.

### @lru_cache Singletons
All settings are accessed through `@lru_cache`-decorated accessor functions, not by instantiating `Settings()` directly in module scope.

```python
@lru_cache()
def get_litellm_settings() -> LiteLLMSettings:
    return LiteLLMSettings()
```

**Test pattern:** To override settings in tests, clear the cache before and after:
```python
from core.config import get_litellm_settings
get_litellm_settings.cache_clear()
monkeypatch.setenv("LITELLM_EMBEDDING_MODEL", "test-model")
# ... run test ...
get_litellm_settings.cache_clear()
```

### Secrets
Use `pydantic.SecretStr` for credentials (`TDX_WEBSERVICES_KEY`, `DB_PASSWORD`, `API_API_KEY`, `AUTH_SECRET_KEY`). Access via `.get_secret_value()`. Never log or serialize secret values.

---

## 3. Storage Client Pattern

All database clients inherit from `BaseStorageClient` in `core/storage_base.py`. The base class manages psycopg 3 connection lifecycle.

```python
class SomeStorageClient(BaseStorageClient):
    def __init__(self, connection_pool=None):
        super().__init__(connection_pool=connection_pool)

    def store_something(self, data: SomeModel) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO ...", (...,))
            conn.commit()
```

- Use `self._get_connection()` as a context manager; it returns a connection from the pool or opens a direct connection if no pool is provided.
- Commit explicitly after writes (`conn.commit()`).
- Never commit in `BaseStorageClient` — commit is the responsibility of the caller method.
- All SQL is raw string literals. No ORM query building.

### Naming Conventions for Storage Files
- `storage_raw.py` — raw article storage
- `storage_chunk.py` — chunk storage (also contains `PostgresClient` used by BM25)
- `storage_vector.py` — embedding storage
- `storage_query_log.py` — query logging
- `storage_{noun}.py` — general pattern for all new storage modules

---

## 4. Dependency Injection Pattern

The `api/` layer uses FastAPI's `Depends()` for all shared resources. The DI providers live in `api/dependencies.py`.

```python
# In dependencies.py
def get_bm25_retriever(request: Request) -> BM25Retriever:
    return request.app.state.bm25_retriever

# In a router
@router.post("/bm25")
def bm25_search(
    body: BM25SearchRequest,
    bm25_retriever: BM25Retriever = Depends(get_bm25_retriever),
    api_key: str = Depends(verify_api_key),
):
    ...
```

Shared resources (retrievers, connection pool, chat service) are initialized once in the lifespan function and stored in `app.state`. They are never re-initialized per request.

For `core/` classes that need dependencies (like `ChatService`), use constructor injection:

```python
chat_service = ChatService(
    bm25_retriever=bm25_retriever,
    vector_retriever=vector_retriever,
    hybrid_search_fn=hybrid_search,  # injected callable
    select_system_prompt_fn=select_system_prompt,
)
```

Injecting callables rather than importing them directly makes mocking straightforward in tests and enforces the `core/ does not import api/` rule.

---

## 5. Error Handling Patterns

### Graceful Degradation
Components that can fail at startup (reranker, HyDE generator) are wrapped in try/except in the lifespan function. If they fail, `app.state` is set to `None` and the dependent endpoints either fall back (hybrid search uses RRF without reranking) or return 503.

```python
try:
    reranker = Reranker()
    app.state.reranker = reranker
except Exception as e:
    logger.error(f"Failed to initialize reranker: {e}", exc_info=True)
    app.state.reranker = None
    # hybrid search will use RRF fallback
```

### Best-Effort Query Logging
Query logging must not cause search requests to fail. Wrap logging calls in try/except:

```python
try:
    log_id = query_log_client.log_query(...)
except Exception as e:
    logger.warning(f"Query logging failed (non-fatal): {e}")
    log_id = None
```

### Structured Logging
Use `utils/logger.py` for all logging. Do not use `print()`. Log at appropriate levels:
- `DEBUG` — internal state, per-chunk details, SQL parameters
- `INFO` — pipeline progress, startup events, request completion
- `WARNING` — non-fatal degradation (empty corpus, reranker unavailable)
- `ERROR` — exceptions that affect the response, always with `exc_info=True`

```python
from utils.logger import get_logger
logger = get_logger(__name__)
```

### HTTP Errors
Raise `fastapi.HTTPException` with specific status codes. Never return error details in 200 responses.

```python
raise HTTPException(status_code=503, detail="BM25 corpus is empty — run the ingestion pipeline first")
```

---

## 6. Testing Conventions

### Framework
`pytest` with `pytest-mock` (mocker fixture) and `pytest-asyncio` for async tests.

### Fixtures
Shared fixtures live in `tests/conftest.py`: `mock_settings`, `mock_tdx_articles`, `mock_chunks`. Test-specific fixtures are defined in the individual test file.

### Mocking Pattern
Mock at the boundary closest to the external dependency. For database calls, patch the psycopg connection/cursor. For LiteLLM calls, patch `litellm.embedding`, `litellm.aembedding`, `litellm.rerank`, etc.

```python
def test_generate_embedding(mocker):
    mock_response = mocker.MagicMock()
    mock_response.data = [mocker.MagicMock(embedding=[0.1] * 3072)]
    mocker.patch("litellm.embedding", return_value=mock_response)
    ...
```

### lru_cache in Tests
Settings use `@lru_cache`. Clear the cache when using `monkeypatch.setenv()` to inject test values:

```python
def test_custom_model(monkeypatch):
    monkeypatch.setenv("LITELLM_CHAT_MODEL", "test-chat")
    get_litellm_settings.cache_clear()
    settings = get_litellm_settings()
    assert settings.CHAT_MODEL == "test-chat"
    get_litellm_settings.cache_clear()  # prevent leaking into other tests
```

### Async Tests
Mark async test functions with `@pytest.mark.asyncio`. For `ChatService.handle_chat()` tests, use `async for chunk in service.handle_chat(...)` with an `asyncio.run()` wrapper or the pytest-asyncio runner.

### No Real Database
Tests do not connect to a live database. All `psycopg.connect()` calls and cursor operations are patched. Verify SQL with `mock_cursor.execute.assert_called_with(expected_sql, expected_params)`.

---

## 7. Database Patterns

### UUIDs
All table primary keys are UUIDs. Use `uuid_utils.uuid7()` (time-ordered UUIDs) for generating new IDs. Never use `uuid.uuid4()` for new PKs unless there is a specific reason to avoid time ordering.

```python
from uuid_utils import uuid7
chunk_id = uuid7()
```

### Alembic Migrations
All schema changes go through Alembic. No schema changes via ad-hoc `psql` or direct SQL scripts.

1. Create the migration: `alembic revision -m "descriptive_name"`
2. Edit the generated file in `alembic/versions/`. Write both `upgrade()` and `downgrade()`.
3. Test round-trip: `alembic upgrade head` then `alembic downgrade -1` then `alembic upgrade head`.
4. Commit the migration file.

Migration files use `op.execute()` for raw SQL. Table creation uses `op.create_table()` with explicit column definitions. Do not use `--autogenerate` (target_metadata is None in `alembic/env.py`).

### Connection Pool Usage
In the API layer, always use the connection pool. Never call `psycopg.connect()` directly in a router. Use the pool via `get_query_log_client()` dependency or by passing the pool to a storage client.

```python
# In api/dependencies.py
def get_query_log_client(request: Request) -> QueryLogClient:
    return QueryLogClient(connection_pool=request.app.state.connection_pool)
```

---

## 8. Naming Conventions

| Item | Convention | Example |
|---|---|---|
| Storage modules | `storage_{noun}.py` | `storage_hyde_log.py` |
| Router files | `{noun_group}.py` | `admin_analytics.py` |
| Request models | `{Method}Request` | `HybridSearchRequest` |
| Response models | `{Noun}Response` | `SearchResponse` |
| Settings classes | `{Domain}Settings` | `ChatSettings` |
| Settings accessors | `get_{domain}_settings()` | `get_chat_settings()` |
| Test files | `test_{module_name}.py` | `test_chat_service.py` |
| Migration files | `{hash}_{description}.py` | `af41a58144ab_add_query_results.py` |

### Environment Variable Prefixes
| Prefix | Domain |
|---|---|
| `TDX_` | TeamDynamix API |
| `DB_` | PostgreSQL RAG database |
| `LITELLM_` | LiteLLM proxy and models |
| `API_` | FastAPI server settings |
| `CHAT_` | Chat endpoint behavior |
| `AUTH_` | Admin JWT authentication |
