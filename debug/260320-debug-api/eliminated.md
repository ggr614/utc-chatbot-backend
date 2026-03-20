# Eliminated Hypotheses — API Layer Debug

Disproven hypotheses are equally valuable — they document what ISN'T the problem.

| # | Hypothesis | Why Disproven |
|---|-----------|---------------|
| 4 | Singleton connection pool race condition | Each uvicorn worker is a separate process; `get_connection_pool` called once during sequential startup |
| 5 | CORS middleware added after routers causes issues | FastAPI/Starlette middleware wraps entire app regardless of registration order |
| 6 | `verify_api_key` async in sync endpoints | FastAPI transparently handles async dependencies in sync endpoints |
| 9 | `hyde_generator.deployment_name` None access | Dependency injection guarantees non-None; attribute always exists on HyDEGenerator |
| 10 | Connection pool not cleaned up on startup failure | `finally` block correctly checks `hasattr` and cleans up |
| 11 | `reranker.model` access when reranker is None | Ternary `reranker.model if reranker else "unknown"` handles None correctly |
| 13 | Path traversal via tag_name in DELETE endpoint | Parameterized SQL prevents injection |
| 16 | System prompt extraction silently fails | Both BM25SearchResult and VectorSearchResult define `system_prompt: Optional[str]` |
| 17 | Command field validation mismatch | Storage client validates against same set as request model |
| 21 | token_usage leaks sensitive info | Standard LLM usage data, not sensitive |
| 24 | `command=None` invalid for BM25 validation | None is valid per CHECK constraint |
