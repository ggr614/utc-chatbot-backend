# Codebase Summary
# UTC Helpdesk Chatbot RAG Backend

---

## 1. Directory Structure

```
utc-chatbot-backend/
├── api/                    FastAPI application (HTTP layer only)
│   ├── auth.py             JWT admin authentication helpers
│   ├── dependencies.py     FastAPI Depends() providers
│   ├── main.py             App factory, lifespan, router registration
│   ├── models/             Pydantic request/response schemas
│   └── routers/            One file per route group
│   └── utils/              HTTP-layer utilities (no core/ imports)
├── alembic/                Database migration management
│   └── versions/           12 migration files (manual SQL)
├── core/                   Business logic — no api/ imports
├── docs/                   Project documentation
├── qa/                     Retrieval evaluation tools
├── tests/                  25 test files, 435 tests
├── utils/                  Shared utilities (logger)
├── alembic.ini             Alembic configuration
├── docker-compose.yml      5-service stack
├── docker-entrypoint.sh    Container startup: migrate then serve
├── Dockerfile              Multi-stage build (builder + runtime)
├── example.env             Environment variable template
├── main.py                 CLI entry point (ingest/process/embed/pipeline/evaluate/sweep)
└── requirements.txt        Pinned Python dependencies
```

---

## 2. core/ Module Inventory

The `core/` directory contains all business logic. Nothing in `core/` imports from `api/`.

| File | Class / Purpose |
|---|---|
| `config.py` | Six `BaseSettings` classes (`TDXSettings`, `DatabaseSettings`, `LiteLLMSettings`, `APISettings`, `ChatSettings`, `AuthSettings`) with `@lru_cache` accessors |
| `schemas.py` | Pydantic data models: `TdxArticle`, `TextChunk`, `VectorRecord` |
| `api_client.py` | TDX REST API client — auth, article fetching, pagination |
| `ingestion.py` | `ArticleProcessor` — fetches from TDX API, deduplicates, stores raw HTML |
| `processing.py` | `TextProcessor` — HTML-to-text via BeautifulSoup/html2text, chunking via LangChain |
| `tokenizer.py` | `Tokenizer` — token counting via `litellm.token_counter()` (local, no API call) |
| `embedding.py` | `EmbeddingGenerator` — async embedding via LiteLLM proxy, batch support |
| `pipeline.py` | `RAGPipeline` — context manager orchestrating ingest → process → embed |
| `bm25_search.py` | `BM25Retriever` — in-memory BM25Okapi, corpus caching, `BM25SearchResult` |
| `vector_search.py` | `VectorRetriever` — pgvector cosine similarity, `VectorSearchResult` |
| `reranker.py` | `Reranker` — Cohere neural reranking via LiteLLM, tracks `last_rerank_latency_ms` |
| `hyde_generator.py` | `HyDEGenerator` — hypothetical document generation via LiteLLM chat |
| `chat_service.py` | `ChatService` — RAG orchestrator: command routing, retrieval, prompt assembly, LLM streaming |
| `storage_base.py` | `BaseStorageClient` — psycopg 3 connection management, shared by all storage classes |
| `storage_raw.py` | Raw article storage (upsert to `articles` table) |
| `storage_chunk.py` | `PostgresClient` + chunk storage (`article_chunks` table) |
| `storage_vector.py` | `VectorStorageClient` base + `VectorStorage` — embedding upsert and KNN queries |
| `storage_query_log.py` | `QueryLogClient` — query log writes and reads |
| `storage_reranker_log.py` | Reranker operation logging (`reranker_logs`, `reranker_results`) |
| `storage_hyde_log.py` | HyDE generation logging (`hyde_logs`) |
| `storage_cache_metrics.py` | Cache hit/miss metrics (`cache_metrics`) |
| `storage_prompt.py` | System prompt CRUD (`tag_system_prompts`) |
| `storage_admin_user.py` | Admin user storage for JWT-authenticated admin UI |

---

## 3. api/ Module Inventory

### api/main.py
App factory and lifespan manager. On startup: creates connection pool, initializes `BM25Retriever` (loads corpus into memory), `VectorRetriever`, `Reranker`, `HyDEGenerator`, and `ChatService`. All stored in `app.state` for injection. On shutdown: closes vector retriever and connection pool.

### api/dependencies.py
FastAPI dependency providers:
- `verify_api_key()` — validates `X-API-Key` header against `API_API_KEY` and `API_ALLOWED_API_KEYS`
- `get_bm25_retriever()` — returns `app.state.bm25_retriever`
- `get_vector_retriever()` — returns `app.state.vector_retriever`
- `get_query_log_client()` — creates `QueryLogClient` using the connection pool

### api/auth.py
JWT-based admin session management: token creation, cookie handling, login form processing.

### api/models/requests.py
`BM25SearchRequest`, `VectorSearchRequest`, `HybridSearchRequest` — Pydantic validation for all search endpoints. Includes field constraints (query length, top_k range, score thresholds) and filtering fields (`status_names`, `category_names`, `is_public`, `tags`).

### api/models/responses.py
`SearchResponse`, `SearchResultChunk`, `HealthResponse`, `ComponentStatus` — unified response shapes for all search and health endpoints.

### api/models/chat.py
OpenAI wire-format models: `ChatMessage`, `ChatCompletionRequest`, `ChatCompletionChunk`, `ChatCompletionChunkChoice`, `ModelObject`, `ModelListResponse`.

### api/routers/search.py
Five search endpoints: `bm25`, `bm25/validate`, `vector`, `hybrid`, `hyde`. Each endpoint logs the query to `query_logs` (best-effort, non-blocking), runs the appropriate retrieval path, and returns a `SearchResponse`. All require `X-API-Key`.

### api/routers/health.py
`GET /health/` — checks BM25 corpus, vector storage, and database pool; returns component-level status with degraded/healthy/unhealthy. `GET /health/ready` — lightweight readiness probe (200/503), no auth.

### api/routers/query_logs.py
`POST /api/v1/query-logs/{id}/response` — idempotent LLM response logging. `GET /api/v1/query-logs/{id}/response` — retrieve stored response.

### api/routers/openai_compat.py
`GET /v1/models` and `POST /v1/chat/completions`. The chat endpoint is an async thin adapter: parses the request, delegates to `ChatService.handle_chat()`, wraps each yielded token as an SSE `data:` chunk in OpenAI format. Requires `X-API-Key`.

### api/routers/admin_prompts.py
HTML admin UI (`GET /admin/prompts`) and JSON API for tag-based system prompt management: list, bulk-save (upsert/delete), delete by tag, and list distinct article categories.

### api/routers/admin_analytics.py
HTML analytics dashboard and JSON API endpoints: overview KPIs, query volume by hour, cache performance breakdown, latency percentiles, popular queries, content statistics.

### api/routers/auth.py
Admin login (`POST /admin/login`) and logout (`POST /admin/logout`) — sets/clears the `admin_session` JWT cookie.

### api/utils/connection_pool.py
`DatabaseConnectionPool` — thread-safe psycopg_pool singleton. `get_connection_pool()` returns the existing pool or creates one. `close_connection_pool()` used during shutdown.

### api/utils/hybrid_search.py
`reciprocal_rank_fusion()` — RRF with configurable `k`. `weighted_score_fusion()` — score normalization + weighted combination. `hybrid_search()` — combines BM25 + vector results, applies fusion, optionally applies neural reranking.

### api/utils/prompt_resolution.py
`select_system_prompt()` — given article tags from search results, queries `tag_system_prompts` for the highest-priority matching prompt; falls back to `__default__`.

---

## 4. Module Dependency Graph

```
main.py (CLI)
  └── core/pipeline.py
        ├── core/ingestion.py → core/api_client.py, core/storage_raw.py
        ├── core/processing.py → core/tokenizer.py, core/storage_chunk.py
        └── core/embedding.py → core/tokenizer.py, core/storage_vector.py

api/main.py (FastAPI)
  ├── api/routers/search.py
  │     ├── api/dependencies.py
  │     ├── api/utils/hybrid_search.py
  │     │     ├── core/bm25_search.py → core/storage_chunk.py, core/schemas.py
  │     │     ├── core/vector_search.py → core/embedding.py, core/storage_vector.py
  │     │     └── core/reranker.py
  │     └── core/hyde_generator.py
  ├── api/routers/openai_compat.py
  │     └── core/chat_service.py
  │           ├── core/bm25_search.py
  │           ├── core/vector_search.py
  │           ├── core/reranker.py
  │           ├── core/storage_query_log.py
  │           └── api/utils/hybrid_search.py (injected via constructor)
  └── api/utils/connection_pool.py → psycopg_pool

All core/ storage modules → core/storage_base.py → core/config.py (DatabaseSettings)
All core/ AI modules → core/config.py (LiteLLMSettings) → litellm proxy
```

---

## 5. Key Classes and Responsibilities

| Class | Location | Responsibility |
|---|---|---|
| `RAGPipeline` | `core/pipeline.py` | Context manager, orchestrates full ingest→process→embed pipeline, returns phase statistics |
| `ArticleProcessor` | `core/ingestion.py` | Fetches TDX articles, computes diff vs. stored articles, stores new/updated, returns counts |
| `TextProcessor` | `core/processing.py` | Converts HTML → clean text → token-bounded chunks via LangChain splitter |
| `EmbeddingGenerator` | `core/embedding.py` | Async batch embedding via `litellm.aembedding()`, with sync wrapper; validates token counts |
| `BM25Retriever` | `core/bm25_search.py` | Loads `article_chunks` corpus into `BM25Okapi`, caches in-process, supports filter predicates |
| `VectorRetriever` | `core/vector_search.py` | Embeds query, runs pgvector `<=>` KNN query via `VectorStorage`, returns scored chunks |
| `Reranker` | `core/reranker.py` | Calls `litellm.rerank()` on a candidate list, re-sorts by reranker score, tracks latency |
| `HyDEGenerator` | `core/hyde_generator.py` | Calls LiteLLM chat to produce a hypothetical TDX article, returns text for embedding |
| `ChatService` | `core/chat_service.py` | Routes commands (q/qlong/bypass/debug/search/follow_up), assembles RAG context, streams LLM response as async generator |
| `BaseStorageClient` | `core/storage_base.py` | Manages psycopg 3 connection lifecycle; all storage classes inherit this |
| `VectorStorage` | `core/storage_vector.py` | Upserts embeddings to `embeddings_openai`, executes KNN queries with optional filters |
| `DatabaseConnectionPool` | `api/utils/connection_pool.py` | Singleton psycopg_pool instance shared across all API request handlers |

---

## 6. Test Coverage Overview

**Total:** 435 tests across 25 files

| File | Focus |
|---|---|
| `test_ingestion.py` | ArticleProcessor: TDX API fetch, dedup, incremental updates |
| `test_processing.py` | TextProcessor: HTML cleaning, chunking, token counting |
| `test_embedding.py` | EmbeddingGenerator: batching, LiteLLM mock, token validation |
| `test_bm25_search.py` | BM25Retriever: corpus loading, scoring, filtering, cache |
| `test_vector_search.py` | VectorRetriever: pgvector query, embedding, filter |
| `test_reranker.py` | Reranker: LiteLLM rerank mock, latency tracking, fallback |
| `test_pipeline.py` | RAGPipeline: phase orchestration, stats |
| `test_storage.py` | Chunk and raw article storage clients |
| `test_storage_vector.py` | VectorStorage: upsert, KNN, cascade |
| `test_storage_query_log.py` | QueryLogClient: write, read, LLM response logging |
| `test_storage_cache_metrics.py` | CacheMetricsClient |
| `test_storage_admin_user.py` | Admin user storage |
| `test_chat_service.py` | ChatService: command routing, RAG assembly, streaming |
| `test_chat_models.py` | OpenAI-compat Pydantic models |
| `test_openai_compat.py` | `/v1/chat/completions` and `/v1/models` endpoint tests |
| `test_prompt_resolution.py` | Tag-to-system-prompt resolution logic |
| `test_config_auth.py` | AuthSettings validation |
| `test_config_chat.py` | ChatSettings validation |
| `test_auth.py` | JWT token creation and validation |
| `test_router_auth.py` | Admin login/logout endpoints |
| `test_logger.py` | Logger setup |
| `test_eval_runner.py` | Retrieval evaluator |
| `test_eval_metrics.py` | MRR, recall, NDCG metrics |
| `test_eval_dataset.py` | QA dataset loading |
| `test_param_sweep.py` | Parameter sweep runner |

**Known gaps:**
- No integration tests against a live database (all DB calls are mocked)
- `api/routers/admin_analytics.py` — SQL queries tested indirectly; no dedicated router-level tests
- `api/routers/admin_prompts.py` — admin UI HTML rendering not tested
- HyDE end-to-end pipeline (mocked at LiteLLM boundary but not full round-trip)
- BM25 corpus loading performance under large corpora (>10k chunks)
