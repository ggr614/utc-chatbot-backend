# System Architecture
# UTC Helpdesk Chatbot RAG Backend

---

## 1. System Component Diagram

```
External                  Docker network: helpdesk-network
─────────────────────────────────────────────────────────────────────────────

  Open WebUI (:3000) ────► LiteLLM proxy (:4000) ◄──── ai providers
  (helpdesk-webui)         (helpdesk-litellm)            (Azure OpenAI,
       │                          │                        AWS Bedrock)
       │                          │
       ▼                          ▼
  FastAPI API (:8000)      PostgreSQL (:5432)
  (helpdesk-api)           (helpdesk-db)
       │                          │
       │                          ├── helpdesk_chatbot DB (RAG tables)
       └──────────────────────────┤
                                  └── litellm DB (LiteLLM request logs)

  Prometheus (:9090) ─────► API /metrics (optional, --profile monitoring)
  (helpdesk-prometheus)

─────────────────────────────────────────────────────────────────────────────
  CLI (outside Docker)     host machine
  python main.py pipeline  connects to PostgreSQL on mapped port
```

All services communicate via the `helpdesk-network` bridge network using Docker internal DNS names (`db`, `litellm`, `api`). The `LITELLM_PROXY_BASE_URL` environment variable passed to the API container is `http://litellm:4000`, not `localhost`.

---

## 2. Service Responsibilities

### db (helpdesk-db)
- Image: `ramsrib/pgvector:16` — PostgreSQL 16 with pgvector extension pre-installed
- Hosts two logical databases on the same instance: `helpdesk_chatbot` (RAG data) and `litellm` (LiteLLM request tracking)
- The LiteLLM database and user (`llmproxy`) are created by the `db-init` one-shot container on first deploy
- Data persisted in Docker volume `helpdesk_pgvector_data`
- Health check: `pg_isready -U postgres -d helpdesk_chatbot`

### db-init (helpdesk-db-init)
- One-shot container (restart: "no") that runs after `db` is healthy
- Creates/updates the LiteLLM PostgreSQL user and database idempotently
- Exits on completion; `litellm` service waits on `service_completed_successfully`

### litellm (helpdesk-litellm)
- Image: `docker.litellm.ai/berriai/litellm:main-stable`
- Routes all AI API calls to upstream providers; model aliases defined in `config.yaml` (mounted read-only)
- Used by the API container for embeddings, chat completions, and reranking
- Also used directly by Open WebUI for model listing
- Health check: HTTP GET to `/health/liveliness`

### api (helpdesk-api)
- Built from project Dockerfile (multi-stage, Python 3.11-slim-bookworm, non-root user)
- FastAPI application served by Uvicorn (4 workers by default)
- Startup: Alembic migrations run first, then Uvicorn starts (via `docker-entrypoint.sh`)
- BM25 corpus loaded into memory once at startup; not reloaded per request
- Health check: `curl http://localhost:8000/health/ready`

### open-webui (helpdesk-webui)
- Image: `ghcr.io/open-webui/open-webui:main`
- Chat UI; connects to LiteLLM proxy as its OpenAI-compatible backend
- Users interact with the chatbot here; requests flow through LiteLLM to the API's `/v1/chat/completions` endpoint
- Ollama disabled (`OLLAMA_BASE_URLS=""`)
- Data persisted in Docker volume `open_webui_data`

### prometheus (helpdesk-prometheus)
- Optional, started with `--profile monitoring`
- 15-day retention, config in `prometheus.yml`

---

## 3. Ingestion Pipeline Data Flow

```
TDX REST API
    │  HTTPS GET /tdwebservices/TDWebApi/api/{app_id}/KnowledgeBase/Articles
    │  Auth: BEID + WebServicesKey headers
    ▼
ArticleProcessor (core/ingestion.py)
    │  Fetches all published articles with pagination
    │  Computes diff: new vs. existing by tdx_article_id
    │  Calls html2text + BeautifulSoup to detect content changes
    ▼
articles table
    │  id (UUID7), tdx_article_id (int, UNIQUE), title, url
    │  content_html, status_name, category_name, is_public
    │  summary, tags (text[]), last_modified_date
    ▼
TextProcessor (core/processing.py)
    │  Reads articles where no chunk exists yet (incremental)
    │  Strips HTML: BeautifulSoup removes tags, html2text converts to Markdown-like text
    │  Splits with LangChain RecursiveCharacterTextSplitter:
    │    - chunk_size set in tokens (not characters) via LiteLLM token counter
    │    - chunk_overlap preserves context across boundaries
    │    - Separators: ["\n\n", "\n", " ", ""]
    ▼
article_chunks table
    │  id (UUID7), parent_article_id (FK→articles CASCADE)
    │  chunk_sequence (0-indexed), text_content, token_count
    │  url, last_modified_date
    ▼
EmbeddingGenerator (core/embedding.py)
    │  Reads chunks with no embedding yet (incremental)
    │  Batches chunks (default 100); calls litellm.aembedding()
    │  Validates token count ≤ EMBED_MAX_TOKENS before sending
    │  Model: text-embedding-3-large via LiteLLM proxy → Azure OpenAI / Bedrock
    ▼
embeddings_openai table
    │  chunk_id (UUID, PK/FK→article_chunks CASCADE)
    │  embedding (vector(3072))  -- pgvector type
    │  created_at
```

All three phases are idempotent. Re-running the pipeline skips already-processed articles/chunks/embeddings. Deleting an article cascades to its chunks and then to its embeddings.

---

## 4. Query Pipeline Data Flow

### BM25 Search
```
HTTP POST /api/v1/search/bm25  {query, top_k, filters}
    │  X-API-Key validated
    ▼
BM25Retriever.search()
    │  BM25Okapi corpus already in memory (loaded at startup)
    │  Tokenizes query (whitespace + punctuation split)
    │  Scores all chunks, applies filter predicates (status, category, is_public, tags)
    │  Returns top_k BM25SearchResult objects
    ▼
SearchResponse  {results: [{rank, score, chunk_id, text_content, source_url, ...}]}
```

### Vector Search
```
HTTP POST /api/v1/search/vector  {query, top_k, min_similarity, filters}
    │  X-API-Key validated
    ▼
VectorRetriever.search()
    │  EmbeddingGenerator embeds query → 3072-dim vector via LiteLLM proxy
    ▼
VectorStorage.search()
    │  SQL: SELECT ... FROM embeddings_openai e
    │       JOIN article_chunks c ON e.chunk_id = c.id
    │       JOIN articles a ON c.parent_article_id = a.id
    │       WHERE ... (filters)
    │       ORDER BY e.embedding <=> %s  LIMIT top_k
    │  pgvector cosine distance operator (<=>)
    ▼
VectorSearchResult list → SearchResponse
```

### Hybrid Search (default query path)
```
HTTP POST /api/v1/search/hybrid  {query, top_k, fetch_k, fusion_method, weights, use_reranker}
    │  X-API-Key validated
    ▼
hybrid_search() in api/utils/hybrid_search.py
    │
    ├── BM25Retriever.search(query, top_k=fetch_k)  ← fetch more for fusion
    ├── VectorRetriever.search(query, top_k=fetch_k)
    │
    ▼
reciprocal_rank_fusion(bm25_results, vector_results, k=RRF_K)
    │  Combined score = Σ(1 / (k + rank_i)) per chunk
    │  Deduplicates across BM25 and vector result sets
    ▼
[optional] Reranker.rerank(query, candidates, top_k)
    │  litellm.rerank() → Cohere Rerank v3.5
    │  Re-sorts by semantic relevance score
    ▼
SearchResponse (top_k results after fusion/reranking)
```

### HyDE Search
```
HTTP POST /api/v1/search/hyde  {query, top_k, ...}
    │  X-API-Key validated
    ▼
HyDEGenerator.generate(query)
    │  Prompt: "Write a UTC IT helpdesk knowledge base article that answers: {query}"
    │  litellm.acompletion() → LLM generates a hypothetical article (~200-400 tokens)
    ▼
hybrid_search(hypothetical_document, ...)  ← same pipeline as above
    │  Query embedding is of the hypothetical document, not the original question
    ▼
SearchResponse
```

---

## 5. Chat Pipeline Data Flow

```
Open WebUI  →  POST /v1/chat/completions
               {model, messages, stream: true}
               X-API-Key header
    │
    ▼
openai_compat.chat_completions() (api/routers/openai_compat.py)
    │  Thin adapter: extracts last user message, email header
    │  Calls ChatService.handle_chat() → async generator
    │
    ▼
ChatService.handle_chat() (core/chat_service.py)
    │
    ├── Command parsing: detects prefix (q/, qlong/, bypass/, debug/, search/, follow_up/)
    │   └── bypass/follow_up: skip RAG, call LLM directly with NO_RAG system prompt
    │
    ├── [if RAG command] hybrid_search(query, fetch_k=CHAT_FETCH_TOP_K)
    │   └── BM25 + vector → RRF fusion → Reranker → top CHAT_TOP_K chunks
    │
    ├── select_system_prompt(article_tags)
    │   └── Queries tag_system_prompts for highest-priority matching tag
    │   └── Falls back to __default__ system prompt
    │
    ├── Context assembly:
    │   └── Top chunks formatted as numbered KB articles with URL citations
    │   └── Token budget: MAX_CONTEXT_TOKENS (4000 by default)
    │
    ├── litellm.acompletion(stream=True, ...)
    │   └── Async generator yields delta chunks from LLM
    │
    ├── Per-chunk SSE formatting:
    │   └── Each token delta → ChatCompletionChunk → JSON → "data: {...}\n\n"
    │
    └── Query logging (best-effort):
        └── query_logs, query_results, llm_responses, reranker_logs written async

    ▼
StreamingResponse (SSE) → Open WebUI renders tokens as they arrive
```

---

## 6. Database Schema Relationships

```
articles (id UUID PK)
    │
    ├─── article_chunks (id UUID PK, parent_article_id FK → articles CASCADE)
    │        │
    │        └─── embeddings_openai (chunk_id UUID PK/FK → article_chunks CASCADE)
    │
    └─── warm_cache_entries (article_id FK → articles CASCADE)
              │
              └─── cache_metrics (cache_entry_id FK → warm_cache_entries SET NULL)

query_logs (id bigserial PK)
    │
    ├─── query_results (query_log_id FK → query_logs CASCADE)
    ├─── llm_responses (query_log_id FK → query_logs UNIQUE CASCADE)
    ├─── hyde_logs     (query_log_id FK → query_logs UNIQUE CASCADE)
    └─── reranker_logs (query_log_id FK → query_logs UNIQUE CASCADE)
              │
              └─── reranker_results (query_log_id FK → query_logs CASCADE)

tag_system_prompts (id UUID PK, tag_name UNIQUE)
    └── Seeded with __default__ at priority 1000
```

Cascade behavior:
- Delete `articles` row → cascades to `article_chunks` → cascades to `embeddings_openai`
- Delete `query_logs` row → cascades to all five child log tables
- Delete `warm_cache_entries` → cascades to `cache_metrics` (SET NULL, not delete)

---

## 7. Authentication Architecture

### Search and Chat Endpoints (X-API-Key)
All endpoints under `/api/v1/` and `/v1/` require the `X-API-Key` header. The `verify_api_key()` dependency in `api/dependencies.py` compares the provided key against:
1. `API_API_KEY` environment variable (primary key)
2. `API_ALLOWED_API_KEYS` comma-separated list (additional keys, e.g. for Open WebUI)

No rate limiting is applied (internal network only).

### Admin UI (JWT Session Cookie)
The admin endpoints (`/admin/prompts`, `/admin/analytics`) require a browser session cookie set by the admin login flow:
1. `POST /admin/login` — verifies username/password against `admin_users` table (bcrypt), issues a JWT signed with `AUTH_SECRET_KEY`, sets `admin_session` cookie
2. `GET /admin/prompts` etc. — validates JWT from cookie via `api/auth.py`
3. `POST /admin/logout` — clears cookie

Token expiry: `AUTH_TOKEN_EXPIRE_MINUTES` (default 120 minutes).

### Health Endpoints (No Auth)
`/health/` and `/health/ready` require no authentication, allowing monitoring systems and Docker health checks to access them freely.

---

## 8. Connection Pool and Resource Management

The API uses a single `psycopg_pool.ConnectionPool` shared across all workers (when using a single-process run) or per-worker (when using Uvicorn multi-worker mode, since each worker is a separate process).

```
APISettings:
  POOL_MIN_SIZE = 5   (connections kept open at all times)
  POOL_MAX_SIZE = 20  (max concurrent connections)
  POOL_TIMEOUT  = 30s (wait time before raising PoolTimeout)
```

Resource initialization order at startup:
1. `DatabaseConnectionPool` — psycopg_pool opened
2. `BM25Retriever` — corpus loaded from `article_chunks` via direct connection; cached in-process
3. `EmbeddingGenerator` + `VectorStorage` + `VectorRetriever` — connection pool assigned
4. `Reranker` — initialized (fails gracefully if LiteLLM unreachable at startup)
5. `HyDEGenerator` — initialized (fails gracefully)
6. `ChatService` — receives all of the above via constructor injection

At shutdown, the lifespan context manager closes the vector retriever then calls `close_connection_pool()`.

---

## 9. Docker Network and Port Mapping

| Service | Internal Port | External Port (default) | Variable |
|---|---|---|---|
| db | 5432 | 5432 | `DB_EXTERNAL_PORT` |
| litellm | 4000 | 4000 | `LITELLM_EXTERNAL_PORT` |
| api | 8000 | 8000 | `API_EXTERNAL_PORT` |
| open-webui | 8080 | 3000 | `OPEN_WEBUI_PORT` |
| prometheus | 9090 | 9090 | `PROMETHEUS_EXTERNAL_PORT` |

Service startup dependency order:
```
db (healthy)
  └── db-init (completed_successfully)
        └── litellm (healthy)
              └── api (depends on db healthy + litellm healthy)
              └── open-webui (depends on litellm healthy)
```

The `api` container depends on both `db` and `litellm` being healthy before it starts, because the startup sequence immediately tries to load the BM25 corpus from PostgreSQL and initialize the embedding generator (which contacts LiteLLM).
