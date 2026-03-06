# RAG Helpdesk Backend

A Retrieval-Augmented Generation (RAG) backend for an IT helpdesk chatbot. Ingests knowledge base articles from TeamDynamix (TDX), processes them into text chunks, generates embeddings via LiteLLM proxy, and stores everything in PostgreSQL with pgvector for hybrid retrieval. Includes a FastAPI REST API with BM25, vector, hybrid, and HyDE search endpoints, neural reranking, admin dashboards, and an evaluation suite.

## Architecture

```
TDX API ─→ Ingestion ─→ Processing ─→ Embedding (LiteLLM) ─→ PostgreSQL + pgvector
                                                                      │
                                                              FastAPI REST API
                                                                      │
                            ┌─────────────┬──────────────┬────────────┤
                          BM25        Vector          Hybrid        HyDE
                        (sparse)     (dense)      (BM25+Vector)  (hypothetical
                                                   + Reranker    doc + hybrid
                                                                  + reranker)
                                                                      │
                                                              Admin UI / Analytics
```

**Pipeline:** `ingest` (TDX API → raw HTML) → `process` (HTML → text chunks) → `embed` (chunks → vectors via LiteLLM)

**Search:** BM25 sparse retrieval, vector semantic retrieval, hybrid fusion (RRF) with Cohere neural reranking, and HyDE (hypothetical document generation + hybrid + reranking)

**All AI calls** (embeddings, chat/HyDE, reranking) route through a LiteLLM proxy using model aliases defined in `config.yaml`.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp example.env .env
# Edit .env with your credentials
```

**Required variables:**

```bash
# TDX API
TDX_WEBSERVICES_KEY=your_key
TDX_BEID=your_beid
TDX_BASE_URL=https://your-instance.teamdynamix.com
TDX_APP_ID=2717

# PostgreSQL
DB_HOST=localhost
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=helpdesk_chatbot

# LiteLLM Proxy (all AI calls go through here)
LITELLM_PROXY_BASE_URL=http://localhost:4000
LITELLM_PROXY_API_KEY=your_litellm_key
LITELLM_EMBEDDING_MODEL=text-embedding-large-3
LITELLM_CHAT_MODEL=gpt-5.2-chat
LITELLM_RERANKER_MODEL=cohere-rerank-v3-5
LITELLM_EMBED_DIM=3072
LITELLM_EMBED_MAX_TOKENS=8191
LITELLM_CHAT_MAX_TOKENS=8191
LITELLM_CHAT_COMPLETION_TOKENS=500
LITELLM_CHAT_TEMPERATURE=0.7

# API Authentication
API_API_KEY=your-secret-api-key
API_ALLOWED_API_KEYS=          # Optional: comma-separated additional keys
```

**Optional API server settings:**

```bash
API_POOL_MIN_SIZE=5            # Connection pool min (default: 5)
API_POOL_MAX_SIZE=20           # Connection pool max (default: 20)
API_POOL_TIMEOUT=30.0          # Connection timeout in seconds
API_HOST=0.0.0.0               # Bind address
API_PORT=8000                  # Server port
API_WORKERS=4                  # Uvicorn worker count
API_LOG_LEVEL=info             # Logging level
```

### 3. Database Setup

The project uses Alembic for database migrations.

```bash
# Apply all migrations (creates tables, extensions, indexes)
alembic upgrade head

# Check current migration status
alembic current

# View migration history
alembic history

# Rollback last migration
alembic downgrade -1

# For existing databases — mark as up-to-date
alembic stamp head
```

### 4. Run the Pipeline

```bash
# Full pipeline: ingest → process → embed
python main.py pipeline

# Individual phases
python main.py ingest
python main.py process
python main.py embed --batch-size 100

# Skip phases
python main.py pipeline --skip-ingestion
python main.py pipeline --skip-processing --skip-embedding
```

### 5. Start the API Server

```bash
# Development (auto-reload)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production (4 workers)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4 --limit-concurrency 100
```

API docs at `http://localhost:8000/docs` (Swagger) or `/redoc` (ReDoc).

## CLI Commands

All commands: `python main.py [--log-level LEVEL] <command> [options]`

Global option `--log-level` must come **before** the subcommand.

| Command | Description | Key Options |
|---------|-------------|-------------|
| `ingest` | Fetch articles from TDX API | `--stats` |
| `process` | Convert HTML to text chunks | `--article-ids ID [ID ...]` |
| `embed` | Generate vector embeddings | `--batch-size N` (default: 100) |
| `pipeline` | Run full pipeline (ingest+process+embed) | `--skip-ingestion`, `--skip-processing`, `--skip-embedding`, `--article-ids`, `--batch-size` |
| `evaluate` | Run retrieval evaluation against QA dataset | `--dataset PATH`, `--methods {bm25,vector,hybrid}`, `--k-values`, `--sample-size N`, `--quality-filter`, `--output-dir` |
| `sweep` | Run vector search parameter sweep | `--dataset PATH`, `--top-k-values`, `--min-similarity-values`, `--primary-metric {mrr,hit_rate_at_5,ndcg_at_5}`, `--output-dir` |

## REST API

### Search Endpoints

All search endpoints require `X-API-Key` header authentication.

| Method | Endpoint | Description | Typical Latency |
|--------|----------|-------------|-----------------|
| POST | `/api/v1/search/bm25` | BM25 keyword-based sparse retrieval | ~50-100ms |
| POST | `/api/v1/search/bm25/validate` | BM25 validation (all results with scores, IDs only) | ~50-100ms |
| POST | `/api/v1/search/vector` | Vector semantic similarity search | ~500ms-1s |
| POST | `/api/v1/search/hybrid` | Hybrid BM25+vector with RRF fusion and neural reranking | ~800ms-1.5s |
| POST | `/api/v1/search/hyde` | HyDE: hypothetical doc generation + hybrid + reranking | ~1.5-2.5s |

All search endpoints support filtering by `status_names`, `category_names`, `is_public`, and `tags`.

### Query Log Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/v1/query-logs/{id}/response` | Yes | Log LLM response for a query (idempotent, 409 on duplicate) |
| GET | `/api/v1/query-logs/{id}/response` | Yes | Retrieve LLM response for a query log |

### Admin Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/prompts` | HTML admin UI for system prompt management |
| GET | `/api/v1/admin/prompts` | List all tag-based system prompts |
| POST | `/api/v1/admin/prompts/bulk-save` | Upsert/delete prompts for multiple tags |
| DELETE | `/api/v1/admin/prompts/{tag_name}` | Delete a prompt by tag (cannot delete `__default__`) |
| GET | `/api/v1/admin/categories` | List distinct article categories |

### Analytics Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/analytics` | HTML analytics dashboard |
| GET | `/api/v1/admin/analytics/overview` | KPI summary (total queries, cache hit rate, latency percentiles) |
| GET | `/api/v1/admin/analytics/query-volume` | Query count by hour (time-series) |
| GET | `/api/v1/admin/analytics/cache-performance` | Cache hit/miss breakdown |
| GET | `/api/v1/admin/analytics/latency` | Latency percentiles (avg, p50, p95, p99) |
| GET | `/api/v1/admin/analytics/popular-queries` | Most frequent queries with stats |
| GET | `/api/v1/admin/analytics/content-stats` | Article, chunk, and embedding counts |

### Health Endpoints (no auth required)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health/` | Detailed health check with component status |
| GET | `/health/ready` | Simple readiness probe for orchestration |

**Authentication:** All search and query-log endpoints require `X-API-Key` header. Set via `API_API_KEY` env var; multiple keys via `API_ALLOWED_API_KEYS` (comma-separated). Admin and health endpoints do not require authentication.

For complete API documentation, request/response schemas, and examples, see [API_README.md](API_README.md).

## Database Schema

All tables use UUIDs as primary keys (content tables) or bigserial (analytics tables). Schema managed with Alembic migrations — see [alembic/versions/](alembic/versions/).

### Content Tables

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `articles` | Raw HTML from TDX | `tdx_article_id` (UNIQUE), `title`, `content_html`, `status_name`, `category_name`, `is_public`, `tags` (text[]) |
| `article_chunks` | Processed text chunks | `parent_article_id` (FK→articles, CASCADE), `chunk_sequence`, `text_content`, `token_count` |
| `embeddings_openai` | Vector embeddings (normalized) | `chunk_id` (PK+FK→article_chunks, CASCADE), `embedding` vector(3072) |
| `warm_cache_entries` | Pre-computed Q&A cache | `canonical_question`, `verified_answer`, `query_embedding`, `article_id` (FK→articles), `is_active` |
| `tag_system_prompts` | Tag-to-system-prompt mapping | `tag_name` (UNIQUE), `system_prompt`, `priority` (higher wins). Seeded with `__default__` |

### Analytics Tables

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `query_logs` | Search request logging | `raw_query`, `email`, `command` (bypass/q/qlong/debug/debuglong/search/follow_up), `cache_result`, `latency_ms` |
| `query_results` | Per-result logging | `query_log_id`, `search_method` (bm25/vector/hybrid/hyde), `rank`, `score`, `chunk_id` |
| `llm_responses` | LLM response logging (1:1 with query_logs) | `response_text`, `model_name`, `llm_latency_ms`, token counts, `citations` (JSONB) |
| `hyde_logs` | HyDE generation logging (1:1) | `hypothetical_document`, `generation_status`, `generation_latency_ms`, `embedding_latency_ms` |
| `reranker_logs` | Reranker operation logging (1:1) | `reranker_status`, `model_name`, `reranker_latency_ms`, `avg_rank_change` |
| `reranker_results` | Per-result reranking details | `rrf_rank`, `rrf_score`, `reranked_rank`, `reranked_score`, `rank_change` |
| `cache_metrics` | Cache performance tracking | `cache_entry_id`, `cache_type`, `latency_ms` |

**Relationships:** `articles` → `article_chunks` → `embeddings_openai` (CASCADE DELETE chain). Analytics tables reference `query_logs` via foreign keys. 1:1 tables (`llm_responses`, `hyde_logs`, `reranker_logs`) enforce uniqueness on `query_log_id`.

## Docker Deployment

### Quick Start

```bash
cp example.env .env.local
# Edit .env.local — set DB_HOST=host.docker.internal for Docker Desktop

# Build and run
docker-compose --env-file .env.local up --build

# Stop
docker-compose down
```

### Docker Compose Services

The `docker-compose.yml` includes PostgreSQL, LiteLLM proxy, FastAPI, and Open WebUI. A separate `docker-compose.prod.yml` provides a stripped-down production configuration.

### Key Features

- Multi-stage build (builder + runtime, ~500-700MB final image)
- Non-root user (`appuser`)
- Built-in health checks
- Automatic Alembic migrations on startup
- Connection pooling for concurrent requests

### Database Connectivity

- **Docker Desktop (macOS/Windows):** `DB_HOST=host.docker.internal`
- **Linux:** Use `--network host` or host IP
- **Production:** Use actual database hostname

### Resource Requirements

- **Memory:** 1-2GB per worker (BM25 corpus loaded into memory)
- **CPU:** 2+ cores recommended
- **Recommended:** 4 uvicorn workers, 100 max concurrent requests

For troubleshooting and production deployment guidance, see the health check, migration, and resource sections in the Docker documentation.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_ingestion.py -v

# Run with debug logging
pytest tests/ -v --log-cli-level=DEBUG
```

Tests use `pytest` with `pytest-mock`. All database and API calls are mocked — no live database required. Fixtures are defined in `tests/conftest.py`.

## Code Quality

```bash
# Format
ruff format .

# Lint
ruff check .

# Auto-fix lint issues
ruff check --fix .
```

## Core Modules

| Module | Class | Purpose |
|--------|-------|---------|
| `core/ingestion.py` | `ArticleProcessor` | Fetches articles from TDX API, handles deduplication |
| `core/processing.py` | `TextProcessor` | HTML→text conversion, chunking via LangChain |
| `core/embedding.py` | `EmbeddingGenerator` | Async embedding generation via LiteLLM |
| `core/hyde_generator.py` | `HyDEGenerator` | Hypothetical document generation for HyDE search |
| `core/reranker.py` | `Reranker` | Neural reranking via Cohere through LiteLLM |
| `core/bm25_search.py` | `BM25Retriever` | BM25 sparse retrieval |
| `core/vector_search.py` | `VectorRetriever` | pgvector cosine similarity retrieval |
| `core/pipeline.py` | `RAGPipeline` | Orchestrates full ingest→process→embed pipeline |
| `core/tokenizer.py` | `Tokenizer` | Token counting via `litellm.token_counter()` |
| `core/config.py` | `Settings` | Pydantic BaseSettings from `.env` |
| `core/storage_*.py` | Various | Storage clients (all inherit `BaseStorageClient`) |
