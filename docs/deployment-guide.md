# Deployment Guide
# UTC Helpdesk Chatbot RAG Backend

---

## 1. Prerequisites

- Docker Desktop (or Docker Engine + Compose plugin) — `docker compose version` should return v2.x
- Git access to this repository
- TDX API credentials: `TDX_WEBSERVICES_KEY` and `TDX_BEID` (obtain from TDX admin)
- LiteLLM proxy credentials or direct API keys (Azure OpenAI or AWS Bedrock)
- At least 4 GB free RAM for the API container (BM25 corpus + 4 Uvicorn workers)
- PostgreSQL port 5432 not already in use on the host (or override `DB_EXTERNAL_PORT`)

For local development without Docker:
- Python 3.11+
- PostgreSQL 16 with pgvector extension (`CREATE EXTENSION vector;`)
- Virtual environment

---

## 2. Environment Setup

Copy the template and fill in all required values:

```bash
cp example.env .env
```

Required variables (no defaults):

| Variable | Description |
|---|---|
| `TDX_WEBSERVICES_KEY` | TDX API authentication key (UUID) |
| `TDX_BEID` | TDX business entity ID (UUID) |
| `DB_PASSWORD` | PostgreSQL password for the RAG database user |
| `LITELLM_DB_PASSWORD` | PostgreSQL password for the LiteLLM database user |
| `LITELLM_PROXY_API_KEY` | API key for the LiteLLM proxy (set this yourself) |
| `LITELLM_MASTER_KEY` | LiteLLM admin master key |
| `LITELLM_SALT_KEY` | LiteLLM salt key for encryption |
| `API_API_KEY` | API key for the FastAPI search endpoints (min 32 chars recommended) |
| `AUTH_SECRET_KEY` | Secret key for admin JWT signing (generate with `openssl rand -hex 32`) |
| `AZURE_API_KEY` | Azure OpenAI API key (or equivalent Bedrock credentials) |

Key optional variables with non-obvious defaults:

```bash
# TDX
TDX_BASE_URL="https://yourdeployment.teamdynamix.com"
TDX_APP_ID=2717   # TDX knowledge base application ID

# LiteLLM model aliases — must match what is defined in config.yaml
LITELLM_EMBEDDING_MODEL="text-embedding-large-3"
LITELLM_CHAT_MODEL="gpt-5.2-chat"
LITELLM_RERANKER_MODEL="cohere-rerank-v3-5"

# API connection pool
API_POOL_MIN_SIZE=5
API_POOL_MAX_SIZE=20
API_WORKERS=4

# Chat endpoint
CHAT_TOP_K=5          # chunks sent to LLM for context
CHAT_FETCH_TOP_K=20   # candidates fetched before fusion/reranking
CHAT_RRF_K=1          # RRF rank-smoothing constant (lower = more weight to top ranks)
```

For Docker Compose, `DB_HOST` is set automatically to `db` in the compose file and does not need to be in `.env`. For local development, set `DB_HOST=localhost`.

---

## 3. Docker Compose Deployment

### First Deploy

```bash
# Start all core services
docker compose up -d

# Verify all containers are running
docker compose ps
```

Expected state after ~90 seconds:

```
NAME                   STATUS
helpdesk-db            Up (healthy)
helpdesk-db-init       Exited (0)   ← expected, one-shot
helpdesk-litellm       Up (healthy)
helpdesk-api           Up (healthy)
helpdesk-webui         Up
```

The `db-init` container exiting with code 0 is correct. It runs once to create the LiteLLM database user and exits.

### With Prometheus Monitoring

```bash
docker compose --profile monitoring up -d
```

This adds `helpdesk-prometheus` on port 9090.

### Stopping and Starting

```bash
# Stop all services (data preserved)
docker compose down

# Stop and remove volumes (destroys all data — use only in dev)
docker compose down -v

# Restart a single service
docker compose restart api

# Rebuild after code changes
docker compose build api
docker compose up -d api
```

### Following Logs

```bash
# All services
docker compose logs -f

# API only (most useful during troubleshooting)
docker compose logs -f api

# LiteLLM to debug model routing
docker compose logs -f litellm
```

---

## 4. Service Startup Order

Docker Compose enforces the following dependency chain:

1. `db` starts and becomes healthy (`pg_isready` passes)
2. `db-init` runs idempotent LiteLLM database/user creation, exits 0
3. `litellm` starts and becomes healthy (`/health/liveliness` passes) — this takes up to 60 seconds on first start as it initializes its database schema
4. `api` and `open-webui` start in parallel after `litellm` is healthy

The `api` container will not start until both `db` (healthy) and `litellm` (healthy) conditions are met. If either dependency is slow, the `api` container waits.

The API's own startup sequence (inside the container, after Alembic migrations):
- Connection pool opened (5 min connections)
- BM25 corpus loaded from `article_chunks` table (fast on empty DB; ~5-30s on large corpus)
- Vector retriever initialized
- Reranker, HyDE generator, ChatService initialized
- Uvicorn starts accepting requests

---

## 5. Database Migrations

Migrations run automatically on container startup via `docker-entrypoint.sh`:

```bash
alembic upgrade head
uvicorn api.main:app ...
```

To run migrations manually (inside the running API container):

```bash
docker compose exec api alembic upgrade head
docker compose exec api alembic current
docker compose exec api alembic history
```

To rollback one migration:

```bash
docker compose exec api alembic downgrade -1
```

To roll back everything (destroys schema and data):

```bash
docker compose exec api alembic downgrade base
```

When adding a new migration during development:

```bash
# From project root with virtual environment active
alembic revision -m "add_new_feature"
# Edit alembic/versions/<hash>_add_new_feature.py
# Implement upgrade() and downgrade()
alembic upgrade head   # test apply
alembic downgrade -1   # test rollback
alembic upgrade head   # apply again
```

Never modify existing migration files after they have been applied to any shared database. Always create a new migration to correct schema issues.

---

## 6. Running the Data Pipeline

The pipeline must be run after deployment to populate the database with TDX articles, chunks, and embeddings. It can also be run on a cron schedule for incremental updates.

All pipeline commands connect directly to PostgreSQL and call the LiteLLM proxy. They can be run from the host with a local virtual environment or inside the API container.

### Inside the API container

```bash
# Full pipeline: ingest → process → embed
docker compose exec api python main.py pipeline

# Individual phases
docker compose exec api python main.py ingest
docker compose exec api python main.py process
docker compose exec api python main.py embed --batch-size 100

# Skip phases already completed
docker compose exec api python main.py pipeline --skip-ingestion
docker compose exec api python main.py pipeline --skip-processing

# Verbose output
docker compose exec api python main.py --log-level DEBUG pipeline
```

### From host (with virtual environment)

```bash
source venv/bin/activate   # or: .venv\Scripts\activate on Windows

# Ensure DB_HOST=localhost in .env for direct connection
python main.py pipeline
```

### Expected output

The pipeline logs progress at INFO level:

```
INFO  ArticleProcessor: Fetched 847 articles from TDX
INFO  ArticleProcessor: 12 new, 3 updated, 832 unchanged
INFO  TextProcessor: Processing 15 articles → chunks
INFO  TextProcessor: Created 89 chunks (avg 312 tokens)
INFO  EmbeddingGenerator: Embedding 89 chunks in batches of 100
INFO  EmbeddingGenerator: Batch 1/1 complete (89 embeddings)
INFO  Pipeline complete: ingest=15 process=89 embed=89
```

### Incremental updates

Re-running the pipeline is safe and efficient. Each phase skips already-processed items:
- Ingest: skips articles with matching `tdx_article_id` and unchanged content hash
- Process: skips articles that already have chunks in `article_chunks`
- Embed: skips chunks that already have entries in `embeddings_openai`

For a scheduled sync, run `python main.py pipeline` on a cron schedule (e.g. nightly).

---

## 7. Development Mode

### Local virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Start the API with auto-reload

```bash
# Ensure PostgreSQL is running and .env has DB_HOST=localhost
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag watches for file changes and restarts Uvicorn automatically. Do not use in production.

### Running tests

```bash
# All tests
pytest tests/ -v

# Specific file
pytest tests/test_chat_service.py -v

# Specific test
pytest tests/test_bm25_search.py::test_search_returns_ranked_results -v

# With debug logging
pytest tests/ -v --log-cli-level=DEBUG

# Stop on first failure
pytest tests/ -x
```

Tests do not require a running database or LiteLLM proxy — all external calls are mocked.

### Linting and formatting

```bash
# Format
ruff format .

# Lint
ruff check .

# Auto-fix lint issues
ruff check --fix .
```

---

## 8. Verifying the Deployment

### Health check

```bash
curl http://localhost:8000/health/
```

Expected healthy response:
```json
{
  "status": "healthy",
  "components": {
    "bm25": {"status": "healthy", "num_chunks": 2847},
    "vector": {"status": "healthy", "num_embeddings": 2847},
    "database_pool": {"status": "healthy", "pool_size": 5}
  }
}
```

If `bm25` shows `"status": "degraded"` with `"num_chunks": 0`, the pipeline has not been run yet.

### Test a search

```bash
curl -X POST http://localhost:8000/api/v1/search/bm25 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"query": "password reset", "top_k": 5}'
```

### API documentation

With the server running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Admin UI

- System prompts: `http://localhost:8000/admin/prompts`
- Analytics: `http://localhost:8000/admin/analytics`

Both require admin login credentials set in the `admin_users` table (see `core/storage_admin_user.py` for the create-user helper).

---

## 9. Monitoring

### Prometheus (optional)

```bash
docker compose --profile monitoring up -d
```

Prometheus scrapes the API's `/metrics` endpoint. Configuration is in `prometheus.yml`. Access the Prometheus UI at `http://localhost:9090`.

### Log aggregation

API logs are written to stdout in JSON-structured format (see `utils/logger.py`). In Docker:

```bash
docker compose logs -f api 2>&1 | grep ERROR
```

Log files are also written to `/app/logs/` inside the container if the directory is volume-mounted.

---

## 10. Troubleshooting

### API container fails to start: "connection refused" to database

The `api` container depends on `db` being healthy, but `psycopg` connection errors at startup usually mean the connection pool itself failed.

Check:
```bash
docker compose logs db | tail -20
docker compose exec db pg_isready -U postgres -d helpdesk_chatbot
```

Ensure `DB_HOST=db` is set (not `localhost`) in the Docker environment. The compose file sets this automatically; check that `.env` is not overriding it with `localhost`.

### LiteLLM container stuck in restart loop

LiteLLM needs its database to be available. Check:
```bash
docker compose logs db-init
docker compose logs litellm | head -50
```

The `db-init` container must exit with code 0 before LiteLLM starts. If `db-init` failed, the LiteLLM database/user was not created.

### BM25 reports 0 chunks at startup

The pipeline has not been run, or the `article_chunks` table is empty. Run:
```bash
docker compose exec api python main.py pipeline
```

Then verify:
```bash
curl http://localhost:8000/health/
```

### Embedding failures ("model not found" from LiteLLM)

The model alias in the `.env` (`LITELLM_EMBEDDING_MODEL`) must match a model alias defined in `config.yaml`. Check:
```bash
docker compose logs litellm | grep -i "model"
curl http://localhost:4000/model/info -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

### Alembic migration fails on container start

If the migration fails, the API container will exit. Check:
```bash
docker compose logs api | grep -i "alembic\|migration\|error"
```

Common causes:
- Database schema already at a newer version than migration files (use `alembic stamp head` to mark as current)
- Connection refused (database not yet ready — check `depends_on` conditions)
- Migration file with invalid SQL (test `alembic upgrade head` locally first)

### Admin UI returns 401 after a period of inactivity

JWT tokens expire after `AUTH_TOKEN_EXPIRE_MINUTES` (default 120). Log in again at `/admin/login`.
