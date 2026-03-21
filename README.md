# UTC Helpdesk Chatbot — RAG Backend

RAG (Retrieval-Augmented Generation) backend for the IT helpdesk chatbot at the University of Tennessee at Chattanooga. Ingests knowledge base articles from TeamDynamix, indexes them with BM25 and pgvector, and serves hybrid search plus an OpenAI-compatible streaming chat endpoint for Open WebUI.

---

## Features

- **Four retrieval modes**: BM25 (keyword), vector (semantic), hybrid (RRF fusion), and HyDE (hypothetical document expansion)
- **Neural reranking**: Optional Cohere Rerank v3.5 via LiteLLM proxy to refine hybrid results
- **OpenAI-compatible chat endpoint**: SSE streaming at `/v1/chat/completions` — drop-in for Open WebUI
- **Tag-based system prompts**: Per-article-category system prompts, manageable through a web admin UI
- **Incremental ingestion pipeline**: CLI pipeline syncs TDX articles without reprocessing unchanged content
- **Full observability**: Query logs, per-result logs, LLM response logs, HyDE logs, reranker logs, analytics dashboard
- **Production-ready deployment**: Docker Compose stack (PostgreSQL 16 + pgvector, LiteLLM proxy, FastAPI, Open WebUI)
- **435 tests** across 25 test files; no live database required for testing

---

## Quick Start

### Prerequisites

- Docker Desktop (Compose v2)
- TDX API credentials (`TDX_WEBSERVICES_KEY`, `TDX_BEID`)
- LiteLLM-compatible AI provider (Azure OpenAI or AWS Bedrock, configured in `config.yaml`)

### Setup

```bash
git clone <repo-url>
cd utc-chatbot-backend

# Configure environment
cp example.env .env
# Edit .env — set TDX credentials, DB passwords, API keys, AI provider keys

# Start all services
docker compose up -d

# Verify health (wait ~90 seconds for services to initialize)
curl http://localhost:8000/health/

# Populate the knowledge base (run once, then schedule for incremental updates)
docker compose exec api python main.py pipeline
```

### Access Points

| URL | Purpose |
|---|---|
| `http://localhost:3000` | Open WebUI chat interface |
| `http://localhost:8000/docs` | FastAPI Swagger UI |
| `http://localhost:8000/health/` | Health check (no auth) |
| `http://localhost:8000/admin/prompts` | System prompt admin UI |
| `http://localhost:8000/admin/analytics` | Analytics dashboard |
| `http://localhost:4000` | LiteLLM proxy admin |

---

## Architecture Overview

```
Open WebUI ──► FastAPI RAG API ──► LiteLLM proxy ──► Azure OpenAI / Bedrock
                     │                                (embeddings, chat, reranking)
                     ▼
              PostgreSQL + pgvector
              (articles, chunks, embeddings, query logs)
```

**Data pipeline:** TDX API → `ArticleProcessor` → raw HTML → `TextProcessor` → text chunks → `EmbeddingGenerator` → 3072-dim vectors

**Query path:** BM25 (in-memory) + pgvector KNN → RRF fusion → Cohere reranking → response

Five Docker services: `db` (PostgreSQL 16 + pgvector), `db-init` (one-shot setup), `litellm` (AI gateway), `api` (FastAPI), `open-webui`. Optional `prometheus` with `--profile monitoring`.

See [docs/system-architecture.md](docs/system-architecture.md) for detailed data flow diagrams.

---

## API Endpoints

All search and chat endpoints require `X-API-Key` header. Health endpoints require no auth.

```bash
# BM25 keyword search
curl -X POST http://localhost:8000/api/v1/search/bm25 \
  -H "X-API-Key: your-key" -H "Content-Type: application/json" \
  -d '{"query": "password reset", "top_k": 5}'

# Hybrid search (BM25 + vector + RRF + reranking)
curl -X POST http://localhost:8000/api/v1/search/hybrid \
  -H "X-API-Key: your-key" -H "Content-Type: application/json" \
  -d '{"query": "VPN connection issues", "top_k": 5}'

# OpenAI-compatible chat (used by Open WebUI)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: your-key" -H "Content-Type: application/json" \
  -d '{"model": "utc-helpdesk", "messages": [{"role": "user", "content": "q/how do I reset my MocsID password?"}], "stream": true}'
```

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/search/bm25` | POST | BM25 keyword search (~50-100ms) |
| `/api/v1/search/vector` | POST | Semantic vector search (~500ms) |
| `/api/v1/search/hybrid` | POST | RRF fusion + reranking (~800ms) |
| `/api/v1/search/hyde` | POST | HyDE + hybrid + reranking (~1.5-2.5s) |
| `/v1/chat/completions` | POST | OpenAI-compat SSE streaming chat |
| `/v1/models` | GET | Model list |
| `/health/` | GET | Component health status |
| `/health/ready` | GET | Readiness probe |
| `/admin/prompts` | GET | System prompt admin UI |
| `/admin/analytics` | GET | Analytics dashboard |

---

## Development

### Local setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp example.env .env        # set DB_HOST=localhost for local dev
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
pytest tests/ -v                    # all 435 tests
pytest tests/test_chat_service.py -v  # single file
pytest tests/ -v --log-cli-level=DEBUG  # with debug output
```

No running database or LiteLLM proxy required — all external calls are mocked.

### Linting

```bash
ruff format .        # format
ruff check .         # lint
ruff check --fix .   # auto-fix
```

### Pipeline CLI

```bash
python main.py pipeline                  # full ingest → process → embed
python main.py ingest                    # TDX fetch only
python main.py process                   # chunk raw articles
python main.py embed --batch-size 100    # generate embeddings
python main.py pipeline --skip-ingestion # process + embed only
python main.py evaluate --dataset data/qa_pairs.jsonl  # retrieval evaluation
```

### Database migrations

```bash
alembic upgrade head      # apply all pending migrations
alembic current           # check current version
alembic downgrade -1      # rollback one migration
alembic revision -m "description"  # create new migration
```

---

## Documentation

| Document | Contents |
|---|---|
| [docs/project-overview-pdr.md](docs/project-overview-pdr.md) | Project purpose, design decisions, technology trade-offs |
| [docs/system-architecture.md](docs/system-architecture.md) | Component diagrams, data flow, database schema, auth |
| [docs/codebase-summary.md](docs/codebase-summary.md) | File inventory, module dependencies, key classes, test coverage |
| [docs/code-standards.md](docs/code-standards.md) | Conventions, patterns, error handling, testing guide |
| [docs/deployment-guide.md](docs/deployment-guide.md) | Docker Compose deployment, pipeline operations, troubleshooting |
| [CLAUDE.md](CLAUDE.md) | AI assistant project guide (architecture, commands, patterns) |

---

## License

Internal project — University of Tennessee at Chattanooga. Not licensed for external distribution.
