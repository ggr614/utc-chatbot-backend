# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) backend service for a helpdesk chatbot. It ingests knowledge base articles from TeamDynamix (TDX) API, processes them into text chunks, generates embeddings, and stores them in PostgreSQL with pgvector for hybrid retrieval (BM25 + vector search).

## Design Principles/Patterns
- Dependency Injection where possible
- Separation of Concerns
- Design for idempotency where appropriate
- Configuration over hardcoding (API keys, model names, chunk sizes, etc.)
- Graceful degradation (handle API failures, rate limits, timeouts)
- Logging and observability
- Immutable data transformations (don't mutate documents mid-pipeline)
- Async-first for I/O-bound operations (embeddings, LLM calls, vector DB queries)
- Retrieval-agnostic interfaces
## Development Commands

Best practices calls for using virtual environments. Remember to use the virtual environment's python when running commands.

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp example.env .env
# Edit .env with your credentials
```

### Database Operations

**Alembic Migrations (Primary Method)**
```bash
# Check current migration version
alembic current

# View migration history
alembic history

# Apply all pending migrations
alembic upgrade head

# Rollback last migration
alembic downgrade -1

# Rollback all migrations (deletes all data)
alembic downgrade base

# Create new migration after schema changes
alembic revision -m "description"

# Mark existing database as up-to-date (for legacy databases)
alembic stamp head
```

**Bootstrap Script (Legacy - being phased out)**
```bash
# Check database status
python main.py bootstrap --status

# Preview changes (dry-run)
python main.py bootstrap --dry-run

# Create tables and extensions
python main.py bootstrap

# Full reset (deletes all data - use with caution)
python main.py bootstrap --full-reset
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_ingestion.py -v

# Run specific test
pytest tests/test_ingestion.py::test_fetch_articles -v

# Run with debug logging
pytest tests/ -v --log-cli-level=DEBUG
```

### Pipeline Operations
```bash
# Run full pipeline with OpenAI embeddings
python main.py pipeline

# Ingest only (fetch from TDX API)
python main.py ingest

# Process only (convert HTML to chunks)
python main.py process

# Process specific articles
python main.py process --article-ids 123 456

# Embed only (generate vectors for chunks)
python main.py embed --batch-size 100

# Skip specific phases
python main.py pipeline --skip-ingestion
python main.py pipeline --skip-processing

# Debug logging
python main.py --log-level DEBUG pipeline
```

### Code Quality
```bash
# Format code with ruff
ruff format .

# Lint code
ruff check .

# Fix linting issues
ruff check --fix .
```

### FastAPI Application
```bash
# Start API server (development mode with auto-reload)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Start API server (production mode with 4 workers)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4 --limit-concurrency 100

# Or run directly with python (development only)
python api/main.py

# Access API documentation (once server is running)
# OpenAPI/Swagger: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc

# Health check
curl http://localhost:8000/health/

# Test BM25 search
curl -X POST http://localhost:8000/api/v1/search/bm25 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"query": "password reset", "top_k": 10}'

# Test vector search
curl -X POST http://localhost:8000/api/v1/search/vector \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"query": "how do I recover my account", "top_k": 10}'

# Test hybrid search
curl -X POST http://localhost:8000/api/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"query": "vpn connection issues", "top_k": 10, "fusion_method": "rrf"}'
```

## Architecture

### Data Flow Pipeline

```
TDX API → Ingestion → articles (raw HTML)
           ↓
        Processing → article_chunks (clean text)
           ↓
        Embedding → embeddings_openai (vectors)
           ↓
        Retrieval → BM25 (sparse) + Vector (dense) hybrid search
```

### Core Modules

**core/ingestion.py** (`ArticleProcessor`)
- Fetches articles from TDX API using `utils/api_client.py`
- Handles deduplication and incremental updates
- Stores raw HTML in `articles` table via `core/storage_raw.py`
- Uses UUID as primary key, `tdx_article_id` as unique constraint

**core/processing.py** (`TextProcessor`)
- Converts HTML to clean text using BeautifulSoup and html2text
- Chunks text using LangChain's RecursiveCharacterTextSplitter
- Counts tokens with tiktoken
- Stores chunks in `article_chunks` table via `core/storage_chunk.py`

**core/embedding.py** (`EmbeddingGenerator`)
- Generates embeddings using Azure OpenAI
- Batch processing with configurable batch size
- Error handling and retry logic for API failures

**core/storage_vector.py** (`VectorStorageClient`, `OpenAIVectorStorage`)
- Stores embeddings in `embeddings_openai` table (normalized structure)
- Uses pgvector for efficient vector operations with cosine similarity
- Joins with `article_chunks` table to retrieve chunk metadata
- CASCADE DELETE via `chunk_id` → `article_chunks.id` → `articles.id`
- Base class `VectorStorageClient` for extensibility

**core/bm25_search.py** (`BM25Retriever`)
- Keyword-based sparse retrieval using BM25 algorithm
- Loads entire corpus into memory (use cache for repeated queries)
- Fast for exact term matching and technical queries

**core/vector_search.py** (`VectorRetriever`)
- Semantic dense retrieval using pgvector cosine similarity
- Embeds query with same model as corpus (OpenAI text-embedding-3-large)
- Better for natural language queries and synonym matching

**core/pipeline.py** (`RAGPipeline`)
- Orchestrates the full pipeline: ingest → process → embed
- Context manager for resource cleanup
- Returns detailed statistics for each phase

### FastAPI Application (api/)

The FastAPI application provides REST API endpoints for search and query logging. It uses connection pooling and shared retrievers for efficient concurrent request handling.

**api/main.py** - Application entry point
- FastAPI app with lifespan management (startup/shutdown)
- Initializes connection pool (5 min, 20 max connections)
- Initializes BM25 and vector retrievers once at startup
- Pre-warms BM25 corpus cache
- Stores shared resources in `app.state` for dependency injection
- Registers search and health routers

**api/dependencies.py** - Dependency injection
- `verify_api_key()`: Validates X-API-Key header, supports multiple keys
- `get_bm25_retriever()`: Returns shared BM25 retriever from app.state
- `get_vector_retriever()`: Returns shared vector retriever from app.state
- `get_query_log_client()`: Creates QueryLogClient with connection pool

**api/routers/search.py** - Search endpoints
- `POST /api/v1/search/bm25`: BM25 keyword search (fast, <100ms)
- `POST /api/v1/search/vector`: Vector semantic search (~500ms-1s, API call)
- `POST /api/v1/search/hybrid`: Hybrid search with RRF or weighted fusion
- All endpoints require API key authentication via X-API-Key header
- Query logging integrated into each endpoint (best-effort, non-blocking)

**api/routers/health.py** - Health check endpoints
- `GET /health/`: Detailed health check with component status (BM25, vector, database pool)
- `GET /health/ready`: Simple readiness probe for Kubernetes/orchestration
- No authentication required (for monitoring systems)

**api/utils/hybrid_search.py** - Hybrid search implementation
- `reciprocal_rank_fusion()`: Rank-based RRF fusion (robust, default)
- `weighted_score_fusion()`: Score-based weighted fusion with normalization
- `hybrid_search()`: Main function that combines BM25 + vector results

**api/utils/connection_pool.py** - Database connection pooling
- `DatabaseConnectionPool`: Thread-safe connection pool using psycopg_pool
- Singleton pattern via `get_connection_pool()`
- Configuration: min_size=5, max_size=20, timeout=30s
- Prevents connection exhaustion under concurrent load

**api/models/requests.py** - Pydantic request models
- `BM25SearchRequest`: Validates BM25 search parameters
- `VectorSearchRequest`: Validates vector search parameters
- `HybridSearchRequest`: Validates hybrid search parameters (fusion method, weights)
- Request validation: query length, top_k range, score thresholds

**api/models/responses.py** - Pydantic response models
- `SearchResponse`: Unified response for all search methods
- `SearchResultChunk`: Individual result with rank, score, chunk data
- `HealthResponse`: Health check status with component details

**Authentication**:
- Simple header-based: `X-API-Key: your-api-key`
- Primary key: `API_API_KEY` environment variable
- Multiple keys: `API_ALLOWED_API_KEYS` (comma-separated)
- Internal network only (no rate limiting)

**Concurrency Strategy**:
- Synchronous endpoints (not async) - compatible with existing codebase
- Connection pooling handles concurrent requests efficiently
- Shared retrievers initialized once at startup (not per-request)
- BM25 corpus loaded into memory once (cached)
- Recommended: 4 uvicorn workers, 100 max concurrent requests
- Resource allocation: ~500MB per worker for BM25 corpus

**Query Logging**:
- All search requests logged to `query_logs` table
- Fields: raw_query, cache_result, latency_ms, user_id, created_at
- Best-effort logging (doesn't fail request if logging fails)
- Uses connection pool for efficient database access

### Database Schema

All tables use UUIDs as primary keys. Schema is managed with **Alembic migrations** (see `alembic/versions/`).

**articles**: Raw HTML storage
- `id` (UUID, PK): Auto-generated unique identifier
- `tdx_article_id` (int, UNIQUE): TDX API article ID
- `title`, `url`, `content_html` (text): Article metadata and content
- `last_modified_date`, `raw_ingestion_date`, `created_at` (timestamps)

**article_chunks**: Processed text chunks
- `id` (UUID, PK): Auto-generated unique identifier
- `parent_article_id` (UUID, FK → articles.id, CASCADE): Links to source article
- `chunk_sequence` (int): Order within article (0-indexed)
- `text_content` (text): Clean text content
- `token_count` (int): Number of tokens
- `url`, `last_modified_date` (text, timestamp): Source metadata

**embeddings_openai**: Normalized vector storage
- `chunk_id` (UUID, PK, FK → article_chunks.id, CASCADE): Links to chunk
- `embedding` (vector(3072)): OpenAI text-embedding-3-large vector
- `created_at` (timestamp): Embedding creation time
- **Important**: This table is normalized - chunk metadata lives in `article_chunks`, joined via `chunk_id`

**warm_cache_entries**: Pre-computed query cache
- `id` (UUID, PK): Cache entry identifier
- `canonical_question`, `verified_answer` (text): Q&A pairs
- `query_embedding` (vector(3072)): For semantic matching
- `article_id` (UUID, FK → articles.id, CASCADE)
- `is_active` (bool): Whether cache entry is active

**cache_metrics**: Cache performance tracking
- `id` (bigserial, PK): Metrics entry
- `cache_entry_id` (UUID, FK → warm_cache_entries, SET NULL)
- `request_timestamp`, `cache_type`, `latency_ms`, `user_id`

**query_logs**: Query analytics
- `id` (bigserial, PK): Log entry
- `raw_query` (text), `query_embedding` (vector(3072))
- `cache_result`, `latency_ms`, `user_id`, `created_at`

### Configuration

All settings are managed via Pydantic in `core/config.py` and loaded from `.env`:

- **TDX**: WEBSERVICES_KEY, BEID, BASE_URL, APP_ID
- **PostgreSQL**: DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
- **Azure OpenAI**: AZURE_OPENAI_API_KEY, AZURE_OPENAI_EMBED_ENDPOINT, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION

Settings are cached with `@lru_cache()` for performance.

### Testing Conventions

Tests use pytest with fixtures defined in `tests/conftest.py`:
- `mock_settings`: Mocked configuration
- `mock_tdx_articles`: Sample TDX API responses
- `mock_chunks`: Sample text chunks
- Tests use `pytest-mock` for mocking database/API calls
- All tests should clean up resources (use context managers)

### Data Models

Pydantic models in `core/schemas.py`:
- `TdxArticle`: Raw article from TDX API
- `TextChunk`: Processed text chunk
- `VectorRecord`: Final embedding record

All models use UUIDs for `id` and `parent_article_id` fields. Use `HttpUrl` from Pydantic for URL validation.

## Key Implementation Patterns

### UUID Usage
- All database primary keys are UUIDs (auto-generated)
- Use `uuid_utils.uuid7()` for generating UUIDs (time-ordered)
- Foreign keys reference UUID fields, not integer IDs
- TDX API uses integer IDs, stored as `tdx_article_id` (UNIQUE constraint)

### Resource Management
- Always use context managers for database clients and pipelines
- Example: `with RAGPipeline() as pipeline:`
- Ensures proper cleanup of connections and resources

### Error Handling
- Use structured logging with `utils/logger.py`
- Log at appropriate levels: DEBUG for details, INFO for progress, ERROR for failures
- Graceful degradation: continue processing remaining items if one fails

### Batch Processing
- Embedding generation uses batches (default 100 chunks)
- Storage operations commit after each batch
- Progress updates every 10% of total items

### Database Operations
- Use transactions for multi-step operations
- Implement idempotent operations (safe to retry)
- Schema changes are managed via Alembic migrations for version control

### Alembic Migrations
- Configuration: `alembic/env.py` uses Pydantic settings from `core/config.py`
- Migrations stored in: `alembic/versions/`
- Connection string: `postgresql+psycopg://user:pass@host:5432/dbname` (psycopg3 dialect)
- `target_metadata = None`: Manual migrations (not autogenerate from ORM models)
- Always test migrations with `upgrade` and `downgrade` before committing

## Working with This Codebase

### Adding a New Embedding Provider
1. Add configuration to `core/config.py`
2. Implement embedding logic in `core/embedding.py`
3. Create Alembic migration for new embeddings table:
   ```bash
   alembic revision -m "add_embeddings_newprovider"
   ```
4. Implement migration with table schema (follow `embeddings_openai` pattern)
5. Create new storage client class in `core/storage_vector.py` inheriting from `VectorStorageClient`
6. Add tests in `tests/test_embedding.py` and `tests/test_storage_vector.py`
7. Update `core/pipeline.py` to support new provider

### Adding a New Retrieval Method
1. Create new retriever class in `core/` (follow pattern from `bm25_search.py`)
2. Implement `search()` and `batch_search()` methods
3. Return results with `rank` and `score`/`similarity` fields
4. Add comprehensive tests in `tests/`
5. Create example script in `examples/`

### Modifying the Database Schema

**Using Alembic Migrations (Recommended)**
1. Update Pydantic models in `core/schemas.py`
2. Create new migration: `alembic revision -m "descriptive_name"`
3. Edit the generated migration file in `alembic/versions/`:
   - Implement `upgrade()` with schema changes (using `op.execute()`, `op.create_table()`, etc.)
   - Implement `downgrade()` to reverse changes
4. Test migration: `alembic upgrade head` (apply), `alembic downgrade -1` (rollback)
5. Update storage clients in `core/storage_*.py`
6. Update all affected tests
7. Commit migration file to version control

**Using Bootstrap Script (Legacy)**
1. Update SQL in `utils/bootstrap_db.py`
2. Update Pydantic models in `core/schemas.py`
3. Update storage clients in `core/storage_*.py`
4. Run `python main.py bootstrap --dry-run` to preview changes
5. Update all affected tests

**Important**: For production databases, always use Alembic migrations for version control and rollback capabilities.

### Testing Database Operations
- Tests use mocked database connections (no real DB needed)
- Use `pytest-mock` to patch `psycopg.connect()`
- Mock cursor return values for SELECT operations
- Verify SQL statements with `mock_cursor.execute.assert_called_with()`

## Notes

- The `api/` directory is empty (FastAPI endpoints not yet implemented)
- The `data/` directory is gitignored and contains generated datasets
- Log files are stored in `logs/` (also gitignored)
- The codebase uses Python 3.11+ features (pattern matching, new type hints)
- All embeddings use the same model for both indexing and querying (critical for vector search)
- **Database migrations**: Use Alembic for all schema changes. `alembic.ini` is gitignored (contains local DB config)
- **Schema normalization**: `embeddings_openai` table stores only `chunk_id` and `embedding`; metadata is in `article_chunks`
- **Storage pattern**: All storage clients inherit from `BaseStorageClient` for connection management
