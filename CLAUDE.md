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

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp example.env .env
# Edit .env with your credentials
```

### Database Operations
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
python main.py pipeline --provider openai

# Run full pipeline with Cohere embeddings
python main.py pipeline --provider cohere

# Ingest only (fetch from TDX API)
python main.py ingest

# Process only (convert HTML to chunks)
python main.py process

# Process specific articles
python main.py process --article-ids 123 456

# Embed only (generate vectors for chunks)
python main.py embed --provider openai --batch-size 100

# Skip specific phases
python main.py pipeline --skip-ingestion --provider openai
python main.py pipeline --skip-processing --provider cohere

# Debug logging
python main.py --log-level DEBUG pipeline --provider openai
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

## Architecture

### Data Flow Pipeline

```
TDX API → Ingestion → articles (raw HTML)
           ↓
        Processing → article_chunks (clean text)
           ↓
        Embedding → embeddings_openai / embeddings_cohere (vectors)
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
- Generates embeddings using Azure OpenAI or AWS Bedrock (Cohere)
- Batch processing with configurable batch size
- Error handling and retry logic for API failures

**core/storage_vector.py** (`VectorStorage`)
- Stores embeddings in `embeddings_openai` or `embeddings_cohere` tables
- Uses pgvector for efficient vector operations
- CASCADE DELETE from parent articles ensures referential integrity

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

### Database Schema

All tables use UUIDs as primary keys for robust identification.

**articles**: Raw HTML storage
- `id` (UUID, PK): Auto-generated unique identifier
- `tdx_article_id` (int, UNIQUE): TDX API article ID
- `content_html` (text): Raw HTML from TDX
- `last_modified_date` (timestamp): From TDX API

**article_chunks**: Processed text chunks
- `id` (UUID, PK): Auto-generated unique identifier
- `parent_article_id` (UUID, FK → articles.id): Links to source article
- `chunk_sequence` (int): Order within article (0-indexed)
- `text_content` (text): Clean text content
- `token_count` (int): Number of tokens

**embeddings_openai** / **embeddings_cohere**: Vector storage
- `chunk_id` (UUID, PK): Auto-generated unique identifier
- `parent_article_id` (UUID, FK → articles.id, CASCADE): Links to source article
- `embedding` (vector): pgvector embedding (3072-dim for OpenAI, 1536-dim for Cohere)
- Foreign key cascades: Deleting an article deletes all embeddings

### Configuration

All settings are managed via Pydantic in `core/config.py` and loaded from `.env`:

- **TDX**: WEBSERVICES_KEY, BEID, BASE_URL, APP_ID
- **PostgreSQL**: DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
- **Azure OpenAI**: AZURE_OPENAI_API_KEY, AZURE_OPENAI_EMBED_ENDPOINT, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
- **AWS Bedrock**: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_EMBED_MODEL_ID

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
- Example: `with RAGPipeline(embedding_provider="openai") as pipeline:`
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
- Bootstrap checks current state before applying changes

## Working with This Codebase

### Adding a New Embedding Provider
1. Add configuration to `core/config.py`
2. Implement embedding logic in `core/embedding.py`
3. Add storage table schema in `utils/bootstrap_db.py`
4. Update `core/storage_vector.py` with new table operations
5. Add tests in `tests/test_embedding.py` and `tests/test_storage_vector.py`

### Adding a New Retrieval Method
1. Create new retriever class in `core/` (follow pattern from `bm25_search.py`)
2. Implement `search()` and `batch_search()` methods
3. Return results with `rank` and `score`/`similarity` fields
4. Add comprehensive tests in `tests/`
5. Create example script in `examples/`

### Modifying the Database Schema
1. Update SQL in `utils/bootstrap_db.py`
2. Update Pydantic models in `core/schemas.py`
3. Update storage clients in `core/storage_*.py`
4. Run `python main.py bootstrap --dry-run` to preview changes
5. Update all affected tests
6. Document migration steps if schema changes are breaking

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
