# RAG Backend Service for Helpdesk Chatbot

A comprehensive RAG (Retrieval-Augmented Generation) backend service that ingests, processes, and embeds knowledge base articles from TeamDynamix (TDX) for use in a helpdesk chatbot.

## Architecture Overview

The system follows a modular pipeline architecture:

```
TDX API → Ingestion → Storage (Raw Articles) → Processing → Storage (Chunks) → Embedding → Storage (Vectors) → Retrieval
```

### Data Flow
1. **Ingestion**: Fetch articles from TDX API and store raw HTML in `articles` table
2. **Processing**: Convert HTML to clean text, count tokens, store in `article_chunks` table
3. **Embedding**: Generate vector embeddings for chunks 
4. **Storage**: Store embeddings with metadata in `embeddings` table 
5. **Retrieval**: Semantic search over embedded chunks (WIP)

## Database Schema

All tables use UUIDs as primary keys for robust identification and foreign key relationships.

### Articles Table (Raw Storage)
```sql
- id (UUID): Article unique identifier (Primary Key, auto-generated)
- tdx_article_id (integer): Original article ID from TDX API (UNIQUE, NOT NULL)
- title (text): Article title
- url (text): Public URL
- content_html (text): Raw HTML content
- last_modified_date (timestamp): Last modification date
- raw_ingestion_date (timestamp): When article was ingested
- created_at (timestamp): Record creation timestamp (auto-generated)
```

### Article Chunks Table (Processed Storage)
```sql
- id (UUID): Unique chunk identifier (Primary Key, auto-generated)
- parent_article_id (UUID): Foreign key to articles table
- chunk_sequence (int): Order within article (0-indexed)
- text_content (text): Clean, processed text content
- token_count (int): Number of tokens in chunk
- url (text): Source article URL
- last_modified_date (timestamp): Last modification date from source article
```

### Embeddings Tables (Vector Storage)

#### Embeddings OpenAI Table
```sql
- chunk_id (UUID): Unique chunk identifier (Primary Key, auto-generated)
- parent_article_id (UUID): Foreign key to articles table (CASCADE DELETE)
- chunk_sequence (int): Order within article
- text_content (text): Clean text content
- token_count (int): Number of tokens
- source_url (text): Article URL
- embedding (vector(3072)): pgvector embedding for OpenAI text-embedding-3-large
- created_at (timestamp): Embedding creation timestamp (auto-generated)
```

#### Embeddings Cohere Table
```sql
- chunk_id (UUID): Unique chunk identifier (Primary Key, auto-generated)
- parent_article_id (UUID): Foreign key to articles table (CASCADE DELETE)
- chunk_sequence (int): Order within article
- text_content (text): Clean text content
- token_count (int): Number of tokens
- source_url (text): Article URL
- embedding (vector(1536)): pgvector embedding for AWS Cohere Embed v4
- created_at (timestamp): Embedding creation timestamp (auto-generated)
```

## Test Coverage

- **Total Tests**: 71
- **Status**: ✅ All Passing
- **Coverage**:
  - TDXClient: 5 tests
  - ArticleProcessor: 12 tests
  - PostgresClient: 13 tests
  - DatabaseBootstrap: 21 tests
  - TextProcessor: 20 tests

## Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the backend directory with the following variables:

```bash
# TDX API Configuration
BASE_URL=https://your-instance.teamdynamix.com
APP_ID=2717
WEBSERVICES_KEY=your_key
BEID=your_beid

# Database Configuration
DB_HOST=localhost
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=your_database

# AWS Bedrock Configuration (for Cohere embeddings)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_EMBED_MODEL_ID=cohere.embed-english-v3
AWS_EMBED_DIM=1536
AWS_MAX_TOKENS=512

# Azure OpenAI Configuration (for OpenAI embeddings)
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_EMBED_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_EMBED_DIM=3072
AZURE_MAX_TOKENS=8191
```

### 3. Bootstrap the Database

```bash
# Check current database status
python main.py bootstrap --status

# Preview changes without applying (dry-run)
python main.py bootstrap --dry-run

# Create tables and extensions
python main.py bootstrap

# Reset database - WARNING: deletes all data!
python main.py bootstrap --full-reset
```

## CLI Usage

The `main.py` CLI provides commands for all pipeline operations, designed for scheduled task execution.

### Command Overview

```bash
python main.py [--log-level LEVEL] <command> [options]
```

**Important:** Global options like `--log-level` must come **before** the subcommand.

### Available Commands

#### 1. Ingest Articles

Fetch articles from TDX API and store them in the database.

```bash
# Basic ingestion
python main.py ingest

# With debug logging
python main.py --log-level DEBUG ingest
```

**Output:**
- New articles count
- Updated articles count
- Unchanged articles count
- Skipped articles count

#### 2. Process Articles

Convert HTML articles to clean text chunks for embedding.

```bash
# Process all articles
python main.py process

# Process specific articles by ID
python main.py process --article-ids 123 456 789
```

**Output:**
- Articles processed count
- Chunks created count

#### 3. Generate Embeddings

Generate vector embeddings for processed text chunks.

```bash
# Use OpenAI embeddings (default)
python main.py embed --provider openai

# Use Cohere embeddings
python main.py embed --provider cohere

# Custom batch size
python main.py embed --provider openai --batch-size 50
```

**Options:**
- `--provider {openai|cohere}` - Embedding provider (default: openai)
- `--batch-size N` - Process N chunks per batch (default: 100)

**Features:**
- Fetches all chunks from the `article_chunks` table
- Processes chunks in configurable batches
- Generates embeddings using the specified provider
- Stores embeddings immediately after each batch
- Shows progress updates during processing
- Handles errors gracefully, continuing with remaining batches

#### 4. Full Pipeline

Run the complete RAG pipeline: ingestion → processing → embedding → storage.

```bash
# Run full pipeline with OpenAI
python main.py pipeline --provider openai

# Run full pipeline with Cohere
python main.py pipeline --provider cohere

# Skip ingestion (use existing articles)
python main.py pipeline --skip-ingestion --provider openai

# Skip processing (use existing chunks)
python main.py pipeline --skip-processing --provider openai

# Dry run (skip embedding generation)
python main.py pipeline --skip-embedding --provider openai

# Process specific articles only
python main.py pipeline --article-ids 123 456 --provider cohere

# Debug logging
python main.py --log-level DEBUG pipeline --provider openai
```

**Options:**
- `--provider {openai|cohere}` - Embedding provider (default: openai)
- `--skip-ingestion` - Skip article ingestion phase
- `--skip-processing` - Skip text processing phase
- `--skip-embedding` - Skip embedding generation phase
- `--article-ids ID [ID ...]` - Process specific article IDs only

**Output:**
- Execution duration
- Ingestion statistics (new, updated, unchanged)
- Processing statistics (articles, chunks)
- Embedding statistics (embeddings generated)
- Storage statistics (embeddings stored)

#### 5. Database Bootstrap

Set up or reset database schema, tables, and extensions.

```bash
# Check database status
python main.py bootstrap --status

# Preview changes (dry-run)
python main.py bootstrap --dry-run

# Create tables and extensions
python main.py bootstrap

# Full reset - WARNING: deletes all data!
python main.py bootstrap --full-reset
```

**Options:**
- `--status` - Check current database state
- `--dry-run` - Preview changes without applying
- `--full-reset` - Drop all tables and recreate (requires confirmation)

### Global Options

- `--log-level {DEBUG|INFO|WARNING|ERROR|CRITICAL}` - Set logging verbosity (default: INFO)
- `--version` - Show version information
- `--help` - Show help message

### Examples for Task Scheduler

```bash
# Daily ingestion job (runs every day at 2 AM)
python main.py --log-level INFO ingest

# Weekly full pipeline with OpenAI (runs every Sunday at 3 AM)
python main.py --log-level INFO pipeline --provider openai

# Incremental update (skip ingestion, process new articles)
python main.py --log-level INFO pipeline --skip-ingestion --provider cohere

# Generate embeddings for already-processed chunks
python main.py --log-level INFO embed --provider openai --batch-size 100
```

### Exit Codes

- `0` - Success
- `1` - General failure
- `130` - User cancelled (Ctrl+C)

All commands log to stdout/stderr with timestamps and can be redirected for monitoring.

## Programmatic Usage

You can also use the modules directly in Python code:

### Running Ingestion

```python
from core.ingestion import ArticleProcessor

processor = ArticleProcessor()
stats = processor.ingest_and_store()
print(f"New: {stats['new_count']}, Updated: {stats['updated_count']}")
```

### Processing Articles

```python
from core.processing import TextProcessor

processor = TextProcessor()

# Convert HTML to clean text
html_content = "<h1>Title</h1><p>Article content here.</p>"
clean_text = processor.process_text(html_content)

# Count tokens
token_count = processor.get_token_count(clean_text)
print(f"Text has {token_count} tokens")
```

### Running Full Pipeline

```python
from core.pipeline import RAGPipeline

# Initialize pipeline with OpenAI embeddings
with RAGPipeline(embedding_provider="openai") as pipeline:
    # Run full pipeline
    stats = pipeline.run_full_pipeline()
    print(f"Duration: {stats['duration_seconds']:.2f}s")
    print(f"New articles: {stats['ingestion']['new_count']}")
```

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_ingestion.py -v
```

## Admin API & Open-webui Integration

The backend includes a FastAPI server that provides:
1. **Admin Dashboard** - Analytics and monitoring UI
2. **OpenAI-compatible API** - Allows Open-webui to connect as if it were an OpenAI API
3. **RAG Retrieval API** - Direct access to vector search

### Starting the API Server

```bash
# From the backend directory
cd utc-chatbot-backend
python -m api.main
```

The server runs on `http://localhost:8000` by default.

### Environment Variables for API

Add these to your `.env` file:

```bash
# API Server Configuration
API_API_KEY=your_secret_api_key    # Required for authentication
API_HOST=0.0.0.0                   # Default: 0.0.0.0
API_PORT=8000                      # Default: 8000
API_WORKERS=4                      # Default: 4
API_LOG_LEVEL=info                 # Default: info
```

### API Endpoints

#### Admin Endpoints (require API key via `X-API-Key` header)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/stats` | GET | Conversation statistics (total, unique users, avg latency) |
| `/admin/users` | GET | Conversations grouped by user |
| `/admin/queries` | GET | Recent queries with optional user filter |
| `/admin/cache` | GET | Cache hit/miss statistics |
| `/admin/conversations/timeline` | GET | Time series data for charting |

**Query Parameters:**
- `days` - Number of days to look back (default: 30)
- `limit` - Max results to return (default: 100)
- `user_id` - Filter by specific user (queries endpoint only)

#### OpenAI-compatible Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions with RAG context injection |
| `/v1/models` | GET | List available models (prefixed with `utc-rag-`) |

#### RAG Retrieval Endpoints (require API key)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/retrieve` | POST | Retrieve relevant KB articles for a query |
| `/chat/log` | POST | Log a query for analytics |

### Admin Dashboard

Access the admin dashboard at `http://localhost:8000/dashboard`

Features:
- Real-time statistics (conversations, users, latency, cache hit rate)
- User activity table with click-to-filter
- Recent queries log
- Configurable time period (7/30/90 days)

### Open-webui Integration

The backend provides an OpenAI-compatible API that allows Open-webui to connect seamlessly.

#### Setup in Open-webui

1. Go to **Settings → Connections**
2. Add a new **OpenAI API** connection:
   - **URL**: `http://localhost:8000/v1` (or `http://host.docker.internal:8000/v1` if Open-webui is in Docker)
   - **API Key**: Your `API_API_KEY` value

3. Select a model prefixed with `utc-rag-` (e.g., `utc-rag-llama3:latest`)

#### How It Works

```
User Query → Open-webui → Our API → RAG Retrieval → Ollama → Response
```

1. User sends a message in Open-webui
2. Open-webui forwards the request to our OpenAI-compatible endpoint
3. Our API extracts the user query and retrieves relevant KB articles
4. The system prompt is augmented with the retrieved context
5. The request is forwarded to Ollama for LLM inference
6. The response is returned in OpenAI format
7. Query is logged for analytics

#### Model Naming

Models are prefixed with `utc-rag-` to distinguish them from direct Ollama connections:
- `utc-rag-llama3:latest` → Routes through our RAG pipeline
- `llama3:latest` → Direct to Ollama (no RAG)

### Database Tables for Analytics

The API uses a `query_logs` table for analytics:

```sql
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    raw_query TEXT NOT NULL,
    cache_result VARCHAR(10) DEFAULT 'miss',
    latency_ms INTEGER,
    user_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);
```

This table is created automatically during bootstrap.

### Example API Calls

#### Get Statistics
```bash
curl -H "X-API-Key: your_key" http://localhost:8000/admin/stats?days=30
```

#### Retrieve RAG Context
```bash
curl -X POST -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I set up Duo Mobile?", "top_k": 3}' \
  http://localhost:8000/chat/retrieve
```

#### Chat Completion (OpenAI format)
```bash
curl -X POST -H "Authorization: Bearer your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "utc-rag-llama3:latest",
    "messages": [{"role": "user", "content": "How do I reset my password?"}]
  }' \
  http://localhost:8000/v1/chat/completions
```

