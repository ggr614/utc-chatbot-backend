# RAG Backend Service for Helpdesk Chatbot

A comprehensive RAG (Retrieval-Augmented Generation) backend service that ingests, processes, and embeds knowledge base articles from TeamDynamix (TDX) for use in a helpdesk chatbot.

## Architecture Overview

The system follows a modular pipeline architecture with a REST API for querying:

```
TDX API → Ingestion → Storage (Raw Articles) → Processing → Storage (Chunks) → Embedding → Storage (Vectors)
                                                                                                    ↓
                                                                                            REST API Layer
                                                                                                    ↓
                                                                    BM25 Search / Vector Search / Hybrid Search
```

### Data Flow

**Pipeline (Data Ingestion):**
1. **Ingestion**: Fetch articles from TDX API and store raw HTML in `articles` table
2. **Processing**: Convert HTML to clean text, count tokens, store in `article_chunks` table
3. **Embedding**: Generate vector embeddings for chunks using Azure OpenAI
4. **Storage**: Store embeddings with metadata in `embeddings_openai` table

**Query (FastAPI REST API):**
5. **BM25 Retrieval**: Fast keyword-based sparse retrieval (~50-100ms)
6. **Vector Retrieval**: Semantic dense retrieval with pgvector (~500ms-1s)
7. **Hybrid Retrieval**: Combined BM25 + vector with fusion algorithms

## Database Schema

All tables use UUIDs as primary keys for robust identification and foreign key relationships.

**Schema Management**: Database schema is managed with Alembic migrations. See [alembic/versions/](alembic/versions/) for migration history.

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

### Embeddings Table (Vector Storage)

#### Embeddings OpenAI Table
```sql
- chunk_id (UUID): Foreign key to article_chunks table (Primary Key, CASCADE DELETE)
- embedding (vector(3072)): pgvector embedding for OpenAI text-embedding-3-large
- created_at (timestamp): Embedding creation timestamp (auto-generated)
```

**Note**: This table is normalized - chunk metadata (parent_article_id, chunk_sequence, text_content, token_count, url) is stored in the `article_chunks` table and joined via `chunk_id` foreign key.

### Cache Tables (Query Optimization)

#### Warm Cache Entries Table
```sql
- id (UUID): Unique cache entry identifier (Primary Key, auto-generated)
- canonical_question (text): Standardized question text
- verified_answer (text): Pre-verified answer
- query_embedding (vector(3072)): Embedding for semantic matching
- article_id (UUID): Foreign key to articles table (CASCADE DELETE)
- is_active (bool): Whether this cache entry is active
```

#### Cache Metrics Table
```sql
- id (bigserial): Unique metrics entry (Primary Key, auto-generated)
- cache_entry_id (UUID): Foreign key to warm_cache_entries (SET NULL on delete)
- request_timestamp (timestamp): When the cache was accessed
- cache_type (text): Type of cache operation
- latency_ms (int): Response latency in milliseconds
- user_id (text): Optional user identifier
```

#### Query Logs Table
```sql
- id (bigserial): Unique log entry (Primary Key, auto-generated)
- raw_query (text): Original user query
- query_embedding (vector(3072)): Query embedding vector
- cache_result (text): Cache hit/miss result
- latency_ms (int): Query latency in milliseconds
- user_id (text): Optional user identifier
- created_at (timestamp): Log creation timestamp
```

## Test Coverage

- **Total Tests**: 50+
- **Status**: ✅ All Passing
- **Coverage**:
  - TDXClient: 5 tests
  - ArticleProcessor: 12 tests
  - PostgresClient: 13 tests
  - TextProcessor: 20 tests

## Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file and configure it:

```bash
cp example.env .env
# Edit .env with your credentials
```

**Required variables:**

```bash
# TDX API Configuration
TDX_WEBSERVICES_KEY=your_key
TDX_BEID=your_beid
TDX_BASE_URL=https://your-instance.teamdynamix.com
TDX_APP_ID=2717

# Database Configuration
DB_HOST=localhost
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=helpdesk_chatbot

# Azure OpenAI Embedding Configuration
EMBEDDING_API_KEY=your_azure_openai_key
EMBEDDING_ENDPOINT=https://your-resource.openai.azure.com/
EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large
EMBEDDING_API_VERSION=2024-02-01
EMBEDDING_EMBED_DIM=3072
EMBEDDING_MAX_TOKENS=8191

# Azure OpenAI Chat Configuration (for LLM responses)
CHAT_API_KEY=your_azure_openai_chat_key
CHAT_ENDPOINT=https://your-resource.openai.azure.com/
CHAT_DEPLOYMENT_NAME=gpt-4o
CHAT_API_VERSION=2024-12-01-preview
CHAT_MAX_TOKENS=8191
CHAT_TEMPERATURE=0.7
CHAT_COMPLETION_TOKENS=500

# API Authentication (for REST API)
API_API_KEY=your-secret-api-key-min-32-chars-recommended
API_ALLOWED_API_KEYS=""  # Optional: comma-separated list for multiple keys
```

**Optional API server configuration:**

```bash
# Connection Pool
API_POOL_MIN_SIZE=5
API_POOL_MAX_SIZE=20
API_POOL_TIMEOUT=30.0

# Server Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_LOG_LEVEL=info
```

### 3. Database Setup with Alembic Migrations

The project uses Alembic for database migrations, providing version control for schema changes.

```bash
# Check current migration status
alembic current

# View migration history
alembic history

# Apply all migrations (creates tables and extensions)
alembic upgrade head

# Rollback to previous migration
alembic downgrade -1

# Rollback all migrations (WARNING: deletes all data!)
alembic downgrade base
```

**For existing databases:** If you already have tables created with a previous setup method, mark the database as up-to-date:
```bash
alembic stamp head
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

Generate vector embeddings for processed text chunks using OpenAI.

```bash
# Generate embeddings
python main.py embed

# Custom batch size
python main.py embed --batch-size 50
```

**Options:**
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
# Run full pipeline
python main.py pipeline

# Skip ingestion (use existing articles)
python main.py pipeline --skip-ingestion

# Skip processing (use existing chunks)
python main.py pipeline --skip-processing

# Dry run (skip embedding generation)
python main.py pipeline --skip-embedding

# Process specific articles only
python main.py pipeline --article-ids 123 456

# Debug logging
python main.py --log-level DEBUG pipeline
```

**Options:**
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

#### 5. Database Management

Manage database schema with version-controlled Alembic migrations:

```bash
# Check current migration version
alembic current

# Apply all pending migrations
alembic upgrade head

# Rollback last migration
alembic downgrade -1

# Create new migration (after schema changes)
alembic revision -m "description"

# View migration history
alembic history
```

### Global Options

- `--log-level {DEBUG|INFO|WARNING|ERROR|CRITICAL}` - Set logging verbosity (default: INFO)
- `--version` - Show version information
- `--help` - Show help message

### Examples for Task Scheduler

```bash
# Daily ingestion job (runs every day at 2 AM)
python main.py --log-level INFO ingest

# Weekly full pipeline (runs every Sunday at 3 AM)
python main.py --log-level INFO pipeline

# Incremental update (skip ingestion, process new articles)
python main.py --log-level INFO pipeline --skip-ingestion

# Generate embeddings for already-processed chunks
python main.py --log-level INFO embed --batch-size 100
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
with RAGPipeline() as pipeline:
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

## REST API

The backend includes a production-ready FastAPI application providing REST API endpoints for search and query operations.

### Quick Start

```bash
# Development mode (auto-reload)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode (4 workers)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Access documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### Available Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/v1/search/bm25` | POST | Yes | BM25 keyword search (~50-100ms) |
| `/api/v1/search/vector` | POST | Yes | Vector semantic search (~500ms-1s) |
| `/api/v1/search/hybrid` | POST | Yes | Hybrid search with RRF/weighted fusion |
| `/health/` | GET | No | Detailed health check with component status |
| `/health/ready` | GET | No | Simple readiness probe for orchestration |

### Authentication

All search endpoints require API key authentication:

```bash
curl -X POST http://localhost:8000/api/v1/search/bm25 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"query": "password reset", "top_k": 5}'
```

### Features

- 🔐 Header-based API key authentication
- 🚀 Fast BM25 search (<100ms after corpus cached)
- 🧠 Semantic vector search with Azure OpenAI embeddings
- 🔀 Hybrid search with Reciprocal Rank Fusion (RRF) or weighted scoring
- 📊 Query logging for analytics
- 🏥 Health checks for monitoring
- ⚡ Connection pooling for concurrent requests
- 📖 OpenAPI/Swagger documentation

**For complete API documentation, examples, and deployment guides, see [API_README.md](API_README.md).**

## Docker Deployment

The application can be containerized using Docker for easy deployment and portability. The Docker setup assumes you have an existing PostgreSQL database with pgvector extension.

### Prerequisites

- Docker installed on your system
- PostgreSQL 14+ with pgvector extension (external database)
- All required environment variables configured

### Quick Start with Docker Compose

1. **Create environment configuration**

```bash
# Copy example environment file
cp example.env .env.local

# Edit .env.local with your actual credentials
# IMPORTANT: Update DB_HOST to point to your PostgreSQL server
# For Docker Desktop (macOS/Windows): use host.docker.internal
# For Linux: use host IP or --network host
# For production: use actual database hostname
```

2. **Build and run with docker-compose**

```bash
# Build and start the API server
docker-compose --env-file .env.local up --build

# View logs
docker-compose logs -f api

# Stop the container
docker-compose down
```

3. **Access the API**

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health/ready

### Build Docker Image Manually

```bash
# Build the Docker image
docker build -t helpdesk-api:1.0.0 .

# Run the container
docker run -d \
  --name helpdesk-api \
  --env-file .env.local \
  -p 8000:8000 \
  helpdesk-api:1.0.0

# View logs
docker logs -f helpdesk-api

# Stop the container
docker stop helpdesk-api
docker rm helpdesk-api
```

### Docker Architecture

**Multi-Stage Build:**
- **Stage 1 (Builder)**: Compiles Python dependencies (gcc, g++, build tools)
- **Stage 2 (Runtime)**: Minimal production image with only runtime dependencies

**Key Features:**
- 🔒 Non-root user (`appuser`) for security
- 📦 Multi-stage build (~500-700MB final image)
- 🏥 Built-in health checks
- 🔄 Automatic database migration on startup
- ⚡ Connection pooling for concurrent requests
- 📝 Comprehensive logging

**Startup Sequence:**
1. Wait for external PostgreSQL database to be ready (max 60 seconds)
2. Run Alembic migrations: `alembic upgrade head`
3. Start uvicorn server with configured workers

### Database Connectivity

**For localhost database access:**

- **Docker Desktop (macOS/Windows):**
  ```bash
  # In .env.local, set:
  DB_HOST=host.docker.internal
  ```

- **Linux:**
  ```bash
  # Option 1: Use host network mode
  docker run --network host --env-file .env.local -p 8000:8000 helpdesk-api:1.0.0

  # Option 2: Use host IP address
  # In .env.local, set:
  DB_HOST=192.168.1.100  # Your host machine IP
  ```

- **Production:**
  ```bash
  # Use actual database hostname or IP
  DB_HOST=db.example.com
  ```

### Development Mode with Hot Reload

To enable code hot-reload during development, uncomment the volume mounts in `docker-compose.yml`:

```yaml
volumes:
  - ./api:/app/api
  - ./core:/app/core
  - ./utils:/app/utils
  - ./logs:/app/logs
```

This allows you to edit code locally and see changes without rebuilding the container.

### Health Checks

**Readiness Probe** (no authentication required):
```bash
curl http://localhost:8000/health/ready
# Response: {"status": "ok"}
```

**Detailed Health Check** (with authentication):
```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/health/
# Response: JSON with BM25, vector, and database pool status
```

### Resource Requirements

- **Memory**: 1-2GB (depending on BM25 corpus size and worker count)
- **CPU**: 2+ cores recommended
- **Disk**: Minimal (source code only, data in PostgreSQL)
- **Network**: Access to PostgreSQL and Azure OpenAI endpoints

### Troubleshooting

**Database Connection Issues:**
```bash
# Check if database is accessible from container
docker exec helpdesk-api pg_isready -h $DB_HOST -U $DB_USER -d $DB_NAME

# View container logs
docker logs helpdesk-api

# Check database connectivity
docker-compose logs api | grep -i "database"
```

**Health Check Failing:**
```bash
# Check if BM25 corpus is loading (can take 30-60 seconds)
docker logs helpdesk-api | grep -i "bm25"

# Manually test health endpoint
docker exec helpdesk-api curl http://localhost:8000/health/ready
```

**Out of Memory:**
```bash
# Reduce number of workers in .env.local
API_WORKERS=2

# Or in docker-compose.yml
environment:
  API_WORKERS: 2
```

**Migration Failures:**
```bash
# Check migration status
docker exec helpdesk-api alembic current

# View migration history
docker exec helpdesk-api alembic history

# Manually run migrations
docker exec helpdesk-api alembic upgrade head
```

### Production Deployment

For production deployment with Kubernetes, AWS ECS, or other container orchestration platforms:

1. **Use managed PostgreSQL** (AWS RDS, Azure Database, etc.) with pgvector
2. **Store secrets securely** (Kubernetes Secrets, AWS Secrets Manager, HashiCorp Vault)
3. **Set resource limits**:
   - Memory: 1-2GB per pod/container
   - CPU: 500m-1000m per pod/container
4. **Configure health probes**:
   - Liveness: `GET /health/ready`
   - Readiness: `GET /health/ready`
5. **Enable TLS/HTTPS** via ingress/load balancer
6. **Implement rate limiting** at API gateway or middleware
7. **Enable log aggregation** (CloudWatch, ELK, Datadog)
8. **Set up monitoring** (Prometheus, Grafana)

**Example Kubernetes deployment excerpt:**
```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: helpdesk-api:1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

## Retrieval System

The backend supports multiple retrieval strategies: BM25 (sparse), vector (dense), and hybrid search.

### BM25 Search (Keyword-Based)

BM25 (Best Matching 25) provides fast keyword-based retrieval using term frequency and document statistics.

#### Basic Usage

```python
from core.bm25_search import BM25Retriever

# Initialize retriever
retriever = BM25Retriever(
    k1=1.5,        # Term frequency saturation (1.2-2.0 typical)
    b=0.75,        # Length normalization (0.75 typical)
    use_cache=True # Cache corpus statistics for faster queries
)

# Single query search
results = retriever.search(
    query="How do I reset my password?",
    top_k=5,              # Return top 5 results
    min_score=0.5         # Minimum relevance score (optional)
)

# Access results
for result in results:
    print(f"[{result.rank}] Score: {result.score:.4f}")
    print(f"Content: {result.chunk.text_content[:100]}...")
    print(f"Source: {result.chunk.source_url}\n")

# Batch search for multiple queries
queries = ["password reset", "VPN setup", "email config"]
batch_results = retriever.batch_search(queries=queries, top_k=3)

# Get retriever statistics
stats = retriever.get_stats()
print(f"Corpus: {stats['num_chunks']} chunks, {stats['num_unique_terms']} terms")
```

#### BM25 Parameters

- **k1** (1.2-2.0): Controls term frequency saturation
  - Higher values give more weight to term frequency
  - Typical value: 1.5

- **b** (0.0-1.0): Controls length normalization
  - 0 = no normalization
  - 1 = full normalization by document length
  - Typical value: 0.75

#### Export Results

```python
# Export to JSON
result_dict = result.to_dict()
# Contains: rank, score, chunk_id, parent_article_id, text_content, source_url, etc.

# Save to file
import json
with open("search_results.json", "w") as f:
    json.dump([r.to_dict() for r in results], f, indent=2)
```

#### Example Script

Run the example script to see BM25 in action:

```bash
python -m examples.bm25_search_example
```

### Vector Search (Semantic)

Vector search provides dense semantic retrieval using OpenAI embeddings and pgvector for cosine similarity.

#### Basic Usage

```python
from core.vector_search import VectorRetriever

# Initialize retriever
retriever = VectorRetriever()

# Single query search
results = retriever.search(
    query="How do I reset my password?",
    top_k=5,                    # Return top 5 results
    min_similarity=0.6          # Minimum cosine similarity (optional)
)

# Access results
for result in results:
    print(f"[{result.rank}] Similarity: {result.similarity:.4f}")
    print(f"Content: {result.chunk.text_content[:100]}...")
    print(f"Source: {result.chunk.source_url}\n")

# Batch search for multiple queries
queries = ["password reset", "VPN setup", "email config"]
batch_results = retriever.batch_search(queries=queries, top_k=3)

# Find similar chunks (recommendations)
similar = retriever.find_similar_to_chunk(
    chunk_id="some-chunk-uuid",
    top_k=5,
    min_similarity=0.6
)

# Get retriever statistics
stats = retriever.get_stats()
print(f"Embeddings: {stats['num_embeddings']}")
print(f"Model: {stats['model']}")
print(f"Dimension: {stats['embedding_dimension']}")

# Clean up
retriever.close()
```

#### How Vector Search Works

1. **Query Embedding**: Converts your search query into a 3072-dimensional vector using OpenAI's `text-embedding-3-large` model
2. **Similarity Search**: Uses pgvector's cosine distance operator (`<=>`) to find semantically similar chunks
3. **Ranking**: Returns results ranked by cosine similarity (0-1, where 1 is identical)

#### Similarity Scores

- **Cosine Similarity**: Ranges from 0 (dissimilar) to 1 (identical)
- **Calculation**: `similarity = 1 - cosine_distance`
- **Typical Thresholds**:
  - > 0.8: Highly relevant
  - 0.6-0.8: Moderately relevant
  - < 0.6: Possibly relevant

#### Key Features

- **Semantic Understanding**: Finds conceptually similar content even without exact keyword matches
- **Synonym Handling**: Understands "login issues" matches "authentication problems"
- **Contextual Search**: Captures meaning and context, not just keywords
- **Efficient**: Leverages pgvector's optimized similarity search

#### Export Results

```python
# Export to JSON
result_dict = result.to_dict()
# Contains: rank, similarity, chunk_id, parent_article_id, text_content, source_url, etc.

# Save to file
import json
with open("search_results.json", "w") as f:
    json.dump([r.to_dict() for r in results], f, indent=2)
```

#### Example Script

Run the example script to see vector search in action:

```bash
python -m examples.vector_search_example
```

#### Context Manager Support

```python
# Automatic cleanup with context manager
with VectorRetriever() as retriever:
    results = retriever.search("password reset", top_k=5)
    # Process results...
# Connections automatically closed
```

#### Vector Search vs BM25

**Use Vector Search when:**
- You need semantic understanding
- Queries use different wording than documents
- Searching for concepts, not keywords
- Users ask questions in natural language

**Use BM25 when:**
- You need exact keyword matches
- Documents contain specific technical terms
- Fast retrieval is critical
- Query terms match document vocabulary

### Hybrid Search

Hybrid search combines BM25 and vector search using fusion algorithms for optimal retrieval.

#### Available via REST API

Hybrid search is available through the FastAPI REST API endpoints. See [API_README.md](API_README.md) for detailed documentation.

#### Fusion Methods

1. **Reciprocal Rank Fusion (RRF)** - Default, recommended
   - Rank-based fusion (robust to score scales)
   - Formula: `score = Σ(1 / (k + rank))`

2. **Weighted Score Fusion**
   - Score-based fusion with normalized weights
   - Formula: `score = (w_bm25 × norm_bm25) + (w_vec × similarity)`

#### Programmatic Usage

```python
from api.utils.hybrid_search import hybrid_search
from core.bm25_search import BM25Retriever
from core.vector_search import VectorRetriever

# Initialize retrievers
bm25 = BM25Retriever(use_cache=True)
vector = VectorRetriever()

# Perform hybrid search with RRF
results = hybrid_search(
    query="password reset issues",
    bm25_retriever=bm25,
    vector_retriever=vector,
    top_k=10,
    fusion_method="rrf",  # or "weighted"
    rrf_k=60,
    bm25_weight=0.5
)

# Access results
for result in results:
    print(f"[{result['rank']}] Score: {result['score']:.4f}")
    print(f"Content: {result['text_content'][:100]}...")
```

**Use Hybrid Search when:**
- You want the best of both keyword and semantic search
- Your queries vary in style (some technical, some natural language)
- You need robust results across different query types
- General-purpose search with balanced precision and recall

