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
5. **Retrieval**: Hybrid search using BM25 (sparse) and vector similarity (dense)

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

## Retrieval System

The backend supports hybrid retrieval combining sparse (BM25) and dense (vector) search strategies.

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

Coming soon: Combine BM25 and vector search with configurable weighting for optimal retrieval.

