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
3. **Embedding**: Generate vector embeddings for chunks (WIP)
4. **Storage**: Store embeddings with metadata in `embeddings` table (WIP)
5. **Retrieval**: Semantic search over embedded chunks (WIP)

## Module Status

### ✅ Working Modules

#### 1. **API Client** (`utils/api_client.py`)
- **Status**: ✅ Fully Implemented & Tested
- **Features**:
  - TDX API authentication with bearer token management
  - Rate limiting (60 requests per 60 seconds)
  - Automatic retry with exponential backoff
  - Connection pooling and session management
  - Article retrieval with error handling
- **Tests**: 5 passing tests

#### 2. **Ingestion Layer** (`core/ingestion.py`)
- **Status**: ✅ Fully Implemented & Tested
- **Features**:
  - Sync articles from TDX API
  - Compare API state vs database state
  - Categorize articles (new, updated, unchanged, skipped)
  - Filter phishing emails category
  - Handle missing/null values gracefully
  - Identify deleted articles
- **Tests**: 12 passing tests

#### 3. **Raw Storage Layer** (`core/storage_raw.py`)
- **Status**: ✅ Fully Implemented & Tested
- **Features**:
  - PostgreSQL connection management with context managers
  - Insert new articles
  - Update existing articles
  - Retrieve article metadata
  - Get existing article IDs
  - Automatic commit/rollback handling
- **Tests**: 13 passing tests

#### 4. **Configuration** (`core/config.py`)
- **Status**: ✅ Implemented
- **Features**:
  - Environment-based configuration
  - Secret management with Pydantic SecretStr
  - Database and API credentials management

#### 5. **Schemas** (`core/schemas.py`)
- **Status**: ✅ Implemented
- **Features**:
  - TdxArticle model with validation
  - TextChunk model for processing
  - VectorRecord model for embeddings

#### 6. **Database Bootstrap** (`utils/bootstrap_db.py`)
- **Status**: ✅ Fully Implemented & Tested
- **Features**:
  - Automated database setup
  - Dry-run mode to preview changes
  - Full-reset mode to rebuild database
  - Status checking
  - Creates articles, article_chunks, and embeddings tables
  - Installs pgvector extension
  - Creates indexes and foreign keys
- **Tests**: 21 passing tests

#### 7. **Processing Layer** (`core/processing.py`)
- **Status**: ✅ Fully Implemented & Tested
- **Features**:
  - HTML to Markdown conversion using html2text
  - Whitespace normalization and text cleanup
  - Token counting using tiktoken (cl100k_base encoding)
  - Configurable HTML converter (ignores links, images, tables)
  - Consistent text processing for embeddings
- **Tests**: 20 passing tests

#### 8. **Tokenizer Utility** (`utils/tokenizer.py`)
- **Status**: ✅ Implemented
- **Features**:
  - Token counting using OpenAI's tiktoken library
  - Supports cl100k_base encoding (GPT-3.5/GPT-4)
  - Accurate token estimation for embedding size limits

### 🚧 Work In Progress (WIP)

#### 9. **Embedding Layer** (`core/embedding.py`)
- **Status**: 🚧 WIP
- **Planned Features**:
  - Integration with AWS/Azure AI endpoints
  - Batch embedding generation
  - Embedding caching
  - Semantic text chunking

#### 10. **Vector Storage Layer** (`core/storage_vector.py`)
- **Status**: 🚧 WIP
- **Planned Features**:
  - pgvector integration
  - Vector similarity search
  - Chunk storage and retrieval

#### 11. **Pipeline** (`core/pipeline.py`)
- **Status**: 🚧 WIP
- **Planned Features**:
  - End-to-end orchestration
  - Error handling and retry logic
  - Progress tracking

## Database Schema

### Articles Table (Raw Storage)
```sql
- id (int): Article ID from TDX (Primary Key)
- title (text): Article title
- url (text): Public URL
- content_html (text): Raw HTML content
- last_modified_date (timestamp): Last modification date
- raw_ingestion_date (timestamp): When article was ingested
- created_at (timestamp): Record creation timestamp
```

### Article Chunks Table (Processed Storage)
```sql
- id (int): Unique chunk ID (Primary Key)
- parent_article_id (int): Reference to articles table
- chunk_sequence (int): Order within article
- text_content (text): Clean, processed text content
- token_count (int): Number of tokens in chunk
- url (text): Source article URL
- last_modified_date (timestamp): Last modification date
```

### Embeddings Table (Vector Storage) - WIP
```sql
- chunk_id (text): Unique chunk identifier (Primary Key)
- parent_article_id (int): Foreign key to articles table
- chunk_sequence (int): Order within article
- text_content (text): Clean text content
- token_count (int): Number of tokens
- source_url (text): Article URL
- embedding (vector(1536)): pgvector embedding
- created_at (timestamp): Embedding creation timestamp
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

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
# TDX API Configuration
BASE_URL=https://your-instance.teamdynamix.com
APP_ID=your_app_id
WEBSERVICES_KEY=your_key
BEID=your_beid

# Database Configuration
DB_HOST=localhost
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=your_database
```

3. Bootstrap the database:
```bash
# Check current database status
python -m utils.bootstrap_db --status

# See what would be created (dry-run)
python -m utils.bootstrap_db --dry-run

# Create tables and extensions
python -m utils.bootstrap_db

# Reset database (WARNING: deletes all data)
python -m utils.bootstrap_db --full-reset
```

## Usage

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

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_ingestion.py -v
```

## Project Structure

```
backend/
├── core/
│   ├── config.py           # ✅ Configuration management
│   ├── schemas.py          # ✅ Pydantic models
│   ├── ingestion.py        # ✅ Article ingestion logic
│   ├── storage_raw.py      # ✅ Raw article storage
│   ├── processing.py       # ✅ Text processing & token counting
│   ├── embedding.py        # 🚧 Embedding generation (WIP)
│   ├── storage_vector.py   # 🚧 Vector storage (WIP)
│   └── pipeline.py         # 🚧 Pipeline orchestration (WIP)
├── utils/
│   ├── api_client.py       # ✅ TDX API client
│   ├── bootstrap_db.py     # ✅ Database setup script
│   ├── tokenizer.py        # ✅ Token counting utility
│   └── logger.py           # Logging utilities
└── tests/
    ├── conftest.py         # ✅ Shared fixtures
    ├── test_ingestion.py   # ✅ Ingestion tests (17 tests)
    ├── test_storage.py     # ✅ Storage tests (13 tests)
    ├── test_bootstrap.py   # ✅ Bootstrap tests (21 tests)
    └── test_processing.py  # ✅ Processing tests (20 tests)
```
