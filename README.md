# RAG Backend Service for Helpdesk Chatbot

A comprehensive RAG (Retrieval-Augmented Generation) backend service that ingests, processes, and embeds knowledge base articles from TeamDynamix (TDX) for use in a helpdesk chatbot.

## Architecture Overview

The system follows a modular pipeline architecture:

```
TDX API → Ingestion → Storage (Raw) → Processing → Embedding → Storage (Vector) → Retrieval
```

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
  - Creates articles and embeddings tables
  - Installs pgvector extension
  - Creates indexes and foreign keys
- **Tests**: 18 passing tests

### 🚧 Work In Progress (WIP)

#### 7. **Processing Layer** (`core/processing.py`)
- **Status**: 🚧 WIP
- **Planned Features**:
  - HTML to Markdown conversion
  - Semantic text chunking
  - Token counting
  - Chunk metadata management

#### 8. **Embedding Layer** (`core/embedding.py`)
- **Status**: 🚧 WIP
- **Planned Features**:
  - Integration with AWS/Azure AI endpoints
  - Batch embedding generation
  - Embedding caching

#### 9. **Vector Storage Layer** (`core/storage_vector.py`)
- **Status**: 🚧 WIP
- **Planned Features**:
  - pgvector integration
  - Vector similarity search
  - Chunk storage and retrieval

#### 10. **Pipeline** (`core/pipeline.py`)
- **Status**: 🚧 WIP
- **Planned Features**:
  - End-to-end orchestration
  - Error handling and retry logic
  - Progress tracking

## Database Schema

### Articles Table (Raw Storage)
```sql
- id (int): Article ID from TDX
- title (text): Article title
- url (text): Public URL
- content_html (text): Raw HTML content
- last_modified_date (timestamp): Last modification date
- raw_ingestion_date (timestamp): When article was ingested
```

### Embeddings Table (Vector Storage) - WIP
```sql
- chunk_id (text): Unique chunk identifier
- parent_article_id (int): Reference to articles table
- chunk_sequence (int): Order within article
- text_content (text): Clean text content
- token_count (int): Number of tokens
- source_url (text): Article URL
- embedding (vector): pgvector embedding
```

## Test Coverage

- **Total Tests**: 48
- **Status**: ✅ All Passing
- **Coverage**:
  - TDXClient: 5 tests
  - ArticleProcessor: 12 tests
  - PostgresClient: 13 tests
  - DatabaseBootstrap: 18 tests

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
│   ├── processing.py       # 🚧 Text processing (WIP)
│   ├── embedding.py        # 🚧 Embedding generation (WIP)
│   ├── storage_vector.py   # 🚧 Vector storage (WIP)
│   └── pipeline.py         # 🚧 Pipeline orchestration (WIP)
├── utils/
│   ├── api_client.py       # ✅ TDX API client
│   ├── bootstrap_db.py     # ✅ Database setup script
│   └── logger.py           # Logging utilities
└── tests/
    ├── conftest.py         # ✅ Shared fixtures
    ├── test_ingestion.py   # ✅ Ingestion tests
    ├── test_storage.py     # ✅ Storage tests
    └── test_bootstrap.py   # ✅ Bootstrap tests
```
