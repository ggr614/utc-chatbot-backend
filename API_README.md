# RAG Helpdesk API Documentation

Comprehensive REST API for helpdesk knowledge base search using Retrieval-Augmented Generation (RAG) with BM25, vector, and hybrid search capabilities.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
- [Request/Response Schemas](#requestresponse-schemas)
- [Search Methods](#search-methods)
- [Error Handling](#error-handling)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Performance](#performance)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

The RAG Helpdesk API provides three search methods for querying a knowledge base:

1. **BM25 Search** - Fast keyword-based sparse retrieval
2. **Vector Search** - Semantic dense retrieval using embeddings
3. **Hybrid Search** - Combined BM25 + vector with fusion algorithms

### Key Features

- 🔐 **Header-based API key authentication**
- 🚀 **Fast BM25 search** (<100ms after corpus cached)
- 🧠 **Semantic vector search** with Azure OpenAI embeddings
- 🔀 **Hybrid search** with Reciprocal Rank Fusion (RRF) or weighted scoring
- 📊 **Query logging** for analytics
- 🏥 **Health checks** for monitoring
- ⚡ **Connection pooling** for concurrent requests
- 📖 **OpenAPI/Swagger documentation** at `/docs`

### Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP + X-API-Key
       ▼
┌─────────────────────────────────────┐
│         FastAPI Application         │
│  ┌─────────────────────────────┐   │
│  │   Connection Pool (5-20)     │   │
│  └─────────────────────────────┘   │
│  ┌──────────┐  ┌──────────────┐   │
│  │   BM25   │  │    Vector    │   │
│  │ Retriever│  │  Retriever   │   │
│  └──────────┘  └──────────────┘   │
└─────────────────────────────────────┘
       │                    │
       ▼                    ▼
┌─────────────┐    ┌────────────────┐
│  PostgreSQL │    │ Azure OpenAI   │
│  + pgvector │    │   Embeddings   │
└─────────────┘    └────────────────┘
```

---

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example environment file
cp example.env .env

# Edit .env and set required variables
nano .env
```

**Minimum required configuration:**

```bash
# Database
DB_HOST=localhost
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=helpdesk_chatbot

# Azure OpenAI (for vector search)
EMBEDDING_API_KEY=your_azure_openai_key
EMBEDDING_ENDPOINT=https://your-resource.openai.azure.com/
EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large
EMBEDDING_API_VERSION=2024-02-01
EMBEDDING_EMBED_DIM=3072
EMBEDDING_MAX_TOKENS=8191

# API Authentication
API_API_KEY=your-secret-api-key-min-32-chars
```

### 3. Start Server

```bash
# Development mode (auto-reload)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode (4 workers)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Test API

```bash
# Health check (no auth required)
curl http://localhost:8000/health/

# BM25 search
curl -X POST http://localhost:8000/api/v1/search/bm25 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key-min-32-chars" \
  -d '{"query": "password reset", "top_k": 5}'
```

### 5. View Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Authentication

All search endpoints require API key authentication via the `X-API-Key` header.

### Header Format

```
X-API-Key: your-secret-api-key-here
```

### Configuration

**Primary API Key:**
```bash
API_API_KEY=your-secret-api-key-min-32-chars-recommended
```

**Multiple API Keys** (comma-separated):
```bash
API_ALLOWED_API_KEYS=client1-key,client2-key,client3-key
```

### Security Considerations

- ✅ Designed for **internal network use**
- ✅ Use **HTTPS/TLS** even on internal network
- ✅ **Rotate keys** using multiple key support (zero-downtime)
- ✅ Keys should be **minimum 32 characters**
- ❌ No rate limiting implemented (add if needed)
- ❌ No JWT/OAuth (simple header-based only)

### Authentication Errors

**401 Unauthorized:**
```json
{
  "detail": "Invalid API key"
}
```

**500 Internal Server Error:**
```json
{
  "detail": "API authentication not properly configured"
}
```

---

## API Endpoints

### Base URL

```
http://localhost:8000
```

### Endpoints Overview

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/` | GET | No | Redirects to `/docs` |
| `/health/` | GET | No | Detailed health check |
| `/health/ready` | GET | No | Readiness probe |
| `/api/v1/search/bm25` | POST | Yes | BM25 keyword search |
| `/api/v1/search/vector` | POST | Yes | Vector semantic search |
| `/api/v1/search/hybrid` | POST | Yes | Hybrid search |

---

### POST /api/v1/search/bm25

Fast keyword-based sparse retrieval using BM25 algorithm.

**Best for:**
- Exact keyword matching
- Technical terms and acronyms
- Identifiers (IDs, codes, specific names)

**Request:**

```json
{
  "query": "password reset troubleshooting",
  "top_k": 10,
  "min_score": 1.0,
  "user_id": "user123"
}
```

**Request Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query (1-1000 chars) |
| `top_k` | integer | No | 10 | Number of results (1-100) |
| `min_score` | float | No | null | Minimum BM25 score threshold |
| `user_id` | string | No | null | User ID for analytics (max 255 chars) |

**Response:** See [SearchResponse Schema](#searchresponse-schema)

**Performance:** ~50-100ms (after corpus cached)

---

### POST /api/v1/search/vector

Semantic dense retrieval using vector embeddings and cosine similarity.

**Best for:**
- Natural language queries
- Conceptual similarity
- Synonym and paraphrase matching

**Request:**

```json
{
  "query": "how do I recover my account access",
  "top_k": 10,
  "min_similarity": 0.7,
  "user_id": "user123"
}
```

**Request Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query (1-1000 chars) |
| `top_k` | integer | No | 10 | Number of results (1-100) |
| `min_similarity` | float | No | null | Minimum cosine similarity (0.0-1.0) |
| `user_id` | string | No | null | User ID for analytics (max 255 chars) |

**Response:** See [SearchResponse Schema](#searchresponse-schema)

**Performance:** ~500ms-1s (includes Azure OpenAI API call for embedding)

**Note:** This endpoint makes an API call to Azure OpenAI to generate the query embedding.

---

### POST /api/v1/search/hybrid

Combined BM25 + vector search with fusion algorithms.

**Best for:**
- General-purpose search
- Balancing precision and recall
- Combining keyword matching and semantic understanding

**Request:**

```json
{
  "query": "vpn connection issues",
  "top_k": 10,
  "fusion_method": "rrf",
  "rrf_k": 60,
  "bm25_weight": 0.5,
  "min_bm25_score": 1.0,
  "min_vector_similarity": 0.7,
  "user_id": "user123"
}
```

**Request Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query (1-1000 chars) |
| `top_k` | integer | No | 10 | Number of results (1-100) |
| `fusion_method` | string | No | "rrf" | Fusion algorithm: "rrf" or "weighted" |
| `rrf_k` | integer | No | 60 | RRF constant (only used with "rrf") |
| `bm25_weight` | float | No | 0.5 | BM25 weight 0-1 (only used with "weighted") |
| `min_bm25_score` | float | No | null | Minimum BM25 score threshold |
| `min_vector_similarity` | float | No | null | Minimum vector similarity (0.0-1.0) |
| `user_id` | string | No | null | User ID for analytics (max 255 chars) |

**Fusion Methods:**

1. **RRF (Reciprocal Rank Fusion)** - Default, recommended
   - Rank-based fusion (robust to score scales)
   - Formula: `score = Σ(1 / (k + rank))`
   - Parameter: `rrf_k` (default 60)

2. **Weighted** - Score-based fusion
   - Combines normalized scores with weights
   - Formula: `score = (w_bm25 × norm_bm25) + (w_vec × similarity)`
   - Parameter: `bm25_weight` (default 0.5)

**Response:** See [SearchResponse Schema](#searchresponse-schema)

**Performance:** ~500ms-1s (dominated by vector embedding API call)

---

### GET /health/

Detailed health check with component status.

**Authentication:** None (intended for monitoring systems)

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-01-28T10:30:00Z",
  "checks": {
    "bm25": {
      "status": "healthy",
      "num_chunks": 1500,
      "cached": true,
      "avg_doc_length": 150.5
    },
    "vector": {
      "status": "healthy",
      "num_embeddings": 1500,
      "embedding_dimension": 3072,
      "model": "text-embedding-3-large"
    },
    "database": {
      "status": "healthy",
      "pool_size": 10,
      "pool_available": 8,
      "requests_waiting": 0
    }
  }
}
```

**Status Values:**
- `healthy` - All components operational
- `degraded` - Some components have issues but system functional
- `unhealthy` - Critical components failed

---

### GET /health/ready

Simple readiness probe for Kubernetes/orchestration.

**Authentication:** None

**Response (200 OK):**

```json
{
  "status": "ready"
}
```

**Response (503 Service Unavailable):**

```json
{
  "status": "not ready",
  "error": "BM25 retriever not initialized"
}
```

---

## Request/Response Schemas

### SearchResponse Schema

Unified response for all search endpoints (BM25, vector, hybrid).

```json
{
  "query": "password reset",
  "method": "bm25",
  "results": [
    {
      "rank": 1,
      "score": 8.5,
      "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
      "parent_article_id": "650e8400-e29b-41d4-a716-446655440000",
      "chunk_sequence": 0,
      "text_content": "To reset your password, navigate to the login page...",
      "token_count": 150,
      "source_url": "https://help.example.com/article/123",
      "last_modified_date": "2025-01-15T10:30:00Z"
    }
  ],
  "total_results": 10,
  "latency_ms": 156,
  "metadata": {
    "min_score": 1.0
  }
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Original query text |
| `method` | string | Search method: "bm25", "vector", or "hybrid" |
| `results` | array | Array of SearchResultChunk objects |
| `total_results` | integer | Number of results returned |
| `latency_ms` | integer | Query processing latency in milliseconds |
| `metadata` | object | Method-specific metadata |

### SearchResultChunk Schema

Individual search result with metadata.

```json
{
  "rank": 1,
  "score": 8.5,
  "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
  "parent_article_id": "650e8400-e29b-41d4-a716-446655440000",
  "chunk_sequence": 0,
  "text_content": "To reset your password, navigate to the login page...",
  "token_count": 150,
  "source_url": "https://help.example.com/article/123",
  "last_modified_date": "2025-01-15T10:30:00Z"
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `rank` | integer | Result ranking position (1-indexed) |
| `score` | float | Relevance score (BM25 score, similarity, or combined) |
| `chunk_id` | UUID | Unique chunk identifier |
| `parent_article_id` | UUID | Source article UUID |
| `chunk_sequence` | integer | Chunk position within article (0-indexed) |
| `text_content` | string | Clean text content of the chunk |
| `token_count` | integer | Number of tokens in chunk |
| `source_url` | string | Source article URL |
| `last_modified_date` | datetime | Article last modified timestamp (ISO 8601) |

**Score Interpretation:**

- **BM25**: Unbounded score (typically 0-20), higher = more relevant
- **Vector**: Cosine similarity (0.0-1.0), higher = more similar
- **Hybrid**: Combined score based on fusion method (interpretation depends on method)

---

## Search Methods

### BM25 Search (Sparse Retrieval)

**Algorithm:** BM25 (Best Matching 25)

**How it works:**
- Loads entire corpus into memory
- Tokenizes query and documents
- Scores based on term frequency and inverse document frequency
- Fast in-memory search (no network calls)

**Parameters:**
- `k1=1.5` - Term frequency saturation (higher = more weight to frequency)
- `b=0.75` - Length normalization (0 = none, 1 = full)

**Pros:**
- ✅ Very fast (~50-100ms)
- ✅ Exact keyword matching
- ✅ No API calls required
- ✅ Works well for technical terms

**Cons:**
- ❌ No semantic understanding
- ❌ High memory usage (corpus in RAM)
- ❌ Doesn't handle synonyms/paraphrases

**Best use cases:**
- Searching for specific error codes
- Technical documentation lookup
- Queries with unique identifiers

---

### Vector Search (Dense Retrieval)

**Algorithm:** Cosine similarity on embeddings

**How it works:**
- Generates query embedding via Azure OpenAI API (~500ms)
- Searches pgvector database using cosine similarity
- Returns semantically similar chunks

**Model:** `text-embedding-3-large` (3072 dimensions)

**Pros:**
- ✅ Semantic understanding
- ✅ Handles synonyms and paraphrases
- ✅ Works with natural language
- ✅ No corpus in memory

**Cons:**
- ❌ Slower (API call required)
- ❌ Less precise for exact matches
- ❌ Requires Azure OpenAI

**Best use cases:**
- Natural language questions
- Conceptual queries
- When exact keywords unknown

---

### Hybrid Search (Combined)

**Algorithm:** BM25 + Vector with fusion

**How it works:**
- Fetches 2× top_k from both BM25 and vector
- Applies fusion algorithm to combine results
- Returns top_k merged results

**Fusion Methods:**

#### 1. Reciprocal Rank Fusion (RRF) - **Recommended**

Formula: `score(chunk) = Σ(1 / (k + rank))`

**Characteristics:**
- Rank-based (robust to score scales)
- No score normalization needed
- Proven in information retrieval research
- Default k=60 (typical value)

**When to use:**
- General-purpose search
- When you want robust results
- Default choice for most use cases

#### 2. Weighted Score Fusion

Formula: `score = (w_bm25 × norm_bm25) + (w_vec × similarity)`

**Characteristics:**
- Score-based (requires normalization)
- BM25 scores normalized to 0-1 (min-max)
- Vector similarities already 0-1
- Configurable weights (default 0.5/0.5)

**When to use:**
- When you want more control over balance
- When you know one method works better
- For A/B testing different weights

**Pros:**
- ✅ Best of both worlds
- ✅ More robust than single method
- ✅ Customizable via fusion method

**Cons:**
- ❌ Slightly slower than BM25 alone
- ❌ Requires tuning fusion parameters

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful (includes 0 results) |
| 400 | Bad Request | Validation error (invalid parameters) |
| 401 | Unauthorized | Invalid or missing API key |
| 500 | Internal Server Error | Database error or unexpected failure |
| 503 | Service Unavailable | Azure OpenAI API failure (temporary) |

### Error Response Format

```json
{
  "detail": "Human-readable error message"
}
```

### Common Errors

**Authentication Error (401):**
```json
{
  "detail": "Invalid API key"
}
```

**Validation Error (400):**
```json
{
  "detail": "Query cannot be empty or whitespace only"
}
```

**Embedding Service Error (503):**
```json
{
  "detail": "Embedding service unavailable"
}
```

**Internal Error (500):**
```json
{
  "detail": "Search failed"
}
```

### Error Handling Best Practices

1. **Implement retry logic** for 503 errors (temporary API failures)
2. **Validate input** client-side before sending requests
3. **Log errors** with request IDs for debugging
4. **Handle 401** by refreshing API key or checking configuration
5. **Monitor error rates** via health endpoint

---

## Configuration

### Environment Variables

**Required:**

```bash
# Database
DB_HOST=localhost
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=helpdesk_chatbot

# Azure OpenAI Embeddings
EMBEDDING_API_KEY=your_key
EMBEDDING_ENDPOINT=https://your-resource.openai.azure.com/
EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large
EMBEDDING_API_VERSION=2024-02-01
EMBEDDING_EMBED_DIM=3072
EMBEDDING_MAX_TOKENS=8191

# API Authentication
API_API_KEY=your-secret-api-key-min-32-chars
```

**Optional:**

```bash
# Multiple API Keys (comma-separated)
API_ALLOWED_API_KEYS=key1,key2,key3

# Connection Pool Settings
API_POOL_MIN_SIZE=5       # Minimum connections (keep warm)
API_POOL_MAX_SIZE=20      # Maximum connections
API_POOL_TIMEOUT=30       # Timeout in seconds

# Server Configuration
API_HOST=0.0.0.0          # Bind host
API_PORT=8000             # Port number
API_WORKERS=4             # Number of workers
API_LOG_LEVEL=info        # Log level (debug, info, warning, error)
```

### Connection Pool Sizing

**Formula:** `max_concurrent_requests = API_POOL_MAX_SIZE / API_WORKERS`

**Recommendations:**

| Workers | Pool Max | Concurrent Requests | Memory (approx) |
|---------|----------|---------------------|-----------------|
| 1 | 10 | 10 | 700MB |
| 2 | 20 | 10 | 1.5GB |
| 4 | 20 | 5 | 2.5GB |
| 4 | 40 | 10 | 2.5GB |

**Notes:**
- Each worker loads BM25 corpus (~500MB)
- Pool is shared across workers
- Higher pool size = more concurrent requests
- Monitor pool exhaustion via `/health/` endpoint

---

## Deployment

### Development Mode

```bash
# Single worker with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Or run directly with Python
python api/main.py
```

### Production Mode

```bash
# Multiple workers (recommended)
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --limit-concurrency 100 \
  --timeout-keep-alive 30 \
  --log-level info
```

### Docker Deployment

**Dockerfile example:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run with gunicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Kubernetes Deployment

**deployment.yaml example:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-helpdesk-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-helpdesk-api
  template:
    metadata:
      labels:
        app: rag-helpdesk-api
    spec:
      containers:
      - name: api
        image: rag-helpdesk-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: API_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: api-key
        livenessProbe:
          httpGet:
            path: /health/
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### Reverse Proxy (nginx)

**nginx.conf example:**

```nginx
upstream api_backend {
    server localhost:8000;
}

server {
    listen 80;
    server_name api.example.com;

    # HTTPS redirect
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint (no auth)
    location /health/ {
        proxy_pass http://api_backend;
        access_log off;
    }
}
```

---

## Performance

### Latency Benchmarks

| Search Method | Cold Start | Warm (Cached) | Network Calls |
|---------------|------------|---------------|---------------|
| BM25 | 1-2s | 50-100ms | 0 |
| Vector | 500ms-1s | 500ms-1s | 1 (embedding API) |
| Hybrid | 1-2s | 500ms-1s | 1 (embedding API) |

**Notes:**
- Cold start: First request after server startup (BM25 corpus load)
- Warm: Subsequent requests (BM25 corpus cached)
- Network calls: Azure OpenAI API for query embeddings

### Throughput

**Tested configuration:** 4 workers, 20 connection pool, 100 concurrent limit

| Search Method | Requests/sec | Concurrent | Latency p95 | Latency p99 |
|---------------|--------------|------------|-------------|-------------|
| BM25 | 200-300 | 100 | 150ms | 300ms |
| Vector | 10-20 | 20 | 1.5s | 2s |
| Hybrid | 10-20 | 20 | 1.5s | 2s |

**Bottlenecks:**
- BM25: CPU (in-memory search)
- Vector: Azure OpenAI API rate limits
- Hybrid: Azure OpenAI API rate limits
- All: Database connection pool exhaustion (monitor via `/health/`)

### Optimization Tips

1. **Increase workers** for BM25 (CPU-bound)
2. **Increase pool size** for concurrent requests
3. **Cache frequent queries** (add Redis layer)
4. **Use batch endpoints** for multiple queries
5. **Monitor Azure OpenAI** rate limits and quotas
6. **Add vector index** (HNSW) for faster searches

### Resource Requirements

**Minimum (1 worker):**
- CPU: 2 cores
- Memory: 1GB
- Database: PostgreSQL 14+ with pgvector

**Recommended (4 workers):**
- CPU: 4-8 cores
- Memory: 3-4GB
- Database: PostgreSQL 14+ with pgvector
- Network: Low latency to Azure OpenAI

**Memory Breakdown:**
- Python runtime: ~200MB
- BM25 corpus per worker: ~500MB
- Connection pool: ~50MB
- Buffers: ~200MB
- **Total per worker:** ~700MB
- **4 workers:** ~2.8GB

---

## Examples

### Example 1: Simple BM25 Search

**Request:**

```bash
curl -X POST http://localhost:8000/api/v1/search/bm25 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "password reset",
    "top_k": 3
  }'
```

**Response:**

```json
{
  "query": "password reset",
  "method": "bm25",
  "results": [
    {
      "rank": 1,
      "score": 12.5,
      "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
      "text_content": "To reset your password, visit the login page and click 'Forgot Password'...",
      "source_url": "https://help.example.com/article/123",
      "token_count": 150,
      "parent_article_id": "650e8400-e29b-41d4-a716-446655440000",
      "chunk_sequence": 0,
      "last_modified_date": "2025-01-15T10:30:00Z"
    }
  ],
  "total_results": 3,
  "latency_ms": 85,
  "metadata": {}
}
```

---

### Example 2: Vector Search with Similarity Threshold

**Request:**

```bash
curl -X POST http://localhost:8000/api/v1/search/vector \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "I cannot access my email",
    "top_k": 5,
    "min_similarity": 0.75,
    "user_id": "user123"
  }'
```

**Response:**

```json
{
  "query": "I cannot access my email",
  "method": "vector",
  "results": [
    {
      "rank": 1,
      "score": 0.89,
      "chunk_id": "660e8400-e29b-41d4-a716-446655440111",
      "text_content": "If you're having trouble accessing your email account, first check...",
      "source_url": "https://help.example.com/article/456",
      "token_count": 200,
      "parent_article_id": "770e8400-e29b-41d4-a716-446655440222",
      "chunk_sequence": 1,
      "last_modified_date": "2025-01-20T15:00:00Z"
    }
  ],
  "total_results": 5,
  "latency_ms": 876,
  "metadata": {
    "min_similarity": 0.75
  }
}
```

---

### Example 3: Hybrid Search with RRF

**Request:**

```bash
curl -X POST http://localhost:8000/api/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "vpn not connecting",
    "top_k": 10,
    "fusion_method": "rrf",
    "rrf_k": 60
  }'
```

**Response:**

```json
{
  "query": "vpn not connecting",
  "method": "hybrid",
  "results": [
    {
      "rank": 1,
      "score": 0.0412,
      "chunk_id": "880e8400-e29b-41d4-a716-446655440333",
      "text_content": "VPN connection troubleshooting: First, ensure your network connection is stable...",
      "source_url": "https://help.example.com/article/789",
      "token_count": 180,
      "parent_article_id": "990e8400-e29b-41d4-a716-446655440444",
      "chunk_sequence": 0,
      "last_modified_date": "2025-01-25T09:00:00Z"
    }
  ],
  "total_results": 10,
  "latency_ms": 932,
  "metadata": {
    "fusion_method": "rrf",
    "rrf_k": 60
  }
}
```

---

### Example 4: Hybrid Search with Weighted Fusion

**Request:**

```bash
curl -X POST http://localhost:8000/api/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "printer error 0x00000709",
    "top_k": 5,
    "fusion_method": "weighted",
    "bm25_weight": 0.7
  }'
```

**Response:**

```json
{
  "query": "printer error 0x00000709",
  "method": "hybrid",
  "results": [
    {
      "rank": 1,
      "score": 0.85,
      "chunk_id": "aa0e8400-e29b-41d4-a716-446655440555",
      "text_content": "Error code 0x00000709 indicates a printer driver issue. To resolve this...",
      "source_url": "https://help.example.com/article/321",
      "token_count": 220,
      "parent_article_id": "bb0e8400-e29b-41d4-a716-446655440666",
      "chunk_sequence": 2,
      "last_modified_date": "2025-01-28T11:30:00Z"
    }
  ],
  "total_results": 5,
  "latency_ms": 845,
  "metadata": {
    "fusion_method": "weighted",
    "bm25_weight": 0.7,
    "vector_weight": 0.3
  }
}
```

**Note:** Higher `bm25_weight` (0.7) gives more importance to exact keyword matching, which is good for specific error codes.

---

### Example 5: Python Client

```python
import requests

class RAGHelpDeskClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }

    def search_bm25(self, query: str, top_k: int = 10, **kwargs):
        """BM25 keyword search"""
        payload = {"query": query, "top_k": top_k, **kwargs}
        response = requests.post(
            f"{self.base_url}/api/v1/search/bm25",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def search_vector(self, query: str, top_k: int = 10, **kwargs):
        """Vector semantic search"""
        payload = {"query": query, "top_k": top_k, **kwargs}
        response = requests.post(
            f"{self.base_url}/api/v1/search/vector",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def search_hybrid(self, query: str, top_k: int = 10,
                     fusion_method: str = "rrf", **kwargs):
        """Hybrid search (BM25 + vector)"""
        payload = {
            "query": query,
            "top_k": top_k,
            "fusion_method": fusion_method,
            **kwargs
        }
        response = requests.post(
            f"{self.base_url}/api/v1/search/hybrid",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def health_check(self):
        """Check API health"""
        response = requests.get(f"{self.base_url}/health/")
        response.raise_for_status()
        return response.json()

# Usage
client = RAGHelpDeskClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# BM25 search
results = client.search_bm25("password reset", top_k=5)
print(f"Found {results['total_results']} results in {results['latency_ms']}ms")

# Hybrid search with weighted fusion
results = client.search_hybrid(
    "how to troubleshoot vpn",
    fusion_method="weighted",
    bm25_weight=0.6
)

# Health check
health = client.health_check()
print(f"Status: {health['status']}")
```

---

## Troubleshooting

### Common Issues

#### 1. "Invalid API key" Error

**Symptoms:**
```json
{"detail": "Invalid API key"}
```

**Solutions:**
- ✅ Check `.env` file has `API_API_KEY` set
- ✅ Verify header is `X-API-Key` (case-sensitive)
- ✅ Ensure no extra whitespace in API key
- ✅ Restart server after changing `.env`

---

#### 2. "BM25 retriever not initialized" Error

**Symptoms:**
```json
{"detail": "BM25 retriever not initialized"}
```

**Solutions:**
- ✅ Check database connection (verify `DB_*` settings)
- ✅ Ensure `article_chunks` table has data
- ✅ Check server startup logs for initialization errors
- ✅ Run `alembic upgrade head` to create tables

---

#### 3. "Embedding service unavailable" Error

**Symptoms:**
```json
{"detail": "Embedding service unavailable"}
```

**Solutions:**
- ✅ Verify Azure OpenAI credentials in `.env`
- ✅ Check Azure OpenAI endpoint is accessible
- ✅ Verify deployment name matches (`EMBEDDING_DEPLOYMENT_NAME`)
- ✅ Check Azure OpenAI quota/rate limits
- ✅ Test with curl: `curl $EMBEDDING_ENDPOINT`

---

#### 4. Slow Vector Search (>2s)

**Symptoms:**
- Vector search taking >2 seconds
- Timeouts on vector/hybrid endpoints

**Solutions:**
- ✅ Check Azure OpenAI latency (network issue?)
- ✅ Verify embedding API version is current
- ✅ Consider increasing `--timeout-keep-alive`
- ✅ Monitor Azure OpenAI rate limits
- ✅ Add pgvector index: `CREATE INDEX ON embeddings_openai USING hnsw(embedding vector_cosine_ops)`

---

#### 5. Connection Pool Exhausted

**Symptoms:**
```json
{"status": "degraded", "checks": {"database": {"pool_available": 0}}}
```

**Solutions:**
- ✅ Increase `API_POOL_MAX_SIZE` in `.env`
- ✅ Reduce `API_WORKERS` (each worker uses connections)
- ✅ Increase `API_POOL_TIMEOUT` for slower queries
- ✅ Monitor `/health/` endpoint for pool stats
- ✅ Check for connection leaks (should auto-close)

---

#### 6. High Memory Usage

**Symptoms:**
- Server using 3-4GB+ memory
- OOM (Out of Memory) errors

**Solutions:**
- ✅ Reduce number of workers (`API_WORKERS=2`)
- ✅ BM25 corpus is loaded per worker (~500MB each)
- ✅ Monitor memory with `ps aux | grep uvicorn`
- ✅ Consider larger instance or fewer workers
- ✅ Clear BM25 cache if needed (restart server)

---

#### 7. No Results Returned

**Symptoms:**
```json
{"results": [], "total_results": 0}
```

**Solutions:**
- ✅ Check query spelling and formatting
- ✅ Verify embeddings exist in database
- ✅ Lower `min_score` or `min_similarity` thresholds
- ✅ Try different search method (BM25 vs vector vs hybrid)
- ✅ Check if corpus is empty: `/health/` → `num_chunks`

---

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# Set in .env
API_LOG_LEVEL=debug

# Or pass to uvicorn
uvicorn api.main:app --log-level debug
```

---

### Health Check Monitoring

Monitor these metrics via `/health/` endpoint:

```python
import requests
import time

def monitor_health(url="http://localhost:8000/health/"):
    while True:
        try:
            response = requests.get(url)
            health = response.json()

            # Check overall status
            if health['status'] != 'healthy':
                print(f"⚠️  Status: {health['status']}")

            # Check database pool
            db = health['checks']['database']
            if db['pool_available'] < 2:
                print(f"⚠️  Pool low: {db['pool_available']} available")

            # Check BM25 cache
            bm25 = health['checks']['bm25']
            if not bm25['cached']:
                print("⚠️  BM25 corpus not cached")

            time.sleep(60)  # Check every minute

        except Exception as e:
            print(f"❌ Health check failed: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_health()
```

---

## Support

For issues, questions, or feature requests:

1. **Documentation**: Check this README and `/docs` endpoint
2. **Health Check**: Run `/health/` to diagnose issues
3. **Logs**: Check server logs with `--log-level debug`
4. **Database**: Verify data exists with `psql` queries

---

## License

Internal use only - not for public distribution.

## Version

API Version: 1.0.0
Last Updated: January 2025
