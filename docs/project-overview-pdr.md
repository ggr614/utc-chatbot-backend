# Project Overview and Design Record (PDR)
# UTC Helpdesk Chatbot RAG Backend

**Organization:** University of Tennessee at Chattanooga (UTC)
**Project type:** RAG (Retrieval-Augmented Generation) backend service
**Status:** Production-deployed, actively developed

---

## 1. Problem Statement

UTC's IT helpdesk Tier 1 support staff handle a large volume of repetitive tickets — password resets, VPN issues, Canvas/Banner access, MocsNet troubleshooting — that are already documented in the TeamDynamix (TDX) knowledge base. Staff currently search that knowledge base manually, which is slow when a customer is waiting on a call and the search UI requires knowing the right keywords.

This project provides an AI-powered retrieval layer on top of those same TDX articles, plus an OpenAI-compatible chat endpoint that Open WebUI can call. Support staff can ask a natural-language question and receive a structured, citation-backed answer in seconds.

---

## 2. High-Level Architecture

The system has two main concerns: **data ingestion** and **query serving**.

```
Data ingestion (offline, CLI):
  TDX REST API
    → ArticleProcessor  (raw HTML → articles table)
    → TextProcessor     (HTML → clean text → article_chunks table)
    → EmbeddingGenerator (chunks → vectors → embeddings_openai table)

Query serving (online, FastAPI):
  Open WebUI / external client
    → POST /v1/chat/completions   (SSE streaming, OpenAI-compat)
    → POST /api/v1/search/hybrid  (BM25 + vector + reranking)
    → POST /api/v1/search/bm25    (sparse keyword)
    → POST /api/v1/search/vector  (dense semantic)
    → POST /api/v1/search/hyde    (HyDE + hybrid + reranking)
```

The pipeline is run separately from the API server; both read and write the same PostgreSQL database.

---

## 3. Current State and Capabilities

- Ingests UTC TDX knowledge base articles (HTML) via TDX REST API with incremental updates
- Processes articles into token-bounded chunks (LangChain `RecursiveCharacterTextSplitter`, LiteLLM token counter)
- Generates and stores 3072-dimensional embeddings (OpenAI `text-embedding-3-large` via LiteLLM proxy)
- Four retrieval modes: BM25, vector, hybrid (RRF fusion), and HyDE
- Optional neural reranking via Cohere Rerank v3.5 through LiteLLM proxy
- OpenAI-compatible SSE chat endpoint with structured RAG response format, command routing, and tag-based system prompt selection
- Admin UI for managing per-tag system prompts
- Analytics dashboard with query volume, latency percentiles, cache stats
- Warm cache for pre-computed answers to high-frequency questions
- Full observability: query logging, result logging, reranker logging, HyDE logging, LLM response logging
- 435 unit tests across 25 test files (pytest + pytest-mock)
- Docker Compose deployment with 5 services

---

## 4. Key Design Decisions and Rationale

### 4.1 LiteLLM Proxy as AI Gateway

All AI API calls (embeddings, chat completions, reranking) route through a self-hosted LiteLLM proxy rather than calling provider APIs directly.

**Rationale:**
- Single point for credential management — no provider keys baked into the API service
- Model aliases (`text-embedding-large-3`, `gpt-5.2-chat`, `cohere-rerank-v3-5`) decouple code from specific model versions; swap the LiteLLM config without touching application code
- LiteLLM handles retry logic, provider failover, and request logging
- Enables switching between Azure OpenAI, AWS Bedrock, and direct OpenAI with a config change only
- The LiteLLM container also manages its own PostgreSQL database for request logging and spend tracking

**Trade-off:** Adds one network hop and a dependency on the LiteLLM container. If it crashes, embeddings and chat fail. Mitigated by Docker health checks and restart policies.

### 4.2 PostgreSQL + pgvector for Vector Storage

pgvector (via the `ramsrib/pgvector:16` image) provides vector similarity search inside PostgreSQL, keeping all data in a single database engine rather than introducing a dedicated vector store like Pinecone or Weaviate.

**Rationale:**
- Eliminates a separate vector database service and its operational overhead
- All data (articles, chunks, embeddings, query logs, cache) lives in one place, enabling JOIN-based queries (e.g., retrieve chunk text alongside its embedding in one query)
- CASCADE deletes propagate correctly: delete an article and its chunks and embeddings vanish automatically
- pgvector's `<=>` cosine distance operator integrates naturally with psycopg 3's raw SQL approach
- The same PostgreSQL instance is shared by LiteLLM (separate database), reducing container count

**Trade-off:** pgvector's ANN index (IVFFlat or HNSW) requires a rebuild when the corpus changes significantly. For a university helpdesk corpus in the low thousands of articles, exact KNN is fast enough that no ANN index is currently required.

### 4.3 Hybrid Search (BM25 + Vector)

Neither BM25 nor vector search alone is sufficient for the helpdesk use case.

- BM25 excels at exact technical term matching: "MocsID", "Banner 9", "VPN profile". These are low-frequency tokens that embeddings often normalize away.
- Vector search excels at paraphrase and natural-language queries: "I can't log in" matches articles about password resets even without the exact words.

Combining both via Reciprocal Rank Fusion (RRF) captures both strengths. The RRF constant `k` is tunable (default 1, lower than the academic default of 60) because the corpus is domain-specific and top-rank precision matters more than recall.

**Trade-off:** BM25 requires loading the entire corpus into memory on startup (~500 MB per worker for large corpora). This is acceptable for the expected article count at UTC. The corpus is cached in memory and not reloaded per request.

### 4.4 Reciprocal Rank Fusion over Weighted Score Fusion

Two fusion methods are implemented (`reciprocal_rank_fusion` and `weighted_score_fusion`). RRF is the default.

**Rationale:** BM25 scores and cosine similarity scores are on different scales and distributions. Normalizing them to a common range introduces a dependency on score distributions, which can be unstable when the corpus or query set changes. RRF is rank-based only — score magnitudes don't matter, just relative ordering — making it more robust across corpus changes.

### 4.5 HyDE (Hypothetical Document Embeddings)

For vague or natural-language queries, the system can generate a hypothetical document that looks like a TDX knowledge base article answering the question, then embed *that* document and run hybrid search. This exploits the observation that a hypothetical answer often lies in a denser region of embedding space than the question itself.

**Rationale:** UTC helpdesk queries typed by support staff in the Open WebUI interface are often conversational. HyDE bridges the vocabulary gap between question style and knowledge-base article style.

**Trade-off:** Adds one LLM generation call before retrieval (~500-1000ms latency). Only exposed as a separate endpoint; the standard `/search/hybrid` does not use HyDE.

### 4.6 SSE Streaming for the Chat Endpoint

The `/v1/chat/completions` endpoint streams tokens using Server-Sent Events (SSE) in OpenAI wire format.

**Rationale:**
- Open WebUI (the chosen chat UI) expects OpenAI-compatible SSE streaming
- Streaming shows the first token quickly, which matters for a support-staff workflow where perceived latency is part of the user experience
- The OpenAI wire format means any OpenAI-compatible client can use this endpoint without modification

**Trade-off:** SSE streaming requires the endpoint to be `async` and the response to be a `StreamingResponse`. The ChatService is therefore async-native.

### 4.7 Raw SQL via psycopg 3 (No ORM)

All database interactions use raw SQL with psycopg 3, not SQLAlchemy or another ORM.

**Rationale:**
- The data model is stable: a small number of tables with well-understood relationships
- Raw SQL gives precise control over query plans; important for the pgvector KNN query which needs the `<=>` operator and `ORDER BY ... LIMIT` to trigger an index scan
- psycopg 3 is the modern async-capable psycopg with connection pool support via `psycopg_pool`
- Alembic handles migrations with manual SQL (no autogenerate from ORM models)

### 4.8 Strict Separation of `core/` from `api/`

The `core/` directory has no imports from `api/`. All business logic (retrieval, embedding, chat) is implemented in `core/` and injected into the API layer via FastAPI's `Depends()` mechanism and `app.state`.

**Rationale:**
- `core/` can be used by the CLI pipeline (`main.py`) and the API server without coupling
- Testability: `core/` classes can be unit-tested without starting the FastAPI application
- The ChatService receives its dependencies (retrievers, settings, helper functions) via constructor injection, making it straightforward to mock in tests

---

## 5. Technology Choices and Trade-offs

| Technology | Choice | Alternative Considered | Rationale |
|---|---|---|---|
| Vector DB | PostgreSQL + pgvector | Pinecone, Weaviate, Chroma | Single DB engine, JOINs, no extra service |
| Embeddings | OpenAI text-embedding-3-large (3072d) | Ada-002, local models | Best retrieval quality; 3072d exploits Matryoshka representation |
| Reranker | Cohere Rerank v3.5 via LiteLLM | Cross-encoder local model | High quality, managed, no GPU required on server |
| BM25 | rank-bm25 (BM25Okapi) | Elasticsearch, Whoosh | Lightweight in-memory, no extra service |
| Chat/LLM | GPT via LiteLLM proxy | Direct API, Ollama | Provider-agnostic via proxy; Bedrock/Azure switchable |
| Framework | FastAPI + Uvicorn | Flask, Django | Native async, Pydantic validation, OpenAPI docs |
| DB driver | psycopg 3 + psycopg_pool | asyncpg, SQLAlchemy | Modern, async-capable, native pool support |
| Migrations | Alembic (manual SQL) | Flyway, Django migrations | Python-native, integrates with psycopg 3 connection string |
| Container | Docker multi-stage build | Buildpacks | Fine-grained control, minimal runtime image, non-root user |
| Chat UI | Open WebUI | Custom UI | OpenAI-compatible, maintained, feature-rich |

---

## 6. What This System Does Not Do

- Does not replace TDX — articles are read-only from TDX; ticket creation is out of scope
- Does not provide real-time article sync; ingestion is run as a scheduled job
- Does not perform access control at the article level; `is_public` filtering is available but the API itself is protected only by a shared key
- Does not store conversation history across sessions (each chat request is independent at the model level, though query logs are stored for analytics)
