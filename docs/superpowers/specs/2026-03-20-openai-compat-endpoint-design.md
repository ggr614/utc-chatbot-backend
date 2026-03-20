# OpenAI-Compatible Chat Completions Endpoint — Design Spec

## Overview

Move LLM calling from the Open WebUI filter (`OpenWebUIFilterV2.py`) into the FastAPI backend by exposing an OpenAI-compatible `/v1/chat/completions` endpoint. The backend becomes the RAG orchestrator: it receives chat requests, performs retrieval, assembles prompts, calls LiteLLM for generation, streams the response, and logs everything.

Open WebUI continues as the frontend but points at the backend instead of LiteLLM directly. The same endpoint serves a future custom frontend with no changes.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | New router in existing FastAPI app (Approach 1) | Reuses shared retrievers, connection pool, logging. No new infrastructure. |
| Frontend | Open WebUI now, custom frontend later | OpenAI compat serves both. |
| Conversation history | Caller sends full history (stateless backend) | Simpler backend. Open WebUI manages history. |
| Conversation logging | Write-only logging, feature-flagged | Log for analytics/eval now, disable when volume is too high in prod. |
| Streaming | SSE from day one | Users already complained about one-shot delivery. |
| RAG trigger | Command-based (`!f`, `!help`, default search) | Same as current filter behavior. Smart auto-detection is YAGNI. |
| OpenAI surface | Minimal — `/v1/models` + `/v1/chat/completions` only | Hardcoded model. No need for full API surface. |
| Model selection | Hardcoded in config, single model | Evaluation pipeline will determine the model later. |
| Transition | Build in parallel, swap with one-line config change | Zero risk to current test bed. |
| Request params | Backend owns generation params (temperature, max_tokens) | Frontend doesn't control knobs — backend config does. |

## New Files

### `api/routers/openai_compat.py`

Thin HTTP adapter. Two endpoints:

- `GET /v1/models` — returns hardcoded model list in OpenAI format
- `POST /v1/chat/completions` — delegates to `ChatService`, streams response as SSE

Responsibilities:
- Parse OpenAI-format request
- Extract latest user message
- Call `ChatService.handle_chat()`
- Convert async generator output to SSE `text/event-stream` response
- Authentication via existing `verify_api_key()` dependency

### `core/chat_service.py`

Framework-agnostic orchestrator. No FastAPI, no HTTP, no SSE knowledge.

Class: `ChatService`

Constructor dependencies (injected):
- `bm25_retriever: BM25Retriever`
- `vector_retriever: VectorRetriever`
- `reranker: Reranker | None`
- `connection_pool` (for query logging)
- `settings` (chat config)

Primary method: `async handle_chat(messages, user_email=None) -> AsyncGenerator[str, None]`

Returns an async generator that yields response text chunks. The router wraps these in SSE framing.

Internal flow:
1. Parse command from latest user message (`!f`, `!help`, or default search)
2. If `!help`: yield help text directly, return
3. If search (default): call hybrid search using shared retrievers
4. Assemble prompt — query sandwich for RAG, plain system prompt for follow-up
5. Resolve system prompt from `tag_system_prompts` table via search metadata
6. Call `litellm.acompletion(stream=True)` with assembled messages
7. Yield content chunks from the streaming response
8. Accumulate full response text during streaming
9. After stream completes: log to `query_logs` / `llm_responses` (best-effort)

## Modified Files

### `api/main.py`

- Import and register `openai_compat.router` at prefix `/v1`
- Initialize `ChatService` in lifespan, store in `app.state.chat_service`

### `api/dependencies.py`

- Add `get_chat_service()` dependency that returns `app.state.chat_service`

### `core/config.py`

New settings class (or section in existing config):

```python
# Chat endpoint settings (CHAT_ prefix)
CHAT_ENABLE_CONVERSATION_LOGGING: bool = True
CHAT_MODEL_ID: str = "utc-helpdesk"
CHAT_TOP_K: int = 5
CHAT_FETCH_TOP_K: int = 20
CHAT_RRF_K: int = 1
CHAT_MIN_VECTOR_SIMILARITY: float = 0.0
CHAT_MAX_CONTEXT_TOKENS: int = 4000
CHAT_REQUEST_TIMEOUT: float = 30.0
```

## Untouched

- All existing search endpoints (`/api/v1/search/*`)
- Health, admin, query_logs routers
- `OpenWebUIFilterV2.py` — stays until cutover, then deleted
- `OpenWebUIFilter.py` — stays until cutover, then deleted

## Request/Response Flow

```
Open WebUI (or future frontend)
  |
  POST /v1/chat/completions  (OpenAI-format, stream: true)
  |
  v
openai_compat router
  |  parse request -> extract messages, model, stream flag
  |  extract latest user message
  |
  v
ChatService.handle_chat()
  |
  +-- 1. Parse command (!f, !help, default search)
  |
  +-- 2. If search: call hybrid search
  |      -> shared BM25 + vector retrievers from app.state
  |      -> returns ranked chunks + query_log_id
  |
  +-- 3. Assemble prompt (query sandwich for RAG, plain for follow-up)
  |      -> resolve system prompt from tag_system_prompts
  |      -> inject context into messages
  |
  +-- 4. Call LiteLLM acompletion(stream=True)
  |      -> returns async generator of chunks
  |
  +-- 5. Yield text chunks (router wraps in SSE OpenAI format)
  |      -> accumulate full response text as chunks stream
  |
  +-- 6. After stream: log to query_logs/llm_responses (best-effort)
         -> feature-flagged via CHAT_ENABLE_CONVERSATION_LOGGING
         -> fire-and-forget, never blocks response
  |
  v
Open WebUI renders streaming tokens
```

## OpenAI Compatibility Details

### GET /v1/models

```json
{
  "object": "list",
  "data": [
    {
      "id": "utc-helpdesk",
      "object": "model",
      "created": 1700000000,
      "owned_by": "utc"
    }
  ]
}
```

`id` is the value from `CHAT_MODEL_ID` config. Open WebUI shows this in its model dropdown.

### POST /v1/chat/completions

**Request** (standard OpenAI format):
```json
{
  "model": "utc-helpdesk",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "How do I reset a password?"}
  ],
  "stream": true
}
```

**Streaming response** (SSE, standard OpenAI format):
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"## QUICK"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Ignored request fields:** `model` (hardcoded), `temperature`, `max_tokens` (backend owns generation params).

**Commands:** Parsed from latest user message content. `!f` = follow-up (no RAG). `!help` = help text (no LLM call). Default = hybrid search + RAG.

## Error Handling

### Retrieval failure (DB down, BM25 empty, vector timeout)

Degrade to follow-up mode — call LLM with no-RAG system prompt, no context. Log the error. User still gets a response.

### LiteLLM streaming failure

```
Try stream=True acompletion
  +-- Success -> relay SSE chunks normally
  +-- Failure -> Try stream=False acompletion (fallback)
                   +-- Success -> wrap full response as SSE (role chunk, content chunk, done)
                   +-- Failure -> OpenAI-format error response
```

Error response format:
```json
{"error": {"message": "...", "type": "server_error", "code": 503}}
```

### Logging failure

Fire-and-forget. Never blocks or fails the response. Log the error server-side.

### Authentication

Same `X-API-Key` header as existing endpoints. Open WebUI sends this via `OPENAI_API_KEYS` config.

## System Prompts

The hardcoded `SYSTEM_PROMPT_RAG` and `SYSTEM_PROMPT_NO_RAG` from `OpenWebUIFilterV2.py` move into `ChatService` as fallbacks. Same content, same behavior.

Runtime resolution: search response metadata includes `system_prompts` dict from `tag_system_prompts` table. Top-ranked article's tag determines the active prompt. Falls back to hardcoded defaults if no tag match.

## Prompt Assembly (Query Sandwich)

Same structure as the current filter:

```
User question: {query}

{system_prompt}

You have access to a knowledge base of IT helpdesk support articles.
Use the following retrieved documents to help answer the user's question:

<knowledge_base>
{formatted_context}
</knowledge_base>

User question: {query}
```

Query appears before and after the context to combat the "lost in the middle" problem.

## Configuration

All new settings use `CHAT_` prefix, loaded from env vars via Pydantic `BaseSettings`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `CHAT_ENABLE_CONVERSATION_LOGGING` | `True` | Feature flag for query/response logging |
| `CHAT_MODEL_ID` | `"utc-helpdesk"` | Model name in /v1/models and request validation |
| `CHAT_TOP_K` | `5` | Results returned to LLM after reranking |
| `CHAT_FETCH_TOP_K` | `20` | Candidates fetched before fusion |
| `CHAT_RRF_K` | `1` | RRF fusion constant |
| `CHAT_MIN_VECTOR_SIMILARITY` | `0.0` | Minimum vector similarity threshold |
| `CHAT_MAX_CONTEXT_TOKENS` | `4000` | Max tokens of retrieved context |
| `CHAT_REQUEST_TIMEOUT` | `30.0` | Timeout for LiteLLM completion call |

## Cutover Plan

Build and test the new endpoint in parallel while the filter continues working.

**When ready to cut over:**

1. Change `docker-compose.yml` (or production equivalent):
   ```yaml
   open-webui:
     environment:
       OPENAI_API_BASE_URLS: http://api:8000/v1    # was http://litellm:4000/v1
       OPENAI_API_KEYS: ${API_API_KEY}              # was ${LITELLM_MASTER_KEY}
   ```

2. Remove the filter from Open WebUI's filter configuration

3. Delete `OpenWebUIFilterV2.py` and `OpenWebUIFilter.py` from the repo

That's it. One config change, one cleanup commit.

## What This Does NOT Include

- Smart follow-up detection (YAGNI — command-based only)
- Full OpenAI API surface (only /v1/models and /v1/chat/completions)
- Frontend temperature/max_tokens passthrough (backend owns generation params)
- Session/conversation storage (caller sends history, backend is stateless)
- Multi-model support (single hardcoded model)
- Changes to existing search, health, admin, or query_log endpoints
