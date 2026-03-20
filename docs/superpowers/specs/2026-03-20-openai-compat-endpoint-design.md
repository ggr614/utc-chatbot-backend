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
| stream: false | Always stream, ignore stream flag | Simplifies router. Open WebUI always sends stream: true. Future frontends can consume SSE too. |
| Sync retrieval | Wrap sync search calls in `asyncio.to_thread()` | Existing BM25/vector/hybrid search are synchronous. `handle_chat` is async. Blocking the event loop would stall streaming. |

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
2. If `!help`: yield help text as a single chunk (router wraps in SSE like any other response), return. No LLM call. Appears as a normal assistant message in Open WebUI.
3. If search (default): call hybrid search via `asyncio.to_thread()` (existing retrievers are synchronous — must not block the event loop)
4. Log query + results to `query_logs` / `query_results` to obtain `query_log_id`
5. Assemble prompt — query sandwich for RAG, plain system prompt for follow-up. Prior conversation turns (assistant/user) are preserved and passed through to LiteLLM. Open WebUI's system message is replaced with the RAG sandwich or no-RAG prompt.
6. Resolve system prompt from `tag_system_prompts` table via search metadata (extract shared prompt resolution logic from search router into a utility)
7. Call `litellm.acompletion(stream=True)` with assembled messages
8. Yield content chunks from the streaming response
9. Accumulate full response text during streaming
10. After stream completes: log LLM response to `llm_responses` using `query_log_id` from step 4. Feature-flagged via `CHAT_ENABLE_CONVERSATION_LOGGING`. Best-effort, fire-and-forget.

**Sync/async boundary:** Steps 3-4 involve synchronous code (BM25Retriever, VectorRetriever, hybrid_search utility, query logging). These are wrapped in `asyncio.to_thread()` to avoid blocking the async event loop. The LiteLLM streaming call (step 7) is natively async.

## Modified Files

### `api/main.py`

- Import and register `openai_compat.router` at prefix `/v1`
- Initialize `ChatService` in lifespan, store in `app.state.chat_service`

### `api/dependencies.py`

- Add `get_chat_service()` dependency that returns `app.state.chat_service`

### `core/config.py`

New `ChatSettings(BaseSettings)` class with `env_prefix="CHAT_"`. This is separate from the existing `LiteLLMSettings` (which uses `LITELLM_` prefix). No namespace collision.

**Generation params** (temperature, max_tokens, model alias) come from the existing `LiteLLMSettings` — `ChatService` reads `LiteLLMSettings.CHAT_MODEL`, `LiteLLMSettings.CHAT_TEMPERATURE`, `LiteLLMSettings.CHAT_COMPLETION_TOKENS`. The new `ChatSettings` only holds chat-endpoint-specific config (retrieval params, logging flag, display model ID, timeouts).

```python
class ChatSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHAT_")

    ENABLE_CONVERSATION_LOGGING: bool = True
    MODEL_ID: str = "utc-helpdesk"          # Display name in /v1/models, not the LiteLLM model alias
    TOP_K: int = 5
    FETCH_TOP_K: int = 20
    RRF_K: int = 1
    MIN_VECTOR_SIMILARITY: float = 0.0
    MAX_CONTEXT_TOKENS: int = 4000
    REQUEST_TIMEOUT: float = 30.0
```

Cached accessor: `get_chat_settings() -> ChatSettings` (same pattern as existing `get_api_settings()`).

### URL Prefix Note

The new router mounts at `/v1/` (what OpenAI clients expect). Existing search endpoints live at `/api/v1/search/`. This split is intentional — `/v1/` is the OpenAI-compatible surface, `/api/v1/` is the internal API.

## Shared Utilities (Extract from Search Router)

System prompt resolution logic (looking up `tag_system_prompts` based on article tags, falling back to defaults) currently lives in the search router. This should be extracted into a shared utility (e.g., `api/utils/prompt_resolution.py`) so both the search router and `ChatService` use the same code path. The search router continues returning `system_prompts` in metadata; `ChatService` calls the utility directly.

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

**Ignored request fields:** `model` (hardcoded), `temperature`, `max_tokens` (backend owns generation params via `LiteLLMSettings`).

**`stream: false` handling:** Always stream regardless of the `stream` field in the request. The router always returns `text/event-stream`. Open WebUI always sends `stream: true`; a future custom frontend should consume SSE as well. This simplifies the router to a single code path.

**Commands:** Parsed from latest user message content. `!f` = follow-up (no RAG). `!help` = help text (no LLM call). Default = hybrid search + RAG. These map to the existing `query_logs.command` CHECK constraint values: default -> `"search"`, `!f` -> `"follow_up"`. `!help` is not logged.

**Conversation history:** All prior assistant/user turns from the request are preserved and passed to LiteLLM. Only the system message is replaced (with RAG sandwich or no-RAG prompt). This ensures follow-up mode (`!f`) has full conversation context.

**Usage field:** The final SSE chunk includes `usage` data (prompt_tokens, completion_tokens, total_tokens) by passing `stream_options={"include_usage": True}` to `litellm.acompletion()`. Open WebUI uses this for token tracking.

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

New `ChatSettings` class with `CHAT_` env prefix (separate from `LiteLLMSettings`):

| Env Var | Field | Default | Purpose |
|---------|-------|---------|---------|
| `CHAT_ENABLE_CONVERSATION_LOGGING` | `ENABLE_CONVERSATION_LOGGING` | `True` | Feature flag for query/response logging |
| `CHAT_MODEL_ID` | `MODEL_ID` | `"utc-helpdesk"` | Display name in /v1/models (not the LiteLLM model alias) |
| `CHAT_TOP_K` | `TOP_K` | `5` | Results returned to LLM after reranking |
| `CHAT_FETCH_TOP_K` | `FETCH_TOP_K` | `20` | Candidates fetched before fusion |
| `CHAT_RRF_K` | `RRF_K` | `1` | RRF fusion constant |
| `CHAT_MIN_VECTOR_SIMILARITY` | `MIN_VECTOR_SIMILARITY` | `0.0` | Minimum vector similarity threshold |
| `CHAT_MAX_CONTEXT_TOKENS` | `MAX_CONTEXT_TOKENS` | `4000` | Max tokens of retrieved context |
| `CHAT_REQUEST_TIMEOUT` | `REQUEST_TIMEOUT` | `30.0` | Timeout for LiteLLM completion call |

**Generation params from `LiteLLMSettings`** (already exist, no new env vars):

| Env Var | Used For |
|---------|----------|
| `LITELLM_CHAT_MODEL` | Actual model alias sent to LiteLLM proxy |
| `LITELLM_CHAT_TEMPERATURE` | Generation temperature |
| `LITELLM_CHAT_COMPLETION_TOKENS` | Max completion tokens |
| `LITELLM_PROXY_BASE_URL` | LiteLLM proxy URL |
| `LITELLM_PROXY_API_KEY` | LiteLLM proxy auth |

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
