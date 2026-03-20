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
| stream: false | Always stream, ignore stream flag | Simplifies router. Open WebUI always sends stream: true. **Future note:** When a custom frontend needs `stream: false` (e.g., using the standard OpenAI Python SDK which expects JSON for non-streaming), revisit this — add a short wrapper that buffers SSE chunks into a single `ChatCompletion` JSON response. Trivial to add later. |
| Sync retrieval | Wrap sync search calls in `asyncio.to_thread()` | Existing BM25/vector/hybrid search are synchronous. `handle_chat` is async. Blocking the event loop would stall streaming. |
| Search strategy | Hybrid search only (BM25 + vector + reranking) | HyDE is intentionally excluded from v1 — it adds latency and complexity. See "What This Does NOT Include" for future enhancement path. |

## New Files

### `api/routers/openai_compat.py`

Thin HTTP adapter. Two endpoints:

- `GET /v1/models` — returns hardcoded model list in OpenAI format
- `POST /v1/chat/completions` — delegates to `ChatService`, streams response as SSE

Responsibilities:
- Parse OpenAI-format request into `ChatCompletionRequest` Pydantic model
- Generate a unique request ID (`f"chatcmpl-{uuid7()}"`) and `created` timestamp (Unix epoch, generated once per request, reused across all SSE chunks)
- Call `ChatService.handle_chat()`
- Wrap async generator output in OpenAI SSE chunk format (including `model`, `created`, `id` fields in every chunk)
- Authentication via existing `verify_api_key()` dependency

### `api/models/chat.py`

Pydantic request/response models for the OpenAI-compatible endpoint. Follows the project's existing pattern in `api/models/requests.py` and `api/models/responses.py`.

```python
class ChatMessage(BaseModel):
    role: Literal["system", "assistant", "user"]
    content: str | None = None  # None allowed for assistant msgs (Open WebUI edge case)

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = True  # Accepted but ignored — always streams
    # temperature, max_tokens, etc. accepted but ignored (backend owns these)

class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: dict  # {"role": "assistant"} or {"content": "..."} or {}
    finish_reason: str | None = None

class ChatCompletionChunk(BaseModel):
    id: str          # "chatcmpl-{uuid7}"
    object: str = "chat.completion.chunk"
    created: int     # Unix timestamp, same for all chunks in a request
    model: str       # CHAT_MODEL_ID value
    choices: list[ChatCompletionChunkChoice]
    usage: dict | None = None  # Only on final chunk

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "utc"

class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]
```

### `core/chat_service.py`

Framework-agnostic orchestrator. No FastAPI, no HTTP, no SSE knowledge.

Class: `ChatService`

Constructor dependencies (injected):
- `bm25_retriever: BM25Retriever`
- `vector_retriever: VectorRetriever`
- `reranker: Reranker | None`
- `connection_pool` (for query logging)
- `chat_settings: ChatSettings`
- `litellm_settings: LiteLLMSettings`

Primary method: `async handle_chat(messages, user_email=None) -> AsyncGenerator[str, None]`

Returns an async generator that yields response text chunks. The router wraps these in SSE framing.

Internal flow:

1. **Parse command** from latest user message (`!f`, `!help`, or default search)
2. **`!help`**: yield help text as a single chunk (router wraps in SSE like any other response), return. No LLM call. Appears as a normal assistant message in Open WebUI. Not logged.
3. **`!f` (follow-up)**: Skip retrieval. Log query to `query_logs` with `command='follow_up'` and null search results (via `asyncio.to_thread()`). If `!f` is sent with no query text after it, pass the conversation history to LiteLLM with an empty latest user message — the model will use prior turns for context.
4. **Default search**: Call `hybrid_search()` via `asyncio.to_thread()` (existing retrievers are synchronous). This returns ranked results with system prompts already attached (see System Prompt Resolution below).
5. **Log query + results**: Call `QueryLogClient.log_query_with_results()` via `asyncio.to_thread()` to write to `query_logs` and individual rows to `query_results` (with `search_method='hybrid'`). Returns `query_log_id` (int, from BIGSERIAL). Feature-flagged via `CHAT_ENABLE_CONVERSATION_LOGGING`.
6. **Log reranker data**: If reranker was used, log to `reranker_logs` and `reranker_results` via `RerankerLogClient` (same pattern as `search.py:636-666`). Feature-flagged with the same logging flag.
7. **Select system prompt**: Use the system prompt from the highest-ranked result after reranking. If no results or no prompt attached, fall back to hardcoded `SYSTEM_PROMPT_RAG`.
8. **Assemble prompt**: Query sandwich for RAG (see Prompt Assembly), plain `SYSTEM_PROMPT_NO_RAG` for follow-up. Prior conversation turns (assistant/user) are preserved and passed through to LiteLLM. Open WebUI's system message is replaced. If Open WebUI sends no system message, insert one at the beginning of the message list.
9. **Format context**: Format search results into `<knowledge_base>` block (see Context Formatting below). Enforce `MAX_CONTEXT_TOKENS` limit (see Context Truncation below).
10. **Call `litellm.acompletion(stream=True)`** with assembled messages. This is natively async. Full call signature (follows same pattern as `core/hyde_generator.py:150-158`):
    ```python
    response = await litellm.acompletion(
        model=f"openai/{litellm_settings.CHAT_MODEL}",  # e.g. "openai/gpt-5.2-chat"
        api_base=litellm_settings.PROXY_BASE_URL,        # LiteLLM proxy URL
        api_key=litellm_settings.PROXY_API_KEY.get_secret_value(),  # unwrap SecretStr
        messages=assembled_messages,
        max_tokens=litellm_settings.CHAT_COMPLETION_TOKENS,
        temperature=litellm_settings.CHAT_TEMPERATURE,
        stream=True,
        stream_options={"include_usage": True},
        num_retries=3,
        timeout=chat_settings.REQUEST_TIMEOUT,
    )
    ```
    **Critical:** The `"openai/"` model prefix tells LiteLLM SDK to use OpenAI-compatible request formatting when routing through the proxy. Without it, LiteLLM tries to infer the provider from the model name, which fails for custom aliases. The `api_base` redirects the call to the proxy instead of OpenAI directly.
11. **Yield content chunks** from the streaming response. Accumulate full response text.
12. **After stream completes**: Log LLM response to `llm_responses` using `query_log_id` from step 5. Best-effort, fire-and-forget.

**Sync/async boundary:** Steps 4-6 involve synchronous code (BM25Retriever, VectorRetriever, hybrid_search utility, query logging clients). These are wrapped in `asyncio.to_thread()` to avoid blocking the async event loop. The LiteLLM streaming call (step 10) is natively async.

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

## System Prompt Resolution

**How it actually works (two-stage pipeline):**

1. **Stage 1 (retriever level):** `BM25Retriever.search()` calls `PromptStorageClient.get_prompts_for_article_ids()` which batch-fetches prompts from `tag_system_prompts` and attaches them to each `BM25SearchResult.system_prompt` field. This happens inside the retriever, not the router.
2. **Stage 2 (consumer selection):** The search router (or `ChatService`) picks which result's prompt wins.

**For ChatService:** Prompts come for free from the search results. `ChatService` selects the system prompt from the **highest-ranked result after reranking**. If the top result has no `system_prompt` (no tag match), fall back to hardcoded `SYSTEM_PROMPT_RAG`.

**For follow-up mode (`!f`):** No search results, so no dynamic prompt. Use hardcoded `SYSTEM_PROMPT_NO_RAG` directly.

**Shared utility extraction:** The boilerplate for extracting the winning prompt from results (currently copy-pasted across search endpoints) should be extracted into a utility (e.g., `api/utils/prompt_resolution.py`). This is the selection logic only — the DB query stays in the retriever.

**Note:** `PromptStorageClient` creates its own standalone DB connection (inherits from `BaseStorageClient`), not from the connection pool. This is acceptable for the search path where it's called within the retriever. For the no-RAG fallback (follow-up mode), `ChatService` does not need to call `PromptStorageClient` at all — it uses the hardcoded fallback directly.

The hardcoded `SYSTEM_PROMPT_RAG` and `SYSTEM_PROMPT_NO_RAG` from `OpenWebUIFilterV2.py` move into `ChatService` as fallback constants. Same content, same behavior.

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

**Message list construction:** Replace the system message in the caller's message list with the assembled sandwich above. If the caller sends no system message, insert one at position 0. All other messages (prior user/assistant turns) are preserved in order.

## Context Formatting

Each search result is formatted as a numbered document block within `<knowledge_base>`:

```
[Document 1]
Source: https://solutions.teamdynamix.com/...
Content:
{text_content}

---

[Document 2]
Source: https://solutions.teamdynamix.com/...
Content:
{text_content}
```

Each block includes:
- Document number (1-indexed, by rank)
- Source URL — accessed via `result["chunk"].source_url` (the hybrid search result dict structure is `{"rank": int, "combined_score": float, "chunk": TextChunk}`)
- Full `text_content` — accessed via `result["chunk"].text_content`

Documents are separated by `---`. Source URLs are always included (the filter had an `ENABLE_CITATIONS` toggle; this is intentionally dropped — citations are always shown).

## Context Truncation

`MAX_CONTEXT_TOKENS` (default 4000) limits the total size of the `<knowledge_base>` block.

**Strategy: drop lowest-ranked chunks that don't fit.**

1. Iterate through results in rank order (highest rank first)
2. For each result, format the document block
3. Estimate tokens using character count heuristic: `len(doc_entry) / 4` (approximately 4 chars per token, same as the filter's approach)
4. If adding the next document would exceed `MAX_CONTEXT_TOKENS * 4` characters, stop — do not include it or any subsequent documents
5. If zero documents fit (first document exceeds limit), include it anyway truncated to the limit

This is the same logic as the filter's `_format_context()` at lines 372-403. The character heuristic is good enough — exact token counting with `litellm.token_counter()` would add latency for negligible accuracy gain at this stage.

## Command Handling

| Command | Parsed As | RAG Search | LLM Call | Logged | `query_logs.command` |
|---------|-----------|------------|----------|--------|---------------------|
| `<query>` (no prefix) | `"search"` | Yes (hybrid) | Yes | Yes | `"search"` |
| `!f <query>` | `"follow_up"` | No | Yes | Yes (null results) | `"follow_up"` |
| `!f` (no query text) | `"follow_up"` | No | Yes (empty user msg, uses conversation history) | Yes (null results) | `"follow_up"` |
| `!help` | `"help"` | No | No | No | N/A |

Commands are case-insensitive, must be at the start of the latest user message. Parsing logic mirrors `OpenWebUIFilterV2._parse_command()`.

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
data: {"id":"chatcmpl-01970a3e-...","object":"chat.completion.chunk","created":1742486400,"model":"utc-helpdesk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-01970a3e-...","object":"chat.completion.chunk","created":1742486400,"model":"utc-helpdesk","choices":[{"index":0,"delta":{"content":"## QUICK"},"finish_reason":null}]}

data: {"id":"chatcmpl-01970a3e-...","object":"chat.completion.chunk","created":1742486400,"model":"utc-helpdesk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1234,"completion_tokens":567,"total_tokens":1801}}

data: [DONE]
```

**SSE chunk fields:**
- `id`: `f"chatcmpl-{uuid7()}"` — generated once per request, reused across all chunks. Uses UUID7 per project convention.
- `created`: Unix timestamp — generated once per request, reused across all chunks.
- `model`: Value from `CHAT_MODEL_ID` config (display name, not LiteLLM alias).
- `usage`: Included on the final chunk only, via `stream_options={"include_usage": True}`.

**Ignored request fields:** `model` (hardcoded), `temperature`, `max_tokens` (backend owns generation params via `LiteLLMSettings`). These fields are accepted in the request model but not passed through.

**`stream: false` handling:** Always stream regardless of the `stream` field in the request. The router always returns `text/event-stream`. Open WebUI always sends `stream: true`. See Decisions table for future note on custom frontend support.

**Conversation history:** All prior assistant/user turns from the request are preserved and passed to LiteLLM. Only the system message is replaced (with RAG sandwich or no-RAG prompt). If no system message exists in the request, one is inserted at position 0. This ensures follow-up mode (`!f`) has full conversation context.

**Usage field:** The final SSE chunk includes `usage` data (prompt_tokens, completion_tokens, total_tokens) by passing `stream_options={"include_usage": True}` to `litellm.acompletion()`. Open WebUI uses this for token tracking.

## Error Handling

### Retrieval failure (DB down, BM25 empty, vector timeout)

Degrade to follow-up mode — call LLM with no-RAG system prompt (`SYSTEM_PROMPT_NO_RAG`), no context. Log the error. User still gets a response.

### LiteLLM streaming failure

```
Try stream=True acompletion
  +-- Success -> relay SSE chunks normally
  +-- Failure -> Try stream=False acompletion (fallback)
                   +-- Success -> wrap full response as SSE (role chunk, content chunk, done)
                   +-- Failure -> OpenAI-format error response
```

**Error delivery format:** Since the endpoint always returns `text/event-stream`, errors are delivered as SSE events (following OpenAI convention for streaming errors):
```
data: {"error": {"message": "LLM service unavailable", "type": "server_error", "code": 503}}

data: [DONE]
```
The HTTP status remains 200 (SSE connection was already established). The error object inside the SSE event signals failure to the client. Open WebUI handles this gracefully.

### Logging failure

Fire-and-forget. Never blocks or fails the response. Log the error server-side.

### Graceful shutdown

In-flight streaming responses may be active when the application shuts down. Rely on uvicorn's graceful shutdown timeout (`--timeout-graceful-shutdown`, default 0) to drain active connections. If the connection pool closes mid-logging, accept that best-effort logging may fail — this is acceptable since logging is already fire-and-forget.

### Authentication

Same `X-API-Key` header as existing endpoints. Open WebUI sends this via `OPENAI_API_KEYS` config.

## Logging Detail

When `CHAT_ENABLE_CONVERSATION_LOGGING=False`, steps 3, 5, 6, and 12 are all skipped — no database writes occur for any chat request. The LLM call and response streaming still work normally.

### What gets logged (when `CHAT_ENABLE_CONVERSATION_LOGGING=True`)

| Table | When | Data |
|-------|------|------|
| `query_logs` | After search (step 5) or for `!f` (step 3) | raw_query, query_embedding (null for !f), email, command, latency_ms |
| `query_results` | After search (step 5) | One row per result: search_method='hybrid', rank, score, chunk_id, parent_article_id |
| `reranker_logs` | After reranking (step 6) | reranker_status, model_name, reranker_latency_ms, avg_rank_change |
| `reranker_results` | After reranking (step 6) | One row per result: rrf_rank, rrf_score, reranked_rank, reranked_score, rank_change |
| `llm_responses` | After stream completes (step 12) | response_text, model_name, llm_latency_ms, token counts, citations JSONB |

### What does NOT get logged

- `!help` commands — no DB writes at all
- Conversation history (prior turns) — not stored, stateless

### Dependencies for logging

`ChatService` needs access to:
- `QueryLogClient` (existing, uses connection pool)
- `RerankerLogClient` (existing, uses connection pool) — only if reranker is available

These are created per-request using the connection pool, same pattern as the search router.

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
| `CHAT_REQUEST_TIMEOUT` | `REQUEST_TIMEOUT` | `30.0` | Timeout for LiteLLM completion call (passed as `timeout` kwarg to `litellm.acompletion()`) |

**Generation params from `LiteLLMSettings`** (already exist, no new env vars):

| Env Var | Used For |
|---------|----------|
| `LITELLM_CHAT_MODEL` | Actual model alias sent to LiteLLM proxy |
| `LITELLM_CHAT_TEMPERATURE` | Generation temperature |
| `LITELLM_CHAT_COMPLETION_TOKENS` | Max completion tokens |
| `LITELLM_PROXY_BASE_URL` | LiteLLM proxy URL |
| `LITELLM_PROXY_API_KEY` | LiteLLM proxy auth |

### Connection Pool Sizing Note

`ChatService` adds a new consumer of the connection pool (query logging, reranker logging). The current pool is min=5, max=20 per worker. With 4 uvicorn workers, that's 80 max connections. Chat requests are longer-lived than search requests (streaming holds a worker for seconds, not milliseconds). Monitor pool utilization after deployment. If contention appears, increase `API_POOL_MAX_SIZE` or reduce workers.

## Operational Notes

**Note on reranker concurrency:** `ChatService` receives a shared `Reranker` instance from `app.state`. The `Reranker.last_rerank_latency_ms` property is instance state that concurrent requests could overwrite. This is a pre-existing concern (same shared instance in the search router). Acceptable at current scale; if it matters later, capture latency as a return value rather than instance state.

**Note on `CHAT_REQUEST_TIMEOUT`:** This value is passed as the `timeout` kwarg to `litellm.acompletion()`. LiteLLM raises its own timeout exception type, which the streaming fallback catches. No outer `asyncio.wait_for()` wrapper.

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
  |  parse into ChatCompletionRequest
  |  generate chatcmpl-{uuid7} ID + created timestamp
  |  extract latest user message
  |
  v
ChatService.handle_chat()
  |
  +-- 1. Parse command (!f, !help, default search)
  |
  +-- 2. If !help: yield help text, return (no logging)
  |
  +-- 3. If !f: log query (command='follow_up', null results)
  |      -> skip to step 8
  |
  +-- 4. If search: call hybrid_search() via asyncio.to_thread()
  |      -> shared BM25 + vector retrievers
  |      -> results have system_prompt attached from retriever
  |
  +-- 5. Log query + results to query_logs/query_results
  |      -> search_method='hybrid'
  |      -> returns query_log_id
  |
  +-- 6. Log reranker data to reranker_logs/reranker_results
  |
  +-- 7. Select system prompt (top-ranked result, fallback to hardcoded)
  |
  +-- 8. Assemble prompt (sandwich for RAG, SYSTEM_PROMPT_NO_RAG for !f)
  |      -> format context with truncation
  |      -> replace/insert system message, preserve conversation history
  |
  +-- 9. Call LiteLLM acompletion(stream=True)
  |      -> with stream_options={"include_usage": True}
  |
  +-- 10. Yield text chunks (router wraps in SSE with id/created/model)
  |       -> accumulate full response text
  |
  +-- 11. After stream: log LLM response to llm_responses
          -> feature-flagged, best-effort
  |
  v
Open WebUI renders streaming tokens
```

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

## Test Plan

### New test files

**`tests/test_chat_service.py`** — Unit tests for `ChatService`:
- Command parsing: default search, `!f`, `!f` with no query, `!help`, case insensitivity
- Context formatting: numbered blocks, source URLs, separator
- Context truncation: fits within limit, drops lowest-ranked, single oversized document
- Prompt assembly: query sandwich structure, system message replacement, system message insertion when missing, conversation history preservation
- System prompt selection: top-ranked result wins, fallback to hardcoded when no prompt attached, fallback when no results
- Follow-up mode: no search called, no-RAG prompt used, conversation history passed through
- Streaming: yields chunks from LiteLLM, accumulates full text
- Streaming fallback: falls back to non-streaming on stream failure
- Logging: query_log written, llm_response written after stream, logging skipped when flag is False

**`tests/test_openai_compat.py`** — Integration tests for the router:
- `GET /v1/models`: returns correct format, model ID from config
- `POST /v1/chat/completions`: SSE format (id, created, model fields present), `data: [DONE]` sentinel
- Auth: 401 without API key, 401 with invalid key
- Error response: SSE error event format on LLM failure
- Edge cases: empty messages list, missing system message

### Mocking strategy

- Mock `litellm.acompletion` — return async generator of fake chunks (same pattern as `tests/test_embedding.py` mocks `litellm.aembedding`)
- Mock `BM25Retriever.search()` and `VectorRetriever.search()` — return canned results with `system_prompt` attached
- Mock `Reranker.rerank()` — return reordered results
- Mock connection pool — use `MagicMock` for `QueryLogClient` and `RerankerLogClient`
- Use `httpx.AsyncClient` with `app` for router integration tests (FastAPI TestClient)

## What This Does NOT Include

- Smart follow-up detection (YAGNI — command-based only)
- Full OpenAI API surface (only /v1/models and /v1/chat/completions)
- Frontend temperature/max_tokens passthrough (backend owns generation params)
- Session/conversation storage (caller sends history, backend is stateless)
- Multi-model support (single hardcoded model)
- Changes to existing search, health, admin, or query_log endpoints
- HyDE search strategy — intentionally excluded from v1. HyDE adds an extra LLM call per query (~1-2s latency) for hypothetical document generation. The `HyDEGenerator` is already initialized in `app.state` and can be added as a `ChatService` constructor dependency in a future iteration. When added, expose as a new command (e.g., `!h`) or make it the default, replacing hybrid.
- `stream: false` JSON response mode — see Decisions table for future note
