# OpenAI-Compatible Chat Completions Endpoint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints that perform RAG retrieval, prompt assembly, and LLM streaming — replacing the Open WebUI filter.

**Architecture:** New FastAPI router (`openai_compat.py`) delegates to a framework-agnostic `ChatService` orchestrator. ChatService calls existing hybrid search (sync, via `asyncio.to_thread()`), assembles query-sandwich prompts, streams LLM responses via `litellm.acompletion(stream=True)`, and logs everything to existing tables. All existing endpoints remain untouched.

**Tech Stack:** FastAPI, LiteLLM, Pydantic v2, SSE (Server-Sent Events), PostgreSQL (pgvector), pytest + pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-03-20-openai-compat-endpoint-design.md`

**Prerequisites:**
- Ensure `pytest-asyncio` is installed: `pip install pytest-asyncio` (used for `@pytest.mark.asyncio` in tests)
- Ensure the `search` and `follow_up` values exist in the `query_logs.command` CHECK constraint (migration `a1b2c3d4e5f6` — verify with `alembic current`)
- After completing all tasks, add `CHAT_*` env vars to `example.env` (see Task 1 for the full list of defaults)

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `core/config.py` (modify) | Add `ChatSettings` class + `get_chat_settings()` accessor |
| Create | `api/models/chat.py` | Pydantic request/response models for OpenAI compat |
| Create | `api/utils/prompt_resolution.py` | Shared system prompt selection from search results |
| Create | `core/chat_service.py` | RAG orchestrator: command parsing, context formatting, prompt assembly, LLM streaming, logging. Receives `hybrid_search` and `select_system_prompt` as constructor callables (avoids `core/` importing from `api/`). |
| Create | `api/routers/openai_compat.py` | Thin HTTP adapter: SSE framing, `/v1/models`, `/v1/chat/completions` |
| Modify | `api/dependencies.py` | Add `get_chat_service()` dependency |
| Modify | `api/main.py` | Register router, initialize `ChatService` in lifespan |
| Create | `tests/test_chat_service.py` | Unit tests for ChatService |
| Create | `tests/test_openai_compat.py` | Integration tests for router |

---

### Task 1: ChatSettings Configuration

**Files:**
- Modify: `core/config.py:64-110` (add after `APISettings`, before accessors)
- Test: `tests/test_config_chat.py`

- [ ] **Step 1: Write failing test for ChatSettings defaults**

```python
# tests/test_config_chat.py
"""Tests for ChatSettings configuration."""
import pytest
from unittest.mock import patch


def test_chat_settings_defaults():
    """ChatSettings should have correct defaults without any env vars."""
    with patch.dict("os.environ", {}, clear=True):
        from core.config import ChatSettings

        settings = ChatSettings()
        assert settings.ENABLE_CONVERSATION_LOGGING is True
        assert settings.MODEL_ID == "utc-helpdesk"
        assert settings.TOP_K == 5
        assert settings.FETCH_TOP_K == 20
        assert settings.RRF_K == 1
        assert settings.MIN_VECTOR_SIMILARITY == 0.0
        assert settings.MAX_CONTEXT_TOKENS == 4000
        assert settings.REQUEST_TIMEOUT == 30.0


def test_chat_settings_from_env():
    """ChatSettings should read from CHAT_ prefixed env vars."""
    env = {
        "CHAT_ENABLE_CONVERSATION_LOGGING": "false",
        "CHAT_MODEL_ID": "test-model",
        "CHAT_TOP_K": "10",
        "CHAT_REQUEST_TIMEOUT": "60.0",
    }
    with patch.dict("os.environ", env, clear=True):
        from core.config import ChatSettings

        settings = ChatSettings()
        assert settings.ENABLE_CONVERSATION_LOGGING is False
        assert settings.MODEL_ID == "test-model"
        assert settings.TOP_K == 10
        assert settings.REQUEST_TIMEOUT == 60.0


def test_get_chat_settings_returns_instance():
    """get_chat_settings() should return a ChatSettings instance."""
    from core.config import get_chat_settings

    settings = get_chat_settings()
    assert isinstance(settings, ChatSettings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config_chat.py -v`
Expected: FAIL with `ImportError: cannot import name 'ChatSettings'`

- [ ] **Step 3: Implement ChatSettings and get_chat_settings()**

Add to `core/config.py` after the `APISettings` class (around line 83) and before the existing accessor functions (around line 86):

```python
class ChatSettings(BaseSettings):
    """Settings for the /v1/chat/completions endpoint."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        env_prefix="CHAT_",
    )

    ENABLE_CONVERSATION_LOGGING: bool = True
    MODEL_ID: str = "utc-helpdesk"
    TOP_K: int = 5
    FETCH_TOP_K: int = 20
    RRF_K: int = 1
    MIN_VECTOR_SIMILARITY: float = 0.0
    MAX_CONTEXT_TOKENS: int = 4000
    REQUEST_TIMEOUT: float = 30.0
```

Add the cached accessor alongside the existing ones (around line 110):

```python
@lru_cache()
def get_chat_settings() -> ChatSettings:
    return ChatSettings()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config_chat.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/config.py tests/test_config_chat.py
git commit -m "feat: add ChatSettings config for /v1/chat/completions endpoint"
```

---

### Task 2: Pydantic Request/Response Models

**Files:**
- Create: `api/models/chat.py`
- Test: `tests/test_chat_models.py`

- [ ] **Step 1: Write failing tests for Pydantic models**

```python
# tests/test_chat_models.py
"""Tests for OpenAI-compatible Pydantic models."""
import pytest
from api.models.chat import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ModelObject,
    ModelListResponse,
)


def test_chat_message_with_content():
    msg = ChatMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_chat_message_none_content():
    """Assistant messages may have None content (Open WebUI edge case)."""
    msg = ChatMessage(role="assistant", content=None)
    assert msg.content is None


def test_chat_message_default_content():
    """Content defaults to None when omitted."""
    msg = ChatMessage(role="assistant")
    assert msg.content is None


def test_chat_completion_request_minimal():
    req = ChatCompletionRequest(
        model="utc-helpdesk",
        messages=[ChatMessage(role="user", content="test")],
    )
    assert req.model == "utc-helpdesk"
    assert req.stream is True  # default
    assert len(req.messages) == 1


def test_chat_completion_request_stream_false_accepted():
    """stream=false is accepted (but ignored by router)."""
    req = ChatCompletionRequest(
        model="utc-helpdesk",
        messages=[ChatMessage(role="user", content="test")],
        stream=False,
    )
    assert req.stream is False


def test_chat_completion_chunk_serialization():
    chunk = ChatCompletionChunk(
        id="chatcmpl-abc123",
        created=1700000000,
        model="utc-helpdesk",
        choices=[
            ChatCompletionChunkChoice(
                delta={"content": "hello"}, finish_reason=None
            )
        ],
    )
    data = chunk.model_dump()
    assert data["id"] == "chatcmpl-abc123"
    assert data["object"] == "chat.completion.chunk"
    assert data["model"] == "utc-helpdesk"
    assert data["choices"][0]["delta"] == {"content": "hello"}
    assert data["usage"] is None


def test_chat_completion_chunk_with_usage():
    chunk = ChatCompletionChunk(
        id="chatcmpl-abc123",
        created=1700000000,
        model="utc-helpdesk",
        choices=[
            ChatCompletionChunkChoice(delta={}, finish_reason="stop")
        ],
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )
    assert chunk.usage["total_tokens"] == 150
    assert chunk.choices[0].finish_reason == "stop"


def test_model_list_response():
    resp = ModelListResponse(
        data=[ModelObject(id="utc-helpdesk", created=1700000000)]
    )
    data = resp.model_dump()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "utc-helpdesk"
    assert data["data"][0]["owned_by"] == "utc"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chat_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.models.chat'`

- [ ] **Step 3: Implement api/models/chat.py**

```python
# api/models/chat.py
"""Pydantic models for the OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "assistant", "user"]
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = True
    temperature: float | None = None
    max_tokens: int | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: dict
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: dict | None = None


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "utc"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chat_models.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add api/models/chat.py tests/test_chat_models.py
git commit -m "feat: add Pydantic models for OpenAI-compatible chat endpoint"
```

---

### Task 3: Command Parsing

**Files:**
- Create: `core/chat_service.py` (partial — command parsing only)
- Test: `tests/test_chat_service.py` (partial — command parsing tests)

- [ ] **Step 1: Write failing tests for command parsing**

```python
# tests/test_chat_service.py
"""Tests for ChatService."""
import pytest
from core.chat_service import ChatService


class TestCommandParsing:
    """Tests for ChatService._parse_command()."""

    def test_default_search(self):
        command, query = ChatService._parse_command("How do I reset a password?")
        assert command is None
        assert query == "How do I reset a password?"

    def test_follow_up_with_query(self):
        command, query = ChatService._parse_command("!f Can you explain step 3?")
        assert command == "follow_up"
        assert query == "Can you explain step 3?"

    def test_follow_up_no_query(self):
        command, query = ChatService._parse_command("!f")
        assert command == "follow_up"
        assert query == ""  # Empty query — LLM uses conversation history

    def test_follow_up_case_insensitive(self):
        command, query = ChatService._parse_command("!F explain more")
        assert command == "follow_up"
        assert query == "explain more"

    def test_help_command(self):
        command, query = ChatService._parse_command("!help")
        assert command == "help"
        assert query == ""

    def test_help_with_trailing_text(self):
        command, query = ChatService._parse_command("!help me")
        assert command == "help"
        assert query == ""

    def test_help_case_insensitive(self):
        command, query = ChatService._parse_command("!HELP")
        assert command == "help"
        assert query == ""

    def test_empty_query(self):
        command, query = ChatService._parse_command("")
        assert command is None
        assert query == ""

    def test_none_query(self):
        command, query = ChatService._parse_command(None)
        assert command is None
        assert query is None

    def test_whitespace_stripped(self):
        command, query = ChatService._parse_command("  How do I reset?  ")
        assert command is None
        assert query == "How do I reset?"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chat_service.py::TestCommandParsing -v`
Expected: FAIL with `ImportError: cannot import name 'ChatService'`

- [ ] **Step 3: Implement command parsing in ChatService**

```python
# core/chat_service.py
"""
ChatService — framework-agnostic RAG orchestrator.

Handles command parsing, hybrid search, prompt assembly,
LLM streaming, and logging for the /v1/chat/completions endpoint.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple


class ChatService:
    """Orchestrates RAG chat: search -> prompt assembly -> LLM -> logging."""

    def __init__(
        self,
        bm25_retriever,
        vector_retriever,
        reranker,
        connection_pool,
        chat_settings,
        litellm_settings,
        hybrid_search_fn: Callable | None = None,
        select_system_prompt_fn: Callable | None = None,
    ):
        self._bm25 = bm25_retriever
        self._vector = vector_retriever
        self._reranker = reranker
        self._pool = connection_pool
        self._chat_settings = chat_settings
        self._litellm_settings = litellm_settings
        self._hybrid_search = hybrid_search_fn
        self._select_system_prompt = select_system_prompt_fn

    @staticmethod
    def _parse_command(query: str | None) -> Tuple[Optional[str], str | None]:
        """Parse command prefix from user query.

        Returns (command, cleaned_query):
        - (None, query) for default search
        - ("follow_up", cleaned) for !f
        - ("help", "") for !help
        """
        if not query or not isinstance(query, str):
            return (None, query)

        stripped = query.strip()
        lower = stripped.lower()

        if lower == "!help" or lower.startswith("!help "):
            return ("help", "")

        if lower == "!f" or lower.startswith("!f "):
            cleaned = stripped[len("!f"):].strip()
            return ("follow_up", cleaned)

        return (None, stripped)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chat_service.py::TestCommandParsing -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/chat_service.py tests/test_chat_service.py
git commit -m "feat: add ChatService with command parsing (!f, !help, default)"
```

---

### Task 4: Context Formatting and Truncation

**Files:**
- Modify: `core/chat_service.py` (add `_format_context()`)
- Test: `tests/test_chat_service.py` (add context formatting tests)

- [ ] **Step 1: Write failing tests for context formatting and truncation**

Append to `tests/test_chat_service.py`:

```python
from unittest.mock import MagicMock
from core.schemas import TextChunk


def _make_result(rank: int, text: str, url: str = "https://example.com/doc") -> dict:
    """Helper to create a hybrid search result dict."""
    chunk = MagicMock(spec=TextChunk)
    chunk.text_content = text
    chunk.source_url = url
    return {"rank": rank, "combined_score": 1.0 / rank, "chunk": chunk}


class TestContextFormatting:
    """Tests for ChatService._format_context()."""

    def test_single_result(self):
        results = [_make_result(1, "Password reset steps here.")]
        context = ChatService._format_context(results, max_context_tokens=4000)
        assert "[Document 1]" in context
        assert "Source: https://example.com/doc" in context
        assert "Password reset steps here." in context

    def test_multiple_results_separated(self):
        results = [
            _make_result(1, "First doc."),
            _make_result(2, "Second doc."),
        ]
        context = ChatService._format_context(results, max_context_tokens=4000)
        assert "[Document 1]" in context
        assert "[Document 2]" in context
        assert "---" in context

    def test_empty_results(self):
        context = ChatService._format_context([], max_context_tokens=4000)
        assert context == ""

    def test_truncation_drops_lowest_ranked(self):
        # Each doc is ~100 chars. With max_context_tokens=30 (~120 chars),
        # only 1 doc should fit.
        results = [
            _make_result(1, "A" * 80),
            _make_result(2, "B" * 80),
        ]
        context = ChatService._format_context(results, max_context_tokens=30)
        assert "[Document 1]" in context
        assert "[Document 2]" not in context

    def test_oversized_single_doc_included_truncated(self):
        # Single doc exceeds limit — still included but truncated.
        results = [_make_result(1, "X" * 20000)]
        context = ChatService._format_context(results, max_context_tokens=100)
        assert "[Document 1]" in context
        assert len(context) <= 100 * 4 + 200  # some overhead for headers
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chat_service.py::TestContextFormatting -v`
Expected: FAIL with `AttributeError: type object 'ChatService' has no attribute '_format_context'`

- [ ] **Step 3: Implement _format_context()**

Add to `core/chat_service.py` in the `ChatService` class:

```python
    @staticmethod
    def _format_context(
        results: list[dict], max_context_tokens: int = 4000
    ) -> str:
        """Format search results into a context string for the knowledge base block.

        Iterates results in rank order. Drops lowest-ranked chunks that exceed
        the token limit (estimated at ~4 chars per token).
        """
        if not results:
            return ""

        max_chars = max_context_tokens * 4
        context_parts: list[str] = []
        total_chars = 0

        for i, result in enumerate(results, 1):
            chunk = result["chunk"]
            text_content = chunk.text_content
            source_url = str(chunk.source_url) if chunk.source_url else ""

            doc_lines = [f"[Document {i}]"]
            if source_url:
                doc_lines.append(f"Source: {source_url}")
            doc_lines.append(f"Content:\n{text_content}")
            doc_entry = "\n".join(doc_lines)

            if total_chars + len(doc_entry) > max_chars:
                if not context_parts:
                    # First doc exceeds limit — include truncated
                    context_parts.append(doc_entry[: max_chars])
                break

            context_parts.append(doc_entry)
            total_chars += len(doc_entry)

        return "\n\n---\n\n".join(context_parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chat_service.py::TestContextFormatting -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/chat_service.py tests/test_chat_service.py
git commit -m "feat: add context formatting with token-based truncation"
```

---

### Task 5: Prompt Assembly (Query Sandwich)

**Files:**
- Modify: `core/chat_service.py` (add system prompts, `_assemble_messages()`)
- Test: `tests/test_chat_service.py` (add prompt assembly tests)

- [ ] **Step 1: Write failing tests for prompt assembly**

Append to `tests/test_chat_service.py`:

```python
class TestPromptAssembly:
    """Tests for ChatService._assemble_messages()."""

    def test_rag_sandwich_replaces_system_message(self):
        messages = [
            {"role": "system", "content": "Original system prompt"},
            {"role": "user", "content": "How do I reset a password?"},
        ]
        result = ChatService._assemble_rag_messages(
            messages=messages,
            query="How do I reset a password?",
            context="[Document 1]\nContent:\nPassword steps...",
            system_prompt=None,  # Use default
        )
        # System message should be replaced with sandwich
        assert result[0]["role"] == "system"
        assert "User question: How do I reset a password?" in result[0]["content"]
        assert "<knowledge_base>" in result[0]["content"]
        assert "Password steps..." in result[0]["content"]
        # User message preserved
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "How do I reset a password?"

    def test_rag_sandwich_inserts_system_when_missing(self):
        messages = [
            {"role": "user", "content": "VPN issues"},
        ]
        result = ChatService._assemble_rag_messages(
            messages=messages,
            query="VPN issues",
            context="[Document 1]\nContent:\nVPN docs...",
        )
        assert result[0]["role"] == "system"
        assert "<knowledge_base>" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_rag_sandwich_preserves_conversation_history(self):
        messages = [
            {"role": "system", "content": "old"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "follow up"},
        ]
        result = ChatService._assemble_rag_messages(
            messages=messages,
            query="follow up",
            context="docs...",
        )
        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "first question"
        assert result[2]["content"] == "first answer"
        assert result[3]["content"] == "follow up"

    def test_rag_sandwich_uses_custom_system_prompt(self):
        messages = [
            {"role": "system", "content": "old"},
            {"role": "user", "content": "test"},
        ]
        result = ChatService._assemble_rag_messages(
            messages=messages,
            query="test",
            context="docs...",
            system_prompt="Custom prompt for this category",
        )
        assert "Custom prompt for this category" in result[0]["content"]

    def test_no_rag_messages_replaces_system(self):
        messages = [
            {"role": "system", "content": "old"},
            {"role": "user", "content": "general question"},
        ]
        result = ChatService._assemble_no_rag_messages(messages)
        assert result[0]["role"] == "system"
        assert "UTC IT Helpdesk Assistant" in result[0]["content"]
        assert "documentation-bound" not in result[0]["content"].lower()

    def test_no_rag_messages_inserts_system_when_missing(self):
        messages = [
            {"role": "user", "content": "general question"},
        ]
        result = ChatService._assemble_no_rag_messages(messages)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chat_service.py::TestPromptAssembly -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement system prompt constants and assembly methods**

Add to `core/chat_service.py`. First, add the system prompt constants at module level (copy from `OpenWebUIFilterV2.py:24-57` and `OpenWebUIFilterV2.py:59-133` respectively):

```python
SYSTEM_PROMPT_NO_RAG = """# UTC IT Helpdesk Assistant - System Prompt

You are the UTC IT Helpdesk Assistant for Tier 1 support staff at the University of Tennessee at Chattanooga (UTC).

## YOUR AUDIENCE

- **Tier 1 support staff**: May have limited IT exposure. Explain jargon inline when used, e.g., "AD (Active Directory)".
- **Always remote**: Staff cannot physically touch equipment.
- **Frame for IT staff**: You're speaking to the employee, not the end user. Never say "you" meaning the customer.

## YOUR ROLE

Provide helpful, concise answers to questions. This could be:
- General IT knowledge questions
- Clarification or follow-up questions about previous responses
- Troubleshooting guidance
- Explanations of concepts or procedures

Keep responses clear and actionable. When uncertain about UTC-specific procedures, acknowledge that and suggest verifying with documentation or escalating.

## CORE GUIDELINES

1. **Be concise**: Staff may be on calls. Get to the point quickly.
2. **Actionable first**: Lead with what to do, explain why if needed.
3. **Explain jargon**: Define technical terms inline on first use, e.g., "MocsID (UTC's single sign-on username)".
4. **Remote-only**: Never suggest physically handling equipment. Physical access = escalation.
5. **Acknowledge uncertainty**: When unsure about UTC-specific procedures, say so rather than guessing.

## UTC CONTEXT

- **UTC** = University of Tennessee at Chattanooga
- **Mocs** = Mockingbirds (mascot), used in system names (MocsNet, MocsID, MocsMail)
- Common systems: Banner, Canvas, TeamDynamix (TDX), Active Directory (AD), Microsoft 365
- Customers = students, faculty, staff"""

SYSTEM_PROMPT_RAG = """# UTC IT Helpdesk Assistant - System Prompt

You are the UTC IT Helpdesk Assistant, a knowledge assistant for Tier 1 support staff at the University of Tennessee at Chattanooga (UTC). Our mascot is the Mockingbird, so many systems use "Mocs" branding (MocsNet, MocsID, etc.).

## YOUR AUDIENCE

- **Tier 1 support staff**: Beginners with limited IT exposure. Explain jargon inline, e.g., "AD (Active Directory)".
- **Always remote**: Staff cannot physically touch equipment. They may have remote desktop/session access only.
- **Often live with customers**: Students, faculty, or staff may be on the line. Speed matters—actionable info first, elaboration second.
- **Perspective varies**: Queries may be written in 1st, 2nd, or 3rd person depending on source and preference. Interpret accordingly.

## YOUR GOAL

Your primary goal is NOT necessarily to solve the problem, but to **set the employee up for success**:
- If the answer is documented → provide the steps
- If not documented → provide a clear checklist of information to gather and which team to escalate to

## RESPONSE FORMAT

Use this exact structure. Include only the sections that apply.

### Template:

## QUICK ANSWER
[1-2 sentences maximum. State whether this can be resolved with documented steps OR needs escalation. Give the single most important action or summary.]

## STEPS
[Numbered list. Only include if the documentation provides a clear procedure. Keep steps concise.]

## IF UNRESOLVED
**Gather:**
- [Specific information the employee should collect]
- [Error messages, screenshots, account details, etc.]

**Escalate to:** [Appropriate team]

## SOURCES
- [URL from retrieved documentation]

### Rules:
- **QUICK ANSWER**: Always include. Maximum 2 sentences.
- **STEPS**: Only include if documentation provides explicit procedure. Do not invent steps.
- **IF UNRESOLVED**: Include when documentation doesn't fully address the issue, OR when additional info is needed regardless.
- **SOURCES**: Always include. List every URL from the retrieved documentation you referenced.

## CORE RULES

1. **Documentation-bound**: Only provide solutions explicitly found in the retrieved documentation. Never invent troubleshooting steps.
2. **Actionable first**: Lead with what to do, explain why later (if at all).
3. **Concise**: Staff are reading while on calls. Every word must earn its place.
4. **Jargon with training wheels**: Use proper IT terminology but define it inline on first use, e.g., "Check their MocsID (UTC's single sign-on username)".
5. **Frame for IT staff**: You are speaking to the employee, not the end user. Never say "you" meaning the customer.
6. **Remote-only solutions**: Never suggest physically handling equipment. If physical access is required, that's an escalation.
7. **Source everything**: Always cite the documentation URLs at the bottom.

## EDGE CASES

**Vague query + scattered context**: If the query is unclear AND the retrieved documentation covers multiple unrelated topics, do your best to answer, then provide a better query suggestion.

**Partially documented**: If documentation covers part of the issue, state what IS documented vs. what requires escalation.

**Not in documentation**: Do not guess. Provide the information-gathering checklist and escalation path.

## UTC CONTEXT

- **UTC** = University of Tennessee at Chattanooga
- **Mocs** = Mockingbirds (mascot), used in system names (MocsNet, MocsID, MocsMail, etc.)
- Common systems: Banner, Canvas, TeamDynamix (TDX), Genetec (access control), Active Directory (AD)
- Customers = students, faculty, staff (never "users" in responses)"""
```

Then add these methods to the `ChatService` class:

```python
    @staticmethod
    def _assemble_rag_messages(
        messages: list[dict],
        query: str,
        context: str,
        system_prompt: str | None = None,
    ) -> list[dict]:
        """Build query-sandwich message list for RAG mode.

        Replaces (or inserts) the system message with the sandwich structure.
        Preserves all other messages (conversation history).
        """
        base_prompt = system_prompt if system_prompt else SYSTEM_PROMPT_RAG

        sandwich = (
            f"User question: {query}\n\n"
            f"{base_prompt}\n\n"
            "You have access to a knowledge base of IT helpdesk support articles.\n"
            "Use the following retrieved documents to help answer the user's question:\n\n"
            f"<knowledge_base>\n{context}\n</knowledge_base>\n\n"
            f"User question: {query}"
        )

        new_messages = []
        system_found = False

        for msg in messages:
            if msg.get("role") == "system":
                new_messages.append({"role": "system", "content": sandwich})
                system_found = True
            else:
                new_messages.append(msg.copy() if isinstance(msg, dict) else msg)

        if not system_found:
            new_messages.insert(0, {"role": "system", "content": sandwich})

        return new_messages

    @staticmethod
    def _assemble_no_rag_messages(messages: list[dict]) -> list[dict]:
        """Build message list for follow-up mode (no RAG context)."""
        new_messages = []
        system_found = False

        for msg in messages:
            if msg.get("role") == "system":
                new_messages.append(
                    {"role": "system", "content": SYSTEM_PROMPT_NO_RAG}
                )
                system_found = True
            else:
                new_messages.append(msg.copy() if isinstance(msg, dict) else msg)

        if not system_found:
            new_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT_NO_RAG})

        return new_messages
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chat_service.py::TestPromptAssembly -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/chat_service.py tests/test_chat_service.py
git commit -m "feat: add query-sandwich prompt assembly and system prompt constants"
```

---

### Task 6: System Prompt Selection Utility

**Files:**
- Create: `api/utils/prompt_resolution.py`
- Test: `tests/test_prompt_resolution.py`

- [ ] **Step 1: Write failing tests for prompt selection**

```python
# tests/test_prompt_resolution.py
"""Tests for system prompt selection from search results."""
import pytest
from unittest.mock import MagicMock
from api.utils.prompt_resolution import select_system_prompt


def _make_result_with_prompt(rank: int, prompt: str | None) -> dict:
    """Helper to create a search result dict with a system_prompt on the result dict."""
    chunk = MagicMock()
    result = {"rank": rank, "combined_score": 1.0, "chunk": chunk}
    if prompt is not None:
        result["system_prompt"] = prompt
    return result


def test_selects_top_ranked_prompt():
    results = [
        _make_result_with_prompt(1, "Top prompt"),
        _make_result_with_prompt(2, "Second prompt"),
    ]
    assert select_system_prompt(results) == "Top prompt"


def test_skips_none_uses_fallback():
    results = [
        _make_result_with_prompt(1, None),
        _make_result_with_prompt(2, "Has prompt"),
    ]
    assert select_system_prompt(results) is None


def test_empty_results_returns_none():
    assert select_system_prompt([]) is None


def test_all_none_returns_none():
    results = [
        _make_result_with_prompt(1, None),
        _make_result_with_prompt(2, None),
    ]
    assert select_system_prompt(results) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prompt_resolution.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement prompt_resolution.py**

```python
# api/utils/prompt_resolution.py
"""Shared utility for selecting the winning system prompt from search results.

The retriever (BM25Retriever) attaches system_prompt to each result via
PromptStorageClient. This utility selects which result's prompt wins.

Rule: Use the top-ranked result's system_prompt. If None, return None
(caller falls back to hardcoded default).
"""

from __future__ import annotations


def select_system_prompt(results: list[dict]) -> str | None:
    """Select the system prompt from the highest-ranked search result.

    Args:
        results: Ranked search results (index 0 = highest rank).
                 System prompt is on the result dict: result["system_prompt"].
                 (Attached by RRF fusion from BM25Retriever's PromptStorageClient.)

    Returns:
        The system prompt string, or None if no prompt available.
    """
    if not results:
        return None

    return results[0].get("system_prompt")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prompt_resolution.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add api/utils/prompt_resolution.py tests/test_prompt_resolution.py
git commit -m "feat: add shared system prompt selection utility"
```

---

### Task 7: Help Text and Get User Query

**Files:**
- Modify: `core/chat_service.py` (add `_get_help_text()`, `_get_user_query()`)
- Test: `tests/test_chat_service.py` (add help and query extraction tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_chat_service.py`:

```python
class TestGetUserQuery:
    """Tests for ChatService._get_user_query()."""

    def test_extracts_last_user_message(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert ChatService._get_user_query(messages) == "second"

    def test_no_user_message(self):
        messages = [{"role": "system", "content": "sys"}]
        assert ChatService._get_user_query(messages) is None

    def test_empty_messages(self):
        assert ChatService._get_user_query([]) is None

    def test_none_content_returns_none(self):
        messages = [{"role": "user", "content": None}]
        assert ChatService._get_user_query(messages) is None


class TestHelpText:
    """Tests for ChatService._get_help_text()."""

    def test_help_text_contains_commands(self):
        text = ChatService._get_help_text()
        assert "!f" in text
        assert "!help" in text
        assert "knowledge base" in text.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chat_service.py::TestGetUserQuery tests/test_chat_service.py::TestHelpText -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement _get_user_query() and _get_help_text()**

Add to `ChatService` class in `core/chat_service.py`:

```python
    @staticmethod
    def _get_user_query(messages: list[dict]) -> str | None:
        """Extract the latest user message content."""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content")
                if content and isinstance(content, str):
                    return content
                return None
        return None

    @staticmethod
    def _get_help_text() -> str:
        """Generate help text for the !help command."""
        return """# RAG Helpdesk Assistant - Command Help

## How It Works

By default, your question is searched against the UTC IT knowledge base using hybrid search (BM25 keyword matching + vector semantic similarity), and the most relevant documents are provided to the LLM to generate an answer.

## Available Commands

### `<your question>` (no command - default)
**Knowledge base search** - Searches the knowledge base and provides context to the LLM.

**Example:**
```
How do I reset a student's password?
```

---

### `!f <your question>`
**Follow-up mode** - Bypasses the knowledge base entirely. Uses only the LLM's built-in knowledge and conversation history.

**When to use:** For general IT questions, clarifications, or follow-up questions about a previous response that don't need UTC-specific documentation.

**Example:**
```
!f Can you explain that last step in more detail?
```

---

### `!help`
**Display this help message**

---

*Need more help? Contact the development team or check the system documentation.*"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chat_service.py::TestGetUserQuery tests/test_chat_service.py::TestHelpText -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/chat_service.py tests/test_chat_service.py
git commit -m "feat: add user query extraction and help text generation"
```

---

### Task 8: ChatService.handle_chat() — Core Orchestration

**Files:**
- Modify: `core/chat_service.py` (add `handle_chat()` async generator)
- Test: `tests/test_chat_service.py` (add orchestration tests)

This is the largest task. `handle_chat()` wires everything together: command parsing, search, prompt assembly, LLM streaming, and logging.

- [ ] **Step 1: Write failing tests for handle_chat()**

Append to `tests/test_chat_service.py`:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from core.config import ChatSettings, LiteLLMSettings


def _make_mock_settings():
    """Create mock ChatSettings and LiteLLMSettings."""
    chat_settings = ChatSettings(
        ENABLE_CONVERSATION_LOGGING=False,  # Disable logging for unit tests
        MODEL_ID="test-model",
        TOP_K=5,
        FETCH_TOP_K=20,
        MAX_CONTEXT_TOKENS=4000,
        REQUEST_TIMEOUT=30.0,
    )
    litellm_settings = MagicMock(spec=LiteLLMSettings)
    litellm_settings.CHAT_MODEL = "gpt-test"
    litellm_settings.CHAT_TEMPERATURE = 0.7
    litellm_settings.CHAT_COMPLETION_TOKENS = 500
    litellm_settings.PROXY_BASE_URL = "http://localhost:4000"
    litellm_settings.PROXY_API_KEY = MagicMock()
    litellm_settings.PROXY_API_KEY.get_secret_value.return_value = "test-key"
    return chat_settings, litellm_settings


def _make_chat_service(chat_settings=None, litellm_settings=None):
    """Create a ChatService with mocked dependencies."""
    cs, ls = _make_mock_settings()
    return ChatService(
        bm25_retriever=MagicMock(),
        vector_retriever=MagicMock(),
        reranker=None,
        connection_pool=MagicMock(),
        chat_settings=chat_settings or cs,
        litellm_settings=litellm_settings or ls,
        hybrid_search_fn=MagicMock(),
        select_system_prompt_fn=MagicMock(return_value=None),
    )


class TestHandleChat:
    """Tests for ChatService.handle_chat() orchestration."""

    @pytest.mark.asyncio
    async def test_help_command_yields_help_text(self):
        service = _make_chat_service()
        messages = [{"role": "user", "content": "!help"}]

        chunks = []
        async for chunk in service.handle_chat(messages):
            chunks.append(chunk)

        full_text = "".join(chunks)
        assert "!f" in full_text
        assert "!help" in full_text

    @pytest.mark.asyncio
    async def test_follow_up_skips_search(self):
        service = _make_chat_service()
        messages = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "!f explain more"},
        ]

        # Mock litellm.acompletion to return a fake stream
        async def fake_stream(*args, **kwargs):
            class FakeChunk:
                def __init__(self, content, finish_reason=None, usage=None):
                    choice = MagicMock()
                    choice.delta.content = content
                    choice.finish_reason = finish_reason
                    self.choices = [choice]
                    self.usage = usage

            yield FakeChunk("Follow-up ")
            yield FakeChunk("response.", finish_reason="stop", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

        with patch("core.chat_service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=fake_stream())
            chunks = []
            async for chunk in service.handle_chat(messages):
                chunks.append(chunk)

        full_text = "".join(chunks)
        assert full_text == "Follow-up response."
        # Search should NOT have been called
        service._bm25.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_search_calls_hybrid(self):
        service = _make_chat_service()
        messages = [{"role": "user", "content": "password reset"}]

        # Mock hybrid_search to return results
        mock_chunk = MagicMock()
        mock_chunk.text_content = "Reset steps..."
        mock_chunk.source_url = "https://example.com"
        search_results = [{"rank": 1, "combined_score": 0.9, "chunk": mock_chunk, "system_prompt": None}]
        reranking_meta = {"reranked": False}

        # Inject mock hybrid_search result via the service's callable
        service._hybrid_search = MagicMock(return_value=(search_results, reranking_meta))
        service._select_system_prompt = MagicMock(return_value=None)

        async def fake_stream(*args, **kwargs):
            class FakeChunk:
                def __init__(self, content, finish_reason=None, usage=None):
                    choice = MagicMock()
                    choice.delta.content = content
                    choice.finish_reason = finish_reason
                    self.choices = [choice]
                    self.usage = usage

            yield FakeChunk("Answer.")
            yield FakeChunk(None, finish_reason="stop", usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11})

        with patch("core.chat_service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=fake_stream())

            chunks = []
            async for chunk in service.handle_chat(messages):
                chunks.append(chunk)

        full_text = "".join(chunks)
        assert full_text == "Answer."
        service._hybrid_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieval_failure_degrades_to_no_rag(self):
        """When hybrid_search raises, should degrade to follow-up mode (no RAG)."""
        service = _make_chat_service()
        messages = [{"role": "user", "content": "password reset"}]

        # hybrid_search raises an exception
        service._hybrid_search = MagicMock(side_effect=RuntimeError("DB down"))

        async def fake_stream(*args, **kwargs):
            class FakeChunk:
                def __init__(self, content, finish_reason=None, usage=None):
                    choice = MagicMock()
                    choice.delta.content = content
                    choice.finish_reason = finish_reason
                    self.choices = [choice]
                    self.usage = usage
            yield FakeChunk("Fallback.", finish_reason="stop")

        with patch("core.chat_service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=fake_stream())

            chunks = []
            async for chunk in service.handle_chat(messages):
                chunks.append(chunk)

        assert "".join(chunks) == "Fallback."
        # LLM should still be called (degraded, not failed)
        mock_litellm.acompletion.assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_fallback_on_stream_failure(self):
        """When stream=True fails, should fall back to stream=False."""
        service = _make_chat_service()
        messages = [{"role": "user", "content": "!f hello"}]

        call_count = 0

        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("stream", False):
                raise Exception("Streaming broken")
            # Non-streaming fallback
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "Fallback response."
            resp.usage = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
            return resp

        with patch("core.chat_service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=mock_acompletion)

            chunks = []
            async for chunk in service.handle_chat(messages):
                chunks.append(chunk)

        assert "".join(chunks) == "Fallback response."
        assert call_count == 2  # First call (stream=True) failed, second (stream=False) succeeded
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chat_service.py::TestHandleChat -v`
Expected: FAIL with `AttributeError: 'ChatService' object has no attribute 'handle_chat'`

- [ ] **Step 3: Implement handle_chat()**

Add imports at the top of `core/chat_service.py`:

```python
import asyncio
import time
import litellm
from typing import AsyncGenerator, Callable
from utils.logger import get_logger
from core.storage_query_log import QueryLogClient
from core.storage_reranker_log import RerankerLogClient

logger = get_logger(__name__)
```

**Note:** `hybrid_search` and `select_system_prompt` are NOT imported here — they come from `api/utils/` which would break architectural layering (`core/` should not import from `api/`). Instead, they are passed as constructor dependencies. `QueryLogClient` and `RerankerLogClient` are imported at module level so tests can patch them as `core.chat_service.QueryLogClient`.

Add the `handle_chat()` method to `ChatService`:

```python
    async def handle_chat(
        self,
        messages: list[dict],
        user_email: str | None = None,
    ) -> AsyncGenerator[str | dict, None]:
        """Orchestrate a RAG chat turn. Yields response text chunks.

        Yields str for content chunks. After all content, yields a dict
        {"usage": {...}} with token counts (if available). The router uses
        this to populate the final SSE chunk's usage field.

        Steps: parse command -> search -> assemble prompt -> stream LLM -> log.
        """
        user_query = self._get_user_query(messages)
        if not user_query:
            return

        command, cleaned_query = self._parse_command(user_query)
        query_log_id = None
        search_results = []
        reranking_metadata = {}

        # --- !help: yield help text, no LLM call ---
        if command == "help":
            yield self._get_help_text()
            return

        # --- !f (follow-up): skip search ---
        if command == "follow_up":
            assembled = self._assemble_no_rag_messages(messages)

        # --- Default: hybrid search + RAG ---
        else:
            try:
                search_results, reranking_metadata = await asyncio.to_thread(
                    self._hybrid_search,
                    query=cleaned_query,
                    bm25_retriever=self._bm25,
                    vector_retriever=self._vector,
                    reranker=self._reranker,
                    top_k=self._chat_settings.TOP_K,
                    fetch_top_k=self._chat_settings.FETCH_TOP_K,
                    rrf_k=self._chat_settings.RRF_K,
                    min_vector_similarity=self._chat_settings.MIN_VECTOR_SIMILARITY,
                )
            except Exception:
                logger.exception("Retrieval failed, degrading to follow-up mode")
                search_results = []

            if search_results:
                system_prompt = self._select_system_prompt(search_results)
                context = self._format_context(
                    search_results, self._chat_settings.MAX_CONTEXT_TOKENS
                )
                assembled = self._assemble_rag_messages(
                    messages, cleaned_query, context, system_prompt
                )
            else:
                assembled = self._assemble_no_rag_messages(messages)

        # --- Logging (feature-flagged, best-effort) ---
        if self._chat_settings.ENABLE_CONVERSATION_LOGGING:
            try:
                query_log_id = await self._log_query(
                    cleaned_query, user_email, command, search_results, reranking_metadata
                )
            except Exception:
                logger.exception("Query logging failed")

        # --- Stream LLM response ---
        llm_start = time.monotonic()
        full_response = []
        usage_data = None

        try:
            response = await litellm.acompletion(
                model=f"openai/{self._litellm_settings.CHAT_MODEL}",
                api_base=self._litellm_settings.PROXY_BASE_URL,
                api_key=self._litellm_settings.PROXY_API_KEY.get_secret_value(),
                messages=assembled,
                max_tokens=self._litellm_settings.CHAT_COMPLETION_TOKENS,
                temperature=self._litellm_settings.CHAT_TEMPERATURE,
                stream=True,
                stream_options={"include_usage": True},
                num_retries=3,
                timeout=self._chat_settings.REQUEST_TIMEOUT,
            )

            async for chunk in response:
                content = chunk.choices[0].delta.content if chunk.choices else None
                if content:
                    full_response.append(content)
                    yield content
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = chunk.usage

        except Exception:
            logger.exception("Streaming failed, attempting non-streaming fallback")
            try:
                response = await litellm.acompletion(
                    model=f"openai/{self._litellm_settings.CHAT_MODEL}",
                    api_base=self._litellm_settings.PROXY_BASE_URL,
                    api_key=self._litellm_settings.PROXY_API_KEY.get_secret_value(),
                    messages=assembled,
                    max_tokens=self._litellm_settings.CHAT_COMPLETION_TOKENS,
                    temperature=self._litellm_settings.CHAT_TEMPERATURE,
                    stream=False,
                    num_retries=3,
                    timeout=self._chat_settings.REQUEST_TIMEOUT,
                )
                content = response.choices[0].message.content
                if content:
                    full_response.append(content)
                    yield content
                if hasattr(response, "usage") and response.usage:
                    usage_data = response.usage
            except Exception:
                logger.exception("Non-streaming fallback also failed")
                raise  # Re-raise — router catches and emits SSE error event

        # --- Yield usage data for router to include in final SSE chunk ---
        if usage_data:
            usage_dict = usage_data if isinstance(usage_data, dict) else {
                "prompt_tokens": getattr(usage_data, "prompt_tokens", 0),
                "completion_tokens": getattr(usage_data, "completion_tokens", 0),
                "total_tokens": getattr(usage_data, "total_tokens", 0),
            }
            yield {"usage": usage_dict}

        # --- Post-stream logging ---
        llm_latency_ms = int((time.monotonic() - llm_start) * 1000)

        if self._chat_settings.ENABLE_CONVERSATION_LOGGING and query_log_id:
            try:
                await self._log_llm_response(
                    query_log_id, "".join(full_response), llm_latency_ms,
                    usage_data, search_results,
                )
            except Exception:
                logger.exception("LLM response logging failed")

    @staticmethod
    def _transform_results_for_logging(search_results: list[dict]) -> list[dict]:
        """Transform hybrid search results to the format QueryLogClient expects.

        hybrid_search returns: [{"rank": int, "combined_score": float, "chunk": TextChunk}]
        QueryLogClient expects: [{"rank": int, "score": float, "chunk_id": UUID, "parent_article_id": UUID}]
        """
        return [
            {
                "rank": r["rank"],
                "score": r["combined_score"],
                "chunk_id": str(r["chunk"].chunk_id),
                "parent_article_id": str(r["chunk"].parent_article_id),
            }
            for r in search_results
        ]

    async def _log_query(
        self,
        query: str | None,
        email: str | None,
        command: str | None,
        search_results: list[dict],
        reranking_metadata: dict,
    ) -> int | None:
        """Log query and results to database. Returns query_log_id."""
        command_value = "follow_up" if command == "follow_up" else "search"
        results_for_logging = self._transform_results_for_logging(search_results)

        def _do_log():
            client = QueryLogClient(connection_pool=self._pool)
            query_log_id = client.log_query_with_results(
                raw_query=query or "",
                cache_result="miss",
                search_method="hybrid",
                results=results_for_logging,
                email=email,
                command=command_value,
            )
            # Log reranker data if available
            rrf_results = reranking_metadata.get("rrf_results_before_reranking", [])
            if reranking_metadata.get("reranked") and self._reranker:
                reranker_client = RerankerLogClient(connection_pool=self._pool)
                reranker_client.log_reranking(
                    query_log_id=query_log_id,
                    rrf_results=rrf_results,
                    reranked_results=search_results,
                    model_name=self._reranker.model if self._reranker else "unknown",
                    reranker_latency_ms=reranking_metadata.get("reranker_latency_ms", 0),
                    reranker_status="success",
                )
            elif reranking_metadata.get("reranking_failed"):
                reranker_client = RerankerLogClient(connection_pool=self._pool)
                reranker_client.log_reranking(
                    query_log_id=query_log_id,
                    rrf_results=rrf_results,
                    reranked_results=rrf_results,
                    model_name=self._reranker.model if self._reranker else "unknown",
                    reranker_latency_ms=reranking_metadata.get("reranker_latency_ms", 0),
                    reranker_status="failed",
                    error_message=reranking_metadata.get("error"),
                )
            return query_log_id

        return await asyncio.to_thread(_do_log)

    async def _log_llm_response(
        self,
        query_log_id: int,
        response_text: str,
        llm_latency_ms: int,
        usage_data,
        search_results: list[dict],
    ) -> None:
        """Log LLM response to llm_responses table."""
        citations = None
        if search_results:
            citations = {
                "num_documents_used": len(search_results),
                "source_urls": [
                    str(r["chunk"].source_url) for r in search_results
                    if hasattr(r["chunk"], "source_url") and r["chunk"].source_url
                ],
                "chunk_ids": [
                    str(r["chunk"].chunk_id) for r in search_results
                    if hasattr(r["chunk"], "chunk_id") and r["chunk"].chunk_id
                ],
            }

        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        if usage_data:
            if isinstance(usage_data, dict):
                prompt_tokens = usage_data.get("prompt_tokens")
                completion_tokens = usage_data.get("completion_tokens")
                total_tokens = usage_data.get("total_tokens")
            else:
                prompt_tokens = getattr(usage_data, "prompt_tokens", None)
                completion_tokens = getattr(usage_data, "completion_tokens", None)
                total_tokens = getattr(usage_data, "total_tokens", None)

        model_name = self._litellm_settings.CHAT_MODEL

        def _do_log():
            client = QueryLogClient(connection_pool=self._pool)
            client.log_llm_response(
                query_log_id=query_log_id,
                response_text=response_text,
                model_name=model_name,
                llm_latency_ms=llm_latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                citations=citations,
            )

        await asyncio.to_thread(_do_log)
```

**Note to implementer:** The `_log_query` method's call to `QueryLogClient.log_query_with_results()` uses the exact parameter order from `core/storage_query_log.py:368-378`: `raw_query, cache_result, search_method, results, latency_ms, email, query_embedding, command`. The `_transform_results_for_logging()` method converts hybrid search result dicts to the format expected by `log_query_results()`: `{"rank", "score", "chunk_id", "parent_article_id"}`. The `log_llm_response` method similarly wraps the existing endpoint logic from `api/routers/query_logs.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chat_service.py::TestHandleChat -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `pytest tests/test_chat_service.py -v`
Expected: All tests PASS (command parsing + context formatting + prompt assembly + help/query + handle_chat)

- [ ] **Step 6: Commit**

```bash
git add core/chat_service.py tests/test_chat_service.py
git commit -m "feat: implement ChatService.handle_chat() with search, streaming, and logging"
```

---

### Task 9: OpenAI-Compatible Router

**Files:**
- Create: `api/routers/openai_compat.py`
- Test: `tests/test_openai_compat.py`

- [ ] **Step 1: Write failing tests for the router**

```python
# tests/test_openai_compat.py
"""Integration tests for the OpenAI-compatible router."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.openai_compat import router
from core.chat_service import ChatService


@pytest.fixture
def mock_chat_service():
    """Create a mock ChatService."""
    service = MagicMock(spec=ChatService)
    return service


@pytest.fixture
def app(mock_chat_service):
    """Create a test FastAPI app with the OpenAI compat router."""
    from api.dependencies import verify_api_key

    test_app = FastAPI()
    test_app.state.chat_service = mock_chat_service

    # Mock chat settings
    mock_settings = MagicMock()
    mock_settings.MODEL_ID = "test-model"

    # Use FastAPI dependency_overrides (not patch) because Depends()
    # captures the function reference at import time
    test_app.dependency_overrides[verify_api_key] = lambda: None

    with patch("api.routers.openai_compat.get_chat_settings", return_value=mock_settings):
        test_app.include_router(router, prefix="/v1")
        yield test_app

    test_app.dependency_overrides.clear()


@pytest.fixture
def client(app):
    return TestClient(app)


class TestModelsEndpoint:
    def test_list_models(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "utc"


class TestChatCompletionsEndpoint:
    def test_streams_sse_response(self, client, mock_chat_service):
        """Chat completions should return SSE-formatted streaming response."""
        async def fake_gen(*args, **kwargs):
            yield "Hello "
            yield "world!"

        mock_chat_service.handle_chat = MagicMock(return_value=fake_gen())

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE events
        lines = response.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) >= 3  # role chunk, content chunks, [DONE]
        assert data_lines[-1] == "data: [DONE]"

        # Check first content chunk has required fields
        first_data = json.loads(data_lines[0].removeprefix("data: "))
        assert "id" in first_data
        assert first_data["id"].startswith("chatcmpl-")
        assert "created" in first_data
        assert first_data["model"] == "test-model"
        assert first_data["object"] == "chat.completion.chunk"

    def test_empty_messages_returns_error(self, client, mock_chat_service):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [],
                "stream": True,
            },
        )
        assert response.status_code == 422  # Pydantic validation


class TestAuthentication:
    def test_missing_api_key_returns_401(self):
        """Without a valid API key, auth should fail with 401."""
        test_app = FastAPI()
        test_app.state.chat_service = MagicMock()

        mock_settings = MagicMock()
        mock_settings.MODEL_ID = "test-model"

        # NO dependency_overrides — real verify_api_key runs
        with patch("api.routers.openai_compat.get_chat_settings", return_value=mock_settings), \
             patch.dict("os.environ", {"API_API_KEY": "real-secret-key-min-32-chars-long!!"}):
            test_app.include_router(router, prefix="/v1")
            client = TestClient(test_app, raise_server_exceptions=False)
            # Send request with wrong API key — should get 401
            response = client.get("/v1/models", headers={"X-API-Key": "wrong-key"})
            assert response.status_code == 401
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_openai_compat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.routers.openai_compat'`

- [ ] **Step 3: Implement the router**

```python
# api/routers/openai_compat.py
"""OpenAI-compatible endpoints for chat completions.

Thin HTTP adapter: parses requests, delegates to ChatService,
wraps output as SSE in OpenAI format.
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from uuid_utils import uuid7

from api.dependencies import verify_api_key
from api.models.chat import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ModelListResponse,
    ModelObject,
)
from core.config import get_chat_settings
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("/models", response_model=ModelListResponse)
def list_models():
    """List available models (OpenAI-compatible)."""
    settings = get_chat_settings()
    return ModelListResponse(
        data=[
            ModelObject(
                id=settings.MODEL_ID,
                created=int(time.time()),
            )
        ]
    )


@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
):
    """OpenAI-compatible chat completions with SSE streaming."""
    settings = get_chat_settings()
    chat_service = request.app.state.chat_service

    request_id = f"chatcmpl-{uuid7()}"
    created = int(time.time())
    model_id = settings.MODEL_ID

    # Extract email from headers if available
    user_email = request.headers.get("X-User-Email")

    messages = [msg.model_dump() for msg in body.messages]

    async def generate_sse():
        # Role chunk
        role_chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model_id,
            choices=[
                ChatCompletionChunkChoice(
                    delta={"role": "assistant"}, finish_reason=None
                )
            ],
        )
        yield f"data: {role_chunk.model_dump_json()}\n\n"

        usage_data = None

        try:
            async for item in chat_service.handle_chat(messages, user_email):
                if isinstance(item, dict) and "usage" in item:
                    # Usage data yielded after all content
                    usage_data = item["usage"]
                    continue
                if not item:
                    continue
                content_chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=model_id,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta={"content": item}, finish_reason=None
                        )
                    ],
                )
                yield f"data: {content_chunk.model_dump_json()}\n\n"

            # Final chunk with finish_reason and usage
            stop_chunk = ChatCompletionChunk(
                id=request_id,
                created=created,
                model=model_id,
                choices=[
                    ChatCompletionChunkChoice(delta={}, finish_reason="stop")
                ],
                usage=usage_data,
            )
            yield f"data: {stop_chunk.model_dump_json()}\n\n"

        except Exception as e:
            logger.exception("Error during chat streaming")
            error_data = json.dumps({
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 503,
                }
            })
            yield f"data: {error_data}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_openai_compat.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add api/routers/openai_compat.py tests/test_openai_compat.py
git commit -m "feat: add OpenAI-compatible /v1/models and /v1/chat/completions router"
```

---

### Task 10: Wire Up — Dependencies, Main App, Lifespan

**Files:**
- Modify: `api/dependencies.py:353-383` (add `get_chat_service()`)
- Modify: `api/main.py:27-28` (add import), `api/main.py:155-163` (add init in lifespan), `api/main.py:217-222` (register router)

- [ ] **Step 1: Add get_chat_service() dependency**

Add to `api/dependencies.py` after the existing dependency functions (around line 383):

```python
def get_chat_service(request: Request):
    """Get the shared ChatService instance from app.state."""
    chat_service = getattr(request.app.state, "chat_service", None)
    if chat_service is None:
        raise HTTPException(
            status_code=503,
            detail="ChatService not initialized",
        )
    return chat_service
```

- [ ] **Step 2: Register router in api/main.py**

Add import at the top of `api/main.py` (around line 27):

```python
from api.routers import search, health, query_logs, admin_prompts, admin_analytics, openai_compat
```

Add router registration after the existing routers (around line 239):

```python
app.include_router(
    openai_compat.router,
    prefix="/v1",
    tags=["OpenAI Compatible"],
)
```

- [ ] **Step 3: Initialize ChatService in lifespan**

Add to the lifespan function in `api/main.py`, after the HyDE generator initialization (around line 155):

```python
        # Initialize ChatService
        logger.info("Initializing ChatService...")
        from core.chat_service import ChatService
        from core.config import get_chat_settings, get_litellm_settings
        from api.utils.hybrid_search import hybrid_search
        from api.utils.prompt_resolution import select_system_prompt

        chat_settings = get_chat_settings()
        litellm_settings = get_litellm_settings()
        chat_service = ChatService(
            bm25_retriever=bm25_retriever,
            vector_retriever=vector_retriever,
            reranker=app.state.reranker,
            connection_pool=connection_pool,
            chat_settings=chat_settings,
            litellm_settings=litellm_settings,
            hybrid_search_fn=hybrid_search,
            select_system_prompt_fn=select_system_prompt,
        )
        app.state.chat_service = chat_service
        logger.info(
            f"ChatService initialized (model_id={chat_settings.MODEL_ID}, "
            f"logging={'enabled' if chat_settings.ENABLE_CONVERSATION_LOGGING else 'disabled'})"
        )
```

- [ ] **Step 4: Run existing tests to check for regressions**

Run: `pytest tests/ -v`
Expected: All existing tests PASS, no regressions

- [ ] **Step 5: Commit**

```bash
git add api/dependencies.py api/main.py
git commit -m "feat: wire up ChatService — dependencies, lifespan init, router registration"
```

---

### Task 11: Logging-Enabled Integration Tests

**Files:**
- Modify: `tests/test_chat_service.py` (add logging tests)

- [ ] **Step 1: Write tests for logging behavior**

Append to `tests/test_chat_service.py`:

```python
class TestLoggingBehavior:
    """Tests for ChatService logging feature flag."""

    @pytest.mark.asyncio
    async def test_logging_disabled_skips_all_db_writes(self):
        """When ENABLE_CONVERSATION_LOGGING=False, no logging calls are made."""
        chat_settings, litellm_settings = _make_mock_settings()
        chat_settings = ChatSettings(ENABLE_CONVERSATION_LOGGING=False)

        service = ChatService(
            bm25_retriever=MagicMock(),
            vector_retriever=MagicMock(),
            reranker=None,
            connection_pool=MagicMock(),
            chat_settings=chat_settings,
            litellm_settings=litellm_settings,
        )

        messages = [{"role": "user", "content": "!f hello"}]

        async def fake_stream(*args, **kwargs):
            class FakeChunk:
                def __init__(self, content, finish_reason=None, usage=None):
                    choice = MagicMock()
                    choice.delta.content = content
                    choice.finish_reason = finish_reason
                    self.choices = [choice]
                    self.usage = usage
            yield FakeChunk("Hi.", finish_reason="stop")

        with patch("core.chat_service.litellm") as mock_litellm, \
             patch("core.chat_service.QueryLogClient") as mock_log_cls:
            mock_litellm.acompletion = AsyncMock(return_value=fake_stream())

            chunks = []
            async for chunk in service.handle_chat(messages):
                chunks.append(chunk)

        # QueryLogClient should never be instantiated
        mock_log_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_help_command_no_logging(self):
        """!help should never trigger logging, even with logging enabled."""
        chat_settings, litellm_settings = _make_mock_settings()
        chat_settings = ChatSettings(ENABLE_CONVERSATION_LOGGING=True)

        service = ChatService(
            bm25_retriever=MagicMock(),
            vector_retriever=MagicMock(),
            reranker=None,
            connection_pool=MagicMock(),
            chat_settings=chat_settings,
            litellm_settings=litellm_settings,
        )

        messages = [{"role": "user", "content": "!help"}]

        with patch("core.chat_service.QueryLogClient") as mock_log_cls:
            chunks = []
            async for chunk in service.handle_chat(messages):
                chunks.append(chunk)

        mock_log_cls.assert_not_called()
        assert "!help" in "".join(chunks)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_chat_service.py::TestLoggingBehavior -v`
Expected: All 2 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_chat_service.py
git commit -m "test: add logging feature flag tests for ChatService"
```

---

### Task 12: Full Suite Verification and Cleanup

**Files:**
- All new and modified files

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run linter**

Run: `ruff check .` and `ruff format .`
Fix any issues.

- [ ] **Step 3: Verify the app starts**

Run: `python -c "from api.main import app; print('App imports OK')"` to check imports resolve.

- [ ] **Step 4: Commit any cleanup**

```bash
git add -A
git commit -m "chore: lint and formatting cleanup"
```

- [ ] **Step 5: Final verification commit message**

```bash
git log --oneline -12
```

Expected: 12 clean commits building up the feature incrementally.

---

## Summary of Commits

| Order | Commit Message |
|-------|---------------|
| 1 | `feat: add ChatSettings config for /v1/chat/completions endpoint` |
| 2 | `feat: add Pydantic models for OpenAI-compatible chat endpoint` |
| 3 | `feat: add ChatService with command parsing (!f, !help, default)` |
| 4 | `feat: add context formatting with token-based truncation` |
| 5 | `feat: add query-sandwich prompt assembly and system prompt constants` |
| 6 | `feat: add shared system prompt selection utility` |
| 7 | `feat: add user query extraction and help text generation` |
| 8 | `feat: implement ChatService.handle_chat() with search, streaming, and logging` |
| 9 | `feat: add OpenAI-compatible /v1/models and /v1/chat/completions router` |
| 10 | `feat: wire up ChatService — dependencies, lifespan init, router registration` |
| 11 | `test: add logging feature flag tests for ChatService` |
| 12 | `chore: lint and formatting cleanup` |
