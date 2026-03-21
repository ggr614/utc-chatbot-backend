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


from unittest.mock import AsyncMock, MagicMock, patch
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


import asyncio
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

        full_text = "".join(c for c in chunks if isinstance(c, str))
        assert full_text == "Follow-up response."
        # Search should NOT have been called
        service._hybrid_search.assert_not_called()

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

        full_text = "".join(c for c in chunks if isinstance(c, str))
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

        assert "".join(c for c in chunks if isinstance(c, str)) == "Fallback."
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

        assert "".join(c for c in chunks if isinstance(c, str)) == "Fallback response."
        assert call_count == 2  # First call (stream=True) failed, second (stream=False) succeeded


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
        assert "!help" in "".join(c for c in chunks if isinstance(c, str))
