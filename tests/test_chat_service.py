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
