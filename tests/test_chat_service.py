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
