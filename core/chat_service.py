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
