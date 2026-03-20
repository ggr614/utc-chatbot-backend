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
