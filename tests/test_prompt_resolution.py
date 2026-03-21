# tests/test_prompt_resolution.py
"""Tests for system prompt selection from search results."""

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
