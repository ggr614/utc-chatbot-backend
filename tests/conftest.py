"""
Pytest configuration and shared fixtures.
"""

import pytest
from datetime import datetime, timezone
from pydantic import HttpUrl

from core.schemas import TdxArticle


@pytest.fixture
def sample_article_data():
    """Sample raw article data from API."""
    return {
        "ID": 123,
        "Subject": "Test Article",
        "Body": "<p>Test content</p>",
        "ModifiedDate": "2024-01-01T00:00:00Z",
        "CategoryName": "IT Help",
    }


@pytest.fixture
def sample_article_model():
    """Sample TdxArticle Pydantic model."""
    return TdxArticle(
        id=123,
        title="Test Article",
        url=HttpUrl(
            "https://utc.teamdynamix.com/TDClient/2717/Portal/KB/ArticleDet?ID=123"
        ),
        content_html="<p>Test content</p>",
        last_modified_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_db_metadata():
    """Sample database metadata."""
    return {
        123: datetime(2024, 1, 1, tzinfo=timezone.utc),
        456: datetime(2024, 1, 2, tzinfo=timezone.utc),
    }


@pytest.fixture
def multiple_articles_data():
    """Multiple sample articles for testing."""
    return [
        {
            "ID": 123,
            "Subject": "First Article",
            "Body": "<p>First content</p>",
            "ModifiedDate": "2024-01-01T00:00:00Z",
            "CategoryName": "IT Help",
        },
        {
            "ID": 456,
            "Subject": "Second Article",
            "Body": "<p>Second content</p>",
            "ModifiedDate": "2024-01-02T00:00:00Z",
            "CategoryName": "Documentation",
        },
        {
            "ID": 789,
            "Subject": "Phishing Warning",
            "Body": "<p>Phishing alert</p>",
            "ModifiedDate": "2024-01-03T00:00:00Z",
            "CategoryName": "Recent Phishing Emails",
        },
    ]


@pytest.fixture
def articles_with_missing_fields():
    """Articles with missing required fields for testing validation."""
    return [
        {
            "ID": None,
            "Subject": "Test",
            "Body": "Content",
            "ModifiedDate": "2024-01-01T00:00:00Z",
        },
        {
            "ID": 123,
            "Subject": None,
            "Body": "Content",
            "ModifiedDate": "2024-01-01T00:00:00Z",
        },
        {
            "ID": 456,
            "Subject": "Test",
            "Body": None,
            "ModifiedDate": "2024-01-01T00:00:00Z",
        },
        {"ID": 789, "Subject": "Test", "Body": "Content", "ModifiedDate": None},
    ]
