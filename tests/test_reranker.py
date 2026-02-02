"""
Tests for Cohere reranker module.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone
from uuid import uuid4
import json

from core.reranker import CohereReranker, RerankerResult
from core.schemas import TextChunk
from pydantic import HttpUrl


@pytest.fixture
def mock_aws_settings():
    """Mock AWS reranker settings."""
    settings = Mock()
    settings.ACCESS_KEY_ID = Mock()
    settings.ACCESS_KEY_ID.get_secret_value = Mock(return_value="test_access_key")
    settings.SECRET_ACCESS_KEY = Mock()
    settings.SECRET_ACCESS_KEY.get_secret_value = Mock(return_value="test_secret_key")
    settings.REGION_NAME = "us-east-1"
    settings.RERANKER_ARN = "cohere.rerank-v3-5:0"
    return settings


@pytest.fixture
def mock_bedrock_client():
    """Create a mock boto3 Bedrock Runtime client."""
    client = Mock()

    # Mock successful response
    response_body = json.dumps(
        {
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.87},
                {"index": 1, "relevance_score": 0.72},
            ]
        }
    )

    client.invoke_model.return_value = {
        "body": Mock(read=Mock(return_value=response_body.encode("utf-8")))
    }

    return client


@pytest.fixture
def sample_fused_results():
    """Create sample fused results from hybrid search."""
    base_time = datetime.now(timezone.utc)

    chunks = [
        TextChunk(
            chunk_id=uuid4(),
            parent_article_id=uuid4(),
            chunk_sequence=0,
            text_content="Password reset instructions for system users.",
            token_count=10,
            source_url=HttpUrl("https://example.com/1"),
            last_modified_date=base_time,
        ),
        TextChunk(
            chunk_id=uuid4(),
            parent_article_id=uuid4(),
            chunk_sequence=0,
            text_content="VPN troubleshooting guide for remote access.",
            token_count=10,
            source_url=HttpUrl("https://example.com/2"),
            last_modified_date=base_time,
        ),
        TextChunk(
            chunk_id=uuid4(),
            parent_article_id=uuid4(),
            chunk_sequence=0,
            text_content="Email configuration steps for new employees.",
            token_count=10,
            source_url=HttpUrl("https://example.com/3"),
            last_modified_date=base_time,
        ),
    ]

    # Simulate RRF fusion output format
    return [
        {"rank": 1, "combined_score": 0.045, "chunk": chunks[0]},
        {"rank": 2, "combined_score": 0.038, "chunk": chunks[1]},
        {"rank": 3, "combined_score": 0.032, "chunk": chunks[2]},
    ]


class TestCohereReranker:
    """Test suite for CohereReranker."""

    @patch("core.reranker.get_aws_reranker_settings")
    @patch("boto3.client")
    def test_initialization_success(
        self, mock_boto_client, mock_get_settings, mock_aws_settings, mock_bedrock_client
    ):
        """Test successful reranker initialization."""
        mock_get_settings.return_value = mock_aws_settings
        mock_boto_client.return_value = mock_bedrock_client

        reranker = CohereReranker()

        assert reranker.model_id == "cohere.rerank-v3-5:0"
        assert reranker.region_name == "us-east-1"
        assert reranker.max_retries == 3
        assert reranker.timeout == 30.0

        mock_boto_client.assert_called_once_with(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="test_access_key",
            aws_secret_access_key="test_secret_key",
        )

    @patch("core.reranker.get_aws_reranker_settings")
    def test_initialization_missing_access_key(self, mock_get_settings):
        """Test initialization fails with missing access key."""
        settings = Mock()
        settings.ACCESS_KEY_ID = None
        settings.SECRET_ACCESS_KEY = Mock()
        settings.SECRET_ACCESS_KEY.get_secret_value = Mock(return_value="secret")
        settings.RERANKER_ARN = "cohere.rerank-v3-5:0"
        mock_get_settings.return_value = settings

        with pytest.raises(ValueError, match="AWS_ACCESS_KEY_ID is not configured"):
            CohereReranker()

    @patch("core.reranker.get_aws_reranker_settings")
    @patch("boto3.client")
    def test_rerank_basic(
        self,
        mock_boto_client,
        mock_get_settings,
        mock_aws_settings,
        mock_bedrock_client,
        sample_fused_results,
    ):
        """Test basic reranking functionality."""
        mock_get_settings.return_value = mock_aws_settings
        mock_boto_client.return_value = mock_bedrock_client

        reranker = CohereReranker()
        results = reranker.rerank(query="password reset help", results=sample_fused_results)

        # Verify results structure
        assert len(results) == 3
        assert all("rank" in r for r in results)
        assert all("combined_score" in r for r in results)
        assert all("chunk" in r for r in results)

        # Verify reranking changed order (based on mock response)
        assert results[0]["rank"] == 1
        assert results[0]["combined_score"] == 0.95  # Top rerank score from mock
        assert results[1]["rank"] == 2
        assert results[1]["combined_score"] == 0.87
        assert results[2]["rank"] == 3
        assert results[2]["combined_score"] == 0.72

        # Verify metadata was added
        assert "metadata" in results[0]
        assert "original_rank" in results[0]["metadata"]
        assert "original_score" in results[0]["metadata"]

        # Verify API call
        mock_bedrock_client.invoke_model.assert_called_once()
        call_args = mock_bedrock_client.invoke_model.call_args
        assert call_args[1]["modelId"] == "cohere.rerank-v3-5:0"

        # Verify request body
        request_body = json.loads(call_args[1]["body"])
        assert request_body["query"] == "password reset help"
        assert len(request_body["documents"]) == 3
        assert request_body["api_version"] == 2

    @patch("core.reranker.get_aws_reranker_settings")
    @patch("boto3.client")
    def test_rerank_empty_results(
        self, mock_boto_client, mock_get_settings, mock_aws_settings, mock_bedrock_client
    ):
        """Test reranking with empty results list."""
        mock_get_settings.return_value = mock_aws_settings
        mock_boto_client.return_value = mock_bedrock_client

        reranker = CohereReranker()
        results = reranker.rerank(query="test", results=[])

        assert results == []
        mock_bedrock_client.invoke_model.assert_not_called()

    @patch("core.reranker.get_aws_reranker_settings")
    @patch("boto3.client")
    def test_rerank_empty_query(
        self,
        mock_boto_client,
        mock_get_settings,
        mock_aws_settings,
        mock_bedrock_client,
        sample_fused_results,
    ):
        """Test reranking with empty query raises ValueError."""
        mock_get_settings.return_value = mock_aws_settings
        mock_boto_client.return_value = mock_bedrock_client

        reranker = CohereReranker()

        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.rerank(query="", results=sample_fused_results)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            reranker.rerank(query="   ", results=sample_fused_results)

    @patch("core.reranker.get_aws_reranker_settings")
    @patch("boto3.client")
    def test_rerank_invalid_results_structure(
        self, mock_boto_client, mock_get_settings, mock_aws_settings, mock_bedrock_client
    ):
        """Test reranking with invalid results structure."""
        mock_get_settings.return_value = mock_aws_settings
        mock_boto_client.return_value = mock_bedrock_client

        reranker = CohereReranker()

        # Test with non-dict items
        with pytest.raises(ValueError, match="Results must be a list of dicts"):
            reranker.rerank(query="test", results=["not", "dicts"])

        # Test with missing chunk field
        with pytest.raises(ValueError, match="Each result must have a 'chunk' field"):
            reranker.rerank(query="test", results=[{"rank": 1, "score": 0.5}])

    @patch("core.reranker.get_aws_reranker_settings")
    @patch("boto3.client")
    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_rerank_retry_on_throttling(
        self, mock_sleep, mock_boto_client, mock_get_settings, mock_aws_settings, sample_fused_results
    ):
        """Test retry logic on throttling errors."""
        from botocore.exceptions import ClientError

        mock_get_settings.return_value = mock_aws_settings

        client = Mock()
        # Fail twice with throttling, succeed on third attempt
        client.invoke_model.side_effect = [
            ClientError({"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}, "invoke_model"),
            ClientError({"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}, "invoke_model"),
            {
                "body": Mock(
                    read=Mock(
                        return_value=json.dumps(
                            {"results": [{"index": 0, "relevance_score": 0.9}]}
                        ).encode("utf-8")
                    )
                )
            },
        ]
        mock_boto_client.return_value = client

        reranker = CohereReranker(max_retries=3)
        results = reranker.rerank(query="test", results=sample_fused_results[:1])

        assert len(results) == 1
        assert results[0]["combined_score"] == 0.9
        assert client.invoke_model.call_count == 3

        # Verify exponential backoff was applied
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0][0][0] == 1.0  # First retry: 1s
        assert mock_sleep.call_args_list[1][0][0] == 2.0  # Second retry: 2s

    @patch("core.reranker.get_aws_reranker_settings")
    @patch("boto3.client")
    def test_rerank_persistent_failure(
        self, mock_boto_client, mock_get_settings, mock_aws_settings, sample_fused_results
    ):
        """Test failure after all retries exhausted."""
        from botocore.exceptions import ClientError

        mock_get_settings.return_value = mock_aws_settings

        client = Mock()
        client.invoke_model.side_effect = ClientError(
            {"Error": {"Code": "ServiceUnavailable", "Message": "Service down"}},
            "invoke_model",
        )
        mock_boto_client.return_value = client

        reranker = CohereReranker(max_retries=2)

        with pytest.raises(RuntimeError, match="Cohere reranking failed after 2 attempts"):
            reranker.rerank(query="test", results=sample_fused_results)

        assert client.invoke_model.call_count == 2

    @patch("core.reranker.get_aws_reranker_settings")
    @patch("boto3.client")
    def test_rerank_non_retryable_error(
        self, mock_boto_client, mock_get_settings, mock_aws_settings, sample_fused_results
    ):
        """Test non-retryable errors fail immediately."""
        from botocore.exceptions import ClientError

        mock_get_settings.return_value = mock_aws_settings

        client = Mock()
        client.invoke_model.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "invoke_model",
        )
        mock_boto_client.return_value = client

        reranker = CohereReranker(max_retries=3)

        with pytest.raises(RuntimeError, match="Cohere reranking failed"):
            reranker.rerank(query="test", results=sample_fused_results)

        # Should only try once (not retryable)
        assert client.invoke_model.call_count == 1

    @patch("core.reranker.get_aws_reranker_settings")
    @patch("boto3.client")
    def test_rerank_limits_to_1000_documents(
        self, mock_boto_client, mock_get_settings, mock_aws_settings, mock_bedrock_client
    ):
        """Test that reranker truncates to 1000 documents."""
        mock_get_settings.return_value = mock_aws_settings
        mock_boto_client.return_value = mock_bedrock_client

        # Create 1500 fake results
        large_results = []
        for i in range(1500):
            chunk = Mock()
            chunk.text_content = f"Document {i}"
            large_results.append({"rank": i + 1, "combined_score": 0.5, "chunk": chunk})

        # Mock response with just first result
        mock_bedrock_client.invoke_model.return_value = {
            "body": Mock(
                read=Mock(
                    return_value=json.dumps(
                        {"results": [{"index": 0, "relevance_score": 0.9}]}
                    ).encode("utf-8")
                )
            )
        }

        reranker = CohereReranker()
        results = reranker.rerank(query="test", results=large_results)

        # Verify only 1000 documents were sent
        call_args = mock_bedrock_client.invoke_model.call_args
        request_body = json.loads(call_args[1]["body"])
        assert len(request_body["documents"]) == 1000

    @patch("core.reranker.get_aws_reranker_settings")
    @patch("boto3.client")
    def test_rerank_top_n_parameter(
        self,
        mock_boto_client,
        mock_get_settings,
        mock_aws_settings,
        mock_bedrock_client,
        sample_fused_results,
    ):
        """Test top_n parameter limits results."""
        mock_get_settings.return_value = mock_aws_settings
        mock_boto_client.return_value = mock_bedrock_client

        # Mock response with 2 results
        mock_bedrock_client.invoke_model.return_value = {
            "body": Mock(
                read=Mock(
                    return_value=json.dumps(
                        {
                            "results": [
                                {"index": 0, "relevance_score": 0.9},
                                {"index": 1, "relevance_score": 0.8},
                            ]
                        }
                    ).encode("utf-8")
                )
            )
        }

        reranker = CohereReranker()
        results = reranker.rerank(query="test", results=sample_fused_results, top_n=2)

        # Verify API was called with top_n=2
        call_args = mock_bedrock_client.invoke_model.call_args
        request_body = json.loads(call_args[1]["body"])
        assert request_body["top_n"] == 2

        # Verify only 2 results returned
        assert len(results) == 2

    @patch("core.reranker.get_aws_reranker_settings")
    @patch("boto3.client")
    def test_rerank_malformed_response(
        self, mock_boto_client, mock_get_settings, mock_aws_settings, sample_fused_results
    ):
        """Test handling of malformed API response."""
        mock_get_settings.return_value = mock_aws_settings

        client = Mock()
        # Return invalid JSON
        client.invoke_model.return_value = {
            "body": Mock(read=Mock(return_value=b"not valid json"))
        }
        mock_boto_client.return_value = client

        reranker = CohereReranker(max_retries=1)

        with pytest.raises(RuntimeError, match="Invalid JSON response"):
            reranker.rerank(query="test", results=sample_fused_results)
