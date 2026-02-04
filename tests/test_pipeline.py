"""
Tests for the pipeline module (RAGPipeline orchestrator).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic import HttpUrl
from datetime import datetime, timezone

from core.pipeline import RAGPipeline
from core.schemas import TdxArticle, TextChunk, VectorRecord


class TestRAGPipeline:
    """Test suite for RAGPipeline orchestrator."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for pipeline components."""
        # Mock database settings
        with patch("core.storage_base.get_database_settings") as mock_db:
            db_settings = Mock()
            db_settings.HOST = "localhost"
            db_settings.USER = "test_user"
            db_settings.PASSWORD.get_secret_value.return_value = "test_password"
            db_settings.NAME = "test_db"
            mock_db.return_value = db_settings

            # Mock embedding settings
            with patch("core.embedding.get_embedding_settings") as mock_embed:
                embed_settings = Mock()
                embed_settings.DEPLOYMENT_NAME = "text-embedding-3-large"
                embed_settings.ENDPOINT = "https://test.openai.azure.com"
                embed_settings.API_VERSION = "2023-05-15"
                embed_settings.MAX_TOKENS = 8000
                embed_settings.EMBED_DIM = 3072
                embed_settings.API_KEY.get_secret_value.return_value = "test-key"
                mock_embed.return_value = embed_settings

                # Mock TDX settings
                with patch("core.api_client.get_tdx_settings") as mock_tdx:
                    tdx_settings = Mock()
                    tdx_settings.BASE_URL = "https://test.teamdynamix.com"
                    tdx_settings.APP_ID = 123
                    tdx_settings.WEBSERVICES_KEY.get_secret_value.return_value = (
                        "test-ws-key"
                    )
                    tdx_settings.BEID.get_secret_value.return_value = "test-beid"
                    mock_tdx.return_value = tdx_settings

                    yield {
                        "db": db_settings,
                        "embed": embed_settings,
                        "tdx": tdx_settings,
                    }

    @pytest.fixture
    def pipeline_openai(self, mock_settings):
        """Create RAGPipeline instance with OpenAI provider."""
        with patch("core.pipeline.ArticleProcessor"):
            with patch("core.pipeline.TextProcessor"):
                with patch("core.pipeline.GenerateEmbeddingsOpenAI"):
                    with patch("core.pipeline.OpenAIVectorStorage"):
                        with patch("core.pipeline.PostgresClient"):
                            with patch("core.pipeline.Tokenizer"):
                                # Mock get_embedding_settings call in process_article_to_chunks
                                with patch(
                                    "core.pipeline.get_embedding_settings"
                                ) as mock_embed_runtime:
                                    mock_embed_runtime.return_value = mock_settings[
                                        "embed"
                                    ]
                                    yield RAGPipeline(embedding_provider="openai")

    def test_init_openai(self, pipeline_openai):
        """Test pipeline initialization with OpenAI provider."""
        assert pipeline_openai.embedding_provider == "openai"
        assert pipeline_openai.skip_ingestion is False
        assert pipeline_openai.skip_processing is False
        assert pipeline_openai.skip_embedding is False
        assert hasattr(pipeline_openai, "article_processor")
        assert hasattr(pipeline_openai, "text_processor")
        assert hasattr(pipeline_openai, "embedder")
        assert hasattr(pipeline_openai, "vector_store")
        assert hasattr(pipeline_openai, "raw_store")

    def test_init_invalid_provider(self, mock_settings):
        """Test that invalid provider raises ValueError."""
        with patch("core.pipeline.PostgresClient"):
            with pytest.raises(ValueError, match="Invalid embedding provider"):
                RAGPipeline(embedding_provider="invalid_provider")

    def test_init_skip_ingestion(self, mock_settings):
        """Test pipeline with ingestion skipped."""
        with patch("core.pipeline.TextProcessor"):
            with patch("core.pipeline.GenerateEmbeddingsOpenAI"):
                with patch("core.pipeline.OpenAIVectorStorage"):
                    with patch("core.pipeline.PostgresClient"):
                        with patch("core.pipeline.Tokenizer"):
                            pipeline = RAGPipeline(
                                embedding_provider="openai", skip_ingestion=True
                            )
                            assert not hasattr(pipeline, "article_processor")

    def test_init_skip_processing(self, mock_settings):
        """Test pipeline with processing skipped."""
        with patch("core.pipeline.ArticleProcessor"):
            with patch("core.pipeline.GenerateEmbeddingsOpenAI"):
                with patch("core.pipeline.OpenAIVectorStorage"):
                    with patch("core.pipeline.PostgresClient"):
                        pipeline = RAGPipeline(
                            embedding_provider="openai", skip_processing=True
                        )
                        assert not hasattr(pipeline, "text_processor")
                        assert not hasattr(pipeline, "tokenizer")

    def test_init_skip_embedding(self, mock_settings):
        """Test pipeline with embedding skipped."""
        with patch("core.pipeline.ArticleProcessor"):
            with patch("core.pipeline.TextProcessor"):
                with patch("core.pipeline.PostgresClient"):
                    with patch("core.pipeline.Tokenizer"):
                        pipeline = RAGPipeline(
                            embedding_provider="openai", skip_embedding=True
                        )
                        assert not hasattr(pipeline, "embedder")
                        assert not hasattr(pipeline, "vector_store")

    def test_generate_chunk_id(self, pipeline_openai):
        """Test chunk ID generation produces valid UUIDs."""
        from uuid import UUID

        chunk_id_1 = pipeline_openai._generate_chunk_id()
        chunk_id_2 = pipeline_openai._generate_chunk_id()

        # IDs should be UUIDs
        assert isinstance(chunk_id_1, UUID)
        assert isinstance(chunk_id_2, UUID)
        # Each call should produce a unique ID
        assert chunk_id_1 != chunk_id_2

    def test_run_ingestion_when_skipped(self, mock_settings):
        """Test that run_ingestion raises error when skipped."""
        with patch("core.pipeline.PostgresClient"):
            pipeline = RAGPipeline(
                embedding_provider="openai",
                skip_ingestion=True,
                skip_processing=True,
                skip_embedding=True,
            )
            with pytest.raises(RuntimeError, match="Cannot run ingestion"):
                pipeline.run_ingestion()

    def test_run_ingestion_success(self, pipeline_openai):
        """Test successful ingestion phase."""
        mock_stats = {
            "new_count": 10,
            "updated_count": 5,
            "unchanged_count": 15,
            "skipped_count": 2,
        }
        pipeline_openai.article_processor.ingest_and_store = Mock(
            return_value=mock_stats
        )

        stats = pipeline_openai.run_ingestion()

        assert stats == mock_stats
        pipeline_openai.article_processor.ingest_and_store.assert_called_once()

    def test_run_processing_when_skipped(self, mock_settings):
        """Test that run_processing raises error when skipped."""
        with patch("core.pipeline.ArticleProcessor"):
            with patch("core.pipeline.PostgresClient"):
                pipeline = RAGPipeline(
                    embedding_provider="openai",
                    skip_processing=True,
                    skip_embedding=True,
                )
                with pytest.raises(RuntimeError, match="Cannot run processing"):
                    pipeline.run_processing()

    def test_process_article_to_chunks(self, pipeline_openai):
        """Test processing a single article into chunks."""
        from uuid import UUID

        test_article_id = UUID("12345678-1234-5678-1234-567812345678")
        article = TdxArticle(
            id=test_article_id,
            tdx_article_id=123,
            title="Test Article",
            url=HttpUrl("https://example.com/123"),
            content_html="<p>Test content</p>",
            last_modified_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            status_name="Approved",
        )

        # Mock text processor
        pipeline_openai.text_processor.process_text = Mock(return_value="Test content")
        pipeline_openai.text_processor.text_to_chunks = Mock(
            return_value=["Chunk 1", "Chunk 2"]
        )

        # Mock tokenizer
        pipeline_openai.tokenizer.num_tokens_from_string = Mock(return_value=50)

        chunks = pipeline_openai.process_article_to_chunks(article)

        assert len(chunks) == 2
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert chunks[0].parent_article_id == test_article_id
        assert chunks[0].chunk_sequence == 0
        assert chunks[1].chunk_sequence == 1
        assert chunks[0].text_content == "Chunk 1"
        assert chunks[1].text_content == "Chunk 2"

        pipeline_openai.text_processor.process_text.assert_called_once_with(
            article.content_html
        )
        pipeline_openai.text_processor.text_to_chunks.assert_called_once()

    def test_run_embedding_when_skipped(self, mock_settings):
        """Test that run_embedding raises error when skipped."""
        with patch("core.pipeline.ArticleProcessor"):
            with patch("core.pipeline.TextProcessor"):
                with patch("core.pipeline.PostgresClient"):
                    with patch("core.pipeline.Tokenizer"):
                        pipeline = RAGPipeline(
                            embedding_provider="openai", skip_embedding=True
                        )
                        chunks = []
                        with pytest.raises(RuntimeError, match="Cannot run embedding"):
                            pipeline.run_embedding(chunks)

    def test_run_embedding_empty_chunks(self, pipeline_openai):
        """Test embedding with empty chunks list."""
        embeddings = pipeline_openai.run_embedding([])
        assert embeddings == []

    def test_run_embedding_success(self, pipeline_openai):
        """Test successful embedding generation."""
        from uuid import UUID

        test_chunk_id_1 = UUID("12345678-1234-5678-1234-567812345671")
        test_chunk_id_2 = UUID("12345678-1234-5678-1234-567812345672")
        test_article_id = UUID("87654321-4321-8765-4321-876543218765")

        chunks = [
            TextChunk(
                chunk_id=test_chunk_id_1,
                parent_article_id=test_article_id,
                chunk_sequence=0,
                text_content="Test content 1",
                token_count=50,
                source_url=HttpUrl("https://example.com"),
                last_modified_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            TextChunk(
                chunk_id=test_chunk_id_2,
                parent_article_id=test_article_id,
                chunk_sequence=1,
                text_content="Test content 2",
                token_count=60,
                source_url=HttpUrl("https://example.com"),
                last_modified_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
        ]

        # Mock embedder
        pipeline_openai.embedder.generate_embedding = Mock(
            side_effect=[[0.1] * 3072, [0.2] * 3072]
        )

        embeddings = pipeline_openai.run_embedding(chunks)

        assert len(embeddings) == 2
        assert all(isinstance(record, VectorRecord) for record, _ in embeddings)
        assert all(isinstance(vector, list) for _, vector in embeddings)
        assert len(embeddings[0][1]) == 3072
        assert pipeline_openai.embedder.generate_embedding.call_count == 2

    def test_run_embedding_continues_on_error(self, pipeline_openai):
        """Test that embedding continues when individual chunks fail."""
        from uuid import UUID

        test_chunk_id_1 = UUID("12345678-1234-5678-1234-567812345671")
        test_chunk_id_2 = UUID("12345678-1234-5678-1234-567812345672")
        test_article_id = UUID("87654321-4321-8765-4321-876543218765")

        chunks = [
            TextChunk(
                chunk_id=test_chunk_id_1,
                parent_article_id=test_article_id,
                chunk_sequence=0,
                text_content="Test content 1",
                token_count=50,
                source_url=HttpUrl("https://example.com"),
                last_modified_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            TextChunk(
                chunk_id=test_chunk_id_2,
                parent_article_id=test_article_id,
                chunk_sequence=1,
                text_content="Test content 2",
                token_count=60,
                source_url=HttpUrl("https://example.com"),
                last_modified_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
        ]

        # Mock embedder - first succeeds, second fails
        pipeline_openai.embedder.generate_embedding = Mock(
            side_effect=[[0.1] * 3072, RuntimeError("Embedding failed")]
        )

        embeddings = pipeline_openai.run_embedding(chunks)

        # Should only have 1 successful embedding
        assert len(embeddings) == 1
        assert embeddings[0][0].chunk_id == test_chunk_id_1

    def test_run_storage_empty_embeddings(self, pipeline_openai):
        """Test storage with empty embeddings list."""
        stored_count = pipeline_openai.run_storage([])
        assert stored_count == 0

    def test_run_storage_success(self, pipeline_openai):
        """Test successful embedding storage."""
        from uuid import UUID

        test_chunk_id = UUID("12345678-1234-5678-1234-567812345678")
        test_article_id = UUID("87654321-4321-8765-4321-876543218765")

        embeddings = [
            (
                VectorRecord(
                    chunk_id=test_chunk_id,
                    parent_article_id=test_article_id,
                    chunk_sequence=0,
                    text_content="Test content",
                    token_count=50,
                    source_url=HttpUrl("https://example.com"),
                ),
                [0.1] * 3072,
            )
        ]

        pipeline_openai.vector_store.insert_embeddings = Mock()

        stored_count = pipeline_openai.run_storage(embeddings)

        assert stored_count == 1
        pipeline_openai.vector_store.insert_embeddings.assert_called_once_with(
            embeddings
        )

    def test_cleanup(self, pipeline_openai):
        """Test pipeline cleanup."""
        pipeline_openai.raw_store.close = Mock()
        pipeline_openai.vector_store.close = Mock()

        pipeline_openai.cleanup()

        pipeline_openai.raw_store.close.assert_called_once()
        pipeline_openai.vector_store.close.assert_called_once()

    def test_context_manager(self, pipeline_openai):
        """Test pipeline as context manager."""
        pipeline_openai.cleanup = Mock()

        with pipeline_openai as p:
            assert p == pipeline_openai

        pipeline_openai.cleanup.assert_called_once()

    def test_run_full_pipeline(self, pipeline_openai):
        """Test running the full pipeline."""
        # Mock cleanup
        pipeline_openai.article_processor.cleanup_non_approved_articles = Mock(
            return_value=0
        )

        # Mock ingestion
        mock_ingestion_stats = {
            "new_count": 10,
            "updated_count": 5,
            "unchanged_count": 15,
            "skipped_count": 2,
        }
        pipeline_openai.article_processor.ingest_and_store = Mock(
            return_value=mock_ingestion_stats
        )

        stats = pipeline_openai.run_full_pipeline()

        assert "ingestion" in stats
        assert "processing" in stats
        assert "embedding" in stats
        assert "storage" in stats
        assert "start_time" in stats
        assert "end_time" in stats
        assert "duration_seconds" in stats
        assert stats["ingestion"] == mock_ingestion_stats
        assert stats["end_time"] is not None
        assert stats["duration_seconds"] > 0
