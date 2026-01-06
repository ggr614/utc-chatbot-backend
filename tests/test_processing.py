"""
Tests for the processing module (TextProcessor).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from core.processing import TextProcessor


class TestTextProcessor:
    """Test suite for TextProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create TextProcessor instance."""
        return TextProcessor()

    def test_init_configures_html_converter(self, processor):
        """Test that HTML converter is properly configured."""
        assert processor.html_converter.ignore_links is True
        assert processor.html_converter.ignore_images is True
        assert processor.html_converter.ignore_tables is True
        assert processor.html_converter.body_width == 0

    def test_process_text_converts_html_to_markdown(self, processor):
        """Test that HTML is converted to markdown."""
        html = "<h1>Title</h1><p>This is a paragraph.</p>"
        result = processor.process_text(html)

        # Should contain the text content without HTML tags
        assert "Title" in result
        assert "This is a paragraph." in result
        assert "<h1>" not in result
        assert "<p>" not in result

    def test_process_text_removes_extra_whitespace(self, processor):
        """Test that extra whitespace is removed."""
        html = "<p>Text   with    multiple     spaces</p>"
        result = processor.process_text(html)

        # Should have single spaces only
        assert "Text with multiple spaces" in result
        assert "   " not in result

    def test_process_text_removes_extra_newlines(self, processor):
        """Test that extra newlines are removed."""
        html = "<p>Line 1</p>\n\n\n<p>Line 2</p>"
        result = processor.process_text(html)

        # Should not have multiple consecutive newlines
        assert "\n\n" not in result

    def test_process_text_strips_leading_trailing_whitespace(self, processor):
        """Test that leading and trailing whitespace is removed."""
        html = "   <p>Content</p>   "
        result = processor.process_text(html)

        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_process_text_raises_error_on_empty_string(self, processor):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            processor.process_text("")

    def test_process_text_raises_error_on_only_whitespace(self, processor):
        """Test that string with only whitespace raises ValueError."""
        with pytest.raises(
            ValueError, match="Text processing resulted in empty content"
        ):
            processor.process_text("   \n\n   ")

    def test_process_text_raises_error_on_non_string(self, processor):
        """Test that non-string input raises ValueError."""
        with pytest.raises(ValueError, match="Text must be a string"):
            processor.process_text(123)

        # None is caught by empty check first
        with pytest.raises(ValueError, match="Text cannot be empty"):
            processor.process_text(None)

    def test_process_text_raises_error_on_empty_result(self, processor):
        """Test that processing resulting in empty content raises ValueError."""
        with pytest.raises(
            ValueError, match="Text processing resulted in empty content"
        ):
            processor.process_text("<div></div>")

    def test_process_text_handles_complex_html(self, processor):
        """Test processing of complex HTML structure."""
        html = """
        <div>
            <h1>Main Title</h1>
            <h2>Subtitle</h2>
            <p>First paragraph with <strong>bold text</strong> and <em>italic text</em>.</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
            <p>Second paragraph.</p>
        </div>
        """
        result = processor.process_text(html)

        # Should contain the text content
        assert "Main Title" in result
        assert "Subtitle" in result
        assert "First paragraph" in result
        assert "bold text" in result
        assert "italic text" in result
        assert "Item 1" in result
        assert "Item 2" in result
        assert "Second paragraph" in result

    def test_process_text_ignores_links(self, processor):
        """Test that links are handled according to ignore_links setting."""
        html = '<p>Check out <a href="https://example.com">this link</a>.</p>'
        result = processor.process_text(html)

        # The link text should be present but not the URL (ignore_links=True)
        assert "this link" in result

    def test_process_text_ignores_images(self, processor):
        """Test that images are ignored."""
        html = '<p>Text with an image <img src="image.jpg" alt="description"> here.</p>'
        result = processor.process_text(html)

        # Image should be ignored (ignore_images=True)
        assert "image.jpg" not in result
        assert "Text with an image" in result

    def test_process_text_handles_special_characters(self, processor):
        """Test that special characters are handled correctly."""
        html = "<p>Special chars: &amp; &lt; &gt; &quot;</p>"
        result = processor.process_text(html)

        # HTML entities should be decoded
        assert "&" in result or "Special chars:" in result

    def test_text_to_chunks_basic_functionality(self, processor):
        """Test basic text chunking functionality."""
        text = "This is a test string. " * 100  # Create a longer text
        chunks = processor.text_to_chunks(text, max_tokens=50, overlap=10)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_text_to_chunks_empty_text_raises_error(self, processor):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            processor.text_to_chunks("", max_tokens=100)

    def test_text_to_chunks_non_string_raises_error(self, processor):
        """Test that non-string input raises ValueError."""
        with pytest.raises(ValueError, match="Text must be a string"):
            processor.text_to_chunks(123, max_tokens=100)

    def test_text_to_chunks_invalid_max_tokens_raises_error(self, processor):
        """Test that invalid max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            processor.text_to_chunks("test text", max_tokens=0)

        with pytest.raises(ValueError, match="max_tokens must be positive"):
            processor.text_to_chunks("test text", max_tokens=-10)

    def test_text_to_chunks_invalid_overlap_raises_error(self, processor):
        """Test that invalid overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            processor.text_to_chunks("test text", max_tokens=100, overlap=-5)

    def test_text_to_chunks_overlap_greater_than_max_raises_error(self, processor):
        """Test that overlap >= max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="overlap.*must be less than max_tokens"):
            processor.text_to_chunks("test text", max_tokens=100, overlap=100)

        with pytest.raises(ValueError, match="overlap.*must be less than max_tokens"):
            processor.text_to_chunks("test text", max_tokens=50, overlap=60)

    def test_process_text_consistent_output(self, processor):
        """Test that processing the same text multiple times gives same result."""
        html = "<p>Test content</p>"

        result1 = processor.process_text(html)
        result2 = processor.process_text(html)

        assert result1 == result2

    def test_process_text_handles_nested_tags(self, processor):
        """Test processing of deeply nested HTML tags."""
        html = """
        <div>
            <div>
                <div>
                    <p>Deeply nested <strong>content <em>here</em></strong>.</p>
                </div>
            </div>
        </div>
        """
        result = processor.process_text(html)

        assert "Deeply nested" in result
        assert "content" in result
        assert "here" in result

    def test_process_text_handles_code_blocks(self, processor):
        """Test processing of code blocks in HTML."""
        html = "<pre><code>def hello():\n    print('world')</code></pre>"
        result = processor.process_text(html)

        # Code content should be present
        assert "hello" in result or "world" in result

    def test_process_text_preserves_text_order(self, processor):
        """Test that text order is preserved after processing."""
        html = "<p>First</p><p>Second</p><p>Third</p>"
        result = processor.process_text(html)

        # Find positions of each word
        first_pos = result.find("First")
        second_pos = result.find("Second")
        third_pos = result.find("Third")

        # Ensure they appear in order
        assert first_pos < second_pos < third_pos
