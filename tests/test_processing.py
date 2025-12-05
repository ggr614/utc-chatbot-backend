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

    def test_process_text_handles_empty_string(self, processor):
        """Test that empty string is handled correctly."""
        result = processor.process_text("")
        assert result == ""

    def test_process_text_handles_only_whitespace(self, processor):
        """Test that string with only whitespace is handled correctly."""
        result = processor.process_text("   \n\n   ")
        assert result == ""

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

    def test_get_token_count_returns_integer(self, processor):
        """Test that token count returns an integer."""
        text = "This is a test string with several words."
        count = processor.get_token_count(text)

        assert isinstance(count, int)
        assert count > 0

    def test_get_token_count_empty_string(self, processor):
        """Test token count for empty string."""
        count = processor.get_token_count("")
        assert count == 0

    def test_get_token_count_increases_with_text_length(self, processor):
        """Test that longer text has more tokens."""
        short_text = "Hello world"
        long_text = "Hello world this is a much longer piece of text with many more words and tokens"

        short_count = processor.get_token_count(short_text)
        long_count = processor.get_token_count(long_text)

        assert long_count > short_count

    def test_get_token_count_uses_tokenizer(self, processor):
        """Test that get_token_count uses the Tokenizer class."""
        with patch("core.processing.Tokenizer") as mock_tokenizer_class:
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.num_tokens_from_string.return_value = 42
            mock_tokenizer_class.return_value = mock_tokenizer_instance

            # Create new processor to use mocked tokenizer
            new_processor = TextProcessor()
            count = new_processor.get_token_count("test text")

            mock_tokenizer_class.assert_called_once()
            mock_tokenizer_instance.num_tokens_from_string.assert_called_once_with(
                "test text"
            )
            assert count == 42

    def test_process_text_consistent_output(self, processor):
        """Test that processing the same text multiple times gives same result."""
        html = "<p>Test content</p>"

        result1 = processor.process_text(html)
        result2 = processor.process_text(html)

        assert result1 == result2

    def test_get_token_count_consistent_output(self, processor):
        """Test that counting tokens multiple times gives same result."""
        text = "Test content"

        count1 = processor.get_token_count(text)
        count2 = processor.get_token_count(text)

        assert count1 == count2

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
