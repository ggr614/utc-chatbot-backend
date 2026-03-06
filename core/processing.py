import html2text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from core.tokenizer import Tokenizer
from utils.logger import get_logger

logger = get_logger(__name__)


class TextProcessor:
    def __init__(self):
        logger.info("Initializing TextProcessor")
        try:
            self.html_converter = html2text.HTML2Text()
            self.html_converter.ignore_links = True
            self.html_converter.ignore_images = True
            self.html_converter.ignore_tables = True
            self.html_converter.body_width = 0  # Disable word wrapping
            logger.debug("TextProcessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TextProcessor: {str(e)}")
            raise RuntimeError(f"TextProcessor initialization failed: {str(e)}") from e

    def process_text(self, text: str) -> str:
        """
        Convert HTML to cleaned plain text.

        Args:
            text: HTML text to process

        Returns:
            Cleaned plain text

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If text processing fails
        """
        if not text:
            logger.warning("Empty text provided to process_text")
            raise ValueError("Text cannot be empty")

        if not isinstance(text, str):
            logger.error(f"Invalid type for text: {type(text)}")
            raise ValueError(f"Text must be a string, got {type(text)}")

        try:
            logger.debug(f"Processing text of length {len(text)}")

            # Convert HTML to Markdown, then clean up
            markdown_text = self.html_converter.handle(text)
            logger.debug(f"Converted HTML to markdown, length: {len(markdown_text)}")

            # Remove extra newlines, leading/trailing whitespace, and multiple spaces
            cleaned_text = " ".join(markdown_text.split()).strip()
            logger.debug(f"Cleaned text, final length: {len(cleaned_text)}")

            if not cleaned_text:
                logger.warning("Text processing resulted in empty string")
                raise ValueError("Text processing resulted in empty content")

            logger.info(
                f"Successfully processed text: {len(text)} -> {len(cleaned_text)} characters"
            )
            return cleaned_text

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to process text: {str(e)}")
            raise RuntimeError(f"Text processing failed: {str(e)}") from e

    def text_to_chunks(
        self, text: str, max_tokens: int, overlap: int = 200
    ) -> List[str]:
        """
        Split text into chunks based on token count.

        Args:
            text: Text to split into chunks
            max_tokens: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks

        Returns:
            List of text chunks

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If chunking fails
        """
        # Validate inputs
        if not text:
            logger.warning("Empty text provided to text_to_chunks")
            raise ValueError("Text cannot be empty")

        if not isinstance(text, str):
            logger.error(f"Invalid type for text: {type(text)}")
            raise ValueError(f"Text must be a string, got {type(text)}")

        if max_tokens <= 0:
            logger.error(f"Invalid max_tokens: {max_tokens}")
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        if overlap < 0:
            logger.error(f"Invalid overlap: {overlap}")
            raise ValueError(f"overlap must be non-negative, got {overlap}")

        if overlap >= max_tokens:
            logger.error(
                f"Overlap ({overlap}) must be less than max_tokens ({max_tokens})"
            )
            raise ValueError(
                f"overlap ({overlap}) must be less than max_tokens ({max_tokens})"
            )

        try:
            logger.debug(
                f"Chunking text of length {len(text)} with max_tokens={max_tokens}, overlap={overlap}"
            )

            tokenizer = Tokenizer()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_tokens,
                chunk_overlap=overlap,
                length_function=tokenizer.num_tokens_from_string,
            )
            chunks = text_splitter.split_text(text)

            logger.info(f"Successfully split text into {len(chunks)} chunks")
            logger.debug(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")

            if not chunks:
                logger.warning("Text splitting resulted in no chunks")

            return chunks

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to chunk text: {str(e)}")
            raise RuntimeError(f"Text chunking failed: {str(e)}") from e
