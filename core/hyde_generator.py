"""
HyDE (Hypothetical Document Embeddings) Generator

This module generates hypothetical documents from user queries using Azure OpenAI.
The hypothetical documents are then embedded and used for semantic search, providing
better document-to-document matching in vector space.
"""

from openai import (
    AsyncAzureOpenAI,
    APIError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
)
from typing import Dict, Optional
from core.config import get_chat_settings
import asyncio
import html
import logging
logger = logging.getLogger(__name__)

# HyDE system prompt for generating hypothetical documents
HYDE_SYSTEM_PROMPT = """You are a technical writer drafting internal IT knowledge base articles for the University of Tennessee Chattanooga.

Given a user question, write a concise 2-4 sentence answer as it would appear in official documentation.

Guidelines:
- Write in declarative, procedural style (not conversational)
- Use second person ("you") for instructions
- State facts directly without hedging or caveats
- Include specific technical terms, menu paths, or system names when relevant
- Do not include greetings, sign-offs, or offer further help
- All guides must be about Information Technology.

Write only the documentation content."""


class HyDEGenerator:
    """
    Generate hypothetical documents from queries using Azure OpenAI.

    HyDE (Hypothetical Document Embeddings) improves retrieval by generating
    hypothetical answers to queries before embedding. This provides better
    semantic alignment between queries and documents in vector space.
    """

    def __init__(self):
        """
        Initialize HyDE generator with Azure OpenAI client.

        Raises:
            ValueError: If required configuration is missing or invalid
            RuntimeError: If Azure OpenAI client initialization fails
        """
        try:
            settings = get_chat_settings()

            # Validate required configuration
            if not settings.DEPLOYMENT_NAME:
                raise ValueError("CHAT_DEPLOYMENT_NAME is not configured")
            if not settings.ENDPOINT:
                raise ValueError("CHAT_ENDPOINT is not configured")
            if not settings.API_VERSION:
                raise ValueError("CHAT_API_VERSION is not configured")
            if not settings.MAX_TOKENS or settings.MAX_TOKENS <= 0:
                raise ValueError("CHAT_MAX_TOKENS must be a positive integer")
            if not settings.COMPLETION_TOKENS or settings.COMPLETION_TOKENS <= 0:
                raise ValueError("CHAT_COMPLETION_TOKENS must be a positive integer")

            self.deployment_name = settings.DEPLOYMENT_NAME
            self.max_tokens = settings.MAX_TOKENS
            self.max_completion_tokens = min(settings.COMPLETION_TOKENS, 500)  # HyDE docs are concise

            # Initialize async client
            try:
                api_key = settings.API_KEY.get_secret_value()
                if not api_key:
                    raise ValueError("CHAT_API_KEY is empty")

                self.client = AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=settings.ENDPOINT,
                    api_version=settings.API_VERSION,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize Azure OpenAI client: {str(e)}"
                ) from e

            # Use embedded HyDE system prompt
            self.system_prompt = HYDE_SYSTEM_PROMPT
            logger.info("HyDE generator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize HyDEGenerator: {str(e)}")
            raise

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by decoding HTML entities and unicode escapes.

        Args:
            text: Raw text that may contain HTML entities or unicode escapes

        Returns:
            Cleaned text with proper characters
        """
        # Decode HTML entities (e.g., &amp; -> &, &quot; -> ")
        text = html.unescape(text)

        # Replace common unicode escapes with actual characters
        replacements = {
            "\u2019": "'",  # Right single quotation mark
            "\u2018": "'",  # Left single quotation mark
            "\u201c": '"',  # Left double quotation mark
            "\u201d": '"',  # Right double quotation mark
            "\u2013": "-",  # En dash
            "\u2014": "—",  # Em dash
            "\u2026": "...",  # Horizontal ellipsis
            "\u00a0": " ",  # Non-breaking space
        }

        for unicode_char, replacement in replacements.items():
            text = text.replace(unicode_char, replacement)

        return text

    async def generate_hypothetical_document(
        self,
        query: str,
        max_retries: int = 3,
    ) -> tuple[str, Optional[Dict[str, int]]]:
        """
        Generate a hypothetical document from a user query.

        Uses Azure OpenAI with the HyDE system prompt to generate a concise
        2-4 sentence answer as it would appear in official documentation.
        Uses model's default temperature.

        Args:
            query: User query to generate hypothetical document for
            max_retries: Maximum retry attempts for API failures (default: 3)

        Returns:
            Tuple of (hypothetical_document, token_usage):
            - hypothetical_document: Generated text (2-4 sentences)
            - token_usage: Dict with prompt_tokens, completion_tokens, total_tokens
              (or None if usage data not available)

        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If generation fails after all retries
        """
        # Input validation
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        cleaned_query = query.strip()

        # Retry logic with exponential backoff
        retry_delay = 1  # Start with 1 second
        last_exception = None

        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"Generating hypothetical document (attempt {attempt + 1}/{max_retries}): "
                    f"query='{cleaned_query[:50]}...'"
                )

                response = await self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_prompt,
                        },
                        {
                            "role": "user",
                            "content": cleaned_query,
                        },
                    ],
                    max_completion_tokens=self.max_completion_tokens,
                )

                # Validate response structure
                if not response or not hasattr(response, "choices"):
                    raise ValueError("Invalid response structure: missing 'choices'")

                if not response.choices or len(response.choices) == 0:
                    raise ValueError("Response choices are empty")

                if not hasattr(response.choices[0], "message"):
                    raise ValueError("Response choice missing 'message'")

                # Extract generated text
                hypothetical_doc = response.choices[0].message.content

                if not hypothetical_doc or not hypothetical_doc.strip():
                    raise ValueError("Generated hypothetical document is empty")

                # Clean the generated text
                hypothetical_doc = self.clean_text(hypothetical_doc.strip())

                # Extract token usage if available
                token_usage = None
                if hasattr(response, 'usage') and response.usage:
                    try:
                        token_usage = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }
                        logger.debug(
                            f"Token usage - prompt: {token_usage['prompt_tokens']}, "
                            f"completion: {token_usage['completion_tokens']}, "
                            f"total: {token_usage['total_tokens']}"
                        )
                    except AttributeError as e:
                        logger.warning(f"Could not extract token usage: {e}")

                logger.debug(
                    f"Generated hypothetical document ({len(hypothetical_doc)} chars): "
                    f"'{hypothetical_doc[:100]}...'"
                )

                return hypothetical_doc, token_usage

            except RateLimitError as e:
                last_exception = e
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(
                        f"Rate limit exceeded after {max_retries} retries"
                    ) from e

            except APITimeoutError as e:
                last_exception = e
                logger.warning(
                    f"API timeout (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(
                        f"API timeout after {max_retries} retries"
                    ) from e

            except AuthenticationError as e:
                # Don't retry authentication errors
                logger.error(f"Authentication failed: {str(e)}")
                raise RuntimeError(
                    "Azure OpenAI authentication failed. Check CHAT_API_KEY"
                ) from e

            except (APIError, ValueError) as e:
                last_exception = e
                logger.error(
                    f"API error (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(
                        f"Hypothetical document generation failed after {max_retries} retries: {str(e)}"
                    ) from e

            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error during generation: {str(e)}", exc_info=True)
                raise RuntimeError(
                    f"Unexpected error generating hypothetical document: {str(e)}"
                ) from e

        # Should not reach here, but just in case
        raise RuntimeError(
            f"Failed to generate hypothetical document after {max_retries} retries"
        ) from last_exception

    def generate_hypothetical_document_sync(
        self,
        query: str,
        max_retries: int = 3,
    ) -> tuple[str, Optional[Dict[str, int]]]:
        """
        Synchronous wrapper for generate_hypothetical_document().

        This method is provided for compatibility with synchronous contexts.
        For async contexts, use generate_hypothetical_document() directly.

        Args:
            query: User query to generate hypothetical document for
            max_retries: Maximum retry attempts for API failures (default: 3)

        Returns:
            Tuple of (hypothetical_document, token_usage):
            - hypothetical_document: Generated text (2-4 sentences)
            - token_usage: Dict with prompt_tokens, completion_tokens, total_tokens
              (or None if usage data not available)

        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If generation fails after all retries
        """
        return asyncio.run(
            self.generate_hypothetical_document(
                query=query,
                max_retries=max_retries,
            )
        )
