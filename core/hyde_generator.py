"""
HyDE (Hypothetical Document Embeddings) Generator

This module generates hypothetical documents from user queries using LiteLLM proxy.
The hypothetical documents are then embedded and used for semantic search, providing
better document-to-document matching in vector space.
"""

import litellm
from litellm.exceptions import (
    RateLimitError,
    Timeout,
    AuthenticationError,
    APIError,
)
from typing import Dict, Optional
from core.config import get_litellm_settings
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
    Generate hypothetical documents from queries using LiteLLM proxy.

    HyDE (Hypothetical Document Embeddings) improves retrieval by generating
    hypothetical answers to queries before embedding. This provides better
    semantic alignment between queries and documents in vector space.
    """

    def __init__(self):
        """
        Initialize HyDE generator with LiteLLM proxy settings.

        Raises:
            ValueError: If required configuration is missing or invalid
            RuntimeError: If initialization fails
        """
        try:
            settings = get_litellm_settings()

            self.model = settings.CHAT_MODEL
            self._proxy_model = f"openai/{self.model}"
            self.api_base = settings.PROXY_BASE_URL
            self.api_key = settings.PROXY_API_KEY.get_secret_value()
            self.max_completion_tokens = min(settings.CHAT_COMPLETION_TOKENS, 500)
            self.deployment_name = self.model  # backward compat for logging
            self.system_prompt = HYDE_SYSTEM_PROMPT

            if not self.api_key:
                raise ValueError("LITELLM_PROXY_API_KEY is empty")

            logger.info(f"HyDE generator initialized: model={self.model}")

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

        Uses LiteLLM proxy with the HyDE system prompt to generate a concise
        2-4 sentence answer as it would appear in official documentation.

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
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        cleaned_query = query.strip()

        try:
            response = await litellm.acompletion(
                model=self._proxy_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": cleaned_query},
                ],
                max_tokens=self.max_completion_tokens,
                api_base=self.api_base,
                api_key=self.api_key,
                num_retries=max_retries,
                timeout=30.0,
            )

            # Extract generated text
            hypothetical_doc = response.choices[0].message.content

            if not hypothetical_doc or not hypothetical_doc.strip():
                raise ValueError("Generated hypothetical document is empty")

            hypothetical_doc = self.clean_text(hypothetical_doc.strip())

            # Extract token usage if available
            token_usage = None
            if hasattr(response, "usage") and response.usage:
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

        except AuthenticationError as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise RuntimeError(
                "LiteLLM authentication failed. Check LITELLM_PROXY_API_KEY"
            ) from e
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {str(e)}")
            raise RuntimeError(
                f"Rate limit exceeded after retries: {str(e)}"
            ) from e
        except Timeout as e:
            logger.error(f"API timeout: {str(e)}")
            raise RuntimeError(f"API timeout after retries: {str(e)}") from e
        except (APIError, ValueError) as e:
            logger.error(f"Generation failed: {str(e)}")
            raise RuntimeError(
                f"Hypothetical document generation failed: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(
                f"Unexpected error during generation: {str(e)}", exc_info=True
            )
            raise RuntimeError(
                f"Unexpected error generating hypothetical document: {str(e)}"
            ) from e

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
            Tuple of (hypothetical_document, token_usage)

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
