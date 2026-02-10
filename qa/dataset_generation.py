from openai import (
    AsyncAzureOpenAI,
    APIError,
    APITimeoutError,
    RateLimitError,
)
from typing import List, Dict, Any
from core.storage_chunk import PostgresClient
from core.schemas import TextChunk
from core.config import get_chat_settings
import json
import asyncio
import html
from datetime import datetime, timezone

import logging

logger = logging.getLogger(__name__)


class GenerateDatasetOpenAI:
    def __init__(self, max_concurrent_requests: int = 50):
        """
        Initialize dataset generator with async support.

        Args:
            max_concurrent_requests: Maximum concurrent API requests (default: 50)
                                   Azure OpenAI allows 2500 RPM, so 50 concurrent is safe
        """
        try:
            settings = get_chat_settings()

            # Validate required configuration
            if not settings.DEPLOYMENT_NAME:
                raise ValueError("DEPLOYMENT_NAME is not configured")
            if not settings.ENDPOINT:
                raise ValueError("ENDPOINT is not configured")
            if not settings.API_VERSION:
                raise ValueError("API_VERSION is not configured")
            if not settings.MAX_TOKENS or settings.MAX_TOKENS <= 0:
                raise ValueError("MAX_TOKENS must be a positive integer")
            if not settings.TEMPERATURE:
                raise ValueError("TEMPERATURE must be configured")
            if not settings.COMPLETION_TOKENS or settings.COMPLETION_TOKENS <= 0:
                raise ValueError("COMPLETION_TOKENS must be a positive integer")

            self.deployment_name = settings.DEPLOYMENT_NAME
            self.max_tokens = settings.MAX_TOKENS
            self.temperature = settings.TEMPERATURE
            self.completion_tokens = settings.COMPLETION_TOKENS
            self.max_concurrent_requests = max_concurrent_requests

            # Initialize async client
            try:
                api_key = settings.API_KEY.get_secret_value()
                if not api_key:
                    raise ValueError("API_KEY is empty")

                self.client = AsyncAzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=settings.ENDPOINT,
                    api_version=settings.API_VERSION,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize Azure OpenAI client: {str(e)}"
                ) from e

        except Exception as e:
            logger.error(f"Failed to initialize GenerateDatasetOpenAI: {str(e)}")
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

    def load_chunks(self, postgresclient: PostgresClient) -> List[TextChunk]:
        # Input validation
        chunks = postgresclient.get_all_chunks()
        return chunks

    async def process_single_chunk(
        self, chunk: TextChunk, idx: int, total: int
    ) -> Dict[str, Any] | None:
        """
        Process a single chunk to generate QA pairs.

        Args:
            chunk: TextChunk to process
            idx: Current index (1-based)
            total: Total number of chunks

        Returns:
            Dictionary with QA pairs and metadata, or None if failed
        """
        try:
            # Clean the text before processing
            clean_content = self.clean_text(chunk.text_content)

            prompt = f"""You are an expert at generating question-answer pairs from IT knowledge base articles for training and evaluation purposes.

Context: This chunk comes from the IT knowledge base of the University of Tennessee at Chattanooga (UTC), a higher education institution. UTC's mascot is the "Mocs," and many campus services and systems use "mocs" in their naming (e.g., MocsNet, MocsID, MyMocsNet, etc.).

Given the following chunk from a UTC IT knowledge base article, generate exactly 3 question-answer pairs that can ONLY be answered using the information provided in this chunk.

IMPORTANT: Use proper apostrophes ('), quotation marks ("), and dashes (- or —) in your responses. Do NOT use unicode escape sequences like \\u2019 or \\u201c.

<chunk>
{clean_content}
</chunk>

For each question-answer pair:
1. The question should be worded differently from the others (vary phrasing, perspective, or specificity)
2. The answer must be directly derivable from the chunk content
3. Assess whether the chunk provides sufficient context to properly answer the question

Return your response as valid JSON in the following format:
{{
    "qa_pairs": [
        {{
            "question": "The question text",
            "answer": "The answer derived from the chunk",
            "sufficient_context": true
        }},
        {{
            "question": "A differently worded question",
            "answer": "The answer derived from the chunk",
            "sufficient_context": true
        }},
        {{
            "question": "Another variation of the question",
            "answer": "The answer derived from the chunk",
            "sufficient_context": false
        }}
    ],
    "chunk_summary": "Brief 1-2 sentence summary of what the chunk covers",
    "overall_quality": "high|medium|low"
}}

Guidelines:
- Set "sufficient_context" to true only if the chunk contains enough information to fully and accurately answer the question without requiring external knowledge
- Set "sufficient_context" to false if the answer would be incomplete, ambiguous, or require assumptions beyond what's in the chunk
- "overall_quality" reflects how useful this chunk is for generating meaningful QA pairs (high = rich content, low = sparse/fragmented)
- Questions can have the same general intent if the chunk only supports one type of question, but must use different wording
- Avoid yes/no questions; prefer questions that require substantive answers
- Questions should reflect how a UTC student, faculty, or staff member might naturally ask for help
- Use natural punctuation: regular apostrophes ('), quotes ("), and dashes (-)

Return only the JSON object, no additional text."""

            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You follow the user's instructions exactly. Use proper punctuation with regular apostrophes and quotes.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=self.completion_tokens,
            )

            # Parse the response
            response_text = response.choices[0].message.content

            # Create metadata-rich output
            result = {
                "chunk_id": str(chunk.chunk_id),
                "parent_article_id": str(chunk.parent_article_id),
                "chunk_sequence": chunk.chunk_sequence,
                "source_url": str(chunk.source_url),
                "token_count": chunk.token_count,
                "last_modified_date": chunk.last_modified_date.isoformat(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "response": response_text,
            }

            if idx % 10 == 0:
                print(f"Processed {idx}/{total} chunks...")

            return result

        except RateLimitError as e:
            logger.warning(f"Rate limit for chunk {idx}, will retry: {str(e)}")
            # Wait and retry once
            await asyncio.sleep(2)
            return await self.process_single_chunk(chunk, idx, total)

        except (APITimeoutError, APIError) as e:
            logger.error(f"API error for chunk {idx}: {str(e)}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error processing chunk {idx}: {str(e)}")
            return None

    async def build_dataset(self, chunks: List[TextChunk]) -> List[Dict[str, Any]]:
        """
        Build dataset with parallel processing for faster generation.

        Args:
            chunks: List of TextChunk objects to process

        Returns:
            List of dictionaries containing QA pairs with metadata
        """
        print(f"Starting parallel processing of {len(chunks)} chunks...")
        print(f"Max concurrent requests: {self.max_concurrent_requests}")

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def process_with_semaphore(chunk: TextChunk, idx: int, total: int):
            async with semaphore:
                return await self.process_single_chunk(chunk, idx, total)

        # Process all chunks concurrently with rate limiting
        tasks = [
            process_with_semaphore(chunk, idx, len(chunks))
            for idx, chunk in enumerate(chunks, 1)
        ]

        results = await asyncio.gather(*tasks)

        # Filter out None results (failed requests)
        json_output = [r for r in results if r is not None]

        print(
            f"\nCompleted: {len(json_output)}/{len(chunks)} chunks processed successfully"
        )

        return json_output

    def save_to_file(self, json_output: List[Dict[str, Any]], filepath: str) -> None:
        """
        Save QA pairs to JSONL (JSON Lines) format.

        Each line is a complete JSON object, making it:
        - Easy to stream/process line-by-line
        - More resilient (one corrupt line doesn't break the file)
        - Easier to append new data

        Args:
            json_output: List of dictionaries to save
            filepath: Path to save the JSONL file
        """
        with open(filepath, "w", encoding="utf-8") as jsonl_file:
            for item in json_output:
                # Write each item as a single line of JSON
                jsonl_file.write(json.dumps(item, ensure_ascii=False) + "\n")
        return None


async def main_async():
    """Async main function for parallel processing."""
    try:
        print("Initializing dataset generation...")
        generator = GenerateDatasetOpenAI(max_concurrent_requests=50)

        print("Connecting to database...")
        postgresclient = PostgresClient()

        print("Loading chunks from database...")
        chunks = generator.load_chunks(postgresclient)
        print(f"Found {len(chunks)} chunks to process")

        if not chunks:
            print(
                "No chunks found in database. Please run the ingestion pipeline first."
            )
            return None

        print("\nGenerating QA pairs with parallel processing...")
        start_time = datetime.now(timezone.utc)

        json_output = await generator.build_dataset(chunks)

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        print(f"\nGeneration completed in {duration:.2f} seconds")
        print(f"Average: {duration / len(chunks):.2f} seconds per chunk")
        print(f"Generated {len(json_output)} QA pair responses")

        print("\nSaving to file...")
        generator.save_to_file(json_output=json_output, filepath=r"data/qa_pairs.jsonl")
        print("Successfully saved QA pairs to data/qa_pairs.jsonl")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        raise

    return None


def main():
    """Synchronous wrapper for async main."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the async main function
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
