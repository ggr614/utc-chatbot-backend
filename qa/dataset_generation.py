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
import re
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

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """
        Strip markdown code fences from LLM response text.

        Azure OpenAI sometimes wraps JSON responses in ```json...``` fences
        despite being told to return raw JSON.

        Args:
            text: Response text that may be wrapped in markdown code fences

        Returns:
            Text with code fences removed, or original text if no fences found
        """
        pattern = r"^```(?:json)?\s*\n?(.*?)\n?\s*```$"
        match = re.match(pattern, text.strip(), re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _validate_response(response_text: str) -> bool:
        """
        Validate that response text contains valid JSON with qa_pairs.

        Args:
            response_text: The (already fence-stripped) response text

        Returns:
            True if valid, False otherwise
        """
        try:
            parsed = json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            return False

        qa_pairs = parsed.get("qa_pairs")
        if not isinstance(qa_pairs, list) or len(qa_pairs) == 0:
            return False

        for pair in qa_pairs:
            if not isinstance(pair, dict):
                return False
            if not pair.get("question") or not pair.get("answer"):
                return False

        return True

    def load_chunks(self, postgresclient: PostgresClient) -> List[TextChunk]:
        # Input validation
        chunks = postgresclient.get_all_chunks()
        return chunks

    async def process_single_chunk(
        self,
        chunk: TextChunk,
        idx: int,
        total: int,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Process a single chunk to generate QA pairs with bounded retry.

        Retries on: RateLimitError, APITimeoutError, content filter,
        empty response, and invalid JSON. Does not retry on generic
        APIError or unexpected exceptions.

        Args:
            chunk: TextChunk to process
            idx: Current index (1-based)
            total: Total number of chunks
            max_retries: Maximum retry attempts (default: 3)

        Returns:
            Dictionary with QA pairs and metadata (status field indicates
            success or failure reason)
        """
        clean_content = self.clean_text(chunk.text_content)
        retry_delay = 1  # Base delay in seconds
        last_failure_reason = "unknown"

        prompt = f"""You are an expert at generating question-answer pairs from IT knowledge base articles for training and evaluation purposes.

Context: This chunk comes from the IT knowledge base of the University of Tennessee at Chattanooga (UTC), a higher education institution. UTC's mascot is the "Mocs," and many campus services and systems use "mocs" in their naming (e.g., MocsNet, MocsID, MyMocsNet, etc.).

Given the following chunk from a UTC IT knowledge base article, generate exactly 3 question-answer pairs that can ONLY be answered using the information provided in this chunk.

IMPORTANT: Use proper apostrophes ('), quotation marks ("), and dashes (- or —) in your responses. Do NOT use unicode escape sequences like \\u2019 or \\u201c.

<chunk>
{clean_content}
</chunk>

## Question Persona

Generate questions as if asked by an entry-level IT helpdesk technician with the following profile:

**Background**
- College-aged (18-23), likely a student worker or recent hire
- No prior helpdesk or IT support experience
- No formal training in troubleshooting methodologies
- Unfamiliar with IT terminology, ticket systems, and escalation paths
- Limited understanding of networking (IP, DNS, DHCP, VLANs), Active Directory, and identity management concepts

**Communication Style**
- Brief, informal, and sometimes vague
- Describes symptoms rather than root causes ("it's not working" vs. specific errors)
- Often omits system details, error codes, or steps already attempted
- Hesitant to use technical language — paraphrases or guesses at terms
- May ask overly broad questions when unsure where to start

**Mindset**
- Lacks confidence — second-guesses themselves and fears breaking things
- Tends to jump to asking for help before fully investigating
- Uncertain about scope of responsibility and when to escalate
- Unfamiliar with SLAs, priority frameworks, and documentation expectations

## Question Difficulty Tiers

Each of the 3 questions MUST come from a different difficulty tier:

1. **Simple lookup** — A straightforward question where the answer is a single fact or step directly stated in the chunk (e.g., "How do I reset a MocsID password?")
2. **Procedural** — A question requiring synthesis of multiple steps or details from the chunk (e.g., "A student can't connect to MocsNet on their laptop — what should I walk them through?")
3. **Contextual reasoning** — A question that requires understanding the broader purpose or implications of the information, or applying it to a scenario not explicitly described (e.g., "A student says their internet works in the library but not their dorm — could this be a MocsNet issue or something else?")

For each question-answer pair:
1. The question should reflect the persona's communication style and knowledge level for its assigned difficulty tier
2. The answer must be directly derivable from the chunk content
3. Assess whether the chunk provides sufficient context to properly answer the question

Return your response as valid JSON in the following format:
{{
    "qa_pairs": [
        {{
            "question": "The question text",
            "answer": "The answer derived from the chunk",
            "difficulty": "simple_lookup",
            "sufficient_context": true
        }},
        {{
            "question": "A differently worded question",
            "answer": "The answer derived from the chunk",
            "difficulty": "procedural",
            "sufficient_context": true
        }},
        {{
            "question": "Another variation of the question",
            "answer": "The answer derived from the chunk",
            "difficulty": "contextual_reasoning",
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
- Avoid yes/no questions; prefer questions that require substantive answers
- Simple lookup questions should use casual, direct phrasing
- Procedural questions can be slightly longer, describing a scenario the technician encountered
- Contextual reasoning questions should reflect genuine confusion or uncertainty about how to apply the information
- Use natural punctuation: regular apostrophes ('), quotes ("), and dashes (-)

Return only the JSON object, no additional text."""

        for attempt in range(max_retries):
            try:
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

                # Check finish_reason
                finish_reason = response.choices[0].finish_reason

                if finish_reason == "content_filter":
                    last_failure_reason = "content_filter"
                    logger.warning(
                        f"Content filter triggered for chunk {idx} "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2**attempt))
                        continue
                    break

                if finish_reason == "length":
                    last_failure_reason = "length_exceeded"
                    logger.warning(
                        f"Response truncated for chunk {idx} "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2**attempt))
                        continue
                    break

                # Extract response content
                response_text = response.choices[0].message.content

                if not response_text or not response_text.strip():
                    last_failure_reason = "empty_response"
                    logger.warning(
                        f"Empty response for chunk {idx} "
                        f"(attempt {attempt + 1}/{max_retries}, "
                        f"finish_reason={finish_reason})"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2**attempt))
                        continue
                    break

                # Strip markdown fences and validate JSON
                response_text = self._strip_markdown_fences(response_text)

                if not self._validate_response(response_text):
                    last_failure_reason = "invalid_json"
                    logger.warning(
                        f"Invalid JSON for chunk {idx} "
                        f"(attempt {attempt + 1}/{max_retries}): "
                        f"'{response_text[:100]}...'"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2**attempt))
                        continue
                    break

                # Success
                if idx % 10 == 0:
                    print(f"Processed {idx}/{total} chunks...")

                return {
                    "chunk_id": str(chunk.chunk_id),
                    "parent_article_id": str(chunk.parent_article_id),
                    "chunk_sequence": chunk.chunk_sequence,
                    "source_url": str(chunk.source_url),
                    "token_count": chunk.token_count,
                    "last_modified_date": chunk.last_modified_date.isoformat(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "response": response_text,
                    "status": "success",
                    "finish_reason": finish_reason,
                }

            except RateLimitError as e:
                last_failure_reason = "rate_limit"
                logger.warning(
                    f"Rate limit for chunk {idx} "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))
                    continue
                break

            except APITimeoutError as e:
                last_failure_reason = "api_timeout"
                logger.warning(
                    f"API timeout for chunk {idx} "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))
                    continue
                break

            except APIError as e:
                last_failure_reason = (
                    f"api_error_{getattr(e, 'status_code', 'unknown')}"
                )
                logger.error(f"API error for chunk {idx}: {e}")
                break

            except Exception as e:
                last_failure_reason = f"exception_{type(e).__name__}"
                logger.error(f"Unexpected error processing chunk {idx}: {e}")
                break

        # All retries exhausted or non-retryable error
        logger.warning(
            f"Failed chunk {idx} after {max_retries} attempts: {last_failure_reason}"
        )
        return {
            "chunk_id": str(chunk.chunk_id),
            "parent_article_id": str(chunk.parent_article_id),
            "chunk_sequence": chunk.chunk_sequence,
            "source_url": str(chunk.source_url),
            "token_count": chunk.token_count,
            "last_modified_date": chunk.last_modified_date.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "response": "",
            "status": last_failure_reason,
            "finish_reason": None,
        }

    async def build_dataset(
        self,
        chunks: List[TextChunk],
        min_token_count: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Build dataset with parallel processing, retry queue, and statistics.

        Args:
            chunks: List of TextChunk objects to process
            min_token_count: Skip chunks below this token count (default: 0)

        Returns:
            List of successful result dictionaries containing QA pairs
        """
        # Filter by minimum token count
        if min_token_count > 0:
            original_count = len(chunks)
            chunks = [c for c in chunks if c.token_count >= min_token_count]
            skipped = original_count - len(chunks)
            if skipped > 0:
                print(
                    f"Skipped {skipped} chunks below {min_token_count} token threshold"
                )

        print(f"Starting parallel processing of {len(chunks)} chunks...")
        print(f"Max concurrent requests: {self.max_concurrent_requests}")

        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def process_with_semaphore(chunk: TextChunk, idx: int, total: int):
            async with semaphore:
                return await self.process_single_chunk(chunk, idx, total)

        # === Pass 1: Initial processing ===
        tasks = [
            process_with_semaphore(chunk, idx, len(chunks))
            for idx, chunk in enumerate(chunks, 1)
        ]
        results = await asyncio.gather(*tasks)

        successes = []
        failures = []
        for i, result in enumerate(results):
            if result is not None and result.get("status") == "success":
                successes.append(result)
            else:
                failures.append((i, chunks[i], result))

        print(f"\nPass 1 complete: {len(successes)} succeeded, {len(failures)} failed")

        # === Pass 2: Retry failed chunks ===
        if failures:
            print(f"\nRetrying {len(failures)} failed chunks (pass 2)...")
            retry_tasks = [
                process_with_semaphore(chunk, original_idx + 1, len(chunks))
                for original_idx, chunk, _ in failures
            ]
            retry_results = await asyncio.gather(*retry_tasks)

            recovered = 0
            final_failures = []
            for (_, _, original_failure), retry_result in zip(failures, retry_results):
                if retry_result is not None and retry_result.get("status") == "success":
                    successes.append(retry_result)
                    recovered += 1
                else:
                    final_failures.append(retry_result or original_failure)

            print(
                f"Pass 2 complete: recovered {recovered}, "
                f"{len(final_failures)} still failed"
            )
        else:
            final_failures = []

        self._print_generation_stats(
            total_chunks=len(chunks),
            successes=successes,
            failures=final_failures,
        )

        return successes

    @staticmethod
    def _print_generation_stats(
        total_chunks: int,
        successes: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
    ) -> None:
        """Print generation statistics summary."""
        failure_reasons: Dict[str, int] = {}
        for f in failures:
            reason = f.get("status", "unknown") if f is not None else "none_returned"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        print("\n" + "=" * 60)
        print("GENERATION STATISTICS")
        print("=" * 60)
        print(f"Total chunks:      {total_chunks}")
        print(
            f"Successful:        {len(successes)} "
            f"({100 * len(successes) / max(total_chunks, 1):.1f}%)"
        )
        print(
            f"Failed:            {len(failures)} "
            f"({100 * len(failures) / max(total_chunks, 1):.1f}%)"
        )

        if failure_reasons:
            print("\nFailure breakdown:")
            for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count}")

        finish_reasons: Dict[str, int] = {}
        for s in successes:
            fr = str(s.get("finish_reason", "unknown"))
            finish_reasons[fr] = finish_reasons.get(fr, 0) + 1

        if finish_reasons:
            print("\nFinish reasons (successes):")
            for reason, count in sorted(finish_reasons.items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count}")

        print("=" * 60)

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


async def main_async(min_token_count: int = 0):
    """Async main function for parallel processing.

    Args:
        min_token_count: Skip chunks below this token count (default: 0)
    """
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

        json_output = await generator.build_dataset(
            chunks, min_token_count=min_token_count
        )

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        print(f"\nGeneration completed in {duration:.2f} seconds")
        print(f"Average: {duration / len(chunks):.2f} seconds per chunk")
        print(f"Generated {len(json_output)} successful QA pair responses")

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
    import argparse

    parser = argparse.ArgumentParser(description="Generate QA dataset from chunks")
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="Skip chunks below this token count (default: 0 = no filter)",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(main_async(min_token_count=args.min_tokens))


if __name__ == "__main__":
    main()
