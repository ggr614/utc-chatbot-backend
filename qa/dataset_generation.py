from openai import (
    AzureOpenAI,
    OpenAIError,
    APIError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
)
from typing import List, Dict, Any
from core.storage_chunk import PostgresClient
from core.schemas import TextChunk
from core.config import get_settings
import json
from utils.tokenizer import Tokenizer

import logging

logger = logging.getLogger(__name__)


class GenerateDatasetOpenAI:
    def __init__(self):
        try:
            settings = get_settings()

            # Validate required configuration
            if not settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME:
                raise ValueError("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME is not configured")
            if not settings.AZURE_OPENAI_CHAT_ENDPOINT:
                raise ValueError("AZURE_OPENAI_CHAT_ENDPOINT is not configured")
            if not settings.AZURE_OPENAI_CHAT_API_VERSION:
                raise ValueError("AZURE_OPENAI_CHAT_API_VERSION is not configured")
            if (
                not settings.AZURE_OPENAI_CHAT_MAX_TOKENS
                or settings.AZURE_OPENAI_CHAT_MAX_TOKENS <= 0
            ):
                raise ValueError(
                    "AZURE_OPENAI_CHAT_MAX_TOKENS must be a positive integer"
                )
            if not settings.AZURE_OPENAI_CHAT_TEMPERATURE:
                raise ValueError("AZURE_OPENAI_CHAT_TEMPERATURE must be configured")
            if (
                not settings.AZURE_OPENAI_CHAT_COMPLETION_TOKENS
                or settings.AZURE_OPENAI_CHAT_COMPLETION_TOKENS <= 0
            ):
                raise ValueError(
                    "AZURE_OPENAI_CHAT_COMPLETION_TOKENS must be a positive integer"
                )

            self.deployment_name = settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
            self.max_tokens = settings.AZURE_OPENAI_CHAT_MAX_TOKENS
            self.temperature = settings.AZURE_OPENAI_CHAT_TEMPERATURE
            self.completion_tokens = settings.AZURE_OPENAI_CHAT_COMPLETION_TOKENS
            self.tokenizer = Tokenizer()

            # Initialize client with error handling
            try:
                api_key = settings.AZURE_OPENAI_CHAT_API_KEY.get_secret_value()
                if not api_key:
                    raise ValueError("AZURE_OPENAI_CHAT_API_KEY is empty")

                self.client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=settings.AZURE_OPENAI_CHAT_ENDPOINT,
                    api_version=settings.AZURE_OPENAI_CHAT_API_VERSION,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize Azure OpenAI client: {str(e)}"
                ) from e

        except Exception as e:
            logger.error(f"Failed to initialize GenerateDatasetOpenAI: {str(e)}")
            raise

    def load_chunks(self, postgresclient: PostgresClient) -> List[TextChunk]:
        # Input validation
        chunks = postgresclient.get_all_chunks()
        return chunks

    def build_dataset(self, chunks: List[TextChunk]) -> List[Dict[str, Any]]:
        json_output = []
        for idx, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {idx}/{len(chunks)}...")
            prompt = f"""You are an expert at generating question-answer pairs from IT knowledge base articles for training and evaluation purposes.

Context: This chunk comes from the IT knowledge base of the University of Tennessee at Chattanooga (UTC), a higher education institution. UTC's mascot is the "Mocs," and many campus services and systems use "mocs" in their naming (e.g., MocsNet, MocsID, MyMocsNet, etc.).

Given the following chunk from a UTC IT knowledge base article, generate exactly 3 question-answer pairs that can ONLY be answered using the information provided in this chunk.

<chunk>
{chunk.text_content}
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

Return only the JSON object, no additional text."""
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You follow the user's instructions exactly.",
                        },
                        {"role": "user", "content": f"{prompt}"},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.completion_tokens,
                )
            except AuthenticationError as e:
                logger.error(f"Authentication failed: {str(e)}")
                raise RuntimeError(
                    f"Azure OpenAI authentication failed: {str(e)}"
                ) from e

            except RateLimitError as e:
                logger.error(f"Rate limit hit: {str(e)}")
                raise RuntimeError(f"Rate limit exceeded: {str(e)}") from e

            except APITimeoutError as e:
                logger.error(f"API timeout: {str(e)}")
                raise RuntimeError(f"API timeout: {str(e)}") from e

            except APIError as e:
                # For server errors (5xx), log and skip this chunk
                status_code = getattr(e, "status_code", None)
                if status_code and 500 <= status_code < 600:
                    logger.warning(
                        f"Server error (5xx) for chunk {idx}, skipping: {str(e)}"
                    )
                    continue
                logger.error(f"API error: {str(e)}")
                raise RuntimeError(f"Azure OpenAI API error: {str(e)}") from e

            except OpenAIError as e:
                logger.error(f"OpenAI client error: {str(e)}")
                raise RuntimeError(f"OpenAI client error: {str(e)}") from e

            except ValueError as e:
                # Don't retry on validation errors
                logger.error(f"Validation error: {str(e)}")
                raise

            json_output.append(response.choices[0].message.content)
        return json_output

    def save_to_file(self, json_output: List[Dict[str, Any]], filepath: str) -> None:
        with open(filepath, "w") as json_file:
            json.dump(json_output, json_file, indent=4)
        return None


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        print("Initializing dataset generation...")
        generator = GenerateDatasetOpenAI()

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

        print("Generating QA pairs (this may take a while)...")
        json_output = generator.build_dataset(chunks)
        print(f"Generated {len(json_output)} QA pair responses")

        print("Saving to file...")
        generator.save_to_file(json_output=json_output, filepath=r"data/qa_pairs.json")
        print("Successfully saved QA pairs to data/qa_pairs.json")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        raise

    return None


if __name__ == "__main__":
    main()
