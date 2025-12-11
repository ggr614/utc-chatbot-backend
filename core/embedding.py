from openai import AzureOpenAI, OpenAIError, APIError, APITimeoutError, RateLimitError, AuthenticationError
import boto3
from botocore.exceptions import ClientError, BotoCoreError, NoCredentialsError
from typing import List
from core.config import get_settings
import json
from utils.tokenizer import Tokenizer
import time
import logging

logger = logging.getLogger(__name__)

class GenerateEmbeddingsOpenAI:
    def __init__(self):
        try:
            settings = get_settings()

            # Validate required configuration
            if not settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME:
                raise ValueError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME is not configured")
            if not settings.AZURE_OPENAI_EMBED_ENDPOINT:
                raise ValueError("AZURE_OPENAI_EMBED_ENDPOINT is not configured")
            if not settings.AZURE_OPENAI_API_VERSION:
                raise ValueError("AZURE_OPENAI_API_VERSION is not configured")
            if not settings.AZURE_MAX_TOKENS or settings.AZURE_MAX_TOKENS <= 0:
                raise ValueError("AZURE_MAX_TOKENS must be a positive integer")
            if not settings.AZURE_EMBED_DIM or settings.AZURE_EMBED_DIM <= 0:
                raise ValueError("AZURE_EMBED_DIM must be a positive integer")

            self.deployment_name = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
            self.max_tokens = settings.AZURE_MAX_TOKENS
            self.expected_dim = settings.AZURE_EMBED_DIM
            self.tokenizer = Tokenizer()

            # Initialize client with error handling
            try:
                api_key = settings.AZURE_OPENAI_API_KEY.get_secret_value()
                if not api_key:
                    raise ValueError("AZURE_OPENAI_API_KEY is empty")

                self.client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=settings.AZURE_OPENAI_EMBED_ENDPOINT,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Azure OpenAI client: {str(e)}") from e

        except Exception as e:
            logger.error(f"Failed to initialize GenerateEmbeddingsOpenAI: {str(e)}")
            raise
    
    def generate_embedding(self, chunk: str) -> List[float]:
        # Input validation
        if not chunk or not chunk.strip():
            raise ValueError("Chunk cannot be empty.")

        # Token count validation with error handling
        try:
            token_count = self.tokenizer.num_tokens_from_string(chunk)
            if token_count > self.max_tokens:
                raise ValueError(
                    f"Chunk is too long. Max tokens allowed: {self.max_tokens} "
                    f"but chunk had {token_count} tokens."
                )
        except Exception as e:
            logger.error(f"Token counting failed: {str(e)}")
            raise RuntimeError(f"Failed to count tokens: {str(e)}") from e

        # Retry logic for API calls
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=chunk,
                    model=self.deployment_name
                )

                # Validate response structure
                if not response or not hasattr(response, 'data'):
                    raise ValueError("Invalid response structure: missing 'data' attribute")

                if not response.data or len(response.data) == 0:
                    raise ValueError("Response data is empty")

                if not hasattr(response.data[0], 'embedding'):
                    raise ValueError("Response data[0] missing 'embedding' attribute")

                embeddings = response.data[0].embedding

                if not embeddings:
                    raise ValueError("Embeddings list is empty")

                # Validate embedding type and dimension
                if not isinstance(embeddings, list):
                    raise ValueError(f"Expected embeddings to be a list, got {type(embeddings)}")

                if not all(isinstance(x, (int, float)) for x in embeddings):
                    raise ValueError("Embeddings contain non-numeric values")

                if len(embeddings) != self.expected_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch. Expected {self.expected_dim}, "
                        f"got {len(embeddings)}"
                    )

                return embeddings

            except AuthenticationError as e:
                logger.error(f"Authentication failed: {str(e)}")
                raise RuntimeError(f"Azure OpenAI authentication failed: {str(e)}") from e

            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts")
                    raise RuntimeError(f"Rate limit exceeded: {str(e)}") from e

            except APITimeoutError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"API timeout, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"API timeout after {max_retries} attempts")
                    raise RuntimeError(f"API timeout: {str(e)}") from e

            except APIError as e:
                # For server errors (5xx), retry
                status_code = getattr(e, 'status_code', None)
                if status_code and 500 <= status_code < 600:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Server error {status_code}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
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

            except Exception as e:
                logger.error(f"Unexpected error generating embeddings: {str(e)}")
                raise RuntimeError(f"Unexpected error: {str(e)}") from e

        raise RuntimeError(f"Failed to generate embeddings after {max_retries} attempts")
    
class GenerateEmbeddingsAWS:
    def __init__(self):
        try:
            settings = get_settings()

            # Validate required configuration
            if not settings.AWS_EMBED_MODEL_ID:
                raise ValueError("AWS_EMBED_MODEL_ID is not configured")
            if not settings.AWS_REGION:
                raise ValueError("AWS_REGION is not configured")
            if not settings.AWS_MAX_TOKENS or settings.AWS_MAX_TOKENS <= 0:
                raise ValueError("AWS_MAX_TOKENS must be a positive integer")
            if not settings.AWS_EMBED_DIM or settings.AWS_EMBED_DIM <= 0:
                raise ValueError("AWS_EMBED_DIM must be a positive integer")

            self.max_tokens = settings.AWS_MAX_TOKENS
            self.model = settings.AWS_EMBED_MODEL_ID
            self.expected_dim = settings.AWS_EMBED_DIM
            self.tokenizer = Tokenizer()

            # Initialize AWS client with error handling
            try:
                aws_access_key = settings.AWS_ACCESS_KEY_ID.get_secret_value()
                aws_secret_key = settings.AWS_SECRET_ACCESS_KEY.get_secret_value()

                if not aws_access_key:
                    raise ValueError("AWS_ACCESS_KEY_ID is empty")
                if not aws_secret_key:
                    raise ValueError("AWS_SECRET_ACCESS_KEY is empty")

                self.client = boto3.client(
                    "bedrock-runtime",
                    aws_access_key_id=aws_access_key,
                    region_name=settings.AWS_REGION,
                    aws_secret_access_key=aws_secret_key
                )
            except NoCredentialsError as e:
                raise RuntimeError(f"AWS credentials not found: {str(e)}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to initialize AWS Bedrock client: {str(e)}") from e

        except Exception as e:
            logger.error(f"Failed to initialize GenerateEmbeddingsAWS: {str(e)}")
            raise
    def generate_embedding(self, chunk: str) -> List[float]:
        # Input validation
        if not chunk or not chunk.strip():
            raise ValueError("Chunk cannot be empty.")

        # Token count validation with error handling
        try:
            token_count = self.tokenizer.num_tokens_from_string(chunk)
            if token_count > self.max_tokens:
                raise ValueError(
                    f"Chunk is too long. Max tokens allowed: {self.max_tokens} "
                    f"but chunk had {token_count} tokens."
                )
        except Exception as e:
            logger.error(f"Token counting failed: {str(e)}")
            raise RuntimeError(f"Failed to count tokens: {str(e)}") from e

        # Retry logic for API calls
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                # Prepare request body
                body = json.dumps({
                    "texts": [chunk],
                    "input_type": "search_document",
                    "embedding_types": ["float"]
                })

                # Invoke AWS Bedrock model
                response = self.client.invoke_model(
                    body=body,
                    modelId=self.model,
                    accept="application/json",
                    contentType="application/json"
                )

                # Validate response structure
                if not response:
                    raise ValueError("Empty response from AWS Bedrock")

                if "body" not in response:
                    raise ValueError("Response missing 'body' key")

                # Parse response body
                try:
                    response_body = json.loads(response["body"].read())
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse response JSON: {str(e)}") from e

                # Extract embeddings
                if "embeddings" not in response_body:
                    raise ValueError(f"No 'embeddings' key in response: {response_body}")

                embeddings_data = response_body["embeddings"]
                if embeddings_data is None:
                    raise ValueError(f"Embeddings is None in response: {response_body}")

                # Handle nested structure
                if not isinstance(embeddings_data, dict):
                    raise ValueError(f"Expected embeddings to be a dict, got {type(embeddings_data)}")

                if "float" not in embeddings_data:
                    raise ValueError(f"No 'float' key in embeddings: {embeddings_data}")

                float_embeddings = embeddings_data["float"]
                if not isinstance(float_embeddings, list) or len(float_embeddings) == 0:
                    raise ValueError(f"Expected non-empty list of embeddings, got {float_embeddings}")

                embeddings = float_embeddings[0]

                # Validate embedding type and dimension
                if not isinstance(embeddings, list):
                    raise ValueError(f"Expected embeddings to be a list, got {type(embeddings)}")

                if not all(isinstance(x, (int, float)) for x in embeddings):
                    raise ValueError("Embeddings contain non-numeric values")

                if len(embeddings) != self.expected_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch. Expected {self.expected_dim}, "
                        f"got {len(embeddings)}"
                    )

                return embeddings

            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', str(e))

                # Handle specific AWS error codes
                if error_code == 'ThrottlingException' or error_code == 'TooManyRequestsException':
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"AWS throttling, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"AWS throttling after {max_retries} attempts")
                        raise RuntimeError(f"AWS rate limit exceeded: {error_message}") from e

                elif error_code == 'AccessDeniedException' or error_code == 'UnauthorizedException':
                    logger.error(f"AWS authentication/authorization failed: {error_message}")
                    raise RuntimeError(f"AWS authentication failed: {error_message}") from e

                elif error_code == 'ServiceUnavailableException' or error_code == 'InternalServerException':
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"AWS service error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"AWS service unavailable after {max_retries} attempts")
                        raise RuntimeError(f"AWS service error: {error_message}") from e

                elif error_code == 'ModelTimeoutException':
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Model timeout, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Model timeout after {max_retries} attempts")
                        raise RuntimeError(f"AWS model timeout: {error_message}") from e

                else:
                    logger.error(f"AWS client error ({error_code}): {error_message}")
                    raise RuntimeError(f"AWS Bedrock error ({error_code}): {error_message}") from e

            except BotoCoreError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Network error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Network error after {max_retries} attempts: {str(e)}")
                    raise RuntimeError(f"AWS network error: {str(e)}") from e

            except ValueError as e:
                # Don't retry on validation errors
                logger.error(f"Validation error: {str(e)}")
                raise

            except Exception as e:
                logger.error(f"Unexpected error generating embeddings: {str(e)}")
                raise RuntimeError(f"Unexpected error: {str(e)}") from e

        raise RuntimeError(f"Failed to generate embeddings after {max_retries} attempts")