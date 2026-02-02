"""
Test script for AWS Bedrock Cohere Reranker connection.

Tests:
1. Load AWS Bedrock Cohere Reranker settings from .env
2. Initialize boto3 client with credentials
3. Test reranker with sample query and documents
4. Display reranked results

Usage:
    python test_cohere_reranker.py
"""

import json
import sys
from core.config import get_aws_reranker_settings
from utils.logger import get_logger

logger = get_logger(__name__)


def test_reranker_connection():
    """Test connection to AWS Bedrock Cohere Reranker."""

    logger.info("=" * 70)
    logger.info("Testing AWS Bedrock Cohere Reranker Connection")
    logger.info("=" * 70)

    # Step 1: Load settings
    logger.info("\n[1/4] Loading AWS Bedrock Reranker settings from .env...")
    try:
        settings = get_aws_reranker_settings()
        logger.info(f"✓ Settings loaded successfully")
        logger.info(f"  - Region: {settings.REGION_NAME}")
        logger.info(f"  - Reranker ARN: {settings.RERANKER_ARN}")
        logger.info(f"  - Access Key ID: {settings.ACCESS_KEY_ID.get_secret_value()[:8]}...")
    except Exception as e:
        logger.error(f"✗ Failed to load settings: {e}")
        logger.error("Make sure you have AWS_* variables set in your .env file:")
        logger.error("  - AWS_ACCESS_KEY_ID")
        logger.error("  - AWS_SECRET_ACCESS_KEY")
        logger.error("  - AWS_REGION_NAME")
        logger.error("  - AWS_RERANKER_ARN")
        return False

    # Step 2: Initialize boto3 client
    logger.info("\n[2/4] Initializing boto3 Bedrock Runtime client...")
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError

        bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=settings.REGION_NAME,
            aws_access_key_id=settings.ACCESS_KEY_ID.get_secret_value(),
            aws_secret_access_key=settings.SECRET_ACCESS_KEY.get_secret_value()
        )
        logger.info(f"✓ Boto3 client initialized successfully")

    except ImportError:
        logger.error("✗ boto3 is not installed. Install it with: pip install boto3")
        return False
    except NoCredentialsError:
        logger.error("✗ AWS credentials not found or invalid")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to initialize boto3 client: {e}")
        return False

    # Step 3: Test reranker with sample data
    logger.info("\n[3/4] Testing reranker with sample query and documents...")

    # Sample query and documents for testing
    test_query = "How do I reset my password?"
    test_documents = [
        "To change your email address, go to Settings > Account > Email and click Update.",
        "Password reset instructions: Click 'Forgot Password' on the login page, enter your email, and follow the link sent to you.",
        "Our office hours are Monday-Friday 9am-5pm EST. Contact us at support@example.com.",
        "To reset your password, navigate to the login screen and select 'Reset Password'. You will receive an email with instructions.",
        "For VPN connection issues, ensure your firewall allows UDP port 1194 and contact IT support.",
    ]

    logger.info(f"Query: '{test_query}'")
    logger.info(f"Documents to rerank: {len(test_documents)}")

    # Prepare request body for Cohere reranker (AWS Bedrock format)
    request_body = {
        "query": test_query,
        "documents": test_documents,
        "top_n": len(test_documents),  # Return all documents, reranked
        "api_version": 2  # Required by AWS Bedrock (integer)
    }

    try:
        # Invoke the reranker model
        response = bedrock_runtime.invoke_model(
            modelId=settings.RERANKER_ARN,
            body=json.dumps(request_body),
            contentType='application/json',
            accept='application/json'
        )

        # Parse response
        response_body = json.loads(response['body'].read())

        logger.info(f"✓ Reranker invoked successfully")

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"✗ AWS ClientError: {error_code}")
        logger.error(f"  Message: {error_message}")

        if error_code == 'AccessDeniedException':
            logger.error("  Hint: Check that your AWS credentials have bedrock:InvokeModel permissions")
        elif error_code == 'ResourceNotFoundException':
            logger.error("  Hint: Check that your RERANKER_ARN is correct")
        elif error_code == 'ValidationException':
            logger.error("  Hint: Check request body format or model ID")

        return False
    except Exception as e:
        logger.error(f"✗ Failed to invoke reranker: {e}")
        return False

    # Step 4: Display results
    logger.info("\n[4/4] Displaying reranked results...")
    logger.info("-" * 70)

    try:
        results = response_body.get('results', [])

        if not results:
            logger.warning("No results returned from reranker")
            return False

        logger.info(f"Reranked {len(results)} documents by relevance:\n")

        for i, result in enumerate(results, 1):
            index = result.get('index', 'N/A')
            relevance_score = result.get('relevance_score', 0.0)

            # Get document text (either from 'document' or from original list)
            if 'document' in result and 'text' in result['document']:
                doc_text = result['document']['text']
            elif isinstance(index, int) and index < len(test_documents):
                doc_text = test_documents[index]
            else:
                doc_text = "N/A"

            logger.info(f"Rank {i}:")
            logger.info(f"  Score: {relevance_score:.4f}")
            logger.info(f"  Index: {index}")
            logger.info(f"  Text: {doc_text[:100]}{'...' if len(doc_text) > 100 else ''}")
            logger.info("")

        # Check if results are properly ranked
        scores = [r.get('relevance_score', 0) for r in results]
        is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

        if is_sorted:
            logger.info("✓ Results are properly sorted by relevance score (descending)")
        else:
            logger.warning("⚠ Results are NOT sorted by relevance score")

        logger.info("-" * 70)
        logger.info("\n✓ ALL TESTS PASSED! AWS Bedrock Cohere Reranker is working correctly.")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to parse results: {e}")
        logger.debug(f"Raw response: {response_body}")
        return False


def main():
    """Main entry point."""
    try:
        success = test_reranker_connection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
