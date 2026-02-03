"""
title: RAG Helpdesk Filter
author: David Wood
version: 1.1.0
date: 2025-01-28
description: Filter function that augments any model with RAG context from the
             Helpdesk knowledge base. Intercepts user queries, retrieves relevant
             documents via hybrid search, and injects them into the conversation
             before the request reaches the LLM.
license: MIT
requirements: requests
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Callable, Awaitable
import requests
import json
import time
import traceback


class Filter:
    """
    RAG Helpdesk Filter - Augments any Open WebUI model with knowledge base context.

    This filter:
    1. Intercepts the user's message (inlet)
    2. Searches the RAG Helpdesk API for relevant documents
    3. Injects retrieved context into the system prompt
    4. Passes the augmented request to the selected model

    Works with ANY model configured in Open WebUI (Ollama, OpenAI, etc.)
    """

    class Valves(BaseModel):
        """Configuration options for the RAG Helpdesk Filter."""

        # Filter Control
        priority: int = Field(
            default=0, description="Filter priority (lower runs first)"
        )

        # RAG API Settings
        RAG_API_BASE_URL: str = Field(
            default="http://localhost:8000",
            description="Base URL of the RAG Helpdesk API",
        )
        RAG_API_KEY: str = Field(
            default="",
            description="API key for the RAG Helpdesk API (X-API-Key header)",
        )

        # Search Parameters
        TOP_K: int = Field(
            default=5,
            description="Number of document chunks to retrieve (1-100)",
            ge=1,
            le=100,
        )
        FUSION_METHOD: str = Field(
            default="rrf",
            description="Fusion method: 'rrf' (Reciprocal Rank Fusion) or 'weighted'",
        )
        RRF_K: int = Field(
            default=60, description="RRF constant (for 'rrf' method)", ge=1
        )
        BM25_WEIGHT: float = Field(
            default=0.5,
            description="BM25 weight for 'weighted' fusion (0.0-1.0)",
            ge=0.0,
            le=1.0,
        )
        MIN_BM25_SCORE: Optional[float] = Field(
            default=None, description="Minimum BM25 score threshold (optional)"
        )
        MIN_VECTOR_SIMILARITY: Optional[float] = Field(
            default=None, description="Minimum vector similarity threshold (optional)"
        )

        # Context Settings
        ENABLE_CITATIONS: bool = Field(
            default=True, description="Include source URLs in context"
        )
        MAX_CONTEXT_TOKENS: int = Field(
            default=4000, description="Maximum approximate tokens for RAG context"
        )

        # Behavior
        REQUEST_TIMEOUT: int = Field(
            default=30, description="Timeout in seconds for RAG API requests"
        )
        GRACEFUL_DEGRADATION: bool = Field(
            default=True,
            description="Continue without RAG on API errors (True) or return error (False)",
        )
        DEBUG_MODE: bool = Field(
            default=False, description="Enable debug logging to console"
        )

        # LLM Response Logging
        ENABLE_LLM_RESPONSE_LOGGING: bool = Field(
            default=False,
            description="Log LLM-generated responses to backend for analytics (requires working query logging)",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name = "RAG Helpdesk Filter"
        # Instance variables to pass data from inlet to outlet
        self._query_log_id = None
        self._search_response_data = None

    def _get_user_query(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract the latest user message from the conversation."""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                # Handle multimodal content (list of parts)
                if isinstance(content, list):
                    text_parts = [
                        part.get("text", "")
                        for part in content
                        if part.get("type") == "text"
                    ]
                    return " ".join(text_parts)
                return content
        return None

    def _search_documents(
        self, query: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Call the RAG Helpdesk API hybrid search endpoint."""
        url = f"{self.valves.RAG_API_BASE_URL.rstrip('/')}/api/v1/search/hybrid"

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.valves.RAG_API_KEY,
        }

        payload = {
            "query": query,
            "top_k": self.valves.TOP_K,
            "fusion_method": self.valves.FUSION_METHOD,
        }

        # Add fusion-specific parameters
        if self.valves.FUSION_METHOD == "rrf":
            payload["rrf_k"] = self.valves.RRF_K
        elif self.valves.FUSION_METHOD == "weighted":
            payload["bm25_weight"] = self.valves.BM25_WEIGHT

        # Add optional thresholds
        if self.valves.MIN_BM25_SCORE is not None:
            payload["min_bm25_score"] = self.valves.MIN_BM25_SCORE
        if self.valves.MIN_VECTOR_SIMILARITY is not None:
            payload["min_vector_similarity"] = self.valves.MIN_VECTOR_SIMILARITY
        if user_id:
            payload["user_id"] = user_id

        # DEBUG: Log full request details
        if self.valves.DEBUG_MODE:
            print(f"[RAG Filter] ========== API REQUEST ==========")
            print(f"[RAG Filter] URL: {url}")
            print(
                f"[RAG Filter] Headers: {json.dumps({k: v[:20] + '...' if k == 'X-API-Key' and len(v) > 20 else v for k, v in headers.items()})}"
            )
            print(f"[RAG Filter] Payload: {json.dumps(payload, indent=2)}")
            print(f"[RAG Filter] Timeout: {self.valves.REQUEST_TIMEOUT}s")

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.valves.REQUEST_TIMEOUT
            )

            # DEBUG: Log response details
            if self.valves.DEBUG_MODE:
                print(f"[RAG Filter] ========== API RESPONSE ==========")
                print(f"[RAG Filter] Status Code: {response.status_code}")
                print(f"[RAG Filter] Response Headers: {dict(response.headers)}")
                print(
                    f"[RAG Filter] Response Body (first 500 chars): {response.text[:500]}"
                )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError as e:
            if self.valves.DEBUG_MODE:
                print(f"[RAG Filter] ========== CONNECTION ERROR ==========")
                print(f"[RAG Filter] Could not connect to: {url}")
                print(f"[RAG Filter] Error type: {type(e).__name__}")
                print(f"[RAG Filter] Error details: {str(e)}")
                print(f"[RAG Filter] Traceback:\n{traceback.format_exc()}")
            raise

        except requests.exceptions.Timeout as e:
            if self.valves.DEBUG_MODE:
                print(f"[RAG Filter] ========== TIMEOUT ERROR ==========")
                print(
                    f"[RAG Filter] Request timed out after {self.valves.REQUEST_TIMEOUT}s"
                )
                print(f"[RAG Filter] URL: {url}")
            raise

        except requests.exceptions.HTTPError as e:
            if self.valves.DEBUG_MODE:
                print(f"[RAG Filter] ========== HTTP ERROR ==========")
                print(f"[RAG Filter] Status Code: {e.response.status_code}")
                print(f"[RAG Filter] Response Body: {e.response.text}")
                print(f"[RAG Filter] Request URL: {url}")
            raise

        except Exception as e:
            if self.valves.DEBUG_MODE:
                print(f"[RAG Filter] ========== UNEXPECTED ERROR ==========")
                print(f"[RAG Filter] Error type: {type(e).__name__}")
                print(f"[RAG Filter] Error message: {str(e)}")
                print(f"[RAG Filter] Traceback:\n{traceback.format_exc()}")
            raise

    def _format_context(self, search_response: Dict[str, Any]) -> str:
        """Format search results into a context string for injection."""
        results = search_response.get("results", [])
        if not results:
            return ""

        context_parts = []
        total_chars = 0
        max_chars = self.valves.MAX_CONTEXT_TOKENS * 4  # ~4 chars per token

        for i, chunk in enumerate(results, 1):
            text_content = chunk.get("text_content", "").strip()
            source_url = chunk.get("source_url", "")

            # Build document entry
            doc_lines = [f"[Document {i}]"]
            if self.valves.ENABLE_CITATIONS and source_url:
                doc_lines.append(f"Source: {source_url}")
            doc_lines.append(f"Content:\n{text_content}")

            doc_entry = "\n".join(doc_lines)

            # Check token limit
            if total_chars + len(doc_entry) > max_chars:
                if self.valves.DEBUG_MODE:
                    print(f"[RAG Filter] Truncating at doc {i} due to token limit")
                break

            context_parts.append(doc_entry)
            total_chars += len(doc_entry)

        return "\n\n---\n\n".join(context_parts)

    def _inject_context_into_messages(
        self, messages: List[Dict[str, Any]], context: str
    ) -> List[Dict[str, Any]]:
        """Inject RAG context into the message list via system prompt."""

        rag_instruction = f"""You have access to a knowledge base of IT helpdesk support articles. 
Use the following retrieved documents to help answer the user's question:

<knowledge_base>
{context}
</knowledge_base>

INSTRUCTIONS:
- If the answer is found in the documents above, use that information to respond accurately.
- If the documents don't contain relevant information, say so and provide general guidance if appropriate.
- When citing information from the documents, reference the source URL if available.
- Be concise but thorough in your response."""

        new_messages = []
        system_found = False

        for msg in messages:
            if msg.get("role") == "system":
                # Append RAG context to existing system prompt
                original_content = msg.get("content", "")
                new_messages.append(
                    {
                        "role": "system",
                        "content": f"{original_content}\n\n{rag_instruction}",
                    }
                )
                system_found = True
            else:
                new_messages.append(msg.copy())

        # If no system message exists, insert one at the beginning
        if not system_found:
            new_messages.insert(0, {"role": "system", "content": rag_instruction})

        return new_messages

    async def inlet(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Inlet filter: Intercepts the request before it reaches the LLM.

        This method:
        1. Extracts the user's query
        2. Searches the RAG Helpdesk API
        3. Injects retrieved context into the messages
        4. Returns the modified request body

        Args:
            body: The request body containing messages
            __user__: User information from Open WebUI
            __event_emitter__: Optional callback for status updates

        Returns:
            Modified request body with RAG context injected
        """
        start_time = time.time()

        # Always log when inlet is triggered (even without debug mode)
        print(f"[RAG Filter] ========== INLET TRIGGERED ==========")
        print(f"[RAG Filter] Debug Mode: {self.valves.DEBUG_MODE}")
        print(f"[RAG Filter] RAG API URL: {self.valves.RAG_API_BASE_URL}")
        print(
            f"[RAG Filter] API Key configured: {'Yes' if self.valves.RAG_API_KEY else 'NO - MISSING!'}"
        )

        if self.valves.DEBUG_MODE:
            print(f"[RAG Filter] User: {__user__}")
            print(f"[RAG Filter] Body keys: {body.keys()}")

        # Check if RAG API is configured
        if not self.valves.RAG_API_KEY:
            error_msg = (
                "RAG API key not configured - please set RAG_API_KEY in filter settings"
            )
            print(f"[RAG Filter] ERROR: {error_msg}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"⚠️ {error_msg}", "done": True},
                    }
                )
            return body

        # Extract user query
        messages = body.get("messages", [])
        user_query = self._get_user_query(messages)

        if not user_query:
            print("[RAG Filter] No user query found in messages, passing through")
            if self.valves.DEBUG_MODE:
                print(f"[RAG Filter] Messages: {json.dumps(messages, indent=2)}")
            return body

        print(
            f"[RAG Filter] Query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}"
        )

        # Emit status update
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "🔍 Searching knowledge base...",
                        "done": False,
                    },
                }
            )

        # Search for relevant documents
        context = ""
        num_docs = 0
        error_msg = None
        error_detail = None

        try:
            user_id = __user__.get("id") if __user__ else None
            search_response = self._search_documents(user_query, user_id)
            context = self._format_context(search_response)
            num_docs = len(search_response.get("results", []))

            # Capture query_log_id for LLM response logging in outlet
            query_log_id = search_response.get("query_log_id")
            if query_log_id and self.valves.ENABLE_LLM_RESPONSE_LOGGING:
                # Store in instance variables for access in outlet (don't pollute body sent to LLM)
                self._query_log_id = query_log_id
                self._search_response_data = {
                    "num_documents": num_docs,
                    "results": search_response.get("results", []),
                }
                if self.valves.DEBUG_MODE:
                    print(
                        f"[RAG Filter] Stored query_log_id {query_log_id} for response logging"
                    )

            print(f"[RAG Filter] SUCCESS: Retrieved {num_docs} documents")
            if self.valves.DEBUG_MODE:
                latency = search_response.get("latency_ms", "?")
                print(f"[RAG Filter] API Latency: {latency}ms")
                print(f"[RAG Filter] Context length: {len(context)} chars")

        except requests.exceptions.Timeout:
            error_msg = "RAG API timeout"
            error_detail = f"Request to {self.valves.RAG_API_BASE_URL} timed out after {self.valves.REQUEST_TIMEOUT}s"
        except requests.exceptions.ConnectionError as e:
            error_msg = "Cannot connect to RAG API"
            error_detail = f"URL: {self.valves.RAG_API_BASE_URL} - {str(e)}"
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                error_msg = "Invalid RAG API key"
                error_detail = (
                    "Check that RAG_API_KEY matches the key configured in the RAG API"
                )
            elif e.response.status_code == 404:
                error_msg = "RAG API endpoint not found"
                error_detail = f"URL {self.valves.RAG_API_BASE_URL}/api/v1/search/hybrid returned 404"
            else:
                error_msg = f"RAG API HTTP {e.response.status_code}"
                error_detail = (
                    e.response.text[:200] if e.response.text else "No response body"
                )
        except json.JSONDecodeError as e:
            error_msg = "Invalid JSON response from RAG API"
            error_detail = str(e)
        except Exception as e:
            error_msg = f"Unexpected error: {type(e).__name__}"
            error_detail = str(e)

        # Handle errors
        if error_msg:
            print(f"[RAG Filter] ========== ERROR ==========")
            print(f"[RAG Filter] Error: {error_msg}")
            print(f"[RAG Filter] Detail: {error_detail}")

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"❌ {error_msg}", "done": True},
                    }
                )

            if not self.valves.GRACEFUL_DEGRADATION:
                # Inject error message so user sees it in the response
                body["messages"] = messages + [
                    {
                        "role": "system",
                        "content": f"[RAG FILTER ERROR]\n{error_msg}\nDetail: {error_detail}\n\nProceeding without knowledge base context.",
                    }
                ]
                return body

            # Graceful degradation: continue without RAG
            print(
                "[RAG Filter] Graceful degradation enabled, continuing without RAG context"
            )
            return body

        # Inject context if we have any
        if context:
            body["messages"] = self._inject_context_into_messages(messages, context)

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"✅ Found {num_docs} relevant documents",
                            "done": True,
                        },
                    }
                )
            print(f"[RAG Filter] Injected context from {num_docs} documents")
        else:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "ℹ️ No matching documents found",
                            "done": True,
                        },
                    }
                )
            print("[RAG Filter] No matching documents found")

        elapsed = int((time.time() - start_time) * 1000)
        print(f"[RAG Filter] Inlet completed in {elapsed}ms")
        print(f"[RAG Filter] =====================================")

        return body

    async def outlet(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Outlet filter: Intercepts the response after the LLM generates it.

        If ENABLE_LLM_RESPONSE_LOGGING is True, logs the full LLM response
        to the backend for analytics and evaluation.

        Args:
            body: The response body from the LLM
            __user__: User information from Open WebUI
            __event_emitter__: Optional callback for status updates

        Returns:
            Response body (unchanged)
        """
        # Check if LLM response logging is enabled
        if not self.valves.ENABLE_LLM_RESPONSE_LOGGING:
            if self.valves.DEBUG_MODE:
                print("[RAG Filter] LLM response logging disabled, skipping outlet")
            return body

        # Extract query_log_id from instance variable (set in inlet)
        query_log_id = self._query_log_id
        if not query_log_id:
            if self.valves.DEBUG_MODE:
                print("[RAG Filter] No query_log_id found, skipping response logging")
            return body

        # Extract LLM response text from body
        try:
            messages = body.get("messages", [])
            if not messages:
                if self.valves.DEBUG_MODE:
                    print("[RAG Filter] No messages in response body")
                return body

            last_message = messages[-1]
            if last_message.get("role") != "assistant":
                if self.valves.DEBUG_MODE:
                    print(
                        f"[RAG Filter] Last message not from assistant: {last_message.get('role')}"
                    )
                return body

            response_text = last_message.get("content", "")
            if not response_text:
                if self.valves.DEBUG_MODE:
                    print("[RAG Filter] Empty response text")
                return body

            # Handle multimodal content (list of parts)
            if isinstance(response_text, list):
                text_parts = [
                    part.get("text", "")
                    for part in response_text
                    if part.get("type") == "text"
                ]
                response_text = " ".join(text_parts)

            if not response_text.strip():
                if self.valves.DEBUG_MODE:
                    print("[RAG Filter] Response text is empty after extraction")
                return body

            print(f"[RAG Filter] Logging LLM response for query_log_id {query_log_id}")
            print(f"[RAG Filter] Response length: {len(response_text)} chars")

            # Extract search data for citations from instance variable
            search_data = self._search_response_data or {}

            # Build citations JSONB
            citations = None
            if search_data:
                results = search_data.get("results", [])
                citations = {
                    "num_documents_used": len(results),
                    "source_urls": [
                        r.get("source_url") for r in results if r.get("source_url")
                    ],
                    "chunk_ids": [
                        str(r.get("chunk_id")) for r in results if r.get("chunk_id")
                    ],
                }

            # Extract model info from body if available
            model_name = body.get("model")  # Open WebUI includes this

            # Prepare request payload
            url = f"{self.valves.RAG_API_BASE_URL.rstrip('/')}/api/v1/query-logs/{query_log_id}/response"

            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.valves.RAG_API_KEY,
            }

            payload = {
                "response_text": response_text,
                "model_name": model_name,
                # Note: LLM latency not easily available in outlet
                # Token counts not available unless model API returns them
            }

            if citations:
                payload["citations"] = citations

            if self.valves.DEBUG_MODE:
                print("[RAG Filter] ========== LLM RESPONSE LOGGING ==========")
                print(f"[RAG Filter] URL: {url}")
                print(f"[RAG Filter] Payload: {json.dumps(payload, indent=2)[:500]}...")

            # Log response (best-effort, non-blocking)
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.valves.REQUEST_TIMEOUT,
                )
                response.raise_for_status()

                result = response.json()
                print(
                    f"[RAG Filter] LLM response logged successfully: ID {result.get('id')}"
                )

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 409:
                    # Already logged - this is OK (idempotent)
                    print(
                        f"[RAG Filter] LLM response already logged for query_log_id {query_log_id}"
                    )
                else:
                    print(
                        f"[RAG Filter] Failed to log LLM response (HTTP {e.response.status_code}): {e.response.text[:200]}"
                    )

            except Exception as e:
                # Don't fail the outlet on logging errors
                print(
                    f"[RAG Filter] Failed to log LLM response: {type(e).__name__}: {str(e)}"
                )
                if self.valves.DEBUG_MODE:
                    print(f"[RAG Filter] Traceback:\n{traceback.format_exc()}")

        except Exception as e:
            # Catch-all: don't fail outlet
            print(
                f"[RAG Filter] Error in outlet LLM logging: {type(e).__name__}: {str(e)}"
            )
            if self.valves.DEBUG_MODE:
                print(f"[RAG Filter] Traceback:\n{traceback.format_exc()}")

        finally:
            # Clean up instance variables after logging
            self._query_log_id = None
            self._search_response_data = None

        return body
