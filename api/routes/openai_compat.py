"""
OpenAI-compatible API endpoint for Open-webui integration.

This allows Open-webui to connect to our backend as if it were an OpenAI API.
We intercept requests, add RAG context, then forward to the actual LLM (Ollama).
"""

import logging
import time
import json
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any, Generator
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.middleware.auth import verify_api_key
from core.storage_vector import OpenAIVectorStorage
from core.embedding import GenerateEmbeddingsOpenAI

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["openai-compatible"])

# Ollama endpoint
OLLAMA_URL = "http://localhost:11434"

# System prompt for UTC Helpdesk
UTC_SYSTEM_PROMPT = """You are a helpful IT support assistant for the University of Tennessee at Chattanooga (UTC).
Use the provided knowledge base articles to answer questions accurately.
If the articles don't contain relevant information, say so and suggest contacting the IT Help Desk at (423) 425-4000.
Always be friendly and professional."""


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


# Lazy-loaded clients
_embed_client = None
_vector_store = None


def get_embed_client():
    global _embed_client
    if _embed_client is None:
        _embed_client = GenerateEmbeddingsOpenAI()
    return _embed_client


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = OpenAIVectorStorage()
    return _vector_store


def log_query(query: str, user_id: Optional[str] = None, cache_result: str = "miss", latency_ms: int = 0):
    """Log query to database for analytics."""
    try:
        from api.services.analytics import AnalyticsService
        analytics = AnalyticsService()

        with analytics.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_logs (raw_query, cache_result, latency_ms, user_id, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    """,
                    (query, cache_result, latency_ms, user_id),
                )
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to log query: {e}")


def retrieve_rag_context(query: str, top_k: int = 3, min_similarity: float = 0.3) -> Optional[str]:
    """Retrieve relevant KB articles for the query."""
    try:
        embed_client = get_embed_client()
        query_embedding = embed_client.generate_embedding(query)

        vector_store = get_vector_store()
        results = vector_store.search_similar_vectors(
            query_vector=query_embedding,
            limit=top_k,
            min_similarity=min_similarity,
        )

        if not results:
            return None

        context_parts = []
        for i, article in enumerate(results, 1):
            context_parts.append(
                f"### Article {i} (Relevance: {article['similarity']:.0%})\n"
                f"**URL:** {article['source_url']}\n\n"
                f"{article['text_content']}\n"
            )

        return "\n---\n".join(context_parts)

    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return None


def call_ollama(model: str, messages: List[Dict], stream: bool = False) -> Dict:
    """Call Ollama API."""
    url = f"{OLLAMA_URL}/api/chat"

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def stream_ollama(model: str, messages: List[Dict]) -> Generator[str, None, None]:
    """Stream response from Ollama."""
    url = f"{OLLAMA_URL}/api/chat"

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=300) as response:
        for line in response:
            if line:
                chunk = json.loads(line.decode("utf-8"))
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    # Format as SSE for OpenAI compatibility
                    delta = {"role": "assistant", "content": content}
                    data = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(data)}\n\n"

                if chunk.get("done"):
                    data = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    yield "data: [DONE]\n\n"


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, req: Request):
    """
    OpenAI-compatible chat completions endpoint.

    Adds RAG context before forwarding to Ollama.
    """
    start_time = time.time()

    # Strip utc-rag- prefix if present (we add this to distinguish our models)
    actual_model = request.model.replace("utc-rag-", "")
    logger.info(f"Chat completion request for model: {request.model} -> {actual_model}")

    # Extract the latest user message for RAG
    user_query = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_query = msg.content
            break

    # Try to get user info from various sources
    user_id = (
        req.headers.get("X-User-Email") or
        req.headers.get("X-Forwarded-User") or
        req.headers.get("X-User-Id") or
        req.headers.get("X-OpenWebUI-User-Email") or
        req.headers.get("X-OpenWebUI-User-Name") or
        "anonymous"
    )

    # Don't use the API key as user ID
    if user_id == "anonymous":
        logger.info("No user identification header found - using 'anonymous'")

    # Build messages list
    messages = []

    # Retrieve RAG context if we have a user query
    if user_query:
        context = retrieve_rag_context(user_query)

        if context:
            system_content = f"""{UTC_SYSTEM_PROMPT}

## Relevant Knowledge Base Articles:

{context}

---
Use the above articles to help answer the user's question. Cite the article URL when referencing specific information."""
        else:
            system_content = UTC_SYSTEM_PROMPT

        messages.append({"role": "system", "content": system_content})

    # Add the rest of the messages (skip any existing system message)
    for msg in request.messages:
        if msg.role != "system":
            messages.append({"role": msg.role, "content": msg.content})

    try:
        if request.stream:
            # Log the query for streaming requests
            if user_query:
                latency_ms = int((time.time() - start_time) * 1000)
                log_query(user_query, user_id=user_id, cache_result="miss", latency_ms=latency_ms)

            return StreamingResponse(
                stream_ollama(actual_model, messages),
                media_type="text/event-stream",
            )
        else:
            result = call_ollama(actual_model, messages, stream=False)

            response_content = result.get("message", {}).get("content", "")

            # Log the query
            if user_query:
                latency_ms = int((time.time() - start_time) * 1000)
                log_query(user_query, user_id=user_id, cache_result="miss", latency_ms=latency_ms)

            return ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=Message(role="assistant", content=response_content),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                ),
            )

    except urllib.error.URLError as e:
        logger.error(f"Ollama connection error: {e}")
        raise HTTPException(status_code=503, detail=f"LLM service unavailable: {e}")
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """List available models (fetches from Ollama, prefixed with utc-rag-)."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

            models = []
            for model in data.get("models", []):
                # Prefix with utc-rag- so users know this goes through our RAG pipeline
                models.append({
                    "id": f"utc-rag-{model['name']}",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "utc-chatbot",
                })

            return {"object": "list", "data": models}

    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return {"object": "list", "data": []}
