"""
title: RAG Helpdesk Filter
author: David Wood
version: 3.0.0
date: 2026-02-25
description: Filter function that augments any model with RAG context. Default behavior
             performs hybrid search (BM25 + vector). Use !f for follow-up mode (LLM-only,
             no RAG). Use !help for command help.
license: MIT
requirements: requests
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Callable, Awaitable, Tuple
import requests
import json
import time
import traceback


# System Prompts
SYSTEM_PROMPT_NO_RAG = """# UTC IT Helpdesk Assistant - System Prompt

You are the UTC IT Helpdesk Assistant for Tier 1 support staff at the University of Tennessee at Chattanooga (UTC).

## YOUR AUDIENCE

- **Tier 1 support staff**: May have limited IT exposure. Explain jargon inline when used, e.g., "AD (Active Directory)".
- **Always remote**: Staff cannot physically touch equipment.
- **Frame for IT staff**: You're speaking to the employee, not the end user. Never say "you" meaning the customer.

## YOUR ROLE

Provide helpful, concise answers to questions. This could be:
- General IT knowledge questions
- Clarification or follow-up questions about previous responses
- Troubleshooting guidance
- Explanations of concepts or procedures

Keep responses clear and actionable. When uncertain about UTC-specific procedures, acknowledge that and suggest verifying with documentation or escalating.

## CORE GUIDELINES

1. **Be concise**: Staff may be on calls. Get to the point quickly.
2. **Actionable first**: Lead with what to do, explain why if needed.
3. **Explain jargon**: Define technical terms inline on first use, e.g., "MocsID (UTC's single sign-on username)".
4. **Remote-only**: Never suggest physically handling equipment. Physical access = escalation.
5. **Acknowledge uncertainty**: When unsure about UTC-specific procedures, say so rather than guessing.

## UTC CONTEXT

- **UTC** = University of Tennessee at Chattanooga
- **Mocs** = Mockingbirds (mascot), used in system names (MocsNet, MocsID, MocsMail)
- Common systems: Banner, Canvas, TeamDynamix (TDX), Active Directory (AD), Microsoft 365
- Customers = students, faculty, staff"""

SYSTEM_PROMPT_RAG = """# UTC IT Helpdesk Assistant - System Prompt

You are the UTC IT Helpdesk Assistant, a knowledge assistant for Tier 1 support staff at the University of Tennessee at Chattanooga (UTC). Our mascot is the Mockingbird, so many systems use "Mocs" branding (MocsNet, MocsID, etc.).

## YOUR AUDIENCE

- **Tier 1 support staff**: Beginners with limited IT exposure. Explain jargon inline, e.g., "AD (Active Directory)".
- **Always remote**: Staff cannot physically touch equipment. They may have remote desktop/session access only.
- **Often live with customers**: Students, faculty, or staff may be on the line. Speed matters—actionable info first, elaboration second.
- **Perspective varies**: Queries may be written in 1st, 2nd, or 3rd person depending on source and preference. Interpret accordingly.

## YOUR GOAL

Your primary goal is NOT necessarily to solve the problem, but to **set the employee up for success**:
- If the answer is documented → provide the steps
- If not documented → provide a clear checklist of information to gather and which team to escalate to

## RESPONSE FORMAT

Use this exact structure. Include only the sections that apply.

### Template:

```
## QUICK ANSWER
[1-2 sentences maximum. State whether this can be resolved with documented steps OR needs escalation. Give the single most important action or summary.]

## STEPS
[Numbered list. Only include if the documentation provides a clear procedure. Keep steps concise.]

## IF UNRESOLVED
**Gather:**
- [Specific information the employee should collect]
- [Error messages, screenshots, account details, etc.]

**Escalate to:** [Appropriate team]

## SOURCES
- [URL from retrieved documentation]
```

### Rules:
- **QUICK ANSWER**: Always include. Maximum 2 sentences.
- **STEPS**: Only include if documentation provides explicit procedure. Do not invent steps.
- **IF UNRESOLVED**: Include when documentation doesn't fully address the issue, OR when additional info is needed regardless.
- **SOURCES**: Always include. List every URL from the retrieved documentation you referenced.

## CORE RULES

1. **Documentation-bound**: Only provide solutions explicitly found in the retrieved documentation. Never invent troubleshooting steps.
2. **Actionable first**: Lead with what to do, explain why later (if at all).
3. **Concise**: Staff are reading while on calls. Every word must earn its place.
4. **Jargon with training wheels**: Use proper IT terminology but define it inline on first use, e.g., "Check their MocsID (UTC's single sign-on username)".
5. **Frame for IT staff**: You are speaking to the employee, not the end user. Never say "you" meaning the customer.
6. **Remote-only solutions**: Never suggest physically handling equipment. If physical access is required, that's an escalation.
7. **Source everything**: Always cite the documentation URLs at the bottom.

## EDGE CASES

**Vague query + scattered context**: If the query is unclear AND the retrieved documentation covers multiple unrelated topics, do your best to answer, then provide a better query suggestion:

```
💡 **Better query:** `[suggested query text]`
```

**Partially documented**: If documentation covers part of the issue, state what IS documented vs. what requires escalation.

**Not in documentation**: Do not guess. Provide the information-gathering checklist and escalation path.

## UTC CONTEXT

- **UTC** = University of Tennessee at Chattanooga
- **Mocs** = Mockingbirds (mascot), used in system names (MocsNet, MocsID, MocsMail, etc.)
- Common systems: Banner, Canvas, TeamDynamix (TDX), Genetec (access control), Active Directory (AD)
- Customers = students, faculty, staff (never "users" in responses)"""


class Filter:
    """
    RAG Helpdesk Filter - Augments any Open WebUI model with knowledge base context.

    This filter:
    1. Intercepts the user's message (inlet)
    2. Searches the RAG Helpdesk API for relevant documents (default)
    3. Injects retrieved context into the system prompt
    4. Passes the augmented request to the selected model

    Use !f to skip search for follow-up questions. Use !help for command help.
    Works with ANY model configured in Open WebUI (Ollama, OpenAI, etc.)
    """

    class Valves(BaseModel):
        """Configuration options for the RAG Helpdesk Filter."""

        RAG_API_BASE_URL: str = Field(
            default="http://api:8000",
            description="Base URL of the RAG Helpdesk API (use http://api:8000 in Docker Compose, http://localhost:8000 for local dev)",
        )
        RAG_API_KEY: str = Field(
            default="",
            description="API key for the RAG Helpdesk API (X-API-Key header)",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.name = "RAG Helpdesk Filter"

        # Hardcoded settings (removed from Valves to prevent user misconfiguration)
        self.priority = 0
        self.TOP_K = 5  # Results returned to LLM after reranking
        self.FETCH_TOP_K = 20  # Candidates fetched from each retriever before fusion
        self.RRF_K = 1
        self.MIN_BM25_SCORE = None
        self.MIN_VECTOR_SIMILARITY = 0.0
        self.ENABLE_CITATIONS = True
        self.MAX_CONTEXT_TOKENS = 4000
        self.REQUEST_TIMEOUT = 30
        self.GRACEFUL_DEGRADATION = True
        self.DEBUG_MODE = True
        self.ENABLE_LLM_RESPONSE_LOGGING = True

        # Instance variables to pass data from inlet to outlet
        self._query_log_id = None
        self._search_response_data = None
        self._command_mode = None  # Track mode: "search" or "follow_up"
        self._original_query = None  # Original query before command strip

    def _parse_command(self, query: str) -> Tuple[Optional[str], str]:
        """
        Parse command from user query.

        Commands (case-insensitive, must be at start of query):
        - !help -> Display help information
        - !f <query> -> Follow-up mode (LLM-only, no RAG search)
        - <query> (no command) -> Hybrid search (default)

        Returns:
            tuple[command, cleaned_query]:
            - command: None (default search), "help", or "follow_up"
            - cleaned_query: Query with command stripped (original if no command)
        """
        if not query or not isinstance(query, str):
            return (None, query)

        query_stripped = query.strip()
        query_lower = query_stripped.lower()

        if query_lower == "!help" or query_lower.startswith("!help "):
            return ("help", "")

        elif query_lower == "!f" or query_lower.startswith("!f "):
            cleaned = query_stripped[len("!f") :].strip()
            return ("follow_up", cleaned if cleaned else query_stripped)

        # No command found — default to search
        return (None, query_stripped)

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

    def _get_help_text(self) -> str:
        """Generate help text explaining all available commands."""
        return """# RAG Helpdesk Assistant - Command Help

## How It Works

By default, your question is searched against the UTC IT knowledge base using hybrid search (BM25 keyword matching + vector semantic similarity), and the most relevant documents are provided to the LLM to generate an answer.

## Available Commands

### `<your question>` (no command — default)
**Knowledge base search** - Searches the knowledge base and provides context to the LLM.

**Example:**
```
How do I reset a student's password?
```

**Performance:** ~500ms-1s

---

### `!f <your question>`
**Follow-up mode** - Bypasses the knowledge base entirely. Uses only the LLM's built-in knowledge and conversation history.

**When to use:** For general IT questions, clarifications, or follow-up questions about a previous response that don't need UTC-specific documentation.

**Example:**
```
!f Can you explain that last step in more detail?
```

**Performance:** Very fast (no knowledge base search)

---

### `!help`
**Display this help message**

---

*Need more help? Contact the development team or check the system documentation.*"""

    def _search_documents(
        self, query: str, email: Optional[str] = None
    ) -> Dict[str, Any]:
        """Call the RAG Helpdesk API hybrid search endpoint."""
        if self.DEBUG_MODE:
            print("[RAG Filter] Using standard hybrid search endpoint")

        url = f"{self.valves.RAG_API_BASE_URL.rstrip('/')}/api/v1/search/hybrid"

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.valves.RAG_API_KEY,
        }

        payload = {
            "query": query,
            "top_k": self.TOP_K,
            "fetch_top_k": self.FETCH_TOP_K,
            "rrf_k": self.RRF_K,
        }

        # Add optional thresholds
        if self.MIN_BM25_SCORE is not None:
            payload["min_bm25_score"] = self.MIN_BM25_SCORE
        if self.MIN_VECTOR_SIMILARITY is not None:
            payload["min_vector_similarity"] = self.MIN_VECTOR_SIMILARITY
        if email:
            payload["email"] = email
        # Add command for query logging
        if hasattr(self, "_command_mode") and self._command_mode:
            payload["command"] = self._command_mode

        # DEBUG: Log full request details
        if self.DEBUG_MODE:
            print(f"[RAG Filter] ========== API REQUEST ==========")
            print(f"[RAG Filter] URL: {url}")
            print(
                f"[RAG Filter] Headers: {json.dumps({k: v[:20] + '...' if k == 'X-API-Key' and len(v) > 20 else v for k, v in headers.items()})}"
            )
            print(f"[RAG Filter] Payload: {json.dumps(payload, indent=2)}")
            print(f"[RAG Filter] Timeout: {self.REQUEST_TIMEOUT}s")

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.REQUEST_TIMEOUT
            )

            # DEBUG: Log response details
            if self.DEBUG_MODE:
                print(f"[RAG Filter] ========== API RESPONSE ==========")
                print(f"[RAG Filter] Status Code: {response.status_code}")
                print(f"[RAG Filter] Response Headers: {dict(response.headers)}")
                print(
                    f"[RAG Filter] Response Body (first 500 chars): {response.text[:500]}"
                )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError as e:
            if self.DEBUG_MODE:
                print(f"[RAG Filter] ========== CONNECTION ERROR ==========")
                print(f"[RAG Filter] Could not connect to: {url}")
                print(f"[RAG Filter] Error type: {type(e).__name__}")
                print(f"[RAG Filter] Error details: {str(e)}")
                print(f"[RAG Filter] Traceback:\n{traceback.format_exc()}")
            raise

        except requests.exceptions.Timeout as e:
            if self.DEBUG_MODE:
                print(f"[RAG Filter] ========== TIMEOUT ERROR ==========")
                print(
                    f"[RAG Filter] Request timed out after {self.REQUEST_TIMEOUT}s"
                )
                print(f"[RAG Filter] URL: {url}")
            raise

        except requests.exceptions.HTTPError as e:
            if self.DEBUG_MODE:
                print(f"[RAG Filter] ========== HTTP ERROR ==========")
                print(f"[RAG Filter] Status Code: {e.response.status_code}")
                print(f"[RAG Filter] Response Body: {e.response.text}")
                print(f"[RAG Filter] Request URL: {url}")
            raise

        except Exception as e:
            if self.DEBUG_MODE:
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
        max_chars = self.MAX_CONTEXT_TOKENS * 4  # ~4 chars per token

        for i, chunk in enumerate(results, 1):
            text_content = chunk.get("text_content", "").strip()
            source_url = chunk.get("source_url", "")

            # Build document entry
            doc_lines = [f"[Document {i}]"]
            if self.ENABLE_CITATIONS and source_url:
                doc_lines.append(f"Source: {source_url}")
            doc_lines.append(f"Content:\n{text_content}")

            doc_entry = "\n".join(doc_lines)

            # Check token limit
            if total_chars + len(doc_entry) > max_chars:
                if self.DEBUG_MODE:
                    print(f"[RAG Filter] Truncating at doc {i} due to token limit")
                break

            context_parts.append(doc_entry)
            total_chars += len(doc_entry)

        return "\n\n---\n\n".join(context_parts)

    def _inject_no_rag_system_prompt(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Inject no-RAG system prompt for follow-up mode."""

        new_messages = []
        system_found = False

        for msg in messages:
            if msg.get("role") == "system":
                # Replace existing system prompt with no-RAG prompt
                new_messages.append(
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_NO_RAG,
                    }
                )
                system_found = True
            else:
                new_messages.append(msg.copy())

        # If no system message exists, insert one at the beginning
        if not system_found:
            new_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT_NO_RAG})

        return new_messages

    def _inject_context_into_messages(
        self,
        messages: List[Dict[str, Any]],
        context: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Inject RAG context into the message list via system prompt.

        Args:
            messages: The conversation message list
            context: Formatted document context string
            system_prompt: Optional system prompt from API. If None, falls back to SYSTEM_PROMPT_RAG.
        """

        # Use API-provided prompt if available, otherwise fall back to hardcoded
        base_prompt = system_prompt if system_prompt else SYSTEM_PROMPT_RAG

        # Build the full instruction with context
        rag_instruction = f"""{base_prompt}

You have access to a knowledge base of IT helpdesk support articles.
Use the following retrieved documents to help answer the user's question:

<knowledge_base>
{context}
</knowledge_base>"""

        new_messages = []
        system_found = False

        for msg in messages:
            if msg.get("role") == "system":
                # Replace existing system prompt with RAG prompt
                new_messages.append(
                    {
                        "role": "system",
                        "content": rag_instruction,
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
        2. Searches the RAG Helpdesk API (default) or skips search (!f)
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
        print(f"[RAG Filter] Debug Mode: {self.DEBUG_MODE}")
        print(f"[RAG Filter] RAG API URL: {self.valves.RAG_API_BASE_URL}")
        print(
            f"[RAG Filter] API Key configured: {'Yes' if self.valves.RAG_API_KEY else 'NO - MISSING!'}"
        )

        if self.DEBUG_MODE:
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
            if self.DEBUG_MODE:
                print(f"[RAG Filter] Messages: {json.dumps(messages, indent=2)}")
            return body

        print(
            f"[RAG Filter] Query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}"
        )

        # Parse command from query
        self._original_query = user_query
        command, cleaned_query = self._parse_command(user_query)

        # Map to command_mode for query logging
        if command == "follow_up":
            self._command_mode = "follow_up"
        elif command == "help":
            self._command_mode = None  # help doesn't log
        else:
            self._command_mode = "search"

        print(f"[RAG Filter] Command Mode: {self._command_mode}")

        # Handle !help command - return help text immediately
        if command == "help":
            print("[RAG Filter] Help command detected - returning help text")
            help_text = self._get_help_text()

            # Create a response with help text as assistant message
            # This bypasses the LLM entirely
            body["messages"] = messages + [
                {
                    "role": "assistant",
                    "content": help_text,
                }
            ]

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "ℹ️ Displaying command help",
                            "done": True,
                        },
                    }
                )

            elapsed = int((time.time() - start_time) * 1000)
            print(f"[RAG Filter] Help command completed in {elapsed}ms")
            print("[RAG Filter] =====================================")

            # Return body with help text - this will skip LLM call
            return body

        # Handle follow-up mode (!f = LLM-only, no RAG)
        if command == "follow_up":
            print("[RAG Filter] Follow-up mode - bypassing RAG retrieval")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "💬 Answering follow-up (no search)",
                            "done": True,
                        },
                    }
                )
            # Inject no-RAG system prompt for general helpdesk assistance
            body["messages"] = self._inject_no_rag_system_prompt(messages)
            print("[RAG Filter] Injected no-RAG system prompt")
            return body

        # Default: use cleaned query for hybrid search
        user_query = cleaned_query
        print(
            f"[RAG Filter] Cleaned Query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}"
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
        resolved_system_prompt = None  # Will hold API-provided system prompt

        try:
            email = __user__.get("email") if __user__ else None
            search_response = self._search_documents(user_query, email)
            context = self._format_context(search_response)
            num_docs = len(search_response.get("results", []))

            # Extract system prompt from API metadata
            api_system_prompts = search_response.get("metadata", {}).get(
                "system_prompts", {}
            )
            if api_system_prompts and search_response.get("results"):
                # Use the prompt from the top-ranked article
                top_article_id = str(
                    search_response["results"][0].get("parent_article_id", "")
                )
                resolved_system_prompt = api_system_prompts.get(top_article_id)
                if resolved_system_prompt:
                    print(
                        f"[RAG Filter] Using API system prompt from article {top_article_id[:8]}... ({len(resolved_system_prompt)} chars)"
                    )
                else:
                    print(
                        "[RAG Filter] Top article not in system_prompts dict, using hardcoded fallback"
                    )
            elif self.DEBUG_MODE:
                print(
                    "[RAG Filter] No system_prompts in API metadata, using hardcoded fallback"
                )

            # Capture query_log_id for LLM response logging in outlet
            query_log_id = search_response.get("query_log_id")
            if query_log_id and self.ENABLE_LLM_RESPONSE_LOGGING:
                # Store in instance variables for access in outlet (don't pollute body sent to LLM)
                self._query_log_id = query_log_id
                self._search_response_data = {
                    "num_documents": num_docs,
                    "results": search_response.get("results", []),
                }
                if self.DEBUG_MODE:
                    print(
                        f"[RAG Filter] Stored query_log_id {query_log_id} for response logging"
                    )

            print(f"[RAG Filter] SUCCESS: Retrieved {num_docs} documents")
            if self.DEBUG_MODE:
                latency = search_response.get("latency_ms", "?")
                print(f"[RAG Filter] API Latency: {latency}ms")
                print(f"[RAG Filter] Context length: {len(context)} chars")

        except requests.exceptions.Timeout:
            error_msg = "RAG API timeout"
            error_detail = f"Request to {self.valves.RAG_API_BASE_URL} timed out after {self.REQUEST_TIMEOUT}s"
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

            if not self.GRACEFUL_DEGRADATION:
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
            body["messages"] = self._inject_context_into_messages(
                messages, context, system_prompt=resolved_system_prompt
            )

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
        print("[RAG Filter] =====================================")

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
        try:
            # Check if LLM response logging is enabled
            if self.ENABLE_LLM_RESPONSE_LOGGING:
                # Extract query_log_id from instance variable (set in inlet)
                query_log_id = self._query_log_id
                if query_log_id:
                    # Extract LLM response text from body
                    self._log_llm_response(body, query_log_id)
                elif self.DEBUG_MODE:
                    print(
                        "[RAG Filter] No query_log_id found, skipping response logging"
                    )
            elif self.DEBUG_MODE:
                print("[RAG Filter] LLM response logging disabled")

        finally:
            # Clean up instance variables after logging
            self._query_log_id = None
            self._search_response_data = None
            self._command_mode = None
            self._original_query = None

        return body

    def _log_llm_response(self, body: Dict[str, Any], query_log_id: int) -> None:
        """Log LLM response to backend for analytics."""
        try:
            messages = body.get("messages", [])
            if not messages:
                if self.DEBUG_MODE:
                    print("[RAG Filter] No messages in response body")
                return

            last_message = messages[-1]
            if last_message.get("role") != "assistant":
                if self.DEBUG_MODE:
                    print(
                        f"[RAG Filter] Last message not from assistant: {last_message.get('role')}"
                    )
                return

            response_text = last_message.get("content", "")
            if not response_text:
                if self.DEBUG_MODE:
                    print("[RAG Filter] Empty response text")
                return

            # Handle multimodal content (list of parts)
            if isinstance(response_text, list):
                text_parts = [
                    part.get("text", "")
                    for part in response_text
                    if part.get("type") == "text"
                ]
                response_text = " ".join(text_parts)

            if not response_text.strip():
                if self.DEBUG_MODE:
                    print("[RAG Filter] Response text is empty after extraction")
                return

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

            if self.DEBUG_MODE:
                print("[RAG Filter] ========== LLM RESPONSE LOGGING ==========")
                print(f"[RAG Filter] URL: {url}")
                print(f"[RAG Filter] Payload: {json.dumps(payload, indent=2)[:500]}...")

            # Log response (best-effort, non-blocking)
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.REQUEST_TIMEOUT,
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
                if self.DEBUG_MODE:
                    print(f"[RAG Filter] Traceback:\n{traceback.format_exc()}")

        except Exception as e:
            # Catch-all: don't fail outlet
            print(
                f"[RAG Filter] Error in outlet LLM logging: {type(e).__name__}: {str(e)}"
            )
            if self.DEBUG_MODE:
                print(f"[RAG Filter] Traceback:\n{traceback.format_exc()}")
