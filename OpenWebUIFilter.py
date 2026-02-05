"""
title: RAG Helpdesk Filter
author: David Wood
version: 2.1.0
date: 2025-02-03
description: Filter function that augments any model with RAG context via command-based
             routing. Use !q for RAG retrieval, !qlong for HyDE (slower, higher recall),
             !debug for RAG + debug info, !debuglong for HyDE + debug info, !help for
             command help, or no command to bypass RAG entirely (fast LLM-only responses).
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
        self._search_metadata = None  # Debug metadata (dict or None)
        self._command_mode = (
            None  # Track mode: "bypass", "q", "qlong", "debug", "debuglong"
        )
        self._original_query = None  # Original query before command strip
        self._use_hyde_search = False  # Command-scoped HyDE toggle

    def _parse_command(self, query: str) -> Tuple[Optional[str], str]:
        """
        Parse command from user query.

        Commands (case-insensitive, must be at start of query):
        - !help -> Display help information
        - !debuglong <query> -> Enable HyDE RAG retrieval + debug output
        - !qlong <query> -> Enable HyDE RAG retrieval (slower, higher recall)
        - !debug <query> -> Enable RAG + debug output
        - !q <query> -> Enable RAG retrieval
        - <query> (no command) -> Bypass RAG

        Returns:
            tuple[command, cleaned_query]:
            - command: None, "help", "debuglong", "qlong", "debug", or "q"
            - cleaned_query: Query with command stripped (original if no command)
        """
        if not query or not isinstance(query, str):
            return (None, query)

        # Trim leading/trailing whitespace
        query_stripped = query.strip()

        # Check for commands (case-insensitive)
        query_lower = query_stripped.lower()

        if query_lower == "!help" or query_lower.startswith("!help "):
            return ("help", "")

        elif query_lower == "!debuglong" or query_lower.startswith("!debuglong "):
            cleaned = query_stripped[len("!debuglong") :].strip()
            return ("debuglong", cleaned if cleaned else query_stripped)

        elif query_lower == "!qlong" or query_lower.startswith("!qlong "):
            cleaned = query_stripped[len("!qlong") :].strip()
            return ("qlong", cleaned if cleaned else query_stripped)

        elif query_lower == "!debug" or query_lower.startswith("!debug "):
            cleaned = query_stripped[len("!debug") :].strip()
            return ("debug", cleaned if cleaned else query_stripped)

        elif query_lower == "!q" or query_lower.startswith("!q "):
            cleaned = query_stripped[len("!q") :].strip()
            return ("q", cleaned if cleaned else query_stripped)

        # No command found
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
        """
        Generate help text explaining all available commands.

        Returns formatted markdown with command descriptions and examples.
        """
        return """# RAG Helpdesk Assistant - Command Help

## Available Commands

### `!q <your question>`
**Standard RAG search** - Fast retrieval from the knowledge base using hybrid search (BM25 + vector similarity).

**When to use:** For most questions about UTC IT procedures, systems, or troubleshooting.

**Example:**
```
!q How do I reset a student's password?
```

**Performance:** Fast (~500ms-1s)

---

### `!qlong <your question>`
**HyDE-enhanced RAG search** - Uses Hypothetical Document Embeddings for higher recall. The system first generates what an ideal answer might look like, then searches for documents similar to that ideal answer.

**When to use:** When standard search doesn't find what you need, or for complex/nuanced questions where you want broader retrieval.

**Example:**
```
!qlong What should I do when a faculty member can't access their Banner account after returning from sabbatical?
```

**Performance:** Slower (~1-3s) due to additional LLM call for document generation

**Simple explanation:** Think of `!q` as searching for documents that match your question's keywords and meaning. `!qlong` first imagines what a perfect answer would say, then finds documents similar to that perfect answer. It's like the difference between searching for "how to fix a flat tire" vs. searching for "remove wheel, patch inner tube, reinflate, reattach wheel" - the second approach might find better results even if the original documents don't use your exact phrasing.

---

### `!debug <your question>`
**RAG search with full diagnostic information** - Same as `!q`, but appends detailed debug information about the search process, including timing, relevance scores, and which documents were retrieved.

**When to use:** When testing the system, evaluating search quality, or troubleshooting why certain results were returned.

**Example:**
```
!debug How do I unlock a MocsID account?
```

---

### `!debuglong <your question>`
**HyDE RAG search with full diagnostic information** - Combines `!qlong` and `!debug`. Uses HyDE for enhanced retrieval and appends comprehensive debug information about the search process.

**When to use:** When you need both higher recall (HyDE) and diagnostic information to understand how the search performed.

**Example:**
```
!debuglong What should I do when a faculty member can't access their Banner account after returning from sabbatical?
```

**Performance:** Slower (~1-3s) due to HyDE generation, plus debug output appended

---

### `<your question>` (no command)
**Direct LLM mode** - Bypasses the knowledge base entirely and uses only the LLM's built-in knowledge.

**When to use:** For general IT questions, clarifications, or follow-up questions that don't need UTC-specific documentation.

**Example:**
```
What does DHCP stand for?
```

**Performance:** Very fast (<100ms, no knowledge base search)

---

### `!help`
**Display this help message**

---

## Quick Comparison: !q vs !qlong

| Feature | !q | !qlong |
|---------|-----|--------|
| **Speed** | Fast (~500ms-1s) | Slower (~1-3s) |
| **Search Method** | Direct hybrid search | HyDE (hypothetical document first) |
| **Best For** | Clear, direct questions | Complex, nuanced questions |
| **Recall** | Good | Higher (casts wider net) |
| **When to Use** | First choice for most queries | When !q doesn't find what you need |

**Pro tip:** Start with `!q` for most questions. If the results aren't quite right, try `!qlong` for a second pass with broader retrieval.

---

*Need more help? Contact the development team or check the system documentation.*"""

    def _extract_debug_metadata(
        self, search_response: Dict[str, Any], context: str, cleaned_query: str
    ) -> Dict[str, Any]:
        """
        Extract comprehensive debug metadata from search response.

        Used by !debug and !debuglong commands to capture all relevant information
        for appending to LLM response.
        """
        metadata = search_response.get("metadata", {})
        results = search_response.get("results", [])

        # Basic query info
        debug_data = {
            "raw_query": self._original_query,
            "cleaned_query": cleaned_query,
            "command": self._command_mode,
            # Search configuration
            "search_method": search_response.get("method", "unknown"),
            "top_k": self.valves.TOP_K,
            "fusion_method": self.valves.FUSION_METHOD,
            "rrf_k": self.valves.RRF_K if self.valves.FUSION_METHOD == "rrf" else None,
            # Performance metrics
            "total_latency_ms": search_response.get("latency_ms", 0),
            # Results overview
            "num_results": len(results),
            "context_length_chars": len(context),
            "context_token_estimate": len(context) // 4,  # ~4 chars per token
            "max_context_tokens": self.valves.MAX_CONTEXT_TOKENS,
            # HyDE-specific (if applicable)
            "hyde_enabled": self._use_hyde_search,
            "hypothetical_document": metadata.get("hypothetical_document"),
            "hyde_latency_ms": metadata.get("hyde_latency_ms"),
            "hyde_token_usage": metadata.get("hyde_token_usage"),
            # Reranking details
            "reranked": metadata.get("reranked", False),
            "reranker_latency_ms": metadata.get("reranker_latency_ms"),
            "reranker_status": metadata.get("reranker_status", "unknown"),
            # Results for display (top_k only)
            "results": [
                {
                    "rank": r.get("rank"),
                    "score": round(r.get("score", 0), 4),
                    "chunk_id": str(r.get("chunk_id", ""))[:8] + "...",  # Truncate UUID
                    "source_url": r.get("source_url"),
                    "text_preview": (r.get("text_content", "")[:100] + "...")
                    if len(r.get("text_content", "")) > 100
                    else r.get("text_content", ""),
                    "token_count": r.get("token_count"),
                }
                for r in results[: self.valves.TOP_K]
            ],
            # Additional metadata (BM25 results, vector results, RRF results if available)
            "bm25_results_count": metadata.get("bm25_results_count"),
            "vector_results_count": metadata.get("vector_results_count"),
            "rrf_results_count": metadata.get("rrf_results_count"),
        }

        return debug_data

    def _format_debug_output(self) -> str:
        """
        Format debug metadata into markdown for appending to LLM response.

        Returns markdown string with comprehensive debug information.
        """
        if not self._search_metadata:
            return ""

        md = self._search_metadata

        # Build markdown output
        output = "\n\n---\n\n## 🔍 Debug Information\n\n"

        # Query Details
        output += "### Query Details\n"
        output += f"- **Raw Query**: `{md.get('raw_query', 'N/A')}`\n"
        output += f"- **Cleaned Query**: `{md.get('cleaned_query', 'N/A')}`\n"
        output += f"- **Command**: `{md.get('command', 'N/A')}`\n\n"

        # Search Configuration
        output += "### Search Configuration\n"
        output += f"- **Method**: {md.get('search_method', 'unknown').upper()}"
        if md.get("hyde_enabled"):
            output += " (HyDE - Hypothetical Document Embeddings)"
        output += "\n"
        output += f"- **Top K**: {md.get('top_k', 'N/A')}\n"
        output += f"- **Fusion Method**: {md.get('fusion_method', 'N/A').upper()}\n"
        if md.get("rrf_k"):
            output += f"- **RRF Constant**: {md.get('rrf_k')}\n"
        output += "\n"

        # Performance Metrics
        output += "### Performance Metrics\n"
        output += f"- **Total Latency**: {md.get('total_latency_ms', 'N/A')}ms\n"

        if md.get("hyde_enabled") and md.get("hyde_latency_ms"):
            output += f"- **HyDE Generation**: {md.get('hyde_latency_ms')}ms\n"

        if md.get("reranked") and md.get("reranker_latency_ms"):
            output += f"- **Reranking**: {md.get('reranker_latency_ms')}ms\n"

        output += "\n"

        # HyDE Generation Details (if applicable)
        if md.get("hyde_enabled") and md.get("hypothetical_document"):
            output += "### HyDE Generation\n"
            output += "- **Status**: Success\n"
            output += "- **Hypothetical Document**:\n"
            hypo_doc = md.get("hypothetical_document", "")
            # Truncate if too long
            if len(hypo_doc) > 300:
                hypo_doc = hypo_doc[:300] + "..."
            output += f"  > {hypo_doc}\n\n"

            if md.get("hyde_token_usage"):
                tokens = md.get("hyde_token_usage", {})
                output += (
                    f"- **Token Usage**: {tokens.get('prompt_tokens', '?')} prompt + "
                )
                output += f"{tokens.get('completion_tokens', '?')} completion = "
                output += f"{tokens.get('total_tokens', '?')} total\n\n"

        # Retrieval Results Count
        if md.get("bm25_results_count") or md.get("vector_results_count"):
            output += "### Retrieval Pipeline\n"
            if md.get("bm25_results_count"):
                output += (
                    f"- **BM25 Results**: {md.get('bm25_results_count')} candidates\n"
                )
            if md.get("vector_results_count"):
                output += f"- **Vector Results**: {md.get('vector_results_count')} candidates\n"
            if md.get("rrf_results_count"):
                output += f"- **RRF Fusion**: {md.get('rrf_results_count')} combined results\n"
            output += "\n"

        # Reranking Details
        if md.get("reranked"):
            output += "### Reranking\n"
            output += f"- **Status**: {md.get('reranker_status', 'unknown').upper()}\n"
            if md.get("reranker_latency_ms"):
                output += f"- **Latency**: {md.get('reranker_latency_ms')}ms\n"
            output += "\n"

        # Final Results (injected into context)
        results = md.get("results", [])
        if results:
            output += "### Final Results (Injected into Context)\n\n"
            for result in results:
                output += (
                    f"**{result.get('rank')}. Relevance: {result.get('score')}**\n"
                )
                output += f"- **Chunk ID**: `{result.get('chunk_id')}`\n"
                if result.get("source_url"):
                    output += f"- **Source**: {result.get('source_url')}\n"
                output += f"- **Tokens**: {result.get('token_count', 'N/A')}\n"
                if result.get("text_preview"):
                    output += f"- **Preview**: {result.get('text_preview')}\n"
                output += "\n"

        # Context Injection Stats
        output += "### Context Injection\n"
        output += f"- **Documents Injected**: {md.get('num_results', 0)}\n"
        output += (
            f"- **Context Length**: {md.get('context_length_chars', 0):,} characters\n"
        )
        output += (
            f"- **Estimated Tokens**: ~{md.get('context_token_estimate', 0)} tokens "
        )
        output += f"(max: {md.get('max_context_tokens', 0)})\n"

        output += "\n---\n"

        return output

    def _search_documents(
        self, query: str, user_id: Optional[str] = None, use_hyde: bool = False
    ) -> Dict[str, Any]:
        """Call the RAG Helpdesk API search endpoint (hybrid or hyde)."""
        # Choose endpoint based on command-scoped HyDE toggle
        if use_hyde:
            endpoint = "hyde"
            if self.valves.DEBUG_MODE:
                print(
                    "[RAG Filter] Using HyDE search endpoint (hypothetical document generation)"
                )
        else:
            endpoint = "hybrid"
            if self.valves.DEBUG_MODE:
                print("[RAG Filter] Using standard hybrid search endpoint")

        url = f"{self.valves.RAG_API_BASE_URL.rstrip('/')}/api/v1/search/{endpoint}"

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
        # Add command for query logging
        if hasattr(self, "_command_mode") and self._command_mode:
            payload["command"] = self._command_mode

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

    def _inject_no_rag_system_prompt(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Inject no-RAG system prompt for bypass mode."""

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
        self, messages: List[Dict[str, Any]], context: str
    ) -> List[Dict[str, Any]]:
        """Inject RAG context into the message list via system prompt."""

        # Build the full instruction with context
        rag_instruction = f"""{SYSTEM_PROMPT_RAG}

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
        self._use_hyde_search = False

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

        # Parse command from query
        self._original_query = user_query
        command, cleaned_query = self._parse_command(user_query)
        self._command_mode = "bypass" if command is None else command
        self._use_hyde_search = command in ("qlong", "debuglong")

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

        # Handle bypass mode (no command = no RAG)
        if command is None:
            print("[RAG Filter] No command detected - bypassing RAG retrieval")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "💬 Using LLM knowledge only (no RAG)",
                            "done": True,
                        },
                    }
                )
            # Inject no-RAG system prompt for general helpdesk assistance
            body["messages"] = self._inject_no_rag_system_prompt(messages)
            print("[RAG Filter] Injected no-RAG system prompt")
            return body

        # Use cleaned query for RAG search (command stripped)
        user_query = cleaned_query
        print(
            f"[RAG Filter] Cleaned Query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}"
        )

        # Emit status update
        if __event_emitter__:
            status_description = (
                "⏳ Running long-form HyDE search..."
                if self._use_hyde_search
                else "🔍 Searching knowledge base..."
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": status_description,
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
            search_response = self._search_documents(
                user_query, user_id, use_hyde=self._use_hyde_search
            )
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

            # Capture comprehensive debug metadata for !debug and !debuglong modes
            if command in ("debug", "debuglong"):
                self._search_metadata = self._extract_debug_metadata(
                    search_response, context, user_query
                )
                if self.valves.DEBUG_MODE:
                    print(f"[RAG Filter] Captured debug metadata for !{command} mode")

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
            if self.valves.ENABLE_LLM_RESPONSE_LOGGING:
                # Extract query_log_id from instance variable (set in inlet)
                query_log_id = self._query_log_id
                if query_log_id:
                    # Extract LLM response text from body
                    self._log_llm_response(body, query_log_id)
                elif self.valves.DEBUG_MODE:
                    print(
                        "[RAG Filter] No query_log_id found, skipping response logging"
                    )
            elif self.valves.DEBUG_MODE:
                print("[RAG Filter] LLM response logging disabled")

            # Check if we need to append debug output (!debug or !debuglong mode)
            if self._search_metadata and self._command_mode in ("debug", "debuglong"):
                self._append_debug_output(body)

        finally:
            # Clean up instance variables after logging
            self._query_log_id = None
            self._search_response_data = None
            self._search_metadata = None
            self._command_mode = None
            self._original_query = None
            self._use_hyde_search = False

        return body

    def _log_llm_response(self, body: Dict[str, Any], query_log_id: int) -> None:
        """Log LLM response to backend for analytics."""
        try:
            messages = body.get("messages", [])
            if not messages:
                if self.valves.DEBUG_MODE:
                    print("[RAG Filter] No messages in response body")
                return

            last_message = messages[-1]
            if last_message.get("role") != "assistant":
                if self.valves.DEBUG_MODE:
                    print(
                        f"[RAG Filter] Last message not from assistant: {last_message.get('role')}"
                    )
                return

            response_text = last_message.get("content", "")
            if not response_text:
                if self.valves.DEBUG_MODE:
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
                if self.valves.DEBUG_MODE:
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

    def _append_debug_output(self, body: Dict[str, Any]) -> None:
        """Append debug information to the LLM response."""
        try:
            print("[RAG Filter] Appending debug output to LLM response")

            # Get last message (LLM response)
            messages = body.get("messages", [])
            if not messages or messages[-1].get("role") != "assistant":
                if self.valves.DEBUG_MODE:
                    print(
                        "[RAG Filter] Cannot append debug output - no assistant message found"
                    )
                return

            last_message = messages[-1]
            content = last_message.get("content", "")

            # Handle multimodal content (list of text/image parts)
            if isinstance(content, list):
                # Find text content part and append debug info
                for part in content:
                    if part.get("type") == "text":
                        original_text = part.get("text", "")
                        debug_output = self._format_debug_output()
                        part["text"] = original_text + debug_output
                        print(
                            f"[RAG Filter] Debug output appended ({len(debug_output)} chars)"
                        )
                        break
            else:
                # Simple string content
                debug_output = self._format_debug_output()
                last_message["content"] = content + debug_output
                print(f"[RAG Filter] Debug output appended ({len(debug_output)} chars)")

        except Exception as e:
            # Don't fail outlet on debug formatting errors
            print(
                f"[RAG Filter] Failed to append debug output: {type(e).__name__}: {str(e)}"
            )
            if self.valves.DEBUG_MODE:
                print(f"[RAG Filter] Traceback:\n{traceback.format_exc()}")
