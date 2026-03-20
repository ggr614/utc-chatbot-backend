# core/chat_service.py
"""
ChatService — framework-agnostic RAG orchestrator.

Handles command parsing, hybrid search, prompt assembly,
LLM streaming, and logging for the /v1/chat/completions endpoint.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple


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

**Vague query + scattered context**: If the query is unclear AND the retrieved documentation covers multiple unrelated topics, do your best to answer, then provide a better query suggestion.

**Partially documented**: If documentation covers part of the issue, state what IS documented vs. what requires escalation.

**Not in documentation**: Do not guess. Provide the information-gathering checklist and escalation path.

## UTC CONTEXT

- **UTC** = University of Tennessee at Chattanooga
- **Mocs** = Mockingbirds (mascot), used in system names (MocsNet, MocsID, MocsMail, etc.)
- Common systems: Banner, Canvas, TeamDynamix (TDX), Genetec (access control), Active Directory (AD)
- Customers = students, faculty, staff (never "users" in responses)"""


class ChatService:
    """Orchestrates RAG chat: search -> prompt assembly -> LLM -> logging."""

    def __init__(
        self,
        bm25_retriever,
        vector_retriever,
        reranker,
        connection_pool,
        chat_settings,
        litellm_settings,
        hybrid_search_fn: Callable | None = None,
        select_system_prompt_fn: Callable | None = None,
    ):
        self._bm25 = bm25_retriever
        self._vector = vector_retriever
        self._reranker = reranker
        self._pool = connection_pool
        self._chat_settings = chat_settings
        self._litellm_settings = litellm_settings
        self._hybrid_search = hybrid_search_fn
        self._select_system_prompt = select_system_prompt_fn

    @staticmethod
    def _parse_command(query: str | None) -> Tuple[Optional[str], str | None]:
        """Parse command prefix from user query.

        Returns (command, cleaned_query):
        - (None, query) for default search
        - ("follow_up", cleaned) for !f
        - ("help", "") for !help
        """
        if not query or not isinstance(query, str):
            return (None, query)

        stripped = query.strip()
        lower = stripped.lower()

        if lower == "!help" or lower.startswith("!help "):
            return ("help", "")

        if lower == "!f" or lower.startswith("!f "):
            cleaned = stripped[len("!f"):].strip()
            return ("follow_up", cleaned)

        return (None, stripped)

    @staticmethod
    def _format_context(
        results: list[dict], max_context_tokens: int = 4000
    ) -> str:
        """Format search results into a context string for the knowledge base block.

        Iterates results in rank order. Drops lowest-ranked chunks that exceed
        the token limit (estimated at ~4 chars per token).
        """
        if not results:
            return ""

        max_chars = max_context_tokens * 4
        context_parts: list[str] = []
        total_chars = 0

        for i, result in enumerate(results, 1):
            chunk = result["chunk"]
            text_content = chunk.text_content
            source_url = str(chunk.source_url) if chunk.source_url else ""

            doc_lines = [f"[Document {i}]"]
            if source_url:
                doc_lines.append(f"Source: {source_url}")
            doc_lines.append(f"Content:\n{text_content}")
            doc_entry = "\n".join(doc_lines)

            if total_chars + len(doc_entry) > max_chars:
                if not context_parts:
                    # First doc exceeds limit — include truncated
                    context_parts.append(doc_entry[: max_chars])
                break

            context_parts.append(doc_entry)
            total_chars += len(doc_entry)

        return "\n\n---\n\n".join(context_parts)

    @staticmethod
    def _assemble_rag_messages(
        messages: list[dict],
        query: str,
        context: str,
        system_prompt: str | None = None,
    ) -> list[dict]:
        """Build query-sandwich message list for RAG mode.

        Replaces (or inserts) the system message with the sandwich structure.
        Preserves all other messages (conversation history).
        """
        base_prompt = system_prompt if system_prompt else SYSTEM_PROMPT_RAG

        sandwich = (
            f"User question: {query}\n\n"
            f"{base_prompt}\n\n"
            "You have access to a knowledge base of IT helpdesk support articles.\n"
            "Use the following retrieved documents to help answer the user's question:\n\n"
            f"<knowledge_base>\n{context}\n</knowledge_base>\n\n"
            f"User question: {query}"
        )

        new_messages = []
        system_found = False

        for msg in messages:
            if msg.get("role") == "system":
                new_messages.append({"role": "system", "content": sandwich})
                system_found = True
            else:
                new_messages.append(msg.copy() if isinstance(msg, dict) else msg)

        if not system_found:
            new_messages.insert(0, {"role": "system", "content": sandwich})

        return new_messages

    @staticmethod
    def _assemble_no_rag_messages(messages: list[dict]) -> list[dict]:
        """Build message list for follow-up mode (no RAG context)."""
        new_messages = []
        system_found = False

        for msg in messages:
            if msg.get("role") == "system":
                new_messages.append(
                    {"role": "system", "content": SYSTEM_PROMPT_NO_RAG}
                )
                system_found = True
            else:
                new_messages.append(msg.copy() if isinstance(msg, dict) else msg)

        if not system_found:
            new_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT_NO_RAG})

        return new_messages

    @staticmethod
    def _get_user_query(messages: list[dict]) -> str | None:
        """Extract the latest user message content."""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content")
                if content and isinstance(content, str):
                    return content
                return None
        return None

    @staticmethod
    def _get_help_text() -> str:
        """Generate help text for the !help command."""
        return """# RAG Helpdesk Assistant - Command Help

## How It Works

By default, your question is searched against the UTC IT knowledge base using hybrid search (BM25 keyword matching + vector semantic similarity), and the most relevant documents are provided to the LLM to generate an answer.

## Available Commands

### `<your question>` (no command - default)
**Knowledge base search** - Searches the knowledge base and provides context to the LLM.

**Example:**
```
How do I reset a student's password?
```

---

### `!f <your question>`
**Follow-up mode** - Bypasses the knowledge base entirely. Uses only the LLM's built-in knowledge and conversation history.

**When to use:** For general IT questions, clarifications, or follow-up questions about a previous response that don't need UTC-specific documentation.

**Example:**
```
!f Can you explain that last step in more detail?
```

---

### `!help`
**Display this help message**

---

*Need more help? Contact the development team or check the system documentation.*"""
