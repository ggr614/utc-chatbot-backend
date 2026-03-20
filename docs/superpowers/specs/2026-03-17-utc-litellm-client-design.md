# utc-litellm-client Design Spec

## Overview

A private internal Python package that eliminates LiteLLM boilerplate across UTC AI projects. Provides pre-built clients for embedding, chat completion, and reranking — all routed through a LiteLLM proxy with an OpenAI-compatible API.

**Target directory:** `C:\Python\utc_llms`
**Package name:** `utc-llms` (existing project, uses `uv` build system)
**Module name:** `utc_llms`

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Distribution | Private internal package (Git install) | Only UTC projects consume it |
| API mode | Proxy-only via OpenAI SDK | All deployments use LiteLLM proxy |
| Model names | Bare aliases, no provider prefixes | Proxy handles routing |
| Architecture | Thin base class + concrete clients | Simple, focused, testable |
| Async strategy | Async-first, auto-generated sync wrappers with separate names (`aembed` / `embed`) | Type-safe, clear at call site |
| Settings | Auto-read from env, accept injectable override | Zero-config production, clean tests |
| Error handling | Decorator adds structured logging, re-raises LiteLLM exceptions as-is | Callers decide how to handle |
| Tokenizer | Excluded | Too trivial (5 lines) to abstract |
| Return types | Raw LiteLLM response objects | Callers extract what they need |

## Package Structure

```
C:\Python\utc_llms\
├── pyproject.toml               # Existing, update dependencies
├── README.md
├── src/
│   └── utc_llms/
│       ├── __init__.py          # Public API exports
│       ├── settings.py          # LiteLLMProxySettings
│       ├── base.py              # BaseLiteLLMClient + _make_sync + @litellm_error_handler
│       ├── embedding.py         # EmbeddingClient
│       ├── completion.py        # CompletionClient
│       └── reranker.py          # RerankClient
└── tests/
    ├── conftest.py              # Shared fixtures (mock settings, etc.)
    ├── test_embedding.py
    ├── test_completion.py
    └── test_reranker.py
```

Note: The existing `src/utc_llms/clients/` directory will be removed. All client modules live directly under `src/utc_llms/`.

## Component Designs

### settings.py — LiteLLMProxySettings

Pydantic `BaseSettings` model that auto-reads from environment variables with `LITELLM_` prefix.

```python
class LiteLLMProxySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LITELLM_", env_file=".env")

    PROXY_BASE_URL: str
    PROXY_API_KEY: SecretStr

    # Model aliases (bare names, no provider prefix)
    EMBEDDING_MODEL: str = "text-embedding-large-3"
    CHAT_MODEL: str = "gpt-5.2-chat"
    RERANKER_MODEL: str = "cohere-rerank-v3-5"

    # Embedding config
    EMBED_DIM: int = 3072
    EMBED_MAX_TOKENS: int = 8191

    # Chat config
    CHAT_MAX_TOKENS: int = 8191
    CHAT_COMPLETION_TOKENS: int = 500
    CHAT_TEMPERATURE: float = 0.7
```

**Cached accessor:**

```python
@lru_cache()
def get_settings() -> LiteLLMProxySettings:
    return LiteLLMProxySettings()
```

### base.py — BaseLiteLLMClient

Holds proxy credentials and provides shared infrastructure.

```python
class BaseLiteLLMClient:
    def __init__(self, model: str, settings: LiteLLMProxySettings | None = None):
        self._settings = settings or get_settings()
        self.model = model
        self.api_base = self._settings.PROXY_BASE_URL
        self.api_key = self._settings.PROXY_API_KEY.get_secret_value()

        if not self.api_key:
            raise ValueError("LITELLM_PROXY_API_KEY is empty")

    def _call_params(self) -> dict:
        """Common params dict for litellm calls."""
        return {
            "model": self.model,
            "api_base": self.api_base,
            "api_key": self.api_key,
        }
```

**_make_sync utility:**

```python
def _make_sync(async_fn):
    """Generate a sync wrapper that calls asyncio.run()."""
    @functools.wraps(async_fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(async_fn(*args, **kwargs))
    wrapper.__doc__ = f"Sync wrapper for {async_fn.__name__}."
    return wrapper
```

**@litellm_error_handler decorator:**

Works with both sync and async functions. On any exception from `litellm.exceptions` (AuthenticationError, RateLimitError, Timeout, APIError):
1. Logs at ERROR with structured context (method name, model, exception type/message)
2. Re-raises the original exception unchanged

On unexpected exceptions:
1. Logs at ERROR with traceback
2. Re-raises unchanged

On success:
1. Logs at DEBUG (method name, model)

The decorator does NOT wrap, transform, or suppress exceptions.

### embedding.py — EmbeddingClient

```python
class EmbeddingClient(BaseLiteLLMClient):
    def __init__(self, settings: LiteLLMProxySettings | None = None):
        s = settings or get_settings()
        super().__init__(model=s.EMBEDDING_MODEL, settings=s)
        self.expected_dim = s.EMBED_DIM
        self.max_tokens = s.EMBED_MAX_TOKENS

    @litellm_error_handler
    async def aembed(self, text: str) -> list[float]:
        """Embed a single text string. Returns vector of expected_dim dimensions."""
        # Calls litellm.aembedding(input=[text], **self._call_params(), num_retries=3, timeout=30.0)
        # Validates response dimension == self.expected_dim

    @litellm_error_handler
    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns list of vectors, order preserved."""
        # Calls litellm.aembedding(input=texts, **self._call_params(), num_retries=3, timeout=60.0)
        # Sorts response by index, validates dimensions

    embed = _make_sync(aembed)
    embed_batch = _make_sync(aembed_batch)
```

**Input validation:** The package validates that inputs are non-empty strings. Token count validation is the caller's responsibility (no tokenizer in this package).

**Output validation:** Checks embedding dimension matches `expected_dim`. Raises `ValueError` on mismatch.

### completion.py — CompletionClient

```python
class CompletionClient(BaseLiteLLMClient):
    def __init__(self, settings: LiteLLMProxySettings | None = None):
        s = settings or get_settings()
        super().__init__(model=s.CHAT_MODEL, settings=s)
        self.max_completion_tokens = s.CHAT_COMPLETION_TOKENS
        self.temperature = s.CHAT_TEMPERATURE

    @litellm_error_handler
    async def acomplete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> litellm.ModelResponse:
        """Send chat completion request. Returns raw ModelResponse."""
        # Calls litellm.acompletion(
        #     messages=messages,
        #     max_tokens=max_tokens or self.max_completion_tokens,
        #     temperature=temperature if temperature is not None else self.temperature,
        #     num_retries=3, timeout=30.0,
        #     **self._call_params()
        # )

    complete = _make_sync(acomplete)
```

**Returns raw `ModelResponse`:** Callers extract `.choices[0].message.content`, `.usage`, or whatever they need. This keeps the client generic for HyDE, chat, summarization, etc.

### reranker.py — RerankClient

```python
class RerankClient(BaseLiteLLMClient):
    def __init__(self, settings: LiteLLMProxySettings | None = None):
        s = settings or get_settings()
        super().__init__(model=s.RERANKER_MODEL, settings=s)

    @litellm_error_handler
    async def arerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> litellm.RerankResponse:
        """Rerank documents by relevance to query. Returns raw RerankResponse."""
        # Calls litellm.arerank(
        #     query=query,
        #     documents=documents,
        #     top_n=top_n or len(documents),
        #     **self._call_params()
        # )

    rerank = _make_sync(arerank)
```

**Accepts plain strings:** The consuming project extracts `text_content` from its domain objects before calling rerank. The package doesn't know about chunks, articles, or search results.

## Public API (__init__.py)

```python
from utc_llms.settings import LiteLLMProxySettings, get_settings
from utc_llms.base import BaseLiteLLMClient, litellm_error_handler
from utc_llms.embedding import EmbeddingClient
from utc_llms.completion import CompletionClient
from utc_llms.reranker import RerankClient
```

## Dependencies

Update `pyproject.toml` to reflect actual needs:

```toml
dependencies = [
    "litellm>=1.60.0",
    "pydantic-settings>=2.13.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=9.0.2",
    "pytest-asyncio>=0.26.0",
    "pytest-mock>=3.14.0",
]
```

Remove `llama-index`, `openai`, and `python-dotenv` from core dependencies (litellm pulls in openai; pydantic-settings handles .env files natively).

## Usage Examples

### Zero-config (production)

```python
from utc_llms import EmbeddingClient, CompletionClient, RerankClient

# Reads LITELLM_PROXY_BASE_URL, LITELLM_PROXY_API_KEY, etc. from env
embedder = EmbeddingClient()
vectors = embedder.embed_batch(["chunk 1", "chunk 2"])

# Async in FastAPI
vectors = await embedder.aembed_batch(["chunk 1", "chunk 2"])

# Chat completion
completer = CompletionClient()
response = completer.complete(messages=[
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is VPN?"},
])
text = response.choices[0].message.content

# Reranking
reranker = RerankClient()
response = reranker.rerank(query="vpn issues", documents=["doc1", "doc2"])
```

### Testing (injectable settings)

```python
from utc_llms import EmbeddingClient, LiteLLMProxySettings

test_settings = LiteLLMProxySettings(
    PROXY_BASE_URL="http://localhost:4000",
    PROXY_API_KEY="test-key",
)
client = EmbeddingClient(settings=test_settings)
```

### Custom base class usage

```python
from utc_llms import BaseLiteLLMClient, litellm_error_handler
import litellm

class SummarizationClient(BaseLiteLLMClient):
    def __init__(self, settings=None):
        s = settings or get_settings()
        super().__init__(model=s.CHAT_MODEL, settings=s)

    @litellm_error_handler
    async def asummarize(self, text: str) -> str:
        response = await litellm.acompletion(
            messages=[
                {"role": "system", "content": "Summarize concisely."},
                {"role": "user", "content": text},
            ],
            max_tokens=200,
            **self._call_params(),
        )
        return response.choices[0].message.content

    summarize = _make_sync(asummarize)
```

## Migration Impact on utc-chatbot-backend

| File | Before | After |
|------|--------|-------|
| `core/embedding.py` | ~180 lines | ~40 lines (domain validation + client) |
| `core/hyde_generator.py` | ~150 lines | ~50 lines (system prompt + text cleaning + client) |
| `core/reranker.py` | ~170 lines | ~60 lines (result parsing + latency tracking + client) |
| `core/config.py` | Contains `LiteLLMSettings` | Remove LiteLLM settings section |

Domain logic stays in the consuming project: HyDE prompt templates, reranker result mapping, embedding dimension validation beyond what the client provides, token counting.

## What the Package Does NOT Do

- Token counting (too trivial to abstract)
- Input token validation (caller's responsibility)
- Response parsing beyond raw LiteLLM types
- Custom retry logic (delegates to LiteLLM's `num_retries`)
- Exception wrapping (logs and re-raises as-is)
- Provider prefix management (proxy handles routing)
- Direct provider calls (proxy-only)
