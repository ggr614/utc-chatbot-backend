import litellm

# Default model for local token counting. Must be a model name recognized by
# litellm's tokenizer (not a proxy alias).  text-embedding-3-large and gpt-4
# both use the cl100k_base tokenizer, so gpt-4 is a safe universal default.
_DEFAULT_TOKEN_COUNTER_MODEL = "gpt-4"


class Tokenizer:
    def __init__(self, model: str | None = None):
        self.model = model or _DEFAULT_TOKEN_COUNTER_MODEL

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        return litellm.token_counter(model=self.model, text=string)
