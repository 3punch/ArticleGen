"""Lazy LiteLLM adapter for the configured OpenAI-compatible provider."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from .config import ModelConfig, ModelConfigurationError


@dataclass(frozen=True)
class ModelCompletion:
    content: str
    tokens_used: int | None
    cost: Decimal | None
    provider: str
    model: str


class PromptTokenLimitExceeded(ValueError):
    """Raised before network I/O if prompt plus maximum output exceeds its hold."""


class ChatModelClient(Protocol):
    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        timeout_seconds: int | None = None,
    ) -> ModelCompletion: ...


def _read(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


class LiteLLMChatClient:
    """Use LiteLLM without importing it until the first billable call is requested."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        timeout_seconds: int | None = None,
    ) -> ModelCompletion:
        self.config.validate_provider()
        model = self.config.model_for(agent_name)
        try:
            import litellm
        except ImportError as exc:  # optional extra is required only for live model use
            raise ModelConfigurationError(
                "LiteLLM is not installed; install the 'projectforge-agents' extra"
            ) from exc

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_tokens = self._count_prompt_tokens(model, messages)
        maximum_request_tokens = input_tokens + self.config.max_output_tokens
        if maximum_request_tokens > self.config.reservation_tokens:
            raise PromptTokenLimitExceeded(
                f"prompt plus max output ({maximum_request_tokens} tokens) exceeds the "
                f"per-call token reservation ({self.config.reservation_tokens})"
            )
        effective_timeout = min(
            self.config.timeout_seconds,
            timeout_seconds if timeout_seconds is not None else self.config.timeout_seconds,
        )
        response = litellm.completion(
            model=model,
            messages=messages,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            temperature=0,
            max_tokens=self.config.max_output_tokens,
            timeout=effective_timeout,
            num_retries=0,
            stream=False,
        )
        choices = _read(response, "choices", []) or []
        if not choices:
            raise ValueError("LLM provider returned no completion choices")
        message = _read(choices[0], "message", {})
        content = _read(message, "content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM provider returned empty assistant content")

        usage = _read(response, "usage", {}) or {}
        token_value = _read(usage, "total_tokens")
        if not isinstance(token_value, (int, float)):
            prompt_tokens = _read(usage, "prompt_tokens", _read(usage, "input_tokens"))
            completion_tokens = _read(usage, "completion_tokens", _read(usage, "output_tokens"))
            if (isinstance(prompt_tokens, (int, float))
                    and isinstance(completion_tokens, (int, float))):
                token_value = prompt_tokens + completion_tokens
        tokens_used = int(token_value) if isinstance(token_value, (int, float)) else None
        cost = self._response_cost(response)
        return ModelCompletion(
            content=content.strip(),
            tokens_used=tokens_used,
            cost=cost,
            provider=self.config.provider,
            model=model,
        )

    @staticmethod
    def _count_prompt_tokens(model: str, messages: list[dict[str, str]]) -> int:
        """Return a network-free upper bound for prompt tokens.

        Each tokenizer token represents at least one UTF-8 byte; counting bytes plus
        fixed chat framing therefore avoids downloading model vocabularies at runtime
        and fails closed for an unknown provider tokenizer. Response usage later settles
        the reservation to provider-reported values when available.
        """
        content_bytes = sum(
            len(message["content"].encode("utf-8")) for message in messages
        )
        chat_overhead = 4 * len(messages) + 2
        return content_bytes + chat_overhead

    @staticmethod
    def _response_cost(response: Any) -> Decimal | None:
        """Read cost explicitly reported by the response without guessing tariffs."""
        hidden = _read(response, "_hidden_params", {}) or {}
        candidate = _read(hidden, "response_cost")
        if candidate is None:
            usage = _read(response, "usage", {}) or {}
            candidate = _read(usage, "cost")
        if candidate is None:
            return None
        try:
            value = Decimal(str(candidate))
        except (InvalidOperation, TypeError, ValueError):
            return None
        if not value.is_finite() or value < 0:
            return None
        # A zero cost from LiteLLM often means the custom provider's tariff is unknown.
        return value if value > 0 else None
