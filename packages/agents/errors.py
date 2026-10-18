"""Model-agent failures that preserve billable response metadata for audit settlement."""

from __future__ import annotations

from .litellm_client import ModelCompletion


class AgentOutputValidationError(ValueError):
    def __init__(self, message: str, *, completion: ModelCompletion) -> None:
        super().__init__(message)
        self.completion = completion
