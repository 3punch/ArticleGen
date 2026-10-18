"""Environment-driven LiteLLM routing and conservative per-call budget policy."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import os
from typing import Mapping


class ModelConfigurationError(RuntimeError):
    """Raised when an LLM call is requested without a usable provider configuration."""


def _positive_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    try:
        value = default if raw is None else int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{key} must be greater than zero")
    return value


def _decimal(env: Mapping[str, str], key: str, default: str) -> Decimal:
    raw = env.get(key, default)
    try:
        value = Decimal(raw)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"{key} must be a decimal number") from exc
    if not value.is_finite() or value < 0:
        raise ValueError(f"{key} must be a finite nonnegative decimal")
    return value


@dataclass(frozen=True)
class ModelConfig:
    """Shared model routing, with per-agent model overrides and fixed call limits.

    Dahl is the initial logical provider. Since its model catalog rotates, the exact
    deployed model ID is supplied by environment rather than guessed or hard-coded.
    LiteLLM routes the OpenAI-compatible Dahl endpoint through its ``openai`` adapter.
    """

    provider: str
    tier: str
    model_id: str | None
    requirement_model_id: str | None
    planner_model_id: str | None
    coder_model_id: str | None
    tester_model_id: str | None
    debugger_model_id: str | None
    reviewer_model_id: str | None
    packager_model_id: str | None
    api_base: str | None
    api_key: str | None
    route_prefix: str
    max_output_tokens: int
    reservation_tokens: int
    reservation_spend: Decimal
    reservation_ttl_seconds: int
    timeout_seconds: int
    confidence_threshold: float

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ModelConfig":
        source = os.environ if env is None else env
        provider = source.get("PROJECTFORGE_LLM_PROVIDER", "dahl").strip().lower()
        if not provider or not provider.replace("-", "").isalnum():
            raise ValueError("PROJECTFORGE_LLM_PROVIDER must be an alphanumeric provider name")
        tier = source.get("PROJECTFORGE_LLM_TIER", "strong").strip().lower()
        if not tier or not tier.replace("-", "").isalnum():
            raise ValueError("PROJECTFORGE_LLM_TIER must be a non-empty tier name")
        provider_model_key = f"PROJECTFORGE_{provider.upper().replace('-', '_')}_{tier.upper().replace('-', '_')}_MODEL"
        model_id = (source.get("PROJECTFORGE_LLM_MODEL") or source.get(provider_model_key) or "").strip() or None
        requirement_model_id = (
            source.get("PROJECTFORGE_REQUIREMENT_MODEL", "").strip()
            or model_id
        )
        planner_model_id = (
            source.get("PROJECTFORGE_PLANNER_MODEL", "").strip()
            or model_id
        )
        coder_model_id = source.get("PROJECTFORGE_CODER_MODEL", "").strip() or model_id
        tester_model_id = source.get("PROJECTFORGE_TESTER_MODEL", "").strip() or model_id
        debugger_model_id = source.get("PROJECTFORGE_DEBUGGER_MODEL", "").strip() or model_id
        reviewer_model_id = source.get("PROJECTFORGE_REVIEWER_MODEL", "").strip() or model_id
        packager_model_id = source.get("PROJECTFORGE_PACKAGER_MODEL", "").strip() or model_id
        default_base = "https://inference.dahl.global/v1" if provider == "dahl" else None
        api_base = (
            source.get("PROJECTFORGE_LLM_BASE_URL")
            or source.get(f"PROJECTFORGE_{provider.upper().replace('-', '_')}_BASE_URL")
            or default_base
        )
        api_key = (
            source.get("PROJECTFORGE_LLM_API_KEY")
            or source.get(f"PROJECTFORGE_{provider.upper().replace('-', '_')}_API_KEY")
            or (source.get("DAHL_API_KEY") if provider == "dahl" else None)
        )
        route_prefix = source.get(
            "PROJECTFORGE_LLM_ROUTE_PREFIX", "openai" if provider == "dahl" else provider,
        ).strip().rstrip("/")
        try:
            confidence_threshold = float(source.get("PROJECTFORGE_REQUIREMENT_CONFIDENCE", "0.70"))
        except (TypeError, ValueError) as exc:
            raise ValueError("PROJECTFORGE_REQUIREMENT_CONFIDENCE must be a number") from exc
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("PROJECTFORGE_REQUIREMENT_CONFIDENCE must be between 0 and 1")
        max_output_tokens = _positive_int(source, "PROJECTFORGE_LLM_MAX_OUTPUT_TOKENS", 4096)
        reservation_tokens = _positive_int(source, "PROJECTFORGE_LLM_RESERVATION_TOKENS", 60000)
        if max_output_tokens >= reservation_tokens:
            raise ValueError("max output tokens must be below the per-call token reservation")
        reservation_spend = _decimal(source, "PROJECTFORGE_LLM_RESERVATION_SPEND", "0.50")
        if reservation_spend <= 0:
            raise ValueError("PROJECTFORGE_LLM_RESERVATION_SPEND must be greater than zero")
        return cls(
            provider=provider,
            tier=tier,
            model_id=model_id,
            requirement_model_id=requirement_model_id,
            planner_model_id=planner_model_id,
            coder_model_id=coder_model_id,
            tester_model_id=tester_model_id,
            debugger_model_id=debugger_model_id,
            reviewer_model_id=reviewer_model_id,
            packager_model_id=packager_model_id,
            api_base=api_base,
            api_key=api_key,
            route_prefix=route_prefix,
            max_output_tokens=max_output_tokens,
            reservation_tokens=reservation_tokens,
            reservation_spend=reservation_spend,
            reservation_ttl_seconds=_positive_int(
                source, "PROJECTFORGE_LLM_RESERVATION_TTL_SECONDS", 900,
            ),
            timeout_seconds=_positive_int(source, "PROJECTFORGE_LLM_TIMEOUT_SECONDS", 60),
            confidence_threshold=confidence_threshold,
        )

    def model_for(self, agent_name: str) -> str:
        model_by_agent = {
            "requirement": self.requirement_model_id,
            "planner": self.planner_model_id,
            "coder": self.coder_model_id,
            "tester": self.tester_model_id,
            "debugger": self.debugger_model_id,
            "reviewer": self.reviewer_model_id,
            "packager": self.packager_model_id,
        }
        if agent_name not in model_by_agent:
            raise ValueError(f"no model route is defined for agent '{agent_name}'")
        model_id = model_by_agent[agent_name]
        if not model_id:
            variable = (
                f"PROJECTFORGE_{self.provider.upper().replace('-', '_')}_"
                f"{self.tier.upper().replace('-', '_')}_MODEL"
            )
            raise ModelConfigurationError(
                f"no model is configured for the {self.tier} {self.provider} tier; "
                f"set {variable} or PROJECTFORGE_LLM_MODEL"
            )
        if "/" in model_id:
            return model_id
        if self.route_prefix:
            return f"{self.route_prefix}/{model_id}"
        return model_id

    def validate_provider(self) -> None:
        if not self.api_key:
            if self.provider == "dahl":
                raise ModelConfigurationError("DAHL_API_KEY is required for Dahl model calls")
            raise ModelConfigurationError(
                "set PROJECTFORGE_LLM_API_KEY (or the provider-specific API key)"
            )
        if not self.api_base:
            raise ModelConfigurationError(
                "set PROJECTFORGE_LLM_BASE_URL for the configured LLM provider"
            )
