"""Provider-neutral result contract shared by real agents and deterministic stubs."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class AgentInvocationResult:
    output: dict[str, Any]
    tokens_used: int | None = 0
    cost: Decimal | None = Decimal("0")
    provider: str | None = None
    model: str | None = None
    llm_calls: int = 0
    is_stub: bool = False
    usage_known: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
