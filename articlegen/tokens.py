"""Token accounting and hard context ceilings.

Free-tier efficiency is enforced here rather than by convention. Two numbers matter:

* ``count_tokens`` - the single estimator used everywhere, so budget decisions and
  the change log always agree with each other.
* ``TokenBudget`` - hard caps. Nothing is ever passed to an agent without being
  measured and, if necessary, truncated or evicted first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional

# Approximation used throughout: ~4 characters per token for prose, slightly
# denser for source code. It is deliberately cheap (no tokenizer dependency) and
# conservative - it rounds up so real usage lands under, not over, the cap.
_CHARS_PER_TOKEN_PROSE = 4.0
_CHARS_PER_TOKEN_CODE = 3.4


def count_tokens(text: str) -> int:
    """Estimate token count for ``text``. Deterministic and dependency-free."""
    if not text:
        return 0
    # Splitting on fenced blocks leaves only prose, so whatever is left over is
    # code (fences included) and gets the denser weighting.
    prose_chars = sum(len(part) for part in re.split(r"```.*?```", text, flags=re.DOTALL))
    code_chars = max(0, len(text) - prose_chars)
    prose_tokens = prose_chars / _CHARS_PER_TOKEN_PROSE
    code_tokens = code_chars / _CHARS_PER_TOKEN_CODE
    return max(1, int(round(prose_tokens + code_tokens)))


def truncate_to_tokens(text: str, max_tokens: int, marker: str = "\n\n[...truncated by token budget...]") -> str:
    """Trim ``text`` so it fits ``max_tokens``, keeping the head (highest signal)."""
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    marker_tokens = count_tokens(marker)
    usable = max(0, max_tokens - marker_tokens)
    if usable <= 0:
        return ""
    # Binary-search a character cut-off that satisfies the token ceiling.
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid]) <= usable:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + marker


@dataclass(frozen=True)
class AgentBudget:
    """Per-agent ceiling on what may enter and leave a single agent call."""

    agent_id: str
    max_input_tokens: int
    max_output_tokens: int


@dataclass
class TokenBudget:
    """Global + per-agent ceilings for one run.

    ``context_hard_cap`` is the strict threshold mentioned in the system rules:
    any assembled context packet above it must be evicted or truncated before use.
    ``context_soft_warn`` triggers a warning in the change log without blocking.
    """

    context_hard_cap: int = 6000
    context_soft_warn: int = 4800
    run_token_ceiling: int = 60000
    reserve_for_response: int = 2000
    agents: Dict[str, AgentBudget] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "TokenBudget":
        return cls(
            context_hard_cap=6000,
            context_soft_warn=4800,
            run_token_ceiling=60000,
            reserve_for_response=2000,
            agents={
                "writer": AgentBudget("writer", 2400, 2000),
                "tester": AgentBudget("tester", 2200, 900),
                "reviewer": AgentBudget("reviewer", 2200, 900),
                "architect": AgentBudget("architect", 2000, 800),
                "synthesizer": AgentBudget("synthesizer", 2600, 2200),
            },
        )

    @classmethod
    def from_dict(cls, data: Dict, base: Optional["TokenBudget"] = None) -> "TokenBudget":
        base = base or cls.default()
        agents = dict(base.agents)
        for agent_id, cfg in (data.get("agents") or {}).items():
            agents[agent_id] = AgentBudget(
                agent_id=agent_id,
                max_input_tokens=int(cfg.get("max_input_tokens", 2000)),
                max_output_tokens=int(cfg.get("max_output_tokens", 1000)),
            )
        return cls(
            context_hard_cap=int(data.get("context_hard_cap", base.context_hard_cap)),
            context_soft_warn=int(data.get("context_soft_warn", base.context_soft_warn)),
            run_token_ceiling=int(data.get("run_token_ceiling", base.run_token_ceiling)),
            reserve_for_response=int(data.get("reserve_for_response", base.reserve_for_response)),
            agents=agents,
        )

    def agent(self, agent_id: str) -> AgentBudget:
        return self.agents.get(agent_id) or AgentBudget(agent_id, self.context_hard_cap, 1000)

    def exceeds_soft_warn(self, tokens: int) -> bool:
        return tokens >= self.context_soft_warn

    def exceeds_hard_cap(self, tokens: int) -> bool:
        return tokens > self.context_hard_cap
