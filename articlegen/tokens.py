"""Token accounting.

Token counting is kept for reporting and observability only. ArticleGen no
longer enforces any token ceiling: context packets are never truncated or
evicted for size, agent input and output are unbounded, and a run is never
stopped because it spent too many tokens.

``count_tokens`` remains the single estimator used everywhere so the change
log and the dashboard always agree with each other. ``truncate_to_tokens`` is
kept as a general-purpose utility; nothing in the pipeline applies a token
budget any more.
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
    """Trim ``text`` so it fits ``max_tokens``, keeping the head (highest signal).

    Kept as a standalone utility. ArticleGen no longer applies a token budget,
    so nothing in the pipeline truncates or evicts context for size.
    """
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    marker_tokens = count_tokens(marker)
    usable = max(0, max_tokens - marker_tokens)
    if usable <= 0:
        return ""
    # Binary-search a character cut-off that satisfies the token budget.
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
    """Per-agent settings. Agent input and output are unbounded: no caps."""

    agent_id: str


@dataclass
class TokenBudget:
    """No ceilings.

    Holds per-agent settings only. Token counts are recorded for reporting;
    they never truncate, evict, or block anything.
    """

    agents: Dict[str, AgentBudget] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "TokenBudget":
        return cls(
            agents={
                "writer": AgentBudget("writer"),
                "tester": AgentBudget("tester"),
                "reviewer": AgentBudget("reviewer"),
                "architect": AgentBudget("architect"),
                "synthesizer": AgentBudget("synthesizer"),
            }
        )

    @classmethod
    def from_dict(cls, data: Dict, base: Optional["TokenBudget"] = None) -> "TokenBudget":
        base = base or cls.default()
        agents = dict(base.agents)
        for agent_id in (data.get("agents") or {}):
            agents.setdefault(agent_id, AgentBudget(agent_id))
        return cls(agents=agents)

    def agent(self, agent_id: str) -> AgentBudget:
        return self.agents.get(agent_id) or AgentBudget(agent_id)
