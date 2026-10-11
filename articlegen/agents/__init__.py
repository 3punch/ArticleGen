"""The five specialised sub-agents, in mandatory execution order."""

from __future__ import annotations

from typing import List, Optional

from ..llm import LLMClient, resolve_client
from ..tokens import TokenBudget
from .architect import Architect
from .base import BaseAgent
from .reviewer import Reviewer
from .synthesizer import Synthesizer
from .tester import Tester
from .writer import Writer

#: Rule 4.2 in claude.md - this order is not negotiable.
PIPELINE_ORDER: List[str] = ["writer", "tester", "reviewer", "architect", "synthesizer"]


def build_agents(client: LLMClient, budget: TokenBudget, *, prompt_dir: Optional[str] = None,
                 rules_path: Optional[str] = None) -> List[BaseAgent]:
    """Instantiate the pipeline in mandatory order."""
    kwargs = {"prompt_dir": prompt_dir, "rules_path": rules_path}
    agents: List[BaseAgent] = [
        Writer(client, budget, **kwargs),
        Tester(client, budget, **kwargs),
        Reviewer(client, budget, **kwargs),
        Architect(client, budget, **kwargs),
        Synthesizer(client, budget, **kwargs),
    ]
    return agents


__all__ = [
    "Architect",
    "BaseAgent",
    "PIPELINE_ORDER",
    "Reviewer",
    "Synthesizer",
    "Tester",
    "Writer",
    "build_agents",
    "resolve_client",
]
