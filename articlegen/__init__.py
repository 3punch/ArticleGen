"""ArticleGen - an automated, iterative multi-agent article/code pipeline.

The package is intentionally dependency-free (standard library only) so the full
orchestration loop can be executed and verified in constrained environments.
"""

from __future__ import annotations

from .models import (
    Brief,
    Draft,
    Finding,
    GateResult,
    Section,
    Severity,
    AgentReport,
    IterationRecord,
    RunResult,
)
from .tokens import TokenBudget, count_tokens
from .gate import verify_brief, verify_draft
from .orchestrator import Orchestrator, OrchestratorConfig
from .prompt2brief import brief_from_prompt
from .llm import ResilientClient, resolve_resilient_client

__version__ = "0.1.0"

__all__ = [
    "AgentReport",
    "Brief",
    "Draft",
    "Finding",
    "GateResult",
    "IterationRecord",
    "Orchestrator",
    "OrchestratorConfig",
    "ResilientClient",
    "RunResult",
    "Section",
    "Severity",
    "TokenBudget",
    "brief_from_prompt",
    "count_tokens",
    "resolve_resilient_client",
    "verify_brief",
    "verify_draft",
    "__version__",
]
