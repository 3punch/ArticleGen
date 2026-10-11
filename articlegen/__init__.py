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
from .orchestrator import Orchestrator

__version__ = "0.1.0"

__all__ = [
    "AgentReport",
    "Brief",
    "Draft",
    "Finding",
    "GateResult",
    "IterationRecord",
    "Orchestrator",
    "RunResult",
    "Section",
    "Severity",
    "TokenBudget",
    "count_tokens",
    "verify_brief",
    "verify_draft",
    "__version__",
]
