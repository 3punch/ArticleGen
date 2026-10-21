"""Research-integrity layer for the article pipeline.

The writing loop (Writer -> Tester -> Reviewer -> Architect -> Synthesizer) is good at
structure and prose. It cannot, on its own, distinguish a demonstrated result from a
planned one, or a real citation from a plausible-looking one. This package adds the
structured stages that can:

* :mod:`articlegen.research.planning`   - ten-part research plan, novelty policy
* :mod:`articlegen.research.sources`    - retrieval, verification, bibliography
* :mod:`articlegen.research.evidence`   - claim/evidence ledger
* :mod:`articlegen.research.codecheck`  - code extraction, execution, tests, hashes
* :mod:`articlegen.research.analysis`   - complexity and benchmark integrity
* :mod:`articlegen.research.validators` - deterministic manuscript checks
* :mod:`articlegen.research.review`     - scientific peer review + revision loop
* :mod:`articlegen.research.quality`    - the three-outcome research gate
* :mod:`articlegen.research.pipeline`   - orchestration of the stages above
"""

from .models import (  # noqa: F401
    CheckStatus,
    ClaimRecord,
    ContributionType,
    EvidenceKind,
    ExperimentRun,
    ExperimentSpec,
    ExperimentStatus,
    GateStatus,
    QualityGateResult,
    ResearchPlan,
    ReviewIssue,
    ReviewReport,
    ReviewSeverity,
    SourceRecord,
    ValidationCheck,
    ValidationReport,
    WorkflowStates,
)

__all__ = [
    "CheckStatus",
    "ClaimRecord",
    "ContributionType",
    "EvidenceKind",
    "ExperimentRun",
    "ExperimentSpec",
    "ExperimentStatus",
    "GateStatus",
    "QualityGateResult",
    "ResearchPlan",
    "ReviewIssue",
    "ReviewReport",
    "ReviewSeverity",
    "SourceRecord",
    "ValidationCheck",
    "ValidationReport",
    "WorkflowStates",
]
