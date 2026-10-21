"""The research quality gate.

Three outcomes, and only three:

* **Ready for final human review** - every required automated check has passed, the
  evidence is traceable, and no unresolved critical/major issue remains. This is not
  a claim that a journal will accept the paper.
* **Needs revision** - one or more important issues remain. The draft can be exported,
  but it must be labelled as needing revision.
* **Incomplete or unverified** - required evidence, source verification, experiment
  execution or validation could not be completed.

There is deliberately no composite numeric score. A writing-quality score cannot
establish scientific validity, so it is not an input to this module at all. A
high-scoring draft with a fabricated reference or an unexecuted experiment is still
incomplete.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..models import ArticleType, Brief, Draft
from .sources import SourceLibrary
from .models import (
    CheckStatus,
    ExperimentRun,
    ExperimentSpec,
    GateStatus,
    QualityGateResult,
    ReviewReport,
    ReviewSeverity,
    ValidationCheck,
    ValidationReport,
    WorkflowStates,
    profile_for,
)


def _required(check: ValidationCheck) -> bool:
    return check.status != CheckStatus.NOT_APPLICABLE


def _blocking_check(check: ValidationCheck) -> bool:
    """Whether a failed/unverified check may coexist with a "ready" verdict.

    Any failed check at moderate severity or above blocks: "Ready for final review"
    must mean every applicable check was evaluated and passed. Minor issues are
    reported but do not block, so a stray unused reference entry cannot make a
    validated manuscript look unverifiable.
    """
    if check.status in (CheckStatus.PASS, CheckStatus.NOT_APPLICABLE):
        return False
    if check.severity == ReviewSeverity.MINOR:
        return False
    return True


def derive_states(brief: Brief, draft: Optional[Draft],
                  validation: Optional[ValidationReport],
                  review: Optional[ReviewReport],
                  library: SourceLibrary,
                  runs: Sequence[ExperimentRun],
                  plan) -> WorkflowStates:
    profile = profile_for(brief.article_type)
    states = WorkflowStates()
    states.draft_generated = bool(
        draft is not None
        and draft.sections
        and all(section.body.strip() for section in draft.sections)
        and draft.word_count > 0
    )
    checks = validation.checks if validation else []
    plan_checks = [c for c in checks if c.category == "plan"]
    states.plan_validated = bool(
        plan is not None
        and getattr(plan, "validated", False)
        and plan_checks
        and all(c.status in (CheckStatus.PASS, CheckStatus.NOT_APPLICABLE)
                for c in plan_checks)
    )
    states.sources_retrieved = bool(library.records)
    states.sources_verified = bool(library.records) and library.report().complete
    successful_runs = [r for r in runs if r.status.value == "ok"]
    states.experiments_executed = bool(successful_runs)
    executed_tests = [t for run in runs for t in run.tests if t.status != "skipped"]
    states.tests_passed = bool(executed_tests) and not any(
        t.status in ("failed", "error") for t in executed_tests)
    technical = [c for c in checks
                 if c.category in ("code", "technical", "performance", "benchmark",
                                   "complexity", "statistics")]
    states.technical_checks_completed = bool(technical) and all(
        c.status in (CheckStatus.PASS, CheckStatus.NOT_APPLICABLE) for c in technical)
    review_check = next((c for c in checks if c.check_id == "review.completed"), None)
    states.peer_review_completed = bool(
        review is not None and review_check is not None
        and review_check.status == CheckStatus.PASS
    )
    states.revision_completed = bool(review is not None and review.revisions)
    states.ready_for_human_review = False  # set by the caller only when the gate is READY
    # Profile requirements are exposed separately so a UI can distinguish
    # "not required" from "required but not done".
    states.required_sources = profile.requires_sources  # type: ignore[attr-defined]
    states.required_experiments = (profile.requires_experiments
                                   or brief.requires_experiments)  # type: ignore[attr-defined]
    return states


def evaluate_quality_gate(*, brief: Brief, draft: Optional[Draft],
                          validation: Optional[ValidationReport],
                          review: Optional[ReviewReport],
                          library: Optional[SourceLibrary] = None,
                          runs: Sequence[ExperimentRun] = (),
                          specs: Sequence[ExperimentSpec] = (),
                          plan=None) -> QualityGateResult:
    """Compute the three-outcome gate from real validation state."""
    library = library or SourceLibrary()
    result = QualityGateResult()
    if draft is None:
        result.status = GateStatus.INCOMPLETE_OR_UNVERIFIED
        result.reasons.append("no draft was generated")
        return result
    if validation is None:
        result.status = GateStatus.INCOMPLETE_OR_UNVERIFIED
        result.reasons.append("no validation report exists; the manuscript is unverified")
        return result

    states = derive_states(brief, draft, validation, review, library, runs, plan)
    result.states = states
    checks = validation.checks
    profile = profile_for(brief.article_type)

    incomplete: List[str] = []
    revision: List[str] = []
    critical: List[str] = []
    for check in checks:
        if not _blocking_check(check):
            continue
        label = f"{check.check_id} ({check.severity.value}): {check.details or check.description}"
        if check.status == CheckStatus.UNVERIFIED:
            # An unverified required check is missing evidence, not a stylistic defect.
            if check.gate_effect == "incomplete" or check.severity == ReviewSeverity.CRITICAL:
                incomplete.append(label)
            else:
                revision.append(label)
            continue
        if check.gate_effect == "incomplete":
            incomplete.append(label)
        else:
            revision.append(label)
        if check.severity == ReviewSeverity.CRITICAL:
            critical.append(f"{check.check_id}: {check.description}")

    blocking_issues = review.blocking() if review is not None else []
    for issue in blocking_issues:
        label = (f"{issue.issue_id} ({issue.severity.value}/{issue.dimension}): "
                 f"{issue.claim[:200]}")
        if issue.requires_user_input:
            incomplete.append(label + " [requires user input]")
        elif issue.dimension in ("data_and_experimental_integrity",
                                 "algorithmic_or_mathematical_correctness",
                                 "methodological_validity") and issue.status == "deferred":
            incomplete.append(label + " [requires new evidence or re-analysis]")
        elif issue.status in ("open", "unresolved", "deferred"):
            revision.append(label)

    if not states.draft_generated:
        incomplete.append("the draft is empty or has an empty section")

    if incomplete:
        result.status = GateStatus.INCOMPLETE_OR_UNVERIFIED
    elif revision:
        result.status = GateStatus.NEEDS_REVISION
    else:
        result.status = GateStatus.READY_FOR_REVIEW
        if not profile.research:
            result.reasons.append(
                "General article: research evidence checks were not applicable; this "
                "status does not assert scholarly validity."
            )

    result.incomplete_reasons = list(dict.fromkeys(incomplete))
    result.revision_reasons = list(dict.fromkeys(revision))
    result.reasons = list(dict.fromkeys(incomplete + revision))
    result.critical_failures = list(dict.fromkeys(critical))
    result.unresolved_issues = [issue.issue_id for issue in blocking_issues]
    states.ready_for_human_review = result.status == GateStatus.READY_FOR_REVIEW
    result.validated = result.status == GateStatus.READY_FOR_REVIEW
    return result


def gate_summary_line(result: QualityGateResult) -> str:
    return f"{result.status.value}: {result.status.label}"


def states_table(result: QualityGateResult) -> List[Dict[str, str]]:
    """Human-readable checklist: state, requirement, and what it means."""
    profile_required = {
        "plan_validated": "Research plan validated",
        "sources_retrieved": "Sources retrieved",
        "sources_verified": "Sources verified",
        "experiments_executed": "Experiments executed",
        "tests_passed": "Tests passed",
        "technical_checks_completed": "Technical checks completed",
        "peer_review_completed": "Peer review completed",
        "revision_completed": "Revision completed",
        "draft_generated": "Draft generated",
        "ready_for_human_review": "Ready for final human review",
    }
    required_sources = bool(getattr(result.states, "required_sources", False))
    required_experiments = bool(getattr(result.states, "required_experiments", False))
    rows: List[Dict[str, str]] = []
    for key, label in profile_required.items():
        value = bool(getattr(result.states, key, False))
        requirement = "required"
        if key in ("sources_retrieved", "sources_verified"):
            requirement = "required" if required_sources else "not required"
        elif key in ("experiments_executed", "tests_passed"):
            requirement = "required" if required_experiments else "not required"
        rows.append({
            "state": key,
            "check": label,
            "requirement": requirement,
            "status": "done" if value else ("not done" if requirement == "required"
                                             else "n/a"),
        })
    return rows
