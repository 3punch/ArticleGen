"""Evidence-labelled export: the draft plus a research-integrity appendix.

The export intentionally puts the validation status *before* the article. A reader
must not be able to download an unverified or incomplete manuscript and mistake it
for a completed study. The appendix records the plan, the verified sources, the
executed results, every failed or unverified check, and the unresolved review issues.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..models import Brief, Draft, Section
from .models import (
    BenchmarkRecord,
    CheckStatus,
    ExperimentRun,
    ExperimentSpec,
    GateStatus,
    QualityGateResult,
    ResearchPlan,
    ReviewIssue,
    ReviewReport,
    SourceRecord,
    ValidationCheck,
    ValidationReport,
    classify_section_role,
    profile_for,
)
from .sources import SourceLibrary


def _cell(text: Any, limit: int = 220) -> str:
    value = str(text or "").replace("|", "\\|").replace("\n", " ").strip()
    return value[:limit]


def find_references_section(draft: Draft):
    return next((section for section in draft.sections
                 if classify_section_role(section.id, section.title) == "references"), None)


def apply_bibliography(draft: Draft, library: SourceLibrary) -> bool:
    """Replace (or add) the references section from the library's records.

    References are generated from verified records only. If the draft has no references
    section and records exist, one is appended: a citation that resolves nowhere is a
    defect, and the bibliography is the one part of the manuscript that must be written
    from records rather than prose. If nothing was retrieved, the section says so
    instead of containing an invented bibliography.
    """
    section = find_references_section(draft)
    if section is None:
        if not library.records:
            return False
        section = Section(
            id="references", title="References", purpose="Generated bibliography.",
            word_target=120, body="",
        )
        draft.sections.append(section)
    bibliography = library.bibliography()
    if bibliography:
        section.body = (
            "Entries below are generated from the verified source records retrieved "
            "for this run.\n\n" + bibliography
        )
    else:
        section.body = (
            "No verified source records were available for this run. No bibliography "
            "is supplied, and claims that would require a citation must be removed or "
            "labelled as unverified."
        )
    return True


def status_banner(gate: Optional[QualityGateResult],
                  review: Optional[ReviewReport],
                  brief: Optional[Brief] = None) -> str:
    if gate is None:
        return ""
    profile = profile_for(brief.article_type) if brief is not None else None
    heading = f"**Research validation status: {gate.status.label}**"
    lines = [f"> {heading}"]
    if gate.status == GateStatus.READY_FOR_REVIEW:
        lines.append(">")
        lines.append("> Every required automated check that applies to this article "
                     "type passed, and no unresolved critical or major issue remains. "
                     "This is a traceability check, not a guarantee of scientific "
                     "correctness or journal acceptance.")
    elif gate.status == GateStatus.NEEDS_REVISION:
        lines.append(">")
        lines.append("> This draft is **not** a validated completed study. "
                     f"{len(gate.reasons)} unresolved issue(s) must be addressed; see "
                     "the research-integrity appendix.")
    else:
        lines.append(">")
        lines.append("> Required evidence, source verification, execution or validation "
                     "could not be completed. Treat this manuscript as a proposal, "
                     "protocol or preliminary draft, not as completed research.")
        experiments_expected = bool(
            (profile is not None and (profile.requires_experiments
                                      or getattr(brief, "requires_experiments", False)))
        )
        if experiments_expected and not gate.states.experiments_executed:
            lines.append(">")
            lines.append("> No experiment executed successfully, so no result is "
                         "reported as established.")
    if gate.critical_failures:
        lines.append(">")
        lines.append("> Critical: " + "; ".join(gate.critical_failures[:4]))
    return "\n".join(lines)


def plan_markdown(plan: Optional[ResearchPlan]) -> str:
    if plan is None:
        return "_No research plan was produced._"
    lines = [
        f"- **Article type:** {plan.article_type.value}",
        f"- **Contribution type:** {plan.contribution_type.label}",
        f"- **Research question:** {plan.research_question or '_(not established)_'}",
        f"- **Contribution:** {plan.contribution or '_(not established)_'}",
        f"- **Gap:** {plan.gap or '_(not established)_'}",
        f"- **Methodology:** {plan.methodology or '_(not established)_'}",
        f"- **Plan decision:** {plan.decision}"
        + (" (needs user direction)" if plan.needs_user_direction else ""),
        f"- **Plan validated:** {'yes' if plan.validated else 'no'}",
    ]
    if plan.evaluation_criteria:
        lines.append("- **Evaluation criteria:** " + "; ".join(plan.evaluation_criteria))
    if plan.limitations:
        lines.append("- **Known limitations:** " + "; ".join(plan.limitations))
    if plan.decision_reasons:
        lines.append("- **Decision reasons:** " + "; ".join(plan.decision_reasons))
    return "\n".join(lines)


def sources_markdown(library: SourceLibrary) -> str:
    report = library.report()
    lines = [
        f"- **Records:** {report.total} "
        f"({report.verified} index-verified, {report.user_provided} user-provided, "
        f"{report.unverified} unverified, {report.metadata_incomplete} incomplete metadata)",
        f"- **Retrieval provider:** {library.retrieval.provider or 'none'} "
        f"(attempted: {'yes' if library.retrieval.attempted else 'no'})",
    ]
    if library.retrieval.errors or library.errors:
        for error in (library.retrieval.errors + library.errors)[:6]:
            lines.append(f"- **Retrieval problem:** {_cell(error, 300)}")
    if library.records:
        lines.append("")
        lines.append("| id | verification | authors | year | title | venue | DOI / URL |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for record in library.records[:40]:
            authors = ", ".join(record.authors[:3]) + (" et al." if len(record.authors) > 3 else "")
            link = f"https://doi.org/{record.doi}" if record.doi else record.url
            lines.append(
                f"| {record.source_id} | {record.verification} | {_cell(authors, 80)} | "
                f"{record.year or ''} | {_cell(record.title)} | {_cell(record.venue, 80)} | "
                f"{_cell(link, 100)} |"
            )
    else:
        lines.append("- **No source records were retrieved.** No bibliography was "
                     "generated and no citation may be presented as verified.")
    return "\n".join(lines)


def experiments_markdown(specs: Sequence[ExperimentSpec],
                         runs: Sequence[ExperimentRun]) -> str:
    if not runs:
        return "_No experiment was executed for this manuscript._"
    lines = [
        "| experiment | status | exit | duration (s) | tests (passed/run) | metrics |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        metrics = ", ".join(f"{k}={v}" for k, v in list(run.metrics.items())[:4]) or "-"
        lines.append(
            f"| {run.experiment_id} | {run.status.value} | {run.exit_code if run.exit_code is not None else '-'} "
            f"| {run.duration_seconds:.3f} | {run.tests_passed}/{run.tests_run} | {_cell(metrics, 160)} |"
        )
    environments = {json.dumps(run.environment, sort_keys=True) for run in runs if run.environment}
    lines.append("")
    for environment in list(environments)[:1]:
        lines.append(f"Execution environment: `{_cell(environment, 500)}`")
    if any(run.status.value != "ok" for run in runs):
        lines.append("")
        lines.append("**Failed or incomplete runs:**")
        for run in runs:
            if run.status.value != "ok":
                lines.append(f"- {run.experiment_id}: {run.status.value} — "
                             f"{_cell(run.error or run.stderr[-300:], 300)}")
    return "\n".join(lines)


def validation_markdown(validation: Optional[ValidationReport]) -> str:
    if validation is None:
        return "_No validation report exists._"
    lines = [
        "| check | status | severity | details |",
        "| --- | --- | --- | --- |",
    ]
    for check in validation.checks:
        # Only non-passing rows: a table of forty green checks helps nobody.
        if check.status == CheckStatus.PASS:
            continue
        lines.append(
            f"| `{check.check_id}` | {check.status.value} | {check.severity.value} | "
            f"{_cell(check.details or check.description, 300)} |"
        )
    summary = validation.summary()
    lines.insert(0, (
        f"Passed {summary['pass']} · failed {summary['fail']} · "
        f"unverified {summary['unverified']} · not applicable "
        f"{summary['not_applicable']}.\n"
    ))
    return "\n".join(lines)


def review_markdown(review: Optional[ReviewReport]) -> str:
    if review is None:
        return "_No scientific review was recorded._"
    blocking = review.blocking()
    lines = [
        f"{len(review.issues)} finding(s); {len(blocking)} unresolved blocking "
        f"issue(s); revision iterations: {review.iterations_run}.",
    ]
    if blocking:
        lines.append("")
        lines.append("| id | severity | dimension | section | claim | correction |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for issue in blocking[:25]:
            lines.append(
                f"| {issue.issue_id} | {issue.severity.value} | {issue.dimension} | "
                f"{_cell(issue.section, 80)} | {_cell(issue.claim, 180)} | "
                f"{_cell(issue.correction, 220)} |"
            )
    if review.revisions:
        lines.append("")
        lines.append("Revision history:")
        for revision in review.revisions:
            lines.append(
                f"- iteration {revision.iteration}: fixed "
                f"{len(revision.applied)}, unresolved {len(revision.unresolved)}, "
                f"deferred {len(revision.deferred)}; sections changed: "
                f"{', '.join(revision.changed_sections) or 'none'}"
            )
    if review.notes:
        lines.append("")
        lines.append("Reviewer notes: " + "; ".join(dict.fromkeys(review.notes))[:1200])
    return "\n".join(lines)


def integrity_appendix(*, brief: Brief, plan: Optional[ResearchPlan],
                       library: SourceLibrary, runs: Sequence[ExperimentRun],
                       specs: Sequence[ExperimentSpec],
                       validation: Optional[ValidationReport],
                       review: Optional[ReviewReport],
                       gate: Optional[QualityGateResult],
                       benchmarks: Sequence[BenchmarkRecord] = ()) -> str:
    sections = [
        "---",
        "",
        "# Research integrity appendix",
        "",
        "This appendix is generated from the run's recorded artefacts, not from model "
        "text. Automated checks reduce preventable errors; they do not establish "
        "scientific validity or predict peer-review acceptance.",
        "",
        "## Validation status",
        "",
    ]
    if gate is not None:
        sections.append(f"- **Status:** {gate.status.label}")
        if gate.reasons:
            sections.append("- **Unresolved issues:**")
            for reason in gate.reasons[:20]:
                sections.append(f"  - {_cell(reason, 400)}")
        states = gate.states
        sections.append(
            "- **Workflow states:** " + ", ".join(
                f"{name}={'yes' if value else 'no'}"
                for name, value in states.to_dict().items()
                if isinstance(value, bool)
            )
        )
    else:
        sections.append("_No quality gate was computed._")

    sections += ["", "## Research plan", "", plan_markdown(plan), "",
                 "## Sources and verification", "", sources_markdown(library), "",
                 "## Experiments and executed results", "",
                 experiments_markdown(specs, runs), ""]
    if benchmarks:
        sections += ["## Benchmark records", ""]
        for record in benchmarks:
            issues = record.metadata_issues()
            sections.append(
                f"- `{record.benchmark_id}` ({record.experiment_id}): "
                f"{record.repetitions} repetition(s); metadata "
                + ("complete" if not issues else "missing: " + ", ".join(issues))
            )
        sections.append("")
    sections += ["## Deterministic validation", "", validation_markdown(validation), "",
                 "## Peer review", "", review_markdown(review), ""]
    profile = profile_for(brief.article_type)
    sections += [
        "## Limits of this validation",
        "",
        "- Execution is process isolation in a temporary directory, not a container "
        "sandbox: it limits accidental damage but is not a security boundary.",
        "- Checks are deterministic heuristics plus a model review. They cannot prove a "
        "theorem, validate a proof step by step, or confirm that a cited source truly "
        "supports every attached claim.",
        "- Unverified items are reported as unverified, never as passed.",
        f"- Article-type profile applied: {profile.label}.",
        "",
    ]
    return "\n".join(sections).rstrip() + "\n"


def manuscript_with_appendix(draft: Draft, *, brief: Brief,
                             plan: Optional[ResearchPlan],
                             library: SourceLibrary,
                             runs: Sequence[ExperimentRun] = (),
                             specs: Sequence[ExperimentSpec] = (),
                             validation: Optional[ValidationReport] = None,
                             review: Optional[ReviewReport] = None,
                             gate: Optional[QualityGateResult] = None,
                             benchmarks: Sequence[BenchmarkRecord] = (),
                             include_appendix: bool = True) -> str:
    body = draft.to_markdown()
    banner = status_banner(gate, review, brief)
    if banner:
        title, _, rest = body.partition("\n")
        body = f"{title}\n\n{banner}\n\n{rest.lstrip()}"
    if not include_appendix or gate is None:
        return body
    return body.rstrip() + "\n\n" + integrity_appendix(
        brief=brief, plan=plan, library=library, runs=runs, specs=specs,
        validation=validation, review=review, gate=gate, benchmarks=benchmarks,
    )


def build_bundle(*, brief: Brief, draft: Optional[Draft], plan: Optional[ResearchPlan],
                 library: SourceLibrary, runs: Sequence[ExperimentRun],
                 specs: Sequence[ExperimentSpec],
                 validation: Optional[ValidationReport],
                 review: Optional[ReviewReport],
                 gate: Optional[QualityGateResult],
                 retrieval: Optional[Dict[str, Any]] = None,
                 code_analysis: Optional[Any] = None,
                 results: Sequence[Any] = (),
                 benchmarks: Sequence[BenchmarkRecord] = (),
                 ledger: Optional[Any] = None,
                 notes: Sequence[str] = ()) -> Dict[str, Any]:
    return {
        "task_id": brief.task_id,
        "article_type": brief.article_type.value,
        "citation_style": brief.citation_style.value,
        "quality_gate": gate.to_dict() if gate else None,
        "plan": plan.to_dict() if plan else None,
        "sources": library.to_dict(),
        "retrieval": retrieval or library.retrieval.to_dict(),
        "experiments": {
            "specs": [spec.to_dict() for spec in specs],
            "runs": [run.to_dict() for run in runs],
        },
        "results": [r.to_dict() if hasattr(r, "to_dict") else r for r in results],
        "benchmarks": [b.to_dict() for b in benchmarks],
        "evidence": ledger.to_dict() if ledger is not None and hasattr(ledger, "to_dict")
        else None,
        "code_analysis": code_analysis.to_dict()
        if code_analysis is not None and hasattr(code_analysis, "to_dict") else None,
        "validation": validation.to_dict() if validation else None,
        "review": review.to_dict() if review else None,
        "notes": list(notes),
        "markdown": manuscript_with_appendix(
            draft, brief=brief, plan=plan, library=library, runs=runs, specs=specs,
            validation=validation, review=review, gate=gate, benchmarks=benchmarks,
        ) if draft is not None else "",
    }
