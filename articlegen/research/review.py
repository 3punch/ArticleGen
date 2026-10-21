"""Scientific peer review and the targeted revision loop.

Two reviewers run:

* a deterministic reviewer built from the failed validation checks, which cannot be
  talked out of a finding; and
* an optional model reviewer that adds semantic issues (unclear contribution,
  citation-claim mismatch, overclaiming) under a strict JSON contract.

The revision pass is deliberately narrow. It may rewrite prose for issues about the
research question, literature framing, internal consistency, limitations and venue
fit. It may **not** "fix" a failed experiment, a code/prose mismatch, missing data,
a statistical error or an unverified source by weakening a sentence: those issues are
deferred and keep blocking the quality gate until the artefact itself changes.

After every revision iteration the validators run again. An issue is only marked
fixed when its underlying check no longer fails; a model saying it fixed something is
not evidence.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..models import ArticleType, Brief, Draft, Section
from ..agents.payload import iter_mapped_sections
from ..llm import LLMClient, ProviderError, ResponseParseError, parse_json
from ..prompts import PromptError, load_template
from . import codecheck, validators
from .models import (
    BenchmarkRecord,
    CheckStatus,
    ExperimentRun,
    ExperimentSpec,
    ReviewIssue,
    ReviewReport,
    ReviewSeverity,
    RevisionSummary,
    ValidationReport,
    dedupe_issues,
)
from .planning import plan_summary_block
from .sources import SourceLibrary

#: Dimensions a prose revision may address. Everything else needs a new artefact.
REVISION_ELIGIBLE_DIMENSIONS = {
    "research_question_and_contribution",
    "literature_and_citations",
    "limitations_and_validity",
    "internal_consistency",
    "writing_and_venue_fit",
}

#: Areas where "softening the wording" would conceal the real error.
EVIDENCE_BOUND_DIMENSIONS = {
    "methodological_validity",
    "algorithmic_or_mathematical_correctness",
    "data_and_experimental_integrity",
    "statistical_and_quantitative_claims",
    "reproducibility",
}

VALID_DIMENSIONS = set(validators.CATEGORY_DIMENSIONS.values()) | {
    "research_question_and_contribution", "literature_and_citations",
    "methodological_validity", "algorithmic_or_mathematical_correctness",
    "data_and_experimental_integrity", "statistical_and_quantitative_claims",
    "reproducibility", "limitations_and_validity", "internal_consistency",
    "writing_and_venue_fit",
}


def _issue_key(issue: ReviewIssue) -> Tuple[str, str, str]:
    return (issue.check_id or "", issue.dimension,
            re.sub(r"\s+", " ", issue.claim.lower())[:100])


def _same_issue(old: ReviewIssue, new: ReviewIssue) -> bool:
    if old.check_id and new.check_id and old.check_id == new.check_id:
        return True
    return (old.dimension == new.dimension
            and re.sub(r"\W+", "", old.claim.lower())[:60]
            == re.sub(r"\W+", "", new.claim.lower())[:60])


def model_issue_from_payload(item: Dict[str, Any], iteration: int = 0) -> Optional[ReviewIssue]:
    if not isinstance(item, dict):
        return None
    claim = str(item.get("claim") or item.get("finding") or "").strip()
    explanation = str(item.get("explanation") or item.get("why") or "").strip()
    if not claim or not explanation:
        return None
    dimension = str(item.get("dimension") or "methodological_validity").strip()
    if dimension not in VALID_DIMENSIONS:
        dimension = "methodological_validity"
    section = str(item.get("section") or "document").strip()[:120]
    severity = ReviewSeverity.parse(item.get("severity"))
    correction = str(item.get("correction") or item.get("action") or "").strip()
    evidence = str(item.get("evidence") or "").strip()[:600]
    auto_fixable = bool(item.get("auto_fixable", False))
    requires_user_input = bool(item.get("requires_user_input", False))
    # A model cannot declare a data or correctness problem auto-fixable, and cannot
    # request user input for a purely stylistic finding.
    if dimension in EVIDENCE_BOUND_DIMENSIONS:
        auto_fixable = False
    if dimension == "writing_and_venue_fit":
        requires_user_input = False
    if not correction:
        correction = ("Revise the affected claim so that it states only what the "
                      "supplied evidence supports.")
    digest = hashlib.sha1(f"{dimension}|{section}|{claim}".encode("utf-8")).hexdigest()[:8]
    return ReviewIssue(
        issue_id=f"M{iteration + 1}-{digest}",
        dimension=dimension,
        severity=severity,
        section=section,
        claim=claim[:600],
        explanation=explanation[:1200],
        evidence=evidence,
        correction=correction[:800],
        auto_fixable=auto_fixable,
        requires_user_input=requires_user_input,
        status="open",
        source="model",
        iteration=iteration,
    )


class ScientificReviewer:
    """Deterministic + optional model review over the whole evidence chain."""

    def __init__(self, client: Optional[LLMClient] = None, *,
                 prompt_dir: Optional[str] = None, model_review: bool = True) -> None:
        self.client = client
        self.prompt_dir = prompt_dir
        self.model_review = model_review and client is not None and client.available

    # -- context ----------------------------------------------------------
    def _user_prompt(self, draft: Draft, brief: Brief, plan, library: SourceLibrary,
                     ledger, runs: Sequence[ExperimentRun],
                     validation: ValidationReport) -> str:
        plan_text = plan_summary_block(plan) if plan is not None else "(no plan)"
        failures = [c for c in validation.checks
                    if c.status in (CheckStatus.FAIL, CheckStatus.UNVERIFIED)]
        failure_text = "\n".join(
            f"- [{c.severity.value}/{c.category}] {c.check_id}: {c.description}"
            f" :: {c.details[:400]}"
            for c in failures[:30]
        ) or "none"
        results = validators.results_from_runs(runs)
        result_text = "\n".join(
            f"- {r.result_id} {r.metric} = {r.value} {r.unit} "
            f"(experiment {r.experiment_id})"
            for r in results[:30]
        ) or "none executed"
        article_type = brief.article_type.value if brief.article_type else "general"
        parts = [
            f"article_type: {article_type}",
            f"venue: {brief.venue or '(not specified)'}",
            f"citation_style: {brief.citation_style.value}",
            "RESEARCH PLAN:",
            plan_text,
            "VERIFIED SOURCES (the only citable records):",
            library.context_block(max_records=12, max_abstract_chars=200),
            "EXECUTED RESULTS (the only reportable results):",
            result_text,
            "DETERMINISTIC CHECK FAILURES (do not contradict these; add issues only):",
            failure_text,
            "MANUSCRIPT:",
            draft.to_markdown()[:14000],
        ]
        return "\n\n".join(parts)

    def review(self, draft: Draft, brief: Brief, *, plan=None,
               library: Optional[SourceLibrary] = None,
               ledger=None, runs: Sequence[ExperimentRun] = (),
               specs: Sequence[ExperimentSpec] = (),
               benchmarks: Sequence[BenchmarkRecord] = (),
               validation: Optional[ValidationReport] = None,
               code_analysis: Optional[codecheck.CodeAnalysis] = None,
               retrieval: Optional[Dict[str, Any]] = None,
               iteration: int = 0) -> Tuple[ValidationReport, ReviewReport]:
        library = library or SourceLibrary()
        if validation is None:
            validation = validators.validate_manuscript(
                draft, brief, plan=plan, library=library, ledger=ledger,
                runs=runs, specs=specs, benchmarks=benchmarks,
                code_analysis=code_analysis, retrieval=retrieval,
                include_review=False,
            )
        issues = validators.issues_from_validation(validation)
        report = ReviewReport(issues=issues)

        if self.model_review:
            try:
                system = load_template("scientific_reviewer", self.prompt_dir)
                user = self._user_prompt(draft, brief, plan, library, ledger, runs,
                                         validation)
                response = self.client.complete(system, user, max_output_tokens=4096)
                payload = parse_json(response.text)
                raw_issues = payload.get("issues") if isinstance(payload, dict) else []
                if isinstance(raw_issues, list):
                    for item in raw_issues[:12]:
                        issue = model_issue_from_payload(item, iteration)
                        if issue is not None:
                            # Do not let a model issue silently duplicate or weaken a
                            # deterministic finding.
                            if not any(_same_issue(issue, existing) for existing in issues):
                                issues.append(issue)
                    report.model_review_ran = True
                report.notes.append("model review completed")
            except (PromptError, ProviderError, ResponseParseError, ValueError) as exc:
                report.notes.append(
                    f"model review unavailable: {type(exc).__name__}: {exc}. "
                    "Only deterministic findings are reported."
                )
            except Exception as exc:  # noqa: BLE001 - review must not abort a run
                report.notes.append(f"model review failed: {type(exc).__name__}: {exc}")
        else:
            report.notes.append(
                "model review skipped (no provider); deterministic review only"
            )

        for issue in issues:
            if issue.severity in (ReviewSeverity.MINOR, ReviewSeverity.MODERATE) \
                    and issue.source == "model" and not issue.auto_fixable:
                issue.status = "open"
        report.issues = sorted(dedupe_issues(issues),
                               key=lambda i: (-i.severity.weight, i.dimension,
                                              i.issue_id))
        return validation, report


class TargetedReviser:
    """Applies a bounded prose revision for eligible review issues."""

    def __init__(self, client: Optional[LLMClient] = None, *,
                 prompt_dir: Optional[str] = None) -> None:
        self.client = client
        self.prompt_dir = prompt_dir

    @property
    def available(self) -> bool:
        return self.client is not None and bool(getattr(self.client, "available", False))

    def eligible(self, issues: Iterable[ReviewIssue]) -> List[ReviewIssue]:
        return [issue for issue in issues
                if issue.is_blocking
                and issue.dimension in REVISION_ELIGIBLE_DIMENSIONS
                and not issue.requires_user_input
                and issue.status != "fixed"]

    def revise(self, draft: Draft, brief: Brief, issues: Sequence[ReviewIssue], *,
               plan=None, library: Optional[SourceLibrary] = None,
               runs: Sequence[ExperimentRun] = (), iteration: int = 0
               ) -> Tuple[Optional[Draft], List[str], List[str], List[str]]:
        """Return ``(new_draft_or_None, changed_section_ids, addressed_ids, notes)``."""
        if not self.available or not issues:
            return None, [], [], ["no eligible revision or no provider"]
        try:
            system = load_template("reviser", self.prompt_dir)
        except PromptError as exc:
            return None, [], [], [f"reviser prompt unavailable: {exc}"]
        results = validators.results_from_runs(runs)
        issue_lines = [
            f"- id={issue.issue_id} [{issue.severity.value}/{issue.dimension}] "
            f"section={issue.section} claim={issue.claim} correction={issue.correction}"
            for issue in issues[:10]
        ]
        sections_text = "\n\n".join(
            f"## section id={section.id} title={section.title}\n{section.body}"
            for section in draft.sections
        )
        user = "\n\n".join([
            "RULES:",
            "- Change only the sections needed to address the listed issues.",
            "- Do not invent results, numbers, sources or quotations.",
            "- Executed results are the only reportable results:\n"
            + ("\n".join(f"  {r.result_id} {r.metric}={r.value} {r.unit}"
                         for r in results[:20]) or "  none"),
            "- Verified citations are the only citable sources:\n"
            + (library.context_block(max_records=12, max_abstract_chars=120)
               if library else "  none"),
            "- If an issue cannot be fixed without new evidence, leave it unresolved "
            "and say so in notes.",
            "ISSUES:",
            "\n".join(issue_lines),
            "PLAN:",
            plan_summary_block(plan) if plan is not None else "(no plan)",
            "SECTIONS:",
            sections_text[:15000],
        ])
        try:
            response = self.client.complete(system, user, max_output_tokens=8192)
            payload = parse_json(response.text)
        except (ProviderError, ResponseParseError) as exc:
            return None, [], [], [f"revision call failed: {exc}"]
        if not isinstance(payload, dict):
            return None, [], [], ["revision reply was not a JSON object"]
        replacements = iter_mapped_sections(payload, brief)
        if not replacements:
            return None, [], [], ["revision reply contained no recognised sections"]
        addressed = [str(i) for i in (payload.get("addressed") or [])]
        notes = [str(n) for n in (payload.get("notes") or [])]
        working = {section.id: section for section in draft.sections}
        changed: List[str] = []
        for item in replacements:
            body = (item.get("body") or "").strip()
            target = working.get(item["id"])
            if target is None or not body or body == target.body.strip():
                continue
            target.body = body
            changed.append(item["id"])
        if not changed:
            return None, [], addressed, notes + ["revision produced no changes"]
        new_draft = Draft(
            task_id=draft.task_id,
            version=draft.version + 1,
            title=draft.title,
            sections=[working[sid] for sid in brief.section_ids if sid in working],
            engine=payload.get("engine", "scientific-reviser"),
        )
        return new_draft, changed, addressed, notes


def run_review_loop(draft: Draft, brief: Brief, *, plan=None,
                    library: Optional[SourceLibrary] = None,
                    ledger=None, runs: Sequence[ExperimentRun] = (),
                    specs: Sequence[ExperimentSpec] = (),
                    benchmarks: Sequence[BenchmarkRecord] = (),
                    retrieval: Optional[Dict[str, Any]] = None,
                    code_analysis: Optional[codecheck.CodeAnalysis] = None,
                    reviewer: Optional[ScientificReviewer] = None,
                    reviser: Optional[TargetedReviser] = None,
                    max_iterations: int = 1,
                    on_iteration: Optional[Callable[[int, Draft, ValidationReport,
                                                     ReviewReport], None]] = None
                    ) -> Tuple[Draft, ValidationReport, ReviewReport]:
    """Review -> targeted revision -> re-validate, up to ``max_iterations``."""
    reviewer = reviewer or ScientificReviewer()
    reviser = reviser or TargetedReviser(reviewer.client)
    library = library or SourceLibrary()
    iterations = max(0, int(max_iterations))
    report = ReviewReport()
    validation: ValidationReport
    previous_issues: List[ReviewIssue] = []
    current = draft

    for iteration in range(0, iterations + 1):
        validation, review_report = reviewer.review(
            current, brief, plan=plan, library=library, ledger=ledger, runs=runs,
            specs=specs, benchmarks=benchmarks, code_analysis=code_analysis,
            retrieval=retrieval, iteration=iteration,
        )
        if previous_issues:
            for old in previous_issues:
                match = next((new for new in review_report.issues
                              if _same_issue(old, new)), None)
                if match is None:
                    old.status = "fixed"
                    match = old
                else:
                    match.status = old.status if old.status == "deferred" else match.status
            fixed_ids = {i.issue_id for i in previous_issues if i.status == "fixed"}
            pending = [i for i in review_report.issues if i.is_blocking]
            summary = RevisionSummary(
                iteration=iteration,
                applied=sorted(fixed_ids),
                unresolved=[i.issue_id for i in pending],
                changed_sections=[],
                notes=[f"re-validated after revision {iteration}"],
            )
            report.revisions.append(summary)
        previous_issues = list(review_report.issues)
        report.issues = review_report.issues
        report.model_review_ran = report.model_review_ran or review_report.model_review_ran
        report.notes.extend(review_report.notes)
        report.notes = list(dict.fromkeys(report.notes))
        if on_iteration is not None:
            on_iteration(iteration, current, validation, review_report)
        if iteration >= iterations:
            break

        eligible = reviser.eligible(review_report.issues)
        deferred = [i for i in review_report.issues
                    if i.is_blocking and i not in eligible]
        for issue in deferred:
            if issue.dimension in EVIDENCE_BOUND_DIMENSIONS:
                issue.status = "deferred"
                report.notes.append(
                    f"{issue.issue_id} deferred: {issue.dimension} cannot be fixed by "
                    "prose revision; the experiment/code/evidence must change"
                )
        if not eligible:
            report.notes.append(
                "no prose-revisible issue remains; remaining issues require new "
                "evidence, code changes or user input"
            )
            break
        new_draft, changed, addressed, notes = reviser.revise(
            current, brief, eligible, plan=plan, library=library, runs=runs,
            iteration=iteration,
        )
        report.revisions.append(RevisionSummary(
            iteration=iteration + 1,
            applied=addressed,
            unresolved=[i.issue_id for i in eligible
                        if i.issue_id not in set(addressed)],
            deferred=[i.issue_id for i in deferred],
            changed_sections=changed,
            notes=notes,
        ))
        if new_draft is None:
            report.notes.append(
                "revision pass produced no usable changes; unresolved issues remain"
            )
            break
        current = new_draft

    report.iterations_run = max((r.iteration for r in report.revisions), default=0)
    # Final validation includes the review checks derived from the final report.
    if not report.issues:
        report.issues = []
    final_validation = validators.validate_manuscript(
        current, brief, plan=plan, library=library, ledger=ledger, runs=runs,
        specs=specs, benchmarks=benchmarks, review=report,
        code_analysis=code_analysis, retrieval=retrieval, include_review=True,
    )
    return current, final_validation, report


def revision_summary_markdown(report: ReviewReport) -> str:
    lines = ["# Scientific review and revision summary", ""]
    counts = {severity.value: len([i for i in report.by_severity(severity)
                                   if i.is_blocking or i.status in ("open", "unresolved")])
              for severity in ReviewSeverity}
    lines.append("## Findings by severity")
    lines.append("")
    lines.append("| severity | unresolved |")
    lines.append("| --- | --- |")
    for severity, count in counts.items():
        lines.append(f"| {severity} | {count} |")
    lines.append("")
    blocking = report.blocking()
    lines.append(f"## Unresolved blocking issues ({len(blocking)})")
    lines.append("")
    if not blocking:
        lines.append("_None._")
    for issue in blocking:
        lines.append(f"### {issue.issue_id} · {issue.severity.value} · {issue.dimension}")
        lines.append("")
        lines.append(f"- **Section:** {issue.section or 'document'}")
        lines.append(f"- **Claim:** {issue.claim}")
        lines.append(f"- **Explanation:** {issue.explanation}")
        if issue.evidence:
            lines.append(f"- **Evidence:** {issue.evidence}")
        lines.append(f"- **Required correction:** {issue.correction}")
        lines.append(f"- **Status:** {issue.status}"
                     + (" (requires user input)" if issue.requires_user_input else ""))
        lines.append("")
    lines.append("## Revision iterations")
    lines.append("")
    if not report.revisions:
        lines.append("_No revision pass was run._")
    for revision in report.revisions:
        lines.append(f"- Iteration {revision.iteration}: "
                     f"fixed={len(revision.applied)}, "
                     f"unresolved={len(revision.unresolved)}, "
                     f"deferred={len(revision.deferred)}, "
                     f"sections changed={', '.join(revision.changed_sections) or 'none'}")
        for note in revision.notes:
            lines.append(f"  - {note}")
    if report.notes:
        lines.append("")
        lines.append("## Reviewer notes")
        for note in dict.fromkeys(report.notes):
            lines.append(f"- {note}")
    return "\n".join(lines).rstrip() + "\n"
