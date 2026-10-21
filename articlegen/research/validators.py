"""Deterministic validation of a research manuscript.

Each check reports one of four statuses and the gate effect it should have:

* ``pass``            - the condition was evaluated and holds;
* ``fail``            - the condition was evaluated and does not hold;
* ``unverified``      - the check could not be performed (for example, no source
  index was reachable). This is never treated as a pass;
* ``not_applicable``  - the selected article type does not require the check.

The validators are deliberately independent of the LLM review. Model review adds
semantic findings on top, but it cannot turn a failed deterministic check into a
pass, and the writing-quality score is not consulted anywhere here.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..models import ArticleType, Brief, Draft
from . import analysis as analysis_module
from . import codecheck
from . import evidence
from .models import (
    BenchmarkRecord,
    CheckStatus,
    ContributionType,
    EvidenceKind,
    ExperimentRun,
    ExperimentSpec,
    ExperimentStatus,
    ReviewIssue,
    ReviewReport,
    ReviewSeverity,
    SourceRecord,
    ValidationCheck,
    ValidationReport,
    classify_section_role,
    profile_for,
)
from .planning import validate_plan
from .sources import CitationResolution, SourceLibrary, split_references

PLACEHOLDER_RE = re.compile(
    r"\b(TODO|TBD|FIXME|XXX|LOREM\s+IPSUM|citation\s+needed|cite\s+this|"
    r"to\s+be\s+(?:determined|completed|supplied)|insert\s+(?:reference|citation)|"
    r"\[citation[^\]]*\]|\[ref\]|\[source\])",
    re.IGNORECASE,
)
TABLE_REF_RE = re.compile(r"\b(Table|Figure)\s+(\d+)\b")
CAPTION_RE = re.compile(r"(?m)^\s*(?:\*\*)?(Table|Figure)\s+(\d+)\s*(?:\*\*)?\s*[:.\-]")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
EQUATION_REF_RE = re.compile(r"\b(?:Eq(?:uation)?\.?)\s*\(?(\d{1,3})\)?")
EQUATION_DEF_RE = re.compile(r"(?m)(?:\\tag\{(\d{1,3})\}|^\s*\(\s*(\d{1,3})\s*\)\s*$)")
METRIC_PAIR_RE = re.compile(
    r"(?P<metric>[A-Za-z][A-Za-z\- ]{2,30}?)\s+"
    r"(?:is|are|equals|reached|reaches|stands?\s+at|was|were|of)\s+"
    r"(?P<value>-?\d[\d,]*(?:\.\d+)?\s*%?)",
    re.IGNORECASE,
)
EMPTY_TABLE_ROW_RE = re.compile(r"(?m)^\s*\|\s*(?:\|\s*)+\s*$")
TABLE_BLOCK_RE = re.compile(r"(?m)((?:^\s*\|.*\|\s*$\n?){2,})")
THEOREM_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?(Theorem|Lemma|Proposition|Corollary|Proof)\b")
PROOF_CLAIM_RE = re.compile(
    r"\b(we\s+prove|proves?|proof\s+of|guarantees?\s+that|mathematically\s+proven|"
    r"for\s+all\s+(?:n|graphs?|inputs?|sizes?)|universally\s+true|establishes?\s+that)\b",
    re.IGNORECASE,
)
UNCERTAINTY_RE = re.compile(
    r"\b(remains?\s+open|not\s+(?:proven|established|shown)|unresolved|conjectur\w+|"
    r"future\s+work|we\s+did\s+not\s+(?:test|prove|show)|open\s+question|"
    r"cannot\s+(?:prove|guarantee|conclude)|pending|planned)\b",
    re.IGNORECASE,
)
COMPLETION_RE = re.compile(
    r"\b(we\s+(?:ran|executed|conducted|measured|implemented|tested)|"
    r"results?\s+(?:show|showed|indicate|were\s+obtained)|"
    r"we\s+(?:found|observed|obtained)|all\s+tests?\s+(?:pass|passed)|"
    r"our\s+(?:experiments?|benchmarks?|tests?)\s+(?:confirm|show|passed))\b",
    re.IGNORECASE,
)
DATA_AVAILABILITY_RE = re.compile(
    r"\b(data\s+availability|dataset\s+(?:is|was)\s+(?:released|available|archived)|"
    r"available\s+at\s+https?://|code\s+availability|replication\s+package)\b",
    re.IGNORECASE,
)
REQUIRED_ROLE_LABELS = {
    "introduction": "Introduction",
    "background": "Background",
    "methods": "Methods",
    "algorithm": "Algorithm",
    "results": "Results",
    "complexity": "Complexity analysis",
    "discussion": "Discussion",
    "limitations": "Limitations",
    "conclusion": "Conclusion",
}

#: Maps a review dimension onto the check category it normally comes from.
CATEGORY_DIMENSIONS = {
    "citations": "literature_and_citations",
    "sources": "literature_and_citations",
    "plan": "research_question_and_contribution",
    "results": "data_and_experimental_integrity",
    "experiments": "data_and_experimental_integrity",
    "code": "algorithmic_or_mathematical_correctness",
    "technical": "algorithmic_or_mathematical_correctness",
    "complexity": "statistical_and_quantitative_claims",
    "performance": "statistical_and_quantitative_claims",
    "benchmark": "reproducibility",
    "statistics": "statistical_and_quantitative_claims",
    "terminology": "internal_consistency",
    "sections": "writing_and_venue_fit",
    "limitations": "limitations_and_validity",
    "numbers": "internal_consistency",
    "structure": "writing_and_venue_fit",
    "novelty": "research_question_and_contribution",
    "review": "internal_consistency",
    "data": "reproducibility",
    "search": "methodological_validity",
}


def _check(check_id: str, description: str, status: CheckStatus, *,
           severity: ReviewSeverity = ReviewSeverity.MAJOR,
           category: str = "general", details: str = "",
           evidence_items: Iterable[str] = (),
           gate_effect: str = "needs_revision",
           scope: Sequence[str] = ()) -> ValidationCheck:
    return ValidationCheck(
        check_id=check_id,
        description=description,
        status=status,
        severity=severity,
        category=category,
        details=details,
        evidence=[str(e)[:400] for e in evidence_items][:10],
        gate_effect=gate_effect,
        article_scope=list(scope),
    )


def _ok(check_id: str, description: str, ok: bool, **kwargs: Any) -> ValidationCheck:
    kwargs.setdefault("severity", ReviewSeverity.MAJOR)
    gate_effect = kwargs.pop("gate_effect", "needs_revision")
    return _check(check_id, description,
                  CheckStatus.PASS if ok else CheckStatus.FAIL,
                  gate_effect=gate_effect, **kwargs)


def _na(check_id: str, description: str, **kwargs: Any) -> ValidationCheck:
    return _check(check_id, description, CheckStatus.NOT_APPLICABLE,
                  gate_effect="none", **kwargs)


def _unverified(check_id: str, description: str, **kwargs: Any) -> ValidationCheck:
    kwargs.setdefault("severity", ReviewSeverity.CRITICAL)
    kwargs["gate_effect"] = kwargs.get("gate_effect", "incomplete")
    return _check(check_id, description, CheckStatus.UNVERIFIED, **kwargs)


def _roles(draft: Draft) -> Dict[str, str]:
    return {section.id: classify_section_role(section.id, section.title)
            for section in draft.sections}


def _role_text(draft: Draft, wanted: Iterable[str]) -> str:
    wanted = set(wanted)
    return "\n\n".join(section.body for section in draft.sections
                       if classify_section_role(section.id, section.title) in wanted)


def _prose(draft: Draft) -> str:
    return evidence.strip_code("\n\n".join(section.body for section in draft.sections))


def _full_text(draft: Draft) -> str:
    return draft.to_markdown()


def _numbers(text: str) -> List[float]:
    values: List[float] = []
    for match in re.finditer(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w.])", text):
        try:
            values.append(float(match.group(1).replace(",", "")))
        except ValueError:
            continue
    return values


def _known_result_values(ledger: evidence.EvidenceLedger,
                         runs: Sequence[ExperimentRun]) -> Set[float]:
    values: Set[float] = set()
    for result in ledger.results:
        try:
            values.add(float(result.value))
        except (TypeError, ValueError):
            continue
    for run in runs:
        for key, value in run.metrics.items():
            if isinstance(value, (int, float)) and not key.endswith("_unit"):
                values.add(float(value))
        for repetition in run.repetition_metrics:
            for value in repetition.values():
                if isinstance(value, (int, float)):
                    values.add(float(value))
    return values


# ---------------------------------------------------------------------------
# Individual check groups
# ---------------------------------------------------------------------------


def _check_plan_checks(plan, brief: Brief, library: SourceLibrary,
                       retrieval: Optional[Dict[str, Any]]) -> List[ValidationCheck]:
    if plan is None:
        if not profile_for(brief.article_type).research:
            return [_na("plan.present",
                        "A research plan is not required for this article type",
                        category="plan")]
        return [_unverified(
            "plan.present", "A structured research plan was validated",
            severity=ReviewSeverity.CRITICAL,
            details="no research plan was produced for a research article",
            category="plan",
        )]
    retrieval = retrieval or {}
    validate_plan(
        plan, brief, library=library,
        retrieval_attempted=bool(retrieval.get("attempted")),
        retrieval_available=bool(retrieval.get("available")),
    )
    return list(plan.issues)


def _check_source_and_citations(draft: Draft, brief: Brief, plan, library: SourceLibrary,
                                retrieval: Optional[Dict[str, Any]],
                                scope: Sequence[str]) -> Tuple[List[ValidationCheck], CitationResolution]:
    checks: List[ValidationCheck] = []
    profile = profile_for(brief.article_type)
    report = library.report()
    resolution = library.resolve_citations(_full_text(draft))
    needs_sources = profile.requires_sources or brief.requires_citations

    if not needs_sources:
        checks.append(_na("citations.resolution",
                          "Citation resolution is required only when sources are required",
                          category="citations"))
        checks.append(_na("sources.verified",
                          "Source verification is required only for research articles",
                          category="sources"))
        return checks, resolution

    retrieval = retrieval or {}
    if not library.records:
        if retrieval.get("attempted") and not retrieval.get("available"):
            checks.append(_unverified(
                "sources.retrieved", "Scholarly sources were retrieved",
                category="sources",
                details=("source retrieval was attempted but the index was unavailable: "
                         + "; ".join(retrieval.get("errors") or ["no provider"])),
                severity=ReviewSeverity.CRITICAL,
            ))
        else:
            checks.append(_ok(
                "sources.retrieved", "Scholarly sources were retrieved", False,
                category="sources", severity=ReviewSeverity.CRITICAL,
                details=("the manuscript requires citations but the source library is "
                         "empty; a bibliography must not be fabricated"),
                gate_effect="incomplete",
            ))
        checks.append(_ok("sources.verified", "Every bibliography entry is verified", False,
                          category="sources", severity=ReviewSeverity.CRITICAL,
                          details="no source records exist to verify",
                          gate_effect="incomplete"))
    else:
        checks.append(_ok("sources.retrieved", "Scholarly sources were retrieved",
                          report.total > 0, category="sources",
                          details=f"{report.total} record(s) in the library"))
        if report.verified == 0:
            checks.append(_unverified(
                "sources.verified", "Every bibliography entry is verified",
                category="sources",
                details=(f"{report.total} record(s) present but none could be verified "
                         f"against an index ({report.user_provided} user-provided, "
                         f"{report.unverified + report.metadata_incomplete} unverified)"),
                severity=ReviewSeverity.CRITICAL,
            ))
        elif report.verified < report.total:
            checks.append(_ok(
                "sources.verified", "Every bibliography entry is verified", False,
                category="sources", severity=ReviewSeverity.MAJOR,
                details=(f"{report.verified}/{report.total} record(s) verified; "
                         f"{report.user_provided} user-provided, "
                         f"{report.unverified} unverified, "
                         f"{report.metadata_incomplete} incomplete metadata"),
                gate_effect="incomplete",
            ))
        else:
            checks.append(_ok("sources.verified", "Every bibliography entry is verified",
                              True, category="sources",
                              details=f"{report.verified} record(s) index-verified"))

    if resolution.fabricated_entries:
        checks.append(_ok(
            "citations.bibliography_grounded",
            "Every reference-list entry corresponds to a retrieved source", False,
            category="citations", severity=ReviewSeverity.CRITICAL,
            details="entries with no matching source record: "
                    + " | ".join(resolution.fabricated_entries[:4]),
            evidence_items=resolution.fabricated_entries[:4],
        ))
    else:
        checks.append(_ok("citations.bibliography_grounded",
                          "Every reference-list entry corresponds to a retrieved source",
                          True, category="citations",
                          details=f"{len(resolution.matched_entries)} matched entry(ies)"))

    if resolution.unresolved:
        checks.append(_ok(
            "citations.resolution",
            "Every in-text citation resolves to a bibliography entry", False,
            category="citations", severity=ReviewSeverity.CRITICAL,
            details="; ".join(resolution.unresolved[:6]),
            evidence_items=resolution.unresolved[:6],
        ))
    elif resolution.has_citations:
        checks.append(_ok("citations.resolution",
                          "Every in-text citation resolves to a bibliography entry",
                          True, category="citations",
                          details=(f"{len(resolution.numeric_markers)} numeric and "
                                   f"{len(resolution.author_year_markers)} author-year "
                                   f"marker(s) resolved")))
    else:
        checks.append(_ok(
            "citations.resolution",
            "Every in-text citation resolves to a bibliography entry", False,
            category="citations", severity=ReviewSeverity.CRITICAL,
            details="the manuscript has no in-text citations even though citations are "
                    "required", gate_effect="incomplete",
        ))

    if resolution.unused_entries:
        checks.append(_ok(
            "citations.all_entries_used", "Every bibliography entry is cited", False,
            category="citations", severity=ReviewSeverity.MINOR,
            details="uncited entries: " + "; ".join(resolution.unused_entries[:5]),
            gate_effect="none",
        ))
    else:
        checks.append(_ok("citations.all_entries_used",
                          "Every bibliography entry is cited", True,
                          category="citations", severity=ReviewSeverity.MINOR,
                          gate_effect="none"))
    return checks, resolution


def _check_results_and_evidence(draft: Draft, brief: Brief, plan, ledger,
                                runs: Sequence[ExperimentRun],
                                specs: Sequence[ExperimentSpec],
                                code_analysis: codecheck.CodeAnalysis,
                                scope: Sequence[str]) -> List[ValidationCheck]:
    checks: List[ValidationCheck] = []
    profile = profile_for(brief.article_type)
    results_text = _role_text(draft, ("results",))
    abstract_text = _role_text(draft, ("abstract",))
    conclusion_text = _role_text(draft, ("conclusion",))
    discussion_text = _role_text(draft, ("discussion",))
    full_prose = _prose(draft)
    successful_runs = [r for r in runs if r.status == ExperimentStatus.OK]
    any_run = [r for r in runs if r.status not in (ExperimentStatus.NOT_RUN,
                                                   ExperimentStatus.SKIPPED)]
    pending = evidence.detect_pending_experiments(
        "\n\n".join([results_text, abstract_text, conclusion_text]))
    completion = evidence.detect_completion_claims(
        "\n\n".join([results_text, abstract_text, conclusion_text]))

    if pending:
        checks.append(_ok(
            "results.no_pending_work", "Results contain no pending or planned work",
            False, category="results", severity=ReviewSeverity.CRITICAL,
            details="planned/pending experiment language appears in the results, "
                    "abstract or conclusion: " + ", ".join(sorted(set(pending))[:5]),
            evidence_items=pending[:5],
        ))
    else:
        checks.append(_ok("results.no_pending_work",
                          "Results contain no pending or planned work", True,
                          category="results", severity=ReviewSeverity.MAJOR))

    requires_experiments = profile.requires_experiments or brief.requires_experiments \
        or bool(plan is not None and plan.required_experiments)
    if results_text and completion and not successful_runs:
        checks.append(_ok(
            "results.evidence",
            "Every reported result has an executed experiment behind it", False,
            category="results", severity=ReviewSeverity.CRITICAL,
            details="the manuscript reports completed findings but no experiment run "
                    "succeeded; this is an invented or unexecuted result",
            evidence_items=completion[:5], gate_effect="incomplete",
        ))
    elif requires_experiments and not successful_runs:
        detail = "no experiment was executed"
        if runs:
            detail = "no experiment run succeeded (statuses: " + ", ".join(
                sorted({r.status.value for r in runs})) + ")"
        if not brief.execute_experiments:
            detail = "experiment execution is disabled for this run"
        checks.append(_ok(
            "results.evidence",
            "Every reported result has an executed experiment behind it", False,
            category="results", severity=ReviewSeverity.CRITICAL,
            details=detail + "; the manuscript cannot be presented as a completed study",
            gate_effect="incomplete",
        ))
    elif results_text:
        checks.append(_ok("results.evidence",
                          "Every reported result has an executed experiment behind it",
                          True, category="results",
                          details=f"{len(successful_runs)} successful run(s)"))
    else:
        checks.append(_na("results.evidence",
                          "Results evidence is required only for studies with results",
                          category="results"))

    if requires_experiments and not results_text:
        checks.append(_ok("results.section_present",
                          "A Results section reports the executed experiments", False,
                          category="results", severity=ReviewSeverity.MAJOR,
                          details="the plan requires experiments but no Results section "
                                  "contains them", gate_effect="needs_revision"))
    else:
        checks.append(_ok("results.section_present",
                          "A Results section reports the executed experiments",
                          bool(results_text) or not requires_experiments,
                          category="results", severity=ReviewSeverity.MAJOR,
                          gate_effect="needs_revision"))

    if results_text and successful_runs:
        known = _known_result_values(ledger, runs)
        untraceable: List[str] = []
        for sentence in evidence.split_sentences(results_text):
            if not re.search(r"\b(result|measure|observ|achiev|obtain|found|"
                             r"performance|latency|runtime|accuracy|error|score)\b",
                             sentence, re.IGNORECASE):
                continue
            for number in _numbers(sentence):
                if number.is_integer() and 0 <= number <= 3:
                    continue
                if not any(abs(number - value) <= max(1e-6, abs(value) * 0.02)
                           for value in known):
                    untraceable.append(f"{number:g}: {sentence[:160]}")
                    break
        if untraceable:
            checks.append(_ok(
                "results.numbers_traceable",
                "Reported numbers match recorded experiment results", False,
                category="results", severity=ReviewSeverity.CRITICAL,
                details=("numbers with no matching recorded result (tolerance 2%): "
                         + " | ".join(untraceable[:4])),
                evidence_items=untraceable[:4],
            ))
        else:
            checks.append(_ok("results.numbers_traceable",
                              "Reported numbers match recorded experiment results", True,
                              category="results",
                              details=f"{len(known)} recorded value(s) available"))
    elif results_text and not successful_runs:
        checks.append(_ok("results.numbers_traceable",
                          "Reported numbers match recorded experiment results", False,
                          category="results", severity=ReviewSeverity.CRITICAL,
                          details="there are no recorded results to trace",
                          gate_effect="incomplete"))
    else:
        checks.append(_na("results.numbers_traceable",
                          "No results section to trace", category="results"))

    # Table cells in the results must be traceable to recorded results, not invented.
    if results_text:
        known_values = _known_result_values(ledger, runs)
        table_cells: List[float] = []
        for match in TABLE_BLOCK_RE.finditer(results_text):
            for line in match.group(1).splitlines()[2:]:
                for cell in [c.strip() for c in line.strip().strip("|").split("|")]:
                    cleaned = cell.replace(",", "").replace("%", "").strip()
                    if re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned):
                        table_cells.append(float(cleaned))
        if table_cells and successful_runs:
            untraceable_cells = [
                value for value in table_cells
                if not any(abs(value - known) <= max(1e-6, abs(known) * 0.02)
                           for known in known_values)
            ]
            checks.append(_ok(
                "tables.numbers_traceable",
                "Result-table values match the recorded experiment results",
                not untraceable_cells, category="numbers",
                severity=ReviewSeverity.CRITICAL,
                details=("table values with no matching recorded result: "
                         + ", ".join(f"{v:g}" for v in untraceable_cells[:8]))
                if untraceable_cells else
                f"{len(table_cells)} table cell(s) match recorded values",
                gate_effect="needs_revision",
            ))
        elif table_cells:
            checks.append(_ok(
                "tables.numbers_traceable",
                "Result-table values match the recorded experiment results", False,
                category="numbers", severity=ReviewSeverity.CRITICAL,
                details="a results table contains values but no experiment was executed",
                gate_effect="incomplete",
            ))
        else:
            checks.append(_na("tables.numbers_traceable", "No numeric result table",
                              category="numbers"))

    # The abstract and conclusion may not disagree about what was demonstrated.
    polarity = None
    reasons: List[str] = []
    if abstract_text and (conclusion_text or discussion_text):
        tail = conclusion_text or discussion_text
        if PROOF_CLAIM_RE.search(abstract_text) and UNCERTAINTY_RE.search(tail):
            polarity = "abstract claims proof while the conclusion reports open work"
            reasons.append("abstract: " + PROOF_CLAIM_RE.search(abstract_text).group(0)
                           + "; conclusion: "
                           + UNCERTAINTY_RE.search(tail).group(0))
        elif COMPLETION_RE.search(abstract_text) and evidence.detect_pending_experiments(tail):
            polarity = "abstract reports completed results while the conclusion reports pending work"
            reasons.append("completed-result language appears in the abstract but the "
                           "conclusion describes the work as pending")
        elif evidence.detect_pending_experiments(abstract_text) and COMPLETION_RE.search(tail):
            polarity = "abstract reports pending work while the conclusion reports completed results"
            reasons.append("pending language in the abstract; completed language in the conclusion")
    if polarity:
        checks.append(_ok("abstract.conclusion_consistency",
                          "Abstract and conclusion agree on what was demonstrated", False,
                          category="numbers", severity=ReviewSeverity.CRITICAL,
                          details=polarity + " (" + "; ".join(reasons) + ")",
                          evidence_items=reasons))
    elif abstract_text:
        checks.append(_ok("abstract.conclusion_consistency",
                          "Abstract and conclusion agree on what was demonstrated", True,
                          category="numbers"))
    else:
        checks.append(_na("abstract.conclusion_consistency",
                          "No abstract and conclusion pair to compare",
                          category="numbers"))

    # Cross-section numeric consistency, including abstract vs results vs conclusion.
    by_role = {
        "abstract": abstract_text, "results": results_text,
        "discussion": discussion_text, "conclusion": conclusion_text,
    }
    seen: Dict[str, Dict[str, List[str]]] = {}
    for role, text in by_role.items():
        for match in METRIC_PAIR_RE.finditer(text):
            metric = re.sub(r"[^a-z0-9 ]+", " ", match.group("metric").lower()).strip()
            metric = re.sub(r"^(the|a|an|our|this|these|those)\s+", "", metric)
            value = re.sub(r"\s+", "", match.group("value")).replace(",", "")
            if metric:
                seen.setdefault(metric, {}).setdefault(value, []).append(role)
    conflicts = {metric: values for metric, values in seen.items() if len(values) > 1}
    if conflicts:
        details = "; ".join(
            f"{metric}: " + ", ".join(f"{value} ({'/'.join(sorted(set(roles)))})"
                                      for value, roles in values.items())
            for metric, values in list(conflicts.items())[:4]
        )
        checks.append(_ok("numbers.cross_section_consistency",
                          "Numbers agree across abstract, results and conclusion", False,
                          category="numbers", severity=ReviewSeverity.CRITICAL,
                          details=details))
    else:
        checks.append(_ok("numbers.cross_section_consistency",
                          "Numbers agree across abstract, results and conclusion", True,
                          category="numbers"))

    # Hypotheses must not be presented as results.
    hypothesis_in_results = [
        m.group(0) for m in evidence.HYPOTHESIS_RE.finditer(results_text)
    ]
    if hypothesis_in_results:
        checks.append(_ok("results.no_hypotheses",
                          "Hypotheses are not presented as findings", False,
                          category="results", severity=ReviewSeverity.MAJOR,
                          details="hypothesis language inside the Results section: "
                                  + ", ".join(sorted(set(hypothesis_in_results))[:4]),
                          evidence_items=hypothesis_in_results[:4]))
    else:
        checks.append(_ok("results.no_hypotheses",
                          "Hypotheses are not presented as findings", True,
                          category="results"))

    # Claim kinds must include explicit labels for anything that is not evidence.
    if ledger.claims:
        counts = ledger.summary()["kinds"]
        unverified = counts.get(EvidenceKind.UNVERIFIED.value, 0)
        total = max(1, ledger.summary()["claims"])
        checks.append(_ok(
            "evidence.labels", "Claims carry an explicit epistemic status",
            unverified / total <= 0.5, category="results",
            severity=ReviewSeverity.MODERATE,
            details=(f"{unverified}/{total} extracted claim(s) have no identifiable "
                     "evidence; label them as hypotheses, assumptions or future work, "
                     "or support them"),
            gate_effect="needs_revision",
        ))
    else:
        checks.append(_na("evidence.labels", "No checkable claims were extracted",
                          category="results"))
    return checks


def _check_code_and_tests(draft: Draft, brief: Brief,
                          code_analysis: codecheck.CodeAnalysis,
                          runs: Sequence[ExperimentRun],
                          scope: Sequence[str]) -> List[ValidationCheck]:
    checks: List[ValidationCheck] = []
    profile = profile_for(brief.article_type)
    research = profile.research
    analysis = code_analysis

    if analysis.compile_errors:
        checks.append(_ok(
            "code.compile", "Displayed code compiles", False, category="code",
            severity=ReviewSeverity.CRITICAL,
            details="; ".join(
                f"{e['section']}#{e['block']}: {e['message']} (line {e['line']})"
                for e in analysis.compile_errors[:4]),
            evidence_items=[e.get("code", "") for e in analysis.compile_errors[:4]],
        ))
    elif analysis.blocks:
        checks.append(_ok("code.compile", "Displayed code compiles", True,
                          category="code", details=f"{len(analysis.blocks)} block(s)"))
    else:
        checks.append(_na("code.compile", "No code blocks to compile", category="code"))

    if analysis.unlabelled_blocks:
        checks.append(_ok("code.language_tags", "Every code block has a language tag",
                          False, category="code", severity=ReviewSeverity.MINOR,
                          details=f"{len(analysis.unlabelled_blocks)} block(s) untagged",
                          gate_effect="none"))
    else:
        checks.append(_na("code.language_tags", "No untagged code blocks", category="code")
                      if not analysis.blocks else
                      _ok("code.language_tags", "Every code block has a language tag", True,
                          category="code", severity=ReviewSeverity.MINOR, gate_effect="none"))

    hash_mismatches = analysis.hash_mismatches
    if hash_mismatches:
        checks.append(_ok(
            "code.displayed_matches_executed",
            "Displayed code matches the code that was executed", False,
            category="code", severity=ReviewSeverity.CRITICAL,
            details="; ".join(m["message"] for m in hash_mismatches[:3]),
        ))
    elif analysis.executable_blocks:
        checks.append(_ok("code.displayed_matches_executed",
                          "Displayed code matches the code that was executed", True,
                          category="code",
                          details="every executed hash corresponds to a displayed block"))
    else:
        checks.append(_na("code.displayed_matches_executed", "No executable code",
                          category="code"))

    if analysis.unexecuted_blocks:
        exempt = []
        for block in analysis.unexecuted_blocks:
            section = next((s for s in draft.sections if s.id == block.section_id), None)
            body_lower = (section.body if section else "").lower()
            labelled = any(marker in body_lower for marker in
                           ("illustrative", "not executed", "untested", "not run",
                            "example only", "pseudo-code", "pseudocode"))
            if block in analysis.illustrative_blocks or labelled:
                exempt.append(block)
        serious = [b for b in analysis.unexecuted_blocks if b not in exempt]
        if serious:
            status_details = (
                f"{len(serious)} executable block(s) were never executed: "
                + ", ".join(f"{b.section_id}#{b.index}" for b in serious[:5])
            )
            checks.append(_ok(
                "code.executed", "Executable code blocks were actually run", False,
                category="code",
                severity=ReviewSeverity.CRITICAL if not analysis.executed_tests else
                ReviewSeverity.MAJOR,
                details=status_details + "; label illustrative snippets explicitly or "
                        "execute them",
                gate_effect="incomplete" if not any(r.status.succeeded for r in runs)
                else "needs_revision",
            ))
        else:
            checks.append(_ok("code.executed", "Executable code blocks were actually run",
                              True, category="code",
                              details=f"{len(exempt)} illustrative block(s) are labelled"))
    else:
        checks.append(_na("code.executed", "No executable code blocks", category="code"))

    # The Erdős–Gyárfás class: prose claims generality, code hard-codes the domain.
    if analysis.generality_gaps:
        worst = max(gap.severity for gap in analysis.generality_gaps)
        checks.append(_ok(
            "code.generality",
            "Implementation matches the claimed generality of the algorithm", False,
            category="technical", severity=worst,
            details=" | ".join(gap.detail for gap in analysis.generality_gaps[:4]),
            evidence_items=[str(gap.domain.values) for gap in analysis.generality_gaps[:4]],
        ))
    elif analysis.executable_blocks:
        checks.append(_ok("code.generality",
                          "Implementation matches the claimed generality of the algorithm",
                          True, category="technical",
                          details="no fixed numeric domain conflicts with the prose"))
    else:
        checks.append(_na("code.generality", "No executable code to compare",
                          category="technical"))

    if analysis.enforcement_gaps:
        checks.append(_ok(
            "code.claim_alignment",
            "Normative claims about behaviour are enforced by the implementation", False,
            category="technical", severity=ReviewSeverity.MAJOR,
            details=" | ".join(f"{gap.claim} - {gap.reason}"
                               for gap in analysis.enforcement_gaps[:3]),
            evidence_items=[gap.claim for gap in analysis.enforcement_gaps[:3]],
        ))
    elif analysis.executable_blocks:
        checks.append(_ok("code.claim_alignment",
                          "Normative claims about behaviour are enforced by the implementation",
                          True, category="technical",
                          details="no unguarded universal claim detected"))
    else:
        checks.append(_na("code.claim_alignment", "No executable code to compare",
                          category="technical"))

    if analysis.missing_dependencies:
        checks.append(_ok(
            "code.dependencies", "Required dependencies are available", False,
            category="technical", severity=ReviewSeverity.CRITICAL,
            details="missing module(s): " + ", ".join(
                sorted(set(analysis.missing_dependencies))),
            gate_effect="incomplete",
        ))
    elif analysis.runtime_errors:
        details = "; ".join(
            f"{err['experiment_id']} [{err['status']}] "
            f"{err.get('error_type') or ''}: {err.get('error', '')[:120]}"
            for err in analysis.runtime_errors[:3]
        )
        checks.append(_ok("code.dependencies", "Required dependencies are available",
                          False, category="technical", severity=ReviewSeverity.MAJOR,
                          details="experiment execution errors: " + details,
                          gate_effect="needs_revision"))
    elif runs:
        checks.append(_ok("code.dependencies", "Required dependencies are available",
                          True, category="technical",
                          details="every executed experiment ran without a dependency error"))
    else:
        checks.append(_na("code.dependencies", "No experiment execution to check",
                          category="technical"))

    test_failures = [t for run in runs for t in run.tests if t.status in ("failed", "error")]
    if test_failures:
        checks.append(_ok(
            "tests.passed", "Executed tests pass", False, category="code",
            severity=ReviewSeverity.CRITICAL,
            details="; ".join(f"{t.name}: {t.status} {t.detail[-120:]}"
                              for t in test_failures[:3]),
        ))
    elif any(run.tests for run in runs):
        checks.append(_ok("tests.passed", "Executed tests pass", True,
                          category="code",
                          details=f"{sum(run.tests_passed for run in runs)} passing test(s)"))
    elif profile.requires_experiments and profile.requires_correctness_argument:
        checks.append(_ok("tests.passed", "Executed tests pass", False,
                          category="code", severity=ReviewSeverity.MAJOR,
                          details="no tests were executed for an algorithmic contribution",
                          gate_effect="incomplete"))
    else:
        checks.append(_na("tests.passed", "No tests were executed", category="code"))

    claim_mismatch = codecheck.tests_claimed_but_not_run(analysis)
    if claim_mismatch:
        severity = ReviewSeverity.CRITICAL if analysis.executed_tests == 0 \
            else ReviewSeverity.MAJOR
        checks.append(_ok(
            "tests.claimed_vs_executed",
            "Claimed tests match the tests that were displayed and executed", False,
            category="code", severity=severity, details=claim_mismatch,
            gate_effect="incomplete" if analysis.executed_tests == 0 else "needs_revision",
        ))
    else:
        checks.append(_ok("tests.claimed_vs_executed",
                          "Claimed tests match the tests that were displayed and executed",
                          True, category="code",
                          details=(f"claimed={analysis.claimed_tests}, "
                                   f"displayed={analysis.displayed_tests}, "
                                   f"executed={analysis.executed_tests}")))
    return checks


def _check_complexity_and_benchmarks(draft: Draft, brief: Brief,
                                     benchmarks: Sequence[BenchmarkRecord],
                                     complexity: analysis_module.ComplexityAnalysis,
                                     scope: Sequence[str]) -> List[ValidationCheck]:
    checks: List[ValidationCheck] = []
    profile = profile_for(brief.article_type)
    if profile.research:
        needs_complexity = profile.requires_complexity_analysis
        has_claims = bool(complexity.claims)
        if has_claims or needs_complexity:
            if not has_claims and needs_complexity:
                checks.append(_ok(
                    "complexity.analysis",
                    "Theoretical complexity analysis is present", False,
                    category="complexity", severity=ReviewSeverity.MAJOR,
                    details="the article type requires a complexity analysis but none "
                            "was found",
                ))
            else:
                problems = []
                if not complexity.analysis_present:
                    problems.append("no paragraph states the input-size parameters, time "
                                    "and memory bounds and assumptions together")
                if complexity.missing_elements:
                    problems.append("missing: " + ", ".join(complexity.missing_elements))
                checks.append(_ok(
                    "complexity.analysis", "Theoretical complexity analysis is present",
                    not problems, category="complexity", severity=ReviewSeverity.MAJOR,
                    details=("; ".join(problems) if problems else
                             "input parameters, time, memory and assumptions present"),
                    evidence_items=complexity.claims[:3],
                ))
        else:
            checks.append(_na("complexity.analysis",
                              "No complexity claim to analyse", category="complexity"))

        if complexity.benchmark_claims_without_records:
            checks.append(_ok(
                "performance.measured",
                "Empirical performance claims are backed by recorded measurements",
                False, category="performance", severity=ReviewSeverity.CRITICAL,
                details="claims with no benchmark record: "
                        + " | ".join(complexity.benchmark_claims_without_records[:4]),
                evidence_items=complexity.benchmark_claims_without_records[:4],
                gate_effect="incomplete",
            ))
        elif complexity.has_performance_claims and benchmarks:
            checks.append(_ok("performance.measured",
                              "Empirical performance claims are backed by recorded measurements",
                              True, category="performance",
                              details=f"{len(benchmarks)} benchmark record(s)"))
        else:
            checks.append(_na("performance.measured",
                              "No empirical performance claim", category="performance"))

        if complexity.untraceable_numbers:
            checks.append(_ok(
                "performance.numbers_traceable",
                "Reported performance numbers appear in the recorded measurements",
                False, category="performance", severity=ReviewSeverity.CRITICAL,
                details=" | ".join(complexity.untraceable_numbers[:3]),
                evidence_items=complexity.untraceable_numbers[:3],
            ))
        else:
            checks.append(_na("performance.numbers_traceable",
                              "No untraceable performance numbers detected",
                              category="performance"))

        if complexity.benchmark_metadata_issues:
            checks.append(_ok(
                "benchmark.metadata",
                "Benchmark metadata is complete enough to reproduce the measurement",
                False, category="benchmark", severity=ReviewSeverity.MAJOR,
                details="; ".join(complexity.benchmark_metadata_issues[:5]),
            ))
        elif benchmarks:
            checks.append(_ok("benchmark.metadata",
                              "Benchmark metadata is complete enough to reproduce the measurement",
                              True, category="benchmark",
                              details=f"{len(benchmarks)} record(s) complete"))
        else:
            checks.append(_na("benchmark.metadata", "No benchmark records",
                              category="benchmark"))

        if complexity.statistical_gaps:
            checks.append(_ok(
                "statistics.reporting",
                "Quantitative comparisons report uncertainty or repetitions", False,
                category="statistics", severity=ReviewSeverity.MAJOR,
                details="; ".join(complexity.statistical_gaps[:3]),
            ))
        else:
            checks.append(_ok("statistics.reporting",
                              "Quantitative comparisons report uncertainty or repetitions",
                              True, category="statistics",
                              details="no unsupported comparative claim detected"))

        if complexity.undefined_terms:
            checks.append(_ok(
                "terminology.defined",
                "Informal or ambiguous terms are operationally defined", False,
                category="terminology", severity=ReviewSeverity.MAJOR,
                details="undefined quality term(s): "
                        + ", ".join(complexity.undefined_terms[:8])
                        + ". Define the metric (e.g. the threshold and measurement) "
                          "before using the term as evidence.",
            ))
        else:
            checks.append(_ok("terminology.defined",
                              "Informal or ambiguous terms are operationally defined",
                              True, category="terminology"))
    else:
        checks.append(_na("complexity.analysis",
                          "Complexity analysis is not required for this article type",
                          category="complexity"))
        checks.append(_na("terminology.defined",
                          "Research terminology checks are not required for this type",
                          category="terminology"))

    if profile.requires_search_strategy:
        checks.append(_ok(
            "search.strategy", "The review reports a reproducible search strategy",
            not complexity.search_strategy_gaps, category="search",
            severity=ReviewSeverity.MAJOR,
            details=("missing: " + ", ".join(complexity.search_strategy_gaps))
            if complexity.search_strategy_gaps else "search protocol elements present",
        ))
    elif complexity.mislabelled_review:
        checks.append(_ok(
            "search.strategy", "Review type is labelled accurately", False,
            category="search", severity=ReviewSeverity.CRITICAL,
            details="; ".join(complexity.mislabelled_review),
        ))
    else:
        checks.append(_na("search.strategy", "Search protocol is not required",
                          category="search"))
    return checks


def _check_structure_and_consistency(draft: Draft, brief: Brief, library: SourceLibrary,
                                     scope: Sequence[str]) -> List[ValidationCheck]:
    checks: List[ValidationCheck] = []
    profile = profile_for(brief.article_type)
    roles = _roles(draft)
    present = set(roles.values())
    missing = [role for role in profile.required_roles if role not in present]
    if profile.research:
        checks.append(_ok(
            "sections.required",
            f"Required sections for a {profile.label.lower()} are present",
            not missing, category="sections", severity=ReviewSeverity.MAJOR,
            details=("missing: " + ", ".join(REQUIRED_ROLE_LABELS.get(r, r)
                                             for r in missing)) if missing else
                    "all required roles present",
            evidence_items=sorted(present),
        ))
    else:
        checks.append(_na("sections.required",
                          "Required-section policy applies to research article types",
                          category="sections"))

    limitations = _role_text(draft, ("limitations",))
    if profile.research:
        substantive = bool(limitations.strip()) and len(limitations.split()) >= 15 and bool(
            re.search(r"\b(limit|assum|cannot|does\s+not|threat|scope|only|"
                      r"unverified|risk|confound|caveat)\b", limitations, re.IGNORECASE))
        checks.append(_ok(
            "limitations.present", "Limitations are present and consistent with the methods",
            substantive, category="limitations", severity=ReviewSeverity.MAJOR,
            details=("no substantive limitations section" if not limitations.strip() else
                     "limitations are too thin or do not state a scope/assumption"),
        ))
    else:
        checks.append(_na("limitations.present", "Limitations are optional for this type",
                          category="limitations"))

    # Placeholders and empty tables.
    full_text = _full_text(draft)
    placeholders = sorted({m.group(0) for m in PLACEHOLDER_RE.finditer(full_text)})
    if placeholders:
        checks.append(_ok(
            "completeness.placeholders",
            "The manuscript contains no placeholders or unresolved citations", False,
            category="structure", severity=ReviewSeverity.CRITICAL,
            details="placeholder markers: " + ", ".join(placeholders[:6]),
            evidence_items=placeholders[:6],
        ))
    else:
        checks.append(_ok("completeness.placeholders",
                          "The manuscript contains no placeholders or unresolved citations",
                          True, category="structure"))

    empty_tables: List[str] = []
    for section in draft.sections:
        for match in TABLE_BLOCK_RE.finditer(section.body):
            lines = [line for line in match.group(1).splitlines() if line.strip()]
            data_rows = [line for line in lines[2:] if line.strip(" |-\t")]
            if len(lines) >= 2 and not data_rows:
                empty_tables.append(f"{section.id}: table with headers but no data rows")
    if empty_tables:
        checks.append(_ok(
            "tables.non_empty", "Tables contain data rather than empty rows", False,
            category="structure", severity=ReviewSeverity.MAJOR,
            details="; ".join(empty_tables[:4]),
        ))
    else:
        checks.append(_ok("tables.non_empty", "Tables contain data rather than empty rows",
                          True, category="structure", severity=ReviewSeverity.MINOR))

    # Table and figure references resolve to captions.
    body_lines = full_text.splitlines()
    caption_lines = [line for line in body_lines if CAPTION_RE.match(line)]
    prose_lines = [line for line in body_lines if not CAPTION_RE.match(line)]
    referenced = {(kind, int(num)) for kind, num in
                  TABLE_REF_RE.findall("\n".join(prose_lines))}
    captions = {(kind, int(num)) for kind, num in
                (m.groups() for line in caption_lines
                 for m in [CAPTION_RE.match(line)] if m)}
    has_images = bool(IMAGE_RE.search(full_text))
    missing_captions = sorted(ref for ref in referenced if ref not in captions
                              and not (ref[0] == "Figure" and has_images))
    if missing_captions:
        checks.append(_ok(
            "tables.figures_exist", "Every referenced table and figure exists", False,
            category="structure", severity=ReviewSeverity.MAJOR,
            details="referenced but no caption: " + ", ".join(
                f"{kind} {num}" for kind, num in missing_captions[:5]),
        ))
    elif referenced:
        checks.append(_ok("tables.figures_exist",
                          "Every referenced table and figure exists", True,
                          category="structure"))
    else:
        checks.append(_na("tables.figures_exist", "No table or figure references",
                          category="structure"))
    unused_captions = sorted(cap for cap in captions if cap not in referenced)
    if unused_captions:
        checks.append(_ok("tables.figures_referenced", "Every table and figure is cited",
                          False, category="structure", severity=ReviewSeverity.MINOR,
                          details="captions never referenced: " + ", ".join(
                              f"{kind} {num}" for kind, num in unused_captions[:5]),
                          gate_effect="none"))
    else:
        checks.append(_na("tables.figures_referenced", "No unreferenced captions",
                          category="structure"))

    # Equation references.
    equation_refs = {int(num) for num in EQUATION_REF_RE.findall(_prose(draft))}
    equation_defs = {int(a or b) for a, b in EQUATION_DEF_RE.findall(_full_text(draft))}
    missing_equations = sorted(ref for ref in equation_refs if ref not in equation_defs)
    if missing_equations:
        checks.append(_ok(
            "equations.resolve", "Every equation reference resolves", False,
            category="structure", severity=ReviewSeverity.MAJOR,
            details="references with no numbered equation: "
                    + ", ".join(f"Eq. {n}" for n in missing_equations[:5]),
        ))
    else:
        checks.append(_ok("equations.resolve", "Every equation reference resolves", True,
                          category="structure") if equation_refs else
                      _na("equations.resolve", "No equation references", category="structure"))

    return checks


def _check_novelty_and_proof(draft: Draft, brief: Brief, plan, library: SourceLibrary,
                             scope: Sequence[str]) -> List[ValidationCheck]:
    checks: List[ValidationCheck] = []
    profile = profile_for(brief.article_type)
    abstract = _role_text(draft, ("abstract",))
    results = _role_text(draft, ("results",))
    conclusion = _role_text(draft, ("conclusion", "discussion"))
    head = "\n\n".join([abstract, conclusion])

    novelty = sorted(set(evidence.detect_novelty_claims(head)))
    if novelty:
        supported = bool(
            plan is not None and plan.validated and plan.contribution_type.is_novel
            and len(plan.gap_evidence) >= 1
            and library.report().verified >= 2
        )
        checks.append(_ok(
            "novelty.claims", "Novelty claims are supported by the literature search",
            supported, category="novelty", severity=ReviewSeverity.CRITICAL,
            details=("unsupported novelty claim(s): " + ", ".join(novelty[:5]) +
                     ". The literature search did not establish the gap.") if not supported
            else f"supported by {library.report().verified} verified source(s)",
            evidence_items=novelty[:5],
        ))
    else:
        checks.append(_ok("novelty.claims", "No unsupported novelty claim", True,
                          category="novelty", severity=ReviewSeverity.MINOR))

    proof_text = "\n\n".join([head, results])
    proof_claims = sorted(set(evidence.detect_proof_claims(proof_text)))
    theorem_structure = bool(THEOREM_RE.search(_full_text(draft)))
    if proof_claims:
        if profile.article_type == ArticleType.THEORETICAL:
            ok = theorem_structure
            detail = ("proof claimed without a theorem/proof structure" if not ok else
                      "theorem/proof structure present")
        else:
            ok = theorem_structure
            detail = ("strong proof or universal guarantee language appears without a "
                      "theorem/proof; successful finite computation is not a proof"
                      if not ok else "theorem/proof structure present")
        checks.append(_ok(
            "proof.claims", "Claims of proof are backed by an actual proof", ok,
            category="technical", severity=ReviewSeverity.CRITICAL,
            details=detail + (" (" + ", ".join(proof_claims[:4]) + ")"),
            evidence_items=proof_claims[:4],
        ))
    else:
        checks.append(_ok("proof.claims", "No unsupported claim of proof", True,
                          category="technical", severity=ReviewSeverity.MINOR))
    return checks


def _check_reproducibility(draft: Draft, brief: Brief, runs: Sequence[ExperimentRun],
                           specs: Sequence[ExperimentSpec],
                           benchmarks: Sequence[BenchmarkRecord],
                           scope: Sequence[str]) -> List[ValidationCheck]:
    checks: List[ValidationCheck] = []
    profile = profile_for(brief.article_type)
    if not profile.research:
        checks.append(_na("experiments.reproducibility",
                          "Reproducibility metadata applies to executed studies",
                          category="reproducibility"))
        return checks
    if not runs:
        checks.append(_na("experiments.reproducibility", "No experiment was executed",
                          category="reproducibility"))
        return checks
    problems: List[str] = []
    for run in runs:
        if not run.status.ran:
            continue
        if not run.environment:
            problems.append(f"{run.experiment_id}: execution environment not captured")
        if not run.code_sha256:
            problems.append(f"{run.experiment_id}: executed code hash missing")
        if not run.started_at:
            problems.append(f"{run.experiment_id}: start time missing")
    for spec in specs:
        if re.search(r"\brandom\b|\bseed\b", spec.code, re.IGNORECASE) and spec.seed is None:
            problems.append(f"{spec.experiment_id}: code uses randomness without a recorded seed")
    if benchmarks:
        for record in benchmarks:
            if record.repetitions <= 0:
                problems.append(f"{record.benchmark_id}: repetition count missing")
    checks.append(_ok(
        "experiments.reproducibility",
        "Executed experiments carry reproducible metadata", not problems,
        category="reproducibility", severity=ReviewSeverity.MAJOR,
        details="; ".join(problems[:5]) if problems else
                f"{len(runs)} run(s) recorded with environment, hash and timestamps",
    ))

    methods_text = _role_text(draft, ("methods",))
    if specs and methods_text:
        mismatches: List[str] = []
        for spec in specs:
            for key, value in spec.parameters.items():
                if not isinstance(value, (int, float)):
                    continue
                match = re.search(rf"\b{re.escape(str(key))}\b\s*(?:=|:|is|of)?\s*"
                                  rf"(-?\d+(?:\.\d+)?)", methods_text, re.IGNORECASE)
                if match and abs(float(match.group(1)) - float(value)) > 1e-9:
                    mismatches.append(f"{key}: prose says {match.group(1)}, executed with {value}")
            if spec.seed is not None:
                match = re.search(r"\bseed\b\s*(?:=|:|of)?\s*(\d+)", methods_text,
                                  re.IGNORECASE)
                if match and int(match.group(1)) != spec.seed:
                    mismatches.append(f"seed: prose says {match.group(1)}, executed with {spec.seed}")
        if mismatches:
            checks.append(_ok(
                "methods.code_match",
                "The methods description matches the code that was executed", False,
                category="technical", severity=ReviewSeverity.CRITICAL,
                details="; ".join(mismatches[:4]),
            ))
        else:
            checks.append(_ok("methods.code_match",
                              "The methods description matches the code that was executed",
                              True, category="technical",
                              details=f"{len(specs)} experiment spec(s) compared"))
    else:
        checks.append(_na("methods.code_match", "No experiment parameters to compare",
                          category="technical"))

    datasets = [spec.dataset for spec in specs if spec.dataset]
    datasets += [record.dataset for record in benchmarks if record.dataset]
    if datasets:
        described = bool(re.search(r"\b(dataset|corpus|benchmark)\b", _prose(draft),
                                   re.IGNORECASE))
        checks.append(_ok(
            "data.provenance", "Data and benchmark sources are identified", described,
            category="data", severity=ReviewSeverity.MAJOR,
            details=("experiments use " + ", ".join(sorted(set(datasets))[:4]) +
                     " but the manuscript does not describe the data or its provenance")
            if not described else "data provenance described",
        ))
    else:
        checks.append(_na("data.provenance", "No external dataset used", category="data"))
    return checks


def _check_review(review: Optional[ReviewReport], scope: Sequence[str]) -> List[ValidationCheck]:
    checks: List[ValidationCheck] = []
    if review is None:
        checks.append(_unverified(
            "review.completed", "Scientific peer review was completed",
            category="review", severity=ReviewSeverity.CRITICAL,
            details="no review report exists",
        ))
        return checks
    checks.append(_ok("review.completed", "Scientific peer review was completed", True,
                      category="review",
                      details=f"{len(review.issues)} issue(s) across "
                              f"{review.iterations_run} revision iteration(s)"))
    blocking = review.blocking()
    if not blocking:
        checks.append(_ok("review.critical_issues", "No unresolved critical review issue",
                          True, category="review"))
        return checks
    for issue in blocking[:15]:
        checks.append(_ok(
            f"review.{issue.issue_id}", f"Review issue resolved: {issue.claim[:90]}",
            False, category="review", severity=issue.severity,
            details=f"[{issue.dimension}] {issue.explanation} Correction: {issue.correction}",
            evidence_items=[issue.evidence] if issue.evidence else [],
            gate_effect="incomplete" if issue.requires_user_input else "needs_revision",
        ))
    return checks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_manuscript(draft: Draft, brief: Brief, *,
                        plan=None,
                        library: Optional[SourceLibrary] = None,
                        ledger: Optional[evidence.EvidenceLedger] = None,
                        runs: Sequence[ExperimentRun] = (),
                        specs: Sequence[ExperimentSpec] = (),
                        benchmarks: Sequence[BenchmarkRecord] = (),
                        review: Optional[ReviewReport] = None,
                        code_analysis: Optional[codecheck.CodeAnalysis] = None,
                        retrieval: Optional[Dict[str, Any]] = None,
                        include_review: bool = True) -> ValidationReport:
    """Run every deterministic check that applies to the selected article type."""
    library = library or SourceLibrary()
    profile = profile_for(brief.article_type)
    scope = [profile.article_type.value]
    report = ValidationReport(article_type=profile.article_type)
    report.notes.append(f"profile: {profile.label} - {profile.description}")

    if code_analysis is None:
        code_analysis = codecheck.analyse_code(
            draft, runs, article_type_research=profile.research, specs=specs)
    if ledger is None:
        ledger = evidence.build_ledger(draft, library,
                                       [r for run in runs for r in _results_from_run(run)])

    for check in _check_plan_checks(plan, brief, library, retrieval):
        report.add(check)

    citation_checks, _resolution = _check_source_and_citations(
        draft, brief, plan, library, retrieval, scope)
    for check in citation_checks:
        report.add(check)

    for check in _check_results_and_evidence(draft, brief, plan, ledger, runs, specs,
                                             code_analysis, scope):
        report.add(check)
    for check in _check_code_and_tests(draft, brief, code_analysis, runs, scope):
        report.add(check)

    complexity = analysis_module.analyse(draft, profile.article_type, benchmarks)
    for check in _check_complexity_and_benchmarks(draft, brief, benchmarks,
                                                  complexity, scope):
        report.add(check)
    for check in _check_structure_and_consistency(draft, brief, library, scope):
        report.add(check)
    for check in _check_novelty_and_proof(draft, brief, plan, library, scope):
        report.add(check)
    for check in _check_reproducibility(draft, brief, runs, specs, benchmarks, scope):
        report.add(check)
    if include_review:
        for check in _check_review(review, scope):
            report.add(check)

    report.notes.append(f"checks: {len(report.checks)}; "
                        f"failed: {len(report.failing())}; "
                        f"unverified: {len(report.unverified_checks())}")
    return report


def _results_from_run(run: ExperimentRun):
    """Extract ``ResultRecord`` objects from one run's metrics (shared helper)."""
    from .models import ResultRecord
    results: List[ResultRecord] = []
    for metric, value in run.metrics.items():
        if metric.endswith("_unit") or not isinstance(value, (int, float)):
            continue
        results.append(ResultRecord(
            result_id="",
            experiment_id=run.experiment_id,
            run_id=run.run_id,
            metric=metric,
            value=value,
            unit=str(run.metrics.get(f"{metric}_unit") or ""),
            n=len(run.repetition_metrics) or run.repetitions,
            provenance="executed",
        ))
    return results


def results_from_runs(runs: Sequence[ExperimentRun]) -> List[Any]:
    results: List[Any] = []
    for run in runs:
        results.extend(_results_from_run(run))
    for index, result in enumerate(results, start=1):
        result.result_id = f"R{index}"
    return results


def issues_from_validation(report: ValidationReport) -> List[ReviewIssue]:
    """Convert failed/unverified checks into peer-review issues.

    A failed deterministic check is not softened into wording advice: the issue keeps
    the check's severity, evidence and required correction, and the gate continues to
    treat it as unresolved until a later validation pass passes the same check.
    """
    issues: List[ReviewIssue] = []
    for index, check in enumerate(report.checks, start=1):
        if check.status not in (CheckStatus.FAIL, CheckStatus.UNVERIFIED):
            continue
        if check.status == CheckStatus.UNVERIFIED and check.severity == ReviewSeverity.MINOR:
            continue
        dimension = CATEGORY_DIMENSIONS.get(check.category,
                                            "methodological_validity")
        correction = _correction_for(check)
        issues.append(ReviewIssue(
            issue_id=f"V{index:02d}-{check.check_id.replace('.', '-')}",
            dimension=dimension,
            severity=check.severity,
            section=_first_evidence(check) or "document",
            claim=check.description + (": " + check.details if check.details else ""),
            explanation=check.details or check.description,
            evidence="; ".join(check.evidence[:3]),
            correction=correction,
            auto_fixable=False,
            requires_user_input=check.gate_effect == "incomplete",
            status="unresolved" if check.status == CheckStatus.FAIL else "open",
            source="deterministic",
            check_id=check.check_id,
        ))
    return issues


def _first_evidence(check: ValidationCheck) -> str:
    if check.evidence:
        return str(check.evidence[0])[:80]
    return ""


_CORRECTIONS = {
    "code": "Fix the implementation or the listing so the code shown is the code that "
            "ran, then execute the tests again.",
    "technical": "Correct the implementation/description mismatch and re-run the "
                 "verification; do not weaken the claim to hide the mismatch.",
    "results": "Run the experiment and report the recorded results, or label the "
               "manuscript as a proposal/protocol and remove the result claim.",
    "citations": "Use only retrieved records for references, or remove the unsupported "
                 "claim and state that the source could not be verified.",
    "sources": "Retrieve and verify the sources, or label the bibliography as "
               "unverified. Do not fabricate references.",
    "novelty": "Either supply literature evidence for the gap or reframe the "
               "contribution as a replication/methods study.",
    "complexity": "State the input-size parameters, time and memory bounds, and the "
                  "assumptions; or remove the complexity claim.",
    "performance": "Record measurements with metadata, or remove the performance claim.",
    "benchmark": "Record hardware, runtime, dependency versions, dataset, repetitions "
                 "and timing methodology for the measurement.",
    "statistics": "Report sample size and dispersion for the comparison, or avoid a "
                  "comparative/significance claim.",
    "terminology": "Give the term an operational definition (metric, threshold, "
                   "measurement) before using it as evidence.",
    "plan": "Revise the research plan with the missing element before drafting.",
    "sections": "Add the section required by the article type and fill it with real "
                "content.",
    "limitations": "Add a substantive limitations/threats-to-validity section that "
                   "matches the methods actually used.",
    "review": "Address the reviewer issue substantively; re-running the validation is "
              "required to close it.",
}


def _correction_for(check: ValidationCheck) -> str:
    return _CORRECTIONS.get(check.category,
                            "Provide the missing evidence or remove the unsupported claim.")
