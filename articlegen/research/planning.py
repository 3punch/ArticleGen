"""The structured research plan and its validation.

The plan is the contract between the research question and the draft. It is built
before any prose is written, it is validated deterministically, and it is what the
quality gate means by "the contribution is defined".

Novelty is never assumed. If the plan claims novelty but the literature search does
not support it, the plan is narrowed or reframed as a replication/methods study, and
the unresolved issue is carried into the review report. The planner may ask the user
for direction rather than guessing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..models import ArticleType, Brief, Section
from .models import (
    ArticleTypeProfile,
    CheckStatus,
    ContributionType,
    ResearchPlan,
    ReviewSeverity,
    SourceRecord,
    ValidationCheck,
    classify_section_role,
    profile_for,
)
from .sources import SourceLibrary

NOVELTY_ASSERTION_RE = re.compile(
    r"\b(first|novel|unprecedented|no\s+prior|has\s+not\s+been|state\s+of\s+the\s+art)\b",
    re.IGNORECASE,
)
ROUTINE_LANGUAGE_RE = re.compile(
    r"\b(re-?implement\w*|reproduc\w*|tutorial|walkthrough|introduction\s+to|"
    r"educational|teaching|summary\s+of\s+existing|wrapper\s+around)\b",
    re.IGNORECASE,
)


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _first(data: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return default


def _clean(text: Any, limit: int = 4000) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def derive_plan(brief: Brief, *, origin: str = "deterministic") -> ResearchPlan:
    """Build the minimum honest plan from the brief without inventing content.

    For a general/educational article this is complete: the "research" fields are
    not required. For a research article the missing fields are left empty so plan
    validation can report exactly what the planner must supply.
    """
    profile = profile_for(brief.article_type)
    contribution_type = ContributionType.parse(brief.contribution_type) if \
        brief.contribution_type else (
            ContributionType.EDUCATIONAL if brief.article_type == ArticleType.EDUCATIONAL
            else ContributionType.ROUTINE_IMPLEMENTATION if not profile.research
            else ContributionType.NOVEL_EMPIRICAL)
    outline = [{
        "id": section.id,
        "title": section.title,
        "purpose": section.purpose,
        "word_target": section.word_target,
        "role": classify_section_role(section.id, section.title),
    } for section in brief.sections]
    plan = ResearchPlan(
        task_id=brief.task_id,
        article_type=brief.article_type,
        title=brief.title,
        problem=brief.objective,
        research_question=brief.research_question,
        background="",
        gap="",
        contribution=brief.contribution_type,
        contribution_type=contribution_type,
        methodology="",
        methodology_justification="",
        required_materials=[],
        required_experiments=list(brief.benchmark.get("experiments") or [])
        if isinstance(brief.benchmark, dict) else [],
        evaluation_criteria=[],
        limitations=[],
        failure_modes=[],
        outline=outline,
        source_queries=[],
        experiments=[],
        origin=origin,
    )
    if not profile.research:
        plan.background = "The article explains established material for its audience."
        plan.gap = "No research gap is claimed; this is an explanatory article."
        plan.methodology = ("Explain the established concept, ground every factual claim "
                            "in a source when citations are required, and avoid novelty "
                            "or performance claims.")
        plan.evaluation_criteria = ["The reader can apply the explained concept."]
        plan.limitations = ["The article makes no empirical contribution and reports no "
                            "new measurements."]
        plan.decision = "proceed"
        plan.validated = True
    return plan


def plan_from_payload(payload: Dict[str, Any], brief: Brief, *,
                      origin: str = "model") -> ResearchPlan:
    """Parse the planner's JSON into a plan without trusting its shape."""
    if not isinstance(payload, dict):
        payload = {}
    data = payload.get("plan") if isinstance(payload.get("plan"), dict) else payload
    article_type = ArticleType.parse(data.get("article_type") or brief.article_type.value)
    contribution_type = ContributionType.parse(
        data.get("contribution_type") or brief.contribution_type)
    outline: List[Dict[str, Any]] = []
    raw_outline = data.get("outline") or data.get("sections") or []
    if isinstance(raw_outline, list):
        for item in raw_outline:
            if not isinstance(item, dict):
                continue
            section_id = _clean(item.get("id") or item.get("section_id"), 80)
            title = _clean(item.get("title"), 160)
            if not title and not section_id:
                continue
            outline.append({
                "id": section_id or re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60],
                "title": title or section_id.replace("-", " ").title(),
                "purpose": _clean(item.get("purpose") or item.get("goal"), 400),
                "word_target": int(item.get("word_target") or 0) or 0,
                "role": _clean(item.get("role"), 40) or
                        classify_section_role(section_id, title),
            })
    plan = ResearchPlan(
        task_id=brief.task_id,
        article_type=article_type,
        title=_clean(data.get("title") or brief.title, 200),
        problem=_clean(_first(data, "problem", "research_problem", "motivation")),
        research_question=_clean(_first(data, "research_question", "question")),
        background=_clean(_first(data, "background", "established_knowledge")),
        gap=_clean(_first(data, "gap", "research_gap", "gap_evidence_statement")),
        gap_evidence=_as_list(_first(data, "gap_evidence", "gap_sources",
                                     "gap_support", default=[])),
        contribution=_clean(_first(data, "contribution", "intended_contribution",
                                   "contribution_statement")),
        contribution_type=contribution_type,
        novelty_claims=_as_list(_first(data, "novelty_claims", "novelty", default=[])),
        methodology=_clean(_first(data, "methodology", "methods")),
        methodology_justification=_clean(_first(data, "methodology_justification",
                                                "method_justification")),
        required_materials=_as_list(_first(data, "required_materials", "materials",
                                           "required_sources", default=[])),
        required_experiments=_as_list(_first(data, "required_experiments", "experiments_required",
                                             default=[])),
        evaluation_criteria=_as_list(_first(data, "evaluation_criteria", "success_metrics",
                                            "evaluation", default=[])),
        limitations=_as_list(_first(data, "limitations", "known_limitations", default=[])),
        failure_modes=_as_list(_first(data, "failure_modes", "failure_conditions", default=[])),
        outline=outline,
        source_queries=_as_list(_first(data, "source_queries", "search_queries",
                                       "literature_queries", default=[])),
        experiments=[dict(item) for item in (data.get("experiments") or [])
                     if isinstance(item, dict)],
        origin=origin,
        raw=payload if isinstance(payload, dict) else {},
    )
    if not plan.outline:
        plan.outline = [{
            "id": section.id, "title": section.title, "purpose": section.purpose,
            "word_target": section.word_target,
            "role": classify_section_role(section.id, section.title),
        } for section in brief.sections]
    return plan


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check(check_id: str, description: str, ok: bool, *, severity: ReviewSeverity,
           details: str = "", evidence: Iterable[str] = (),
           gate_effect: str = "needs_revision", unverified: bool = False,
           not_applicable: bool = False) -> ValidationCheck:
    if not_applicable:
        status = CheckStatus.NOT_APPLICABLE
    elif unverified:
        status = CheckStatus.UNVERIFIED
    else:
        status = CheckStatus.PASS if ok else CheckStatus.FAIL
    return ValidationCheck(
        check_id=check_id, description=description, status=status,
        severity=severity, category="plan", details=details,
        evidence=list(evidence), gate_effect=gate_effect,
    )


def _roles(plan: ResearchPlan) -> set:
    roles = set()
    for item in plan.outline:
        role = str(item.get("role") or "").strip()
        if not role:
            role = classify_section_role(str(item.get("id") or ""),
                                         str(item.get("title") or ""))
        roles.add(role)
    return roles


def validate_plan(plan: ResearchPlan, brief: Brief, *,
                  library: Optional[SourceLibrary] = None,
                  retrieval_attempted: bool = False,
                  retrieval_available: bool = False) -> ResearchPlan:
    """Deterministically validate and, if necessary, narrow the plan."""
    profile = profile_for(brief.article_type if brief.article_type != ArticleType.GENERAL
                          else plan.article_type)
    checks: List[ValidationCheck] = []
    research = profile.research
    library = library or SourceLibrary()
    report = library.report()
    verified_sources = report.verified
    usable_sources = report.usable
    scope = [profile.article_type.value]

    def add(check: ValidationCheck) -> None:
        check.article_scope = scope
        checks.append(check)

    if not research:
        add(_check("plan.research_scope", "General article: research-plan fields are not "
                   "required", True, severity=ReviewSeverity.MINOR,
                   gate_effect="none", not_applicable=True))
    else:
        add(_check("plan.problem", "Research problem and motivation are stated",
                   len(plan.problem) >= 40, severity=ReviewSeverity.MAJOR,
                   details=f"{len(plan.problem)} characters supplied",
                   gate_effect="incomplete"))
        add(_check("plan.research_question",
                   "Research question is precise and answerable",
                   len(plan.research_question) >= 20 and _question_like(plan.research_question),
                   severity=ReviewSeverity.CRITICAL,
                   details=plan.research_question or "no research question supplied",
                   gate_effect="incomplete"))
        add(_check("plan.background", "Relevant background and established knowledge are "
                   "stated", len(plan.background) >= 40,
                   severity=ReviewSeverity.MAJOR, gate_effect="needs_revision"))
        add(_check("plan.contribution", "Intended original contribution is stated",
                   len(plan.contribution) >= 30, severity=ReviewSeverity.CRITICAL,
                   details=plan.contribution or "no contribution statement",
                   gate_effect="incomplete"))
        add(_check("plan.methodology", "Methodology and its justification are stated",
                   len(plan.methodology) >= 30 and len(plan.methodology_justification) >= 20,
                   severity=ReviewSeverity.MAJOR,
                   details=(f"methodology={len(plan.methodology)} chars, "
                            f"justification={len(plan.methodology_justification)} chars"),
                   gate_effect="needs_revision"))
        add(_check("plan.limitations", "Known limitations and failure modes are stated",
                   len(plan.limitations) >= 1 and len(plan.failure_modes) >= 1,
                   severity=ReviewSeverity.MAJOR, gate_effect="needs_revision"))
        add(_check("plan.evaluation", "Evaluation criteria and success metrics are defined",
                   len(plan.evaluation_criteria) >= 1,
                   severity=ReviewSeverity.MAJOR, gate_effect="needs_revision"))

        if profile.requires_sources:
            if not retrieval_attempted:
                add(_check("plan.sources_retrieved", "Scholarly sources were retrieved",
                           False, severity=ReviewSeverity.CRITICAL,
                           details="source retrieval was not attempted",
                           unverified=True, gate_effect="incomplete"))
            elif not retrieval_available and not library.records:
                add(_check("plan.sources_retrieved", "Scholarly sources were retrieved",
                           False, severity=ReviewSeverity.CRITICAL,
                           details="the source index was unavailable, so no bibliography "
                                   "could be verified", unverified=True,
                           gate_effect="incomplete"))
            else:
                add(_check("plan.sources_retrieved", "Scholarly sources were retrieved",
                           len(library.records) > 0, severity=ReviewSeverity.CRITICAL,
                           details=f"{len(library.records)} record(s); {verified_sources} "
                                   f"index-verified, {report.user_provided} user-provided",
                           gate_effect="incomplete"))
            add(_check("plan.source_coverage",
                       "Enough independent sources support the background and gap",
                       usable_sources >= 3, severity=ReviewSeverity.MAJOR,
                       details=f"{usable_sources} usable source(s)", gate_effect="needs_revision"))

            gap_supported = bool(plan.gap_evidence) and usable_sources > 0
            add(_check("plan.gap_established",
                       "The research gap is supported by evidence",
                       bool(plan.gap) and gap_supported,
                       severity=ReviewSeverity.MAJOR,
                       details=("gap statement present" if plan.gap else "no gap statement")
                       + ("; " + ", ".join(plan.gap_evidence[:4]) if plan.gap_evidence else
                          "; no supporting sources or reasoning"),
                       evidence=plan.gap_evidence[:6],
                       gate_effect="needs_revision"))
        else:
            add(_check("plan.sources_retrieved", "Scholarly sources required for this type",
                       True, severity=ReviewSeverity.MINOR, gate_effect="none",
                       not_applicable=True))

        if profile.requires_experiments:
            has_design = bool(plan.required_experiments or plan.experiments)
            add(_check("plan.experiment_design", "Experiments required by the plan are "
                       "specified before drafting", has_design,
                       severity=ReviewSeverity.CRITICAL,
                       details=("design present" if has_design else
                                "no experiment specified in the plan"),
                       gate_effect="incomplete"))
        if profile.requires_complexity_analysis:
            mentions = bool(re.search(r"complexit|asymptot|O\s*\(|performance",
                                      " ".join([plan.methodology, plan.evaluation_criteria
                                                and " ".join(plan.evaluation_criteria) or "",
                                                plan.contribution]),
                                      re.IGNORECASE))
            add(_check("plan.complexity_analysis_planned",
                       "Complexity analysis is part of the plan",
                       mentions, severity=ReviewSeverity.MAJOR,
                       details="plan must state the input-size parameters and bounds",
                       gate_effect="needs_revision"))
        if profile.requires_search_strategy:
            strategy = bool(re.search(r"search|database|inclusion|exclusion|screening",
                                      plan.methodology, re.IGNORECASE))
            add(_check("plan.search_strategy",
                       "Systematic-review search strategy is specified",
                       strategy, severity=ReviewSeverity.CRITICAL,
                       gate_effect="needs_revision"))
        if profile.requires_correctness_argument:
            add(_check("plan.correctness_argument",
                       "Plan commits to a correctness argument or proof",
                       bool(re.search(r"correct|proof|theorem|invariant|argue",
                                      f"{plan.methodology} {plan.contribution}",
                                      re.IGNORECASE)),
                       severity=ReviewSeverity.MAJOR, gate_effect="needs_revision"))

    # -- outline consistency ---------------------------------------------
    roles = _roles(plan)
    if research and profile.required_roles:
        missing_roles = [role for role in profile.required_roles if role not in roles]
        add(_check("plan.outline_roles",
                   f"Outline covers the sections required by the "
                   f"{profile.label.lower()} article type",
                   not missing_roles, severity=ReviewSeverity.MAJOR,
                   details=("missing roles: " + ", ".join(missing_roles))
                   if missing_roles else "all required roles present",
                   evidence=sorted(roles), gate_effect="needs_revision"))
    outline_ids = [str(item.get("id")) for item in plan.outline if item.get("id")]
    brief_ids = brief.section_ids
    if research:
        diverged = [sid for sid in plan.section_ids() if sid not in brief_ids]
        add(_check("plan.outline_matches_brief",
                   "Plan outline is consistent with the approved brief outline",
                   not diverged and bool(outline_ids),
                   severity=ReviewSeverity.MAJOR,
                   details=(f"plan adds or renames sections not in the brief: "
                            f"{', '.join(diverged)}") if diverged else "",
                   gate_effect="needs_revision"))

    # -- novelty policy ---------------------------------------------------
    novelty_statements = list(plan.novelty_claims)
    if plan.contribution_type.is_novel and novelty_statements:
        pass
    assertion_text = " ".join([plan.contribution] + novelty_statements)
    asserts_novelty = bool(NOVELTY_ASSERTION_RE.search(assertion_text)) or \
        bool(plan.novelty_claims)
    if asserts_novelty and research:
        supported = bool(plan.gap_evidence) and verified_sources >= 2
        if plan.contribution_type == ContributionType.ROUTINE_IMPLEMENTATION:
            plan.contribution_type = ContributionType.REPLICATION
            plan.decision = "narrow_claims"
            plan.decision_reasons.append(
                "novelty language was attached to a routine implementation; the plan "
                "was reframed as a replication"
            )
            add(_check("plan.novelty_supported",
                       "Novelty claims are supported by the literature search",
                       False, severity=ReviewSeverity.MAJOR,
                       details="a routine implementation cannot claim novelty; claims "
                               "were narrowed to a replication",
                       gate_effect="needs_revision"))
        elif not supported:
            plan.decision = "narrow_claims"
            plan.decision_reasons.append(
                "novelty claim has insufficient literature evidence "
                f"({verified_sources} verified source(s), "
                f"{len(plan.gap_evidence)} gap-evidence item(s))"
            )
            if brief.novelty_claim:
                plan.needs_user_direction = True
                plan.decision_reasons.append(
                    "the user explicitly requested a novelty claim; additional "
                    "literature direction is required"
                )
            add(_check("plan.novelty_supported",
                       "Novelty claims are supported by the literature search",
                       False, severity=ReviewSeverity.CRITICAL,
                       details="remove the novelty claim, or supply literature evidence "
                               "that establishes the gap",
                       gate_effect="needs_revision"))
        else:
            add(_check("plan.novelty_supported",
                       "Novelty claims are supported by the literature search", True,
                       severity=ReviewSeverity.CRITICAL,
                       details=f"{verified_sources} verified source(s), "
                               f"{len(plan.gap_evidence)} gap-evidence item(s)",
                       gate_effect="needs_revision"))

    # A contribution that is routine language must not inherit a novel label.
    if research and plan.contribution_type.is_novel \
            and ROUTINE_LANGUAGE_RE.search(plan.contribution or ""):
        plan.contribution_type = ContributionType.REPLICATION
        plan.decision = "narrow_claims"
        plan.decision_reasons.append(
            "contribution text describes reproduction or explanation, not a novel result"
        )
        add(_check("plan.contribution_type",
                   "Contribution type matches the described contribution", False,
                   severity=ReviewSeverity.MAJOR,
                   details="retitled as a replication/methods study",
                   gate_effect="needs_revision"))
    else:
        add(_check("plan.contribution_type",
                   "Contribution type matches the described contribution", True,
                   severity=ReviewSeverity.MAJOR, gate_effect="needs_revision"))

    # -- decision ---------------------------------------------------------
    if plan.needs_user_direction:
        plan.decision = "needs_user_direction"
    elif plan.decision not in ("narrow_claims", "reframe_replication"):
        plan.decision = "proceed"
    plan.issues = checks
    plan.validated = all(c.status in (CheckStatus.PASS, CheckStatus.NOT_APPLICABLE)
                         for c in checks)
    if not plan.validated and plan.decision == "proceed":
        failures = [c for c in checks if c.status == CheckStatus.FAIL]
        if any("source" in c.check_id or "gap" in c.check_id for c in failures):
            plan.decision = "narrow_claims"
            plan.decision_reasons.append(
                "the literature evidence did not establish a gap; claims were narrowed"
            )
    if plan.decision == "narrow_claims":
        plan.validated = False
    return plan


def _question_like(text: str) -> bool:
    token = (text or "").strip()
    if token.endswith("?"):
        return True
    return bool(re.search(r"\b(whether|how|why|what|which|does|do|can|is|are|"
                          r"to\s+what\s+extent)\b", token, re.IGNORECASE))


def plan_summary_block(plan: ResearchPlan, max_chars: int = 1800) -> str:
    """Compact plan for agent context packets."""
    lines = [
        f"article_type: {plan.article_type.value}",
        f"research_question: {plan.research_question or '(not established)'}",
        f"contribution ({plan.contribution_type.label}): {plan.contribution or '(none)'}",
        f"gap: {plan.gap or '(not established)'}",
        f"methodology: {plan.methodology or '(none)'}",
        f"evaluation: {'; '.join(plan.evaluation_criteria) or '(none)'}",
        f"limitations: {'; '.join(plan.limitations) or '(none)'}",
        f"decision: {plan.decision}",
    ]
    if plan.required_experiments:
        lines.append("required experiments: " + "; ".join(plan.required_experiments))
    text = "\n".join(lines)
    return text[:max_chars]


def outline_additions(plan: ResearchPlan, brief: Brief) -> List[Section]:
    """Required-role sections the plan proposes but the brief outline lacks.

    Returns only well-formed additions. The caller decides whether to append them
    to the brief; nothing is silently inserted into a validated brief here.
    """
    profile = profile_for(plan.article_type)
    existing_roles = {classify_section_role(s.id, s.title) for s in brief.sections}
    existing_ids = set(brief.section_ids)
    additions: List[Section] = []
    for item in plan.outline:
        section_id = str(item.get("id") or "").strip()
        title = str(item.get("title") or "").strip()
        purpose = str(item.get("purpose") or "").strip()
        role = str(item.get("role") or classify_section_role(section_id, title))
        if role not in profile.required_roles or role in existing_roles:
            continue
        if not section_id or not title or section_id in existing_ids:
            continue
        if len(purpose) < 15:
            purpose = f"Report the {role.replace('-', ' ')} required by the research plan."
        additions.append(Section(
            id=section_id, title=title, purpose=purpose,
            word_target=int(item.get("word_target") or 0),
        ))
        existing_roles.add(role)
        existing_ids.add(section_id)
    return additions
