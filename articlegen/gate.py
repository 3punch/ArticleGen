"""Stage 1 - Input Verification (decision gate).

The gate is the only thing allowed to route a task back before any sub-agent
spends tokens. It checks formatting, structure and minimal requirements, and it
reports *specific*, actionable issues rather than a bare pass/fail.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Tuple
from urllib.parse import urlparse

from .models import (
    TASK_ID_RE,
    Brief,
    Draft,
    Finding,
    GateResult,
    Severity,
)

# Markers that mean "this input is not finished".
PLACEHOLDER_RE = re.compile(
    r"\b(TODO|TBD|FIXME|XXX|PLACEHOLDER|Lorem\s+ipsum|fill\s+me\s+in|coming\s+soon)\b",
    re.IGNORECASE,
)
FENCE_RE = re.compile(r"^\s*```(.*)$", re.MULTILINE)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
ANCHOR_RE = re.compile(r"^\s*#[a-z0-9][a-z0-9-]*$", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)\]>]+")
CITATION_RE = re.compile(r"\[\d+\]|\([A-Z][A-Za-z]+,\s*\d{4}\)|https?://")

MIN_TITLE = 3
MAX_TITLE = 120
MIN_OBJECTIVE = 40
MIN_PURPOSE = 15
MIN_TARGET_WORDS = 200
MAX_TARGET_WORDS = 12000
MIN_SECTIONS = 3
MAX_SECTIONS = 12
MAX_LINE_LENGTH = 400
WORD_COUNT_TOLERANCE = 0.35


def _finding(stage: str, severity: Severity, category: str, target: str, claim: str, action: str,
             evidence: str = "") -> Finding:
    return Finding(
        agent="gate",
        severity=severity,
        category=category,
        section=target,
        claim=claim,
        action=action,
        evidence=evidence,
        auto_fixable=False,
    )


def _check_task_id(brief: Brief, issues: List[Finding]) -> None:
    if not brief.task_id:
        issues.append(_finding("brief", Severity.BLOCKER, "structure", "task_id",
                               "task_id is missing", "Add a task_id so runs can be tracked."))
    elif not TASK_ID_RE.match(brief.task_id):
        issues.append(_finding(
            "brief", Severity.BLOCKER, "structure", "task_id",
            f"task_id '{brief.task_id}' is not a valid identifier",
            "Use 3-49 lowercase characters: a-z, 0-9, '-' or '_'; must start alphanumeric.",
        ))


def _check_text_fields(brief: Brief, issues: List[Finding]) -> None:
    if not brief.title:
        issues.append(_finding("brief", Severity.BLOCKER, "structure", "title",
                               "title is missing", "Provide a title for the deliverable."))
    elif not (MIN_TITLE <= len(brief.title) <= MAX_TITLE):
        issues.append(_finding(
            "brief", Severity.BLOCKER, "formatting", "title",
            f"title length {len(brief.title)} is outside {MIN_TITLE}-{MAX_TITLE} characters",
            "Rewrite the title to fit the required length.", brief.title[:80],
        ))

    if not brief.objective:
        issues.append(_finding("brief", Severity.BLOCKER, "structure", "objective",
                               "objective is missing", "State what the deliverable must achieve."))
    elif len(brief.objective) < MIN_OBJECTIVE:
        issues.append(_finding(
            "brief", Severity.BLOCKER, "structure", "objective",
            f"objective is only {len(brief.objective)} characters (minimum {MIN_OBJECTIVE})",
            "Expand the objective: state the goal, the reader takeaway and the scope.",
        ))

    if not brief.audience:
        issues.append(_finding("brief", Severity.BLOCKER, "structure", "audience",
                               "audience is missing", "Name the target reader explicitly."))

    if brief.target_words <= 0:
        issues.append(_finding("brief", Severity.BLOCKER, "structure", "target_words",
                               "target_words is missing", "Set an integer word target."))
    elif not (MIN_TARGET_WORDS <= brief.target_words <= MAX_TARGET_WORDS):
        issues.append(_finding(
            "brief", Severity.BLOCKER, "structure", "target_words",
            f"target_words {brief.target_words} is outside {MIN_TARGET_WORDS}-{MAX_TARGET_WORDS}",
            "Choose a target inside the supported range.",
        ))

    for field_name, value in (("title", brief.title), ("objective", brief.objective),
                              ("audience", brief.audience)):
        if value and PLACEHOLDER_RE.search(value):
            issues.append(_finding(
                "brief", Severity.BLOCKER, "formatting", field_name,
                f"{field_name} contains an unfinished placeholder",
                "Replace the placeholder with real content before running the pipeline.",
                PLACEHOLDER_RE.search(value).group(0),
            ))


def _check_sections(brief: Brief, issues: List[Finding]) -> None:
    sections = brief.sections
    if not sections:
        issues.append(_finding("brief", Severity.BLOCKER, "structure", "sections",
                               "no outline sections supplied",
                               f"Provide {MIN_SECTIONS}-{MAX_SECTIONS} sections with id, title and purpose."))
        return
    if len(sections) < MIN_SECTIONS:
        issues.append(_finding(
            "brief", Severity.BLOCKER, "structure", "sections",
            f"only {len(sections)} sections supplied (minimum {MIN_SECTIONS})",
            "Add more sections so the outline covers the objective.",
        ))
    if len(sections) > MAX_SECTIONS:
        issues.append(_finding(
            "brief", Severity.MAJOR, "structure", "sections",
            f"{len(sections)} sections exceed the maximum of {MAX_SECTIONS}",
            "Merge related sections to keep the outline navigable.",
        ))

    seen_ids: Dict[str, int] = {}
    seen_titles: Dict[str, str] = {}
    total_target = 0
    for index, section in enumerate(sections, start=1):
        where = f"sections[{index - 1}]"
        if not section.id:
            issues.append(_finding("brief", Severity.BLOCKER, "structure", where,
                                   "section is missing an id", "Give every section a stable kebab-case id."))
        elif not re.match(r"^[a-z0-9][a-z0-9-]{1,40}$", section.id):
            issues.append(_finding(
                "brief", Severity.BLOCKER, "formatting", where,
                f"section id '{section.id}' is not kebab-case",
                "Use lowercase letters, digits and single hyphens.",
            ))
        elif section.id in seen_ids:
            issues.append(_finding(
                "brief", Severity.BLOCKER, "structure", where,
                f"duplicate section id '{section.id}'",
                "Section ids must be unique; rename the duplicate.",
            ))
        else:
            seen_ids[section.id] = index

        if not section.title:
            issues.append(_finding("brief", Severity.BLOCKER, "structure", where,
                                   "section is missing a title", "Add a heading for the section."))
        elif len(section.title) > MAX_TITLE:
            issues.append(_finding("brief", Severity.MINOR, "formatting", where,
                                   f"section title is {len(section.title)} characters (max {MAX_TITLE})",
                                   "Shorten the heading."))
        elif section.title.strip().lower() in seen_titles:
            issues.append(_finding(
                "brief", Severity.MAJOR, "structure", where,
                f"duplicate section title '{section.title}'",
                "Differentiate the headings so the outline is unambiguous.",
            ))
        else:
            seen_titles[section.title.strip().lower()] = section.id

        if not section.purpose or len(section.purpose) < MIN_PURPOSE:
            issues.append(_finding(
                "brief", Severity.MAJOR, "structure", where,
                f"section purpose is too short ({len(section.purpose or '')} chars, minimum {MIN_PURPOSE})",
                "Describe what this section must accomplish for the reader.",
            ))

        for field_name, value in (("section title", section.title),
                                  ("section purpose", section.purpose)):
            placeholder = PLACEHOLDER_RE.search(value)
            if placeholder:
                issues.append(_finding(
                    "brief", Severity.BLOCKER, "formatting", where,
                    f"{field_name} contains an unfinished placeholder",
                    "Replace the placeholder with specific content before running the pipeline.",
                    placeholder.group(0),
                ))

        if section.body and PLACEHOLDER_RE.search(section.body):
            issues.append(_finding(
                "brief", Severity.BLOCKER, "formatting", section.id or where,
                "seed body contains an unfinished placeholder",
                "Remove placeholder text; the Writer will generate real prose.",
            ))
        if section.word_target < 0:
            issues.append(_finding("brief", Severity.MAJOR, "structure", where,
                                   "negative word_target", "Use a positive integer word target."))
        total_target += max(0, section.word_target)

    if brief.target_words > 0 and total_target > 0:
        ratio = total_target / brief.target_words
        if ratio < 0.5 or ratio > 2.0:
            issues.append(_finding(
                "brief", Severity.MINOR, "structure", "sections",
                f"per-section word targets sum to {total_target}, "
                f"{ratio:.1f}x the overall target of {brief.target_words}",
                "Rebalance section word targets against the overall target.",
            ))


def _check_markdown(text: str, label: str, issues: List[Finding]) -> None:
    """Formatting checks shared by brief seed bodies and produced drafts."""
    if not text:
        return
    fences = FENCE_RE.findall(text)
    if len(fences) % 2 != 0:
        issues.append(_finding(
            "draft", Severity.BLOCKER, "formatting", label,
            f"unbalanced fenced code blocks ({len(fences)} fence markers)",
            "Close every code fence with a matching ``` line.",
        ))
    for line_number, line in enumerate(text.splitlines(), start=1):
        if "\t" in line:
            issues.append(_finding(
                "draft", Severity.MINOR, "formatting", label,
                f"tab character on line {line_number}",
                "Use spaces for indentation in Markdown.", line.strip()[:60],
            ))
            break
        if len(line) > MAX_LINE_LENGTH:
            issues.append(_finding(
                "draft", Severity.MINOR, "formatting", label,
                f"line {line_number} is {len(line)} characters (max {MAX_LINE_LENGTH})",
                "Wrap or split the line.",
            ))
            break
        if line.rstrip() != line:
            issues.append(_finding(
                "draft", Severity.NOTE, "formatting", label,
                f"trailing whitespace on line {line_number}",
                "Strip trailing whitespace.",
            ))
            break


def _check_code_blocks(text: str, label: str, issues: List[Finding]) -> None:
    blocks = re.findall(r"```([^\n]*)\n(.*?)```", text, flags=re.DOTALL)
    for index, (lang, body) in enumerate(blocks, start=1):
        lang = lang.strip()
        if not lang:
            issues.append(_finding(
                "draft", Severity.MAJOR, "formatting", label,
                f"code block {index} has no language tag",
                "Tag the fence, e.g. ```python, so highlighting and validation work.",
            ))
        if not body.strip():
            issues.append(_finding(
                "draft", Severity.MAJOR, "structure", label,
                f"code block {index} is empty",
                "Remove the empty fence or add the intended code.",
            ))
        if lang.lower() in {"python", "py", "python3"} and body.strip():
            try:
                compile(body, f"<{label}:block{index}>", "exec")
            except SyntaxError as exc:  # pragma: no cover - exercised by tests
                issues.append(_finding(
                    "draft", Severity.BLOCKER, "correctness", label,
                    f"code block {index} does not compile: {exc.msg} (line {exc.lineno})",
                    "Fix the syntax error before the draft can be accepted.",
                    (exc.text or "").strip()[:80],
                ))


def _check_links(text: str, label: str, issues: List[Finding]) -> None:
    headings = [m.group(2).strip() for m in HEADING_RE.finditer(text)]
    slugs = {re.sub(r"[^a-z0-9]+", "-", h.lower()).strip("-") for h in headings}
    for anchor_text, target in MARKDOWN_LINK_RE.findall(text):
        target = target.strip()
        if not target:
            issues.append(_finding("draft", Severity.MAJOR, "formatting", label,
                                   f"link '{anchor_text[:40]}' has an empty target",
                                   "Supply a destination for the link."))
        elif ANCHOR_RE.match(target):
            slug = target[1:]
            if slug and slug not in slugs:
                issues.append(_finding(
                    "draft", Severity.MINOR, "correctness", label,
                    f"internal anchor '{target}' does not match any heading",
                    "Point the anchor at an existing heading.",
                ))


def _check_citations(text: str, issues: List[Finding]) -> None:
    found = CITATION_RE.findall(text)
    if not found:
        issues.append(_finding(
            "draft", Severity.MAJOR, "citations", "document",
            "brief requires citations but none were found",
            "Add numbered references, author-year markers or source URLs.",
        ))
        return
    for url in URL_RE.findall(text):
        parsed = urlparse(url.rstrip(".,;"))
        if not parsed.scheme or not parsed.netloc or "." not in parsed.netloc:
            issues.append(_finding(
                "draft", Severity.MAJOR, "citations", "document",
                f"malformed citation URL '{url[:60]}'",
                "Use a fully qualified http(s) URL.",
            ))


def verify_brief(brief: Brief) -> GateResult:
    """Validate the incoming task input. Returns a routing decision."""
    issues: List[Finding] = []
    checks: List[str] = []

    _check_task_id(brief, issues)
    checks.append("task_id")
    _check_text_fields(brief, issues)
    checks.append("text_fields")
    _check_sections(brief, issues)
    checks.append("sections")
    for section in brief.sections:
        _check_markdown(section.body, section.id or "section", issues)
        _check_code_blocks(section.body, section.id or "section", issues)
    checks.append("markdown_formatting")
    checks.append("code_blocks")
    checks.append("placeholders")

    blockers = [i for i in issues if i.severity == Severity.BLOCKER]
    routed_back = bool(blockers)
    reason = ""
    if routed_back:
        reason = (
            f"{len(blockers)} blocking issue(s) in task input; routed back for correction: "
            + "; ".join(sorted({b.category for b in blockers}))
        )

    return GateResult(
        ok=not routed_back,
        stage="input_verification",
        issues=issues,
        checks_run=checks,
        routed_back=routed_back,
        reason=reason,
    )


def verify_draft(draft: Draft, brief: Brief) -> GateResult:
    """Validate a produced draft against the brief before it is accepted."""
    issues: List[Finding] = []
    checks: List[str] = []

    draft_map = draft.section_map()
    missing = [sid for sid in brief.section_ids if sid not in draft_map]
    extra = [sid for sid in draft_map if sid not in set(brief.section_ids)]
    if missing:
        issues.append(_finding("draft", Severity.BLOCKER, "structure", ",".join(missing),
                               f"draft is missing outline section(s): {', '.join(missing)}",
                               "Restore the missing sections from the outline."))
    if extra:
        issues.append(_finding("draft", Severity.MAJOR, "structure", ",".join(extra),
                               f"draft contains section(s) not in the outline: {', '.join(extra)}",
                               "Remove the extra sections or update the outline."))
    checks.append("outline_alignment")

    for section in draft.sections:
        if not section.body.strip():
            issues.append(_finding("draft", Severity.BLOCKER, "structure", section.id,
                                   f"section '{section.title}' is empty",
                                   "The Writer must produce content for every outline section."))
            continue
        if PLACEHOLDER_RE.search(section.body):
            issues.append(_finding(
                "draft", Severity.BLOCKER, "formatting", section.id,
                "section contains an unfinished placeholder",
                "Replace placeholder text with finished prose.",
                PLACEHOLDER_RE.search(section.body).group(0),
            ))
        target = section.word_target or 0
        if target and abs(section.word_count - target) / target > WORD_COUNT_TOLERANCE:
            issues.append(_finding(
                "draft", Severity.MINOR, "structure", section.id,
                f"section is {section.word_count} words against a target of {target}",
                "Expand or trim the section to within tolerance of its target.",
            ))
    checks.append("section_completeness")

    for section in draft.sections:
        _check_markdown(section.body, section.id, issues)
        _check_code_blocks(section.body, section.id, issues)
        _check_links(section.body, section.id, issues)
    checks.append("markdown_formatting")
    checks.append("code_blocks")
    checks.append("links")

    full_text = draft.to_markdown()
    if brief.requires_citations:
        _check_citations(full_text, issues)
        checks.append("citations")

    if brief.target_words:
        deviation = abs(draft.word_count - brief.target_words) / brief.target_words
        if deviation > 0.5:
            issues.append(_finding(
                "draft", Severity.MAJOR, "structure", "document",
                f"draft is {draft.word_count} words against an overall target of {brief.target_words}",
                "Rebalance section lengths to approach the overall target.",
            ))
        checks.append("word_count")

    blockers = [i for i in issues if i.severity == Severity.BLOCKER]
    return GateResult(
        ok=not blockers,
        stage="draft_verification",
        issues=issues,
        checks_run=checks,
        routed_back=bool(blockers),
        reason=f"{len(blockers)} blocking issue(s) in draft" if blockers else "draft acceptable",
    )


GATE_CHECKS: Dict[str, Callable[..., GateResult]] = {
    "brief": verify_brief,
    "draft": verify_draft,
}


def summarise_gate(result: GateResult) -> Tuple[int, int, int]:
    """(blockers, majors, minors) - used for compact change-log lines."""
    blockers = len([i for i in result.issues if i.severity == Severity.BLOCKER])
    majors = len([i for i in result.issues if i.severity == Severity.MAJOR])
    minors = len([i for i in result.issues if i.severity in (Severity.MINOR, Severity.NOTE)])
    return blockers, majors, minors
