"""Turns dashboard form values into a validated ``Brief``.

Streamlit form widgets give back loose strings and numbers. This module owns the
messy part - parsing the sections editor, coercing types, surfacing parse errors -
so the app script stays declarative and this logic stays unit-testable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..gate import verify_brief
from ..models import Brief, GateResult

SECTION_HEADER = "id | Title | What this section must accomplish | word_target"

SECTION_TEMPLATE = """introduction | Introduction | Frame the problem and state what the reader gains | 200
core-concept | The Core Concept | Explain the mechanism precisely enough to implement it | 450
worked-example | Worked Example | Show the concept applied end to end with real inputs and outputs | 350
tradeoffs | Trade-offs and Limits | Name what this approach costs and when it is the wrong choice | 200"""


class SectionParseError(ValueError):
    """Raised when a line of the sections editor cannot be parsed."""


def parse_sections(text: str) -> List[Dict[str, Any]]:
    """Parse the ``id | Title | purpose | word_target`` editor format.

    Blank lines and ``#`` comments are ignored. A malformed line raises with its
    1-based line number, because "sections are wrong" is not actionable in a UI.
    """
    sections: List[Dict[str, Any]] = []
    for raw_index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [part.strip() for part in stripped.split("|")]
        if len(parts) != 4:
            raise SectionParseError(
                f"line {raw_index}: expected 4 fields separated by '|', found {len(parts)}"
            )
        section_id, title, purpose, word_target = parts
        if not section_id:
            raise SectionParseError(f"line {raw_index}: section id is empty")
        if not title:
            raise SectionParseError(f"line {raw_index}: section title is empty")
        try:
            target = int(word_target)
        except ValueError:
            raise SectionParseError(
                f"line {raw_index}: word_target '{word_target}' is not an integer"
            ) from None
        sections.append({
            "id": section_id,
            "title": title,
            "purpose": purpose,
            "word_target": target,
        })
    return sections


def sections_to_text(sections: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{s.get('id', '')} | {s.get('title', '')} | {s.get('purpose', '')} | "
        f"{s.get('word_target', 0)}"
        for s in sections
    )


def brief_to_text(brief: Brief) -> str:
    """Render an existing brief back into the editor format (for the Load button)."""
    return sections_to_text([s.to_dict() for s in brief.sections])


def build_brief_dict(*, task_id: str, title: str, objective: str, audience: str,
                     mode: str = "article", tone: str = "", target_words: int = 0,
                     sections_text: str = "", constraints_text: str = "",
                     requires_citations: bool = False,
                     article_type: str = "general", citation_style: str = "apa",
                     research_question: str = "", contribution_type: str = "",
                     venue: str = "", requires_experiments: bool = False,
                     execute_experiments: bool = True, review_iterations: int = 1,
                     source_retrieval: str = "",
                     novelty_claim: bool = False) -> Dict[str, Any]:
    """Assemble a brief dict from raw form values, without validating it.

    Deliberately permissive: the gate owns validation, and running a half-built dict
    through the gate is what produces useful errors for the user.
    """
    constraints = [
        line.strip().lstrip("-* ").strip()
        for line in (constraints_text or "").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    section_parse_error = ""
    try:
        sections = parse_sections(sections_text or "")
    except SectionParseError as exc:
        sections = []
        section_parse_error = str(exc)

    return {
        "task_id": (task_id or "").strip(),
        "title": (title or "").strip(),
        "objective": (objective or "").strip(),
        "audience": (audience or "").strip(),
        "mode": (mode or "article").strip().lower(),
        "tone": (tone or "").strip(),
        "target_words": int(target_words or 0),
        "requires_citations": bool(requires_citations),
        "constraints": constraints,
        "sections": sections,
        # Research settings. "general" keeps the classic article workflow unchanged;
        # any research type activates planning, sources, execution and the quality gate.
        "article_type": (article_type or "general").strip().lower(),
        "citation_style": (citation_style or "apa").strip().lower(),
        "venue": (venue or "").strip(),
        "research_question": (research_question or "").strip(),
        "contribution_type": (contribution_type or "").strip().lower(),
        "requires_experiments": bool(requires_experiments),
        "execute_experiments": bool(execute_experiments),
        "review_iterations": max(0, int(review_iterations or 0)),
        "source_retrieval": (source_retrieval or "").strip().lower(),
        "novelty_claim": bool(novelty_claim),
        # Internal diagnostic consumed by validate(); preserving it here lets the
        # UI show the bad line instead of reducing every parse error to "no sections".
        "_section_parse_error": section_parse_error,
    }


def validate(data: Dict[str, Any]) -> Tuple[Optional[Brief], GateResult, List[str]]:
    """Build a Brief and run the decision gate on it.

    Returns ``(brief_or_None, gate_result, parse_errors)``. ``brief`` is ``None`` only
    when the dict could not be coerced at all; otherwise the gate decides.
    """
    parse_errors: List[str] = []
    section_parse_error = data.get("_section_parse_error")
    if section_parse_error:
        parse_errors.append(f"sections: {section_parse_error}")

    try:
        brief = Brief.from_dict(data)
    except (TypeError, ValueError) as exc:
        parse_errors.append(f"could not build a brief from the form: {exc}")
        return None, GateResult(ok=False, stage="form", reason=str(parse_errors[0])), parse_errors

    return brief, verify_brief(brief), parse_errors


def format_issues(gate: GateResult) -> str:
    """Render gate issues as Markdown for display in the UI."""
    if not gate.issues:
        return "_No issues — the gate passes._"
    lines = ["| severity | category | target | issue | required action |",
             "| --- | --- | --- | --- | --- |"]
    for issue in sorted(gate.issues, key=lambda f: (-f.severity.weight, f.category)):
        claim = issue.claim.replace("|", "\\|")
        action = issue.action.replace("|", "\\|")
        lines.append(f"| **{issue.severity.value}** | {issue.category} | "
                     f"{issue.section or '-'} | {claim} | {action} |")
    return "\n".join(lines)


def _coerce_number(value: Any, cast) -> Optional[Any]:
    """Cast a JSONL field to one dtype, or None when it is absent/malformed.

    pyarrow refuses a column that mixes types - e.g. floats for iteration events
    and "" for the RUN_START/RUN_END events that carry no score. Returning None
    keeps the column a single nullable dtype instead of degrading to object.
    """
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def changelog_rows(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten change_log.jsonl entries into uniform, Arrow-safe table rows."""
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        rows.append({
            "timestamp": str(entry.get("timestamp", "")),
            "event": str(entry.get("event", "")),
            "task": str(entry.get("task_id", "")),
            "iteration": _coerce_number(entry.get("iteration"), int),
            "score": _coerce_number(entry.get("score"), float),
            "applied": _coerce_number(entry.get("applied"), int),
            "deferred": _coerce_number(entry.get("deferred"), int),
            "tokens": _coerce_number(entry.get("tokens_total"), int),
        })
    return rows
