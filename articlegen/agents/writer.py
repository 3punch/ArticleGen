"""Agent 1 - Writer / Developer.

Produces section content. Two engines:

* an LLM call when a provider is configured;
* a **deterministic scaffold generator** otherwise.

The scaffold generator is explicit about what it is. It emits structured, coherent
placeholder-free prose derived from the brief so that the rest of the loop (validation,
review, structure, synthesis, scoring, convergence) can be exercised and tested without
credentials. It is labelled ``offline-heuristic`` in every report and in the change
log, and it never fabricates citations, statistics or command output - because it does
not make factual claims at all.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ..llm import coerce_findings
from ..models import AgentReport, AgentStatus, Brief, Draft, Finding, Mode, Section, Severity
from .base import BaseAgent

_SENTENCE_TEMPLATES: List[str] = [
    "This section covers {purpose}.",
    "For {audience}, the practical question is how {title_l} changes what actually gets built and shipped.",
    "The stated objective - {objective} - depends on this part being settled before the rest follows.",
    "Three forces shape the outcome here: the constraints inherited from the brief, the cost of getting it wrong, and the cost of correcting it later.",
    "Every statement in this section is either derived from the brief or is explicitly flagged for verification before publication.",
    "Where a trade-off exists it is named rather than smoothed over, because an unstated trade-off is the one that resurfaces during review.",
    "What follows moves from the general shape of the problem to the specific decisions that fall out of it.",
    "Teams that skip this step tend to pay for it later, when the missing reasoning has to be reconstructed under deadline.",
    "The details are ordered by how strongly they constrain the choices that come after them.",
    "Nothing here assumes familiarity with the wider system beyond what the brief already states.",
    "The distinction that matters is between what is fixed by the brief and what remains a genuine design decision.",
    "Readers who only take one thing from this section should take the constraint, not the recommendation.",
]

_CODE_TEMPLATE = '''```python
def {fn}(payload: dict) -> dict:
    """Return the validated payload for the "{title}" step."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    return {{key: value for key, value in payload.items() if value is not None}}
```'''

FILL_THRESHOLD = 0.8
MAX_FILL_FACTOR = 1.3


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "step"


def _sentences(section: Section, brief: Brief, count: int, offset: int) -> List[str]:
    values = {
        "purpose": (section.purpose or section.title).strip().rstrip("."),
        "title_l": section.title.strip().lower(),
        "audience": brief.audience.strip().lower(),
        "objective": brief.objective.strip().rstrip("."),
    }
    out: List[str] = []
    for index in range(count):
        template = _SENTENCE_TEMPLATES[(offset + index) % len(_SENTENCE_TEMPLATES)]
        out.append(template.format(**values))
    return out


def scaffold_body(section: Section, brief: Brief, *, include_code: bool = False) -> str:
    """Deterministic section body derived from the brief. No randomness, no facts."""
    target = section.word_target or max(80, brief.target_words // max(1, len(brief.sections)))
    include_code = include_code or brief.mode == Mode.CODE

    paragraphs: List[str] = []
    words = 0
    offset = len(section.id) % len(_SENTENCE_TEMPLATES)
    round_index = 0
    guard = 0
    while words < target and guard < 40:
        guard += 1
        size = 3 + (round_index % 2)
        paragraph = " ".join(_sentences(section, brief, size, offset + round_index * size))
        paragraphs.append(paragraph)
        words += len(paragraph.split())
        round_index += 1
        if words > target * MAX_FILL_FACTOR:
            break

    if include_code:
        paragraphs.append(
            "The following listing shows the validation entry point for this step."
        )
        paragraphs.append(_CODE_TEMPLATE.format(fn=_slugify(section.title), title=section.title))

    return "\n\n".join(paragraphs).strip()


class Writer(BaseAgent):
    agent_id = "writer"
    role = "Writer/Developer"
    draft_required = True

    def deterministic(self, brief: Brief, draft: Draft, iteration: int,
                      peer_reports: List[AgentReport]) -> Dict[str, Any]:
        sections: List[Dict[str, str]] = []
        notes: List[str] = []
        target_ids = {f.section for r in peer_reports for f in r.findings
                      if f.agent in ("reviewer", "tester") and f.category in
                      ("readability", "clarity", "completeness", "flow", "redundancy")}

        for section in draft.sections:
            target = section.word_target or max(80, brief.target_words // max(1, len(draft.sections)))
            needs_fill = (not section.body.strip()) or section.word_count < target * FILL_THRESHOLD
            if not needs_fill and section.id not in target_ids:
                continue
            body = scaffold_body(section, brief)
            sections.append({"id": section.id, "title": section.title, "body": body})

        if not sections:
            notes.append("no section required new content this iteration")

        return {
            "sections": sections,
            "notes": notes,
            "summary": (
                f"deterministic scaffold generated for {len(sections)} section(s); "
                f"engine=offline-heuristic, no external facts asserted"
            ),
        }

    def compose(self, *, payload: Dict[str, Any], brief: Brief, draft: Draft, iteration: int,
                status: AgentStatus, notes: List[str], engine: str, tokens_in: int,
                tokens_out: int, peer_reports: List[AgentReport]) -> AgentReport:
        findings: List[Finding] = []
        produced = payload.get("sections") or []
        valid_ids = set(brief.section_ids)
        usable: List[Dict[str, str]] = []

        for item in produced:
            if not isinstance(item, dict):
                continue
            section_id = str(item.get("id", "")).strip()
            body = str(item.get("body", "") or "").strip()
            if section_id not in valid_ids:
                # Inventing sections breaks traceability; refuse rather than pass it on.
                findings.append(Finding(
                    agent=self.agent_id, severity=Severity.MAJOR,
                    category="outline", section=section_id or "-",
                    claim=f"writer returned section id '{section_id}' that is not in the outline",
                    action="Drop the section or add it to the brief outline.",
                    iteration=iteration,
                ))
                continue
            if not body:
                continue
            usable.append({"id": section_id, "body": body})

        notes = notes + [str(n) for n in (payload.get("notes") or [])]
        if engine.startswith("offline"):
            notes.append("content is deterministic scaffold prose, not model-generated writing")

        findings.extend(coerce_findings(payload.get("findings"), self.agent_id, iteration))

        return self.base_report(
            iteration=iteration, status=status, notes=notes, engine=engine,
            tokens_in=tokens_in, tokens_out=tokens_out, findings=findings,
            summary=str(payload.get("summary") or f"{len(usable)} section(s) drafted"),
            payload={"sections": usable},
        )
