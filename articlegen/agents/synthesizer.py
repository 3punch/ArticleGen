"""Agent 5 - Synthesizer / Refiner.

The only agent permitted to emit a new master draft (rule R2). It merges the Writer's
content with the other agents' findings in a fixed order - structure, then correctness,
then content, then readability - so a later step can never silently undo an earlier one.

Every incoming finding lands in exactly one of ``applied``, ``deferred`` or
``rejected``. Nothing is dropped on the floor: that is what keeps the loop converging
instead of oscillating on the same defects.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..models import AgentReport, AgentStatus, Brief, Draft, Finding, Section, Severity
from .base import BaseAgent
from .reviewer import FILLER_PATTERNS
from .writer import apply_mechanical_polish

CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "can't": "cannot", "won't": "will not",
    "it's": "it is", "that's": "that is", "there's": "there is",
    "we're": "we are", "they're": "they are", "you're": "you are",
    "i'm": "I am", "let's": "let us", "we've": "we have", "they've": "they have",
}
DUPLICATE_WORD_RE = re.compile(r"\b(\w+)(\s+\1)+\b", re.IGNORECASE)
ANCHOR_LINK_RE = re.compile(r"\[([^\]]+)\]\(#[^)]*\)")
H1H2_RE = re.compile(r"^(#{1,2})(\s)", re.MULTILINE)


def _apply_text_fixes(body: str, fix_types: set) -> str:
    if not body:
        return body
    text = body
    if "dedupe_words" in fix_types:
        text = DUPLICATE_WORD_RE.sub(r"\1", text)
    if "remove_fillers" in fix_types:
        for pattern, replacement in FILLER_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        text = re.sub(r"[ ]{2,}", " ", text)
        # Do not sentence-capitalize the scheme in a URL following punctuation.
        text = re.sub(
            r"([.!?]) (?![A-Za-z][A-Za-z0-9+.-]*://)([a-z])",
            lambda m: f"{m.group(1)} {m.group(2).upper()}",
            text,
        )
        text = re.sub(r"\b,\s*([.!?])", r"\1", text)
    if "expand_contractions" in fix_types:
        for contraction, expanded in CONTRACTIONS.items():
            text = re.sub(rf"\b{re.escape(contraction)}\b", expanded, text, flags=re.IGNORECASE)
    if "drop_anchor" in fix_types:
        text = ANCHOR_LINK_RE.sub(r"\1", text)
    if "demote_headings" in fix_types:
        text = H1H2_RE.sub(r"###\2", text)
    return text.strip()


class Synthesizer(BaseAgent):
    agent_id = "synthesizer"
    role = "Synthesizer/Refiner"
    draft_required = True

    def deterministic(self, brief: Brief, draft: Draft, iteration: int,
                      peer_reports: List[AgentReport],
                      open_findings: Optional[List[Finding]] = None) -> Dict[str, Any]:
        brief_map = {s.id: s for s in brief.sections}
        working: Dict[str, Section] = {
            s.id: Section(id=s.id, title=s.title, purpose=s.purpose, body=s.body,
                          word_target=s.word_target)
            for s in draft.sections
        }
        applied: List[str] = []
        deferred: List[Dict[str, str]] = []
        rejected: List[Dict[str, str]] = []
        text_fixes: Dict[str, set] = {}

        reports_by_agent: Dict[str, AgentReport] = {r.agent_id: r for r in peer_reports}

        # -- 1. Content from the Writer ---------------------------------
        writer = reports_by_agent.get("writer")
        if writer:
            for item in writer.payload.get("sections") or []:
                section_id = item.get("id")
                body = (item.get("body") or "").strip()
                if section_id not in brief_map:
                    rejected.append({"claim": f"writer section '{section_id}'",
                                     "reason": "id is not in the approved outline"})
                    continue
                if not body:
                    rejected.append({"claim": f"writer body for '{section_id}'",
                                     "reason": "empty body would delete the section"})
                    continue
                if section_id not in working:
                    source = brief_map[section_id]
                    working[section_id] = Section(id=section_id, title=source.title,
                                                  purpose=source.purpose,
                                                  word_target=source.word_target)
                working[section_id].body = body
                applied.append(f"writer:rewrote {section_id}")

        # -- 2. Structural corrections from the Architect ---------------
        architect = reports_by_agent.get("architect")
        if architect:
            for finding in architect.findings:
                fix = finding.fix or {}
                kind = fix.get("type")
                if not finding.auto_fixable or not kind:
                    deferred.append({"claim": finding.claim,
                                     "reason": "structural change needs human judgement"})
                    continue
                if kind == "add_sections":
                    for section_id in fix.get("ids") or []:
                        source = brief_map.get(section_id)
                        if not source:
                            continue
                        working[section_id] = Section(
                            id=section_id, title=source.title, purpose=source.purpose,
                            body="", word_target=source.word_target,
                        )
                        applied.append(f"architect:restored {section_id}")
                elif kind == "remove_sections":
                    for section_id in fix.get("ids") or []:
                        if working.pop(section_id, None) is not None:
                            applied.append(f"architect:removed {section_id}")
                elif kind == "reorder":
                    applied.append("architect:reordered to outline sequence")
                elif kind == "demote_headings":
                    text_fixes.setdefault(fix.get("section", "*"), set()).add("demote_headings")
                    applied.append(f"architect:demoted headings in {fix.get('section')}")
                else:
                    deferred.append({"claim": finding.claim,
                                     "reason": f"no mechanical fix for '{kind}'"})

        # -- 3. Safety net: the outline is authoritative -----------------
        for section_id, source in brief_map.items():
            if section_id not in working:
                working[section_id] = Section(
                    id=section_id, title=source.title, purpose=source.purpose,
                    body="", word_target=source.word_target,
                )
                applied.append(f"synthesizer:restored empty outline section {section_id}")

        # -- 4. Readability and mechanical correctness fixes ------------
        for agent_id in ("reviewer", "tester"):
            report = reports_by_agent.get(agent_id)
            if not report:
                continue
            for finding in report.findings:
                fix = finding.fix or {}
                kind = fix.get("type")
                if finding.auto_fixable and kind:
                    target = fix.get("section", "*")
                    text_fixes.setdefault(target, set()).add(kind)
                    applied.append(f"{agent_id}:{kind}@{target}")
                    continue
                if finding.severity in (Severity.NOTE,):
                    deferred.append({"claim": finding.claim, "reason": "advisory note; deferred"})
                    continue
                deferred.append({
                    "claim": finding.claim,
                    "reason": "requires content or judgement change in a later iteration",
                })

        # -- 5. Apply the accumulated text fixes ------------------------
        global_fixes = text_fixes.get("*", set())
        for section_id, section in working.items():
            fixes = global_fixes | text_fixes.get(section_id, set())
            if fixes:
                section.body = _apply_text_fixes(section.body, fixes)

        # -- 5b. Shared mechanical polish -------------------------------
        # Runs every iteration on every section: it is measured, converging
        # (once applied the same finding does not reproduce) and it stops the
        # Reviewer from re-raising rhythm defects the fixes already settle.
        for section_id, section in working.items():
            polished, operations = apply_mechanical_polish(section.body)
            if operations:
                section.body = polished
                applied.append(f"polish:{'+'.join(operations)}@{section_id}")

        # -- 6. Emit in canonical outline order -------------------------
        ordered = [working[sid].to_dict() for sid in brief.section_ids if sid in working]
        return {
            "title": draft.title or brief.title,
            "sections": ordered,
            "applied": applied,
            "deferred": deferred,
            "rejected": rejected,
            "summary": (
                f"merged {len(applied)} change(s); {len(deferred)} deferred, {len(rejected)} rejected; "
                f"{len(ordered)} section(s) in final draft"
            ),
        }

    def compose(self, *, payload: Dict[str, Any], brief: Brief, draft: Draft, iteration: int,
                status: AgentStatus, notes: List[str], engine: str, tokens_in: int,
                tokens_out: int, peer_reports: List[AgentReport],
                open_findings: Optional[List[Finding]] = None) -> AgentReport:
        sections = payload.get("sections") or []
        return self.base_report(
            iteration=iteration, status=status, notes=notes, engine=engine,
            tokens_in=tokens_in, tokens_out=tokens_out, findings=[],
            summary=str(payload.get("summary") or f"{len(sections)} section(s) merged"),
            payload={
                "sections": sections,
                "title": payload.get("title") or draft.title,
                "applied": payload.get("applied") or [],
                "deferred": payload.get("deferred") or [],
                "rejected": payload.get("rejected") or [],
            },
        )


def build_draft(payload: Dict[str, Any], brief: Brief, previous_version: int) -> Optional[Draft]:
    """Turn a Synthesizer payload into the new master draft."""
    sections = payload.get("sections")
    if not isinstance(sections, list) or not sections:
        return None
    parsed: List[Section] = []
    seen = set()
    for item in sections:
        if not isinstance(item, dict):
            return None
        section_id, body = item.get("id"), item.get("body")
        if (not isinstance(section_id, str) or section_id not in brief.section_ids
                or section_id in seen or not isinstance(body, str)):
            return None
        seen.add(section_id)
        try:
            section = Section.from_dict(item)
        except (TypeError, ValueError):
            return None
        parsed.append(section)
    if seen != set(brief.section_ids) or not any(s.body.strip() for s in parsed):
        return None
    return Draft(
        task_id=brief.task_id,
        version=previous_version + 1,
        title=str(payload.get("title") or brief.title),
        sections=parsed,
        engine=payload.get("engine", "synthesizer"),
    )
