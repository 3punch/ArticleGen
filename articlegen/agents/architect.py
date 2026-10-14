"""Agent 4 - Architect / Structure.

Owns global shape only. Every finding it can express mechanically carries a ``fix``
payload so the Synthesizer can apply it without re-deriving the decision; judgement
calls stay ``auto_fixable=False`` and are deferred instead.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..llm import coerce_findings
from ..models import AgentReport, AgentStatus, Brief, Draft, Finding, Severity
from .base import BaseAgent

H1_IN_BODY_RE = re.compile(r"^#{1,2}\s+", re.MULTILINE)
BALANCE_TOLERANCE = 2.0
OVERALL_DEVIATION = 0.40
OPENING_WORDS = 4


class Architect(BaseAgent):
    agent_id = "architect"
    role = "Architect/Structure"
    draft_required = False

    def deterministic(self, brief: Brief, draft: Draft, iteration: int,
                      peer_reports: List[AgentReport],
                      open_findings: Optional[List[Finding]] = None) -> Dict[str, Any]:
        findings: List[Finding] = []
        brief_ids = brief.section_ids
        draft_ids = [s.id for s in draft.sections]
        draft_map = draft.section_map()

        missing = [sid for sid in brief_ids if sid not in draft_map]
        extra = [sid for sid in draft_ids if sid not in set(brief_ids)]

        if missing:
            findings.append(Finding(
                self.agent_id, Severity.BLOCKER, "outline", ",".join(missing),
                f"outline section(s) absent from the draft: {', '.join(missing)}",
                "Restore the missing sections so the draft matches the approved outline.",
                auto_fixable=True, fix={"type": "add_sections", "ids": missing},
                iteration=iteration,
            ))
        if extra:
            findings.append(Finding(
                self.agent_id, Severity.MAJOR, "outline", ",".join(extra),
                f"draft contains section(s) outside the outline: {', '.join(extra)}",
                "Remove them, or route the outline change back to the human.",
                auto_fixable=True, fix={"type": "remove_sections", "ids": extra},
                iteration=iteration,
            ))

        # Ordering: compare the relative order of the ids both sides agree on.
        shared_brief = [sid for sid in brief_ids if sid in draft_map]
        shared_draft = [sid for sid in draft_ids if sid in set(brief_ids)]
        if shared_brief != shared_draft:
            findings.append(Finding(
                self.agent_id, Severity.MAJOR, "narrative", "document",
                "section order diverges from the approved outline",
                f"Reorder to: {' -> '.join(shared_brief)}",
                auto_fixable=True, fix={"type": "reorder", "order": shared_brief},
                iteration=iteration,
            ))

        # Balance: per-section allocation versus the intended share.
        total_words = draft.word_count or 1
        target_total = sum(s.word_target for s in brief.sections) or brief.target_words or 0
        if target_total:
            for section in brief.sections:
                if section.id in missing or not section.word_target:
                    continue
                actual = draft_map[section.id].word_count
                expected_share = section.word_target / target_total
                actual_share = actual / total_words
                if actual > 0 and (actual_share / max(expected_share, 1e-6)) > BALANCE_TOLERANCE:
                    findings.append(Finding(
                        self.agent_id, Severity.MAJOR, "balance", section.id,
                        f"section holds {actual_share:.0%} of the document against an intended "
                        f"{expected_share:.0%} ({actual} words vs target {section.word_target})",
                        "Rebalance length so no section crowds out the others.",
                        iteration=iteration,
                    ))

        if brief.target_words and draft.word_count:
            deviation = abs(draft.word_count - brief.target_words) / brief.target_words
            if deviation > OVERALL_DEVIATION:
                direction = "over" if draft.word_count > brief.target_words else "under"
                findings.append(Finding(
                    self.agent_id, Severity.MAJOR, "balance", "document",
                    f"draft is {draft.word_count} words, {deviation:.0%} {direction} the "
                    f"{brief.target_words}-word target",
                    "Expand thin sections or trim overlong ones to approach the target.",
                    iteration=iteration,
                ))

        # Heading discipline inside section bodies.
        for section in draft.sections:
            if H1_IN_BODY_RE.search(section.body):
                findings.append(Finding(
                    self.agent_id, Severity.MINOR, "headings", section.id,
                    "section body declares an H1/H2, which collides with the document outline",
                    "Demote in-body headings to H3 or deeper.",
                    auto_fixable=True, fix={"type": "demote_headings", "section": section.id},
                    iteration=iteration,
                ))

        # Repetitive openings across sections.
        openings: Dict[str, List[str]] = {}
        for section in draft.sections:
            words = section.body.split()[:OPENING_WORDS]
            if words:
                openings.setdefault(" ".join(w.lower().strip(",.") for w in words), []).append(section.id)
        for opening, ids in openings.items():
            if len(ids) > 1:
                findings.append(Finding(
                    self.agent_id, Severity.MINOR, "narrative", ",".join(ids),
                    f"{len(ids)} sections open with the same phrase ('{opening}')",
                    "Vary the openings so sections do not read as copies.",
                    iteration=iteration,
                ))

        # Coverage of the stated objective.
        haystack = draft.to_markdown().lower()
        objective_terms = [w for w in re.findall(r"[a-z]{6,}", brief.objective.lower())]
        uncovered = [term for term in set(objective_terms) if term not in haystack]
        if objective_terms and len(uncovered) / len(set(objective_terms)) > 0.5:
            findings.append(Finding(
                self.agent_id, Severity.MAJOR, "coverage", "document",
                f"objective vocabulary largely absent from the draft "
                f"({len(uncovered)}/{len(set(objective_terms))} key terms uncovered)",
                "Address the uncovered parts of the objective, or narrow the objective.",
                iteration=iteration,
            ))

        target_order = shared_brief if shared_brief != shared_draft else []
        return {
            "findings": [f.to_dict() for f in findings],
            "target_order": target_order,
            "coverage": {"addressed": [], "missing": uncovered if objective_terms else []},
            "summary": (
                f"{len(findings)} structural issue(s); outline {len(draft_ids)}/{len(brief_ids)} "
                f"sections present, {draft.word_count} words"
            ),
        }

    def compose(self, *, payload: Dict[str, Any], brief: Brief, draft: Draft, iteration: int,
                status: AgentStatus, notes: List[str], engine: str, tokens_in: int,
                tokens_out: int, peer_reports: List[AgentReport]) -> AgentReport:
        raw = payload.get("findings")
        if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "severity" in raw[0]:
            findings = coerce_findings(raw, self.agent_id, iteration)
        else:
            findings = [Finding.from_dict(f) for f in (raw or []) if isinstance(f, dict)]
        return self.base_report(
            iteration=iteration, status=status, notes=notes, engine=engine,
            tokens_in=tokens_in, tokens_out=tokens_out, findings=findings,
            summary=str(payload.get("summary") or f"{len(findings)} structural issue(s)"),
            payload={
                "target_order": payload.get("target_order") or [],
                "coverage": payload.get("coverage") or {},
            },
        )
