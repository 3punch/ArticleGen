"""Quality scoring and the convergence decision.

The score is deliberately derived from things that are *counted* - finding weights,
outline coverage, word-count deviation, measured readability - rather than from a model
rating itself. That keeps runs reproducible (rule R10) and makes a score movement
traceable to a specific finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import AgentReport, Brief, Draft, Finding, GateResult

WEIGHTS = {
    "correctness": 0.35,
    "readability": 0.25,
    "structure": 0.25,
    "completeness": 0.15,
}

#: Finding weight that zeroes a component. Scaled by document size so a long
#: document is not punished merely for having more surface area.
PENALTY_SCALE_PER_SECTION = 1.6


@dataclass
class ScoreCard:
    total: float
    components: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        return {"total": round(self.total, 4),
                **{key: round(value, 4) for key, value in self.components.items()}}


def _penalty(findings: List[Finding], sections: int) -> float:
    """Map finding weights onto a 0..1 penalty, larger documents absorbing more."""
    scale = PENALTY_SCALE_PER_SECTION * max(1, sections)
    weight = sum(f.severity.weight for f in findings)
    return min(1.0, weight / scale)


def draft_is_publishable(draft: Draft, brief: Brief) -> bool:
    """True when the draft is a real article, not an empty or stub outline.

    Non-completeness weights sum to 0.85, so a hollow draft used to meet the
    default score target whenever the other agents were silent.
    """
    if not draft.sections or any(not section.body.strip() for section in draft.sections):
        return False
    if brief.target_words and draft.word_count < int(brief.target_words * 0.5):
        return False
    return True


def score_iteration(brief: Brief, draft: Draft, reports: List[AgentReport],
                    gate_result: Optional[GateResult] = None) -> ScoreCard:
    sections = max(1, len(brief.sections))
    by_agent = {r.agent_id: r for r in reports}
    notes: List[str] = []
    if not any(section.body.strip() for section in draft.sections):
        notes.append("draft has no article text; an empty outline cannot meet the target")
        return ScoreCard(total=0.0, components={key: 0.0 for key in WEIGHTS}, notes=notes)

    # -- correctness ------------------------------------------------------
    tester = by_agent.get("tester")
    correctness = 1.0 - _penalty(tester.findings if tester else [], sections)

    # -- readability ------------------------------------------------------
    reviewer = by_agent.get("reviewer")
    readability = 1.0 - _penalty(reviewer.findings if reviewer else [], sections)
    if reviewer and isinstance(reviewer.payload.get("metrics"), dict):
        metrics = reviewer.payload["metrics"]
        avg = float(metrics.get("avg_sentence_words", 0) or 0)
        if avg > 24:
            readability *= max(0.7, 24.0 / avg)

    # -- structure --------------------------------------------------------
    architect = by_agent.get("architect")
    structure = 1.0 - _penalty(architect.findings if architect else [], sections)
    draft_ids = {s.id for s in draft.sections}
    if brief.section_ids:
        structure *= len([sid for sid in brief.section_ids if sid in draft_ids]) / len(brief.section_ids)
    if draft_ids - set(brief.section_ids):
        structure *= 0.9
        notes.append("draft contains sections outside the outline")

    # -- completeness -----------------------------------------------------
    completeness = 1.0
    filled = [s for s in draft.sections if s.body.strip()]
    if draft.sections:
        completeness *= len(filled) / len(draft.sections)
    if brief.target_words:
        deviation = abs(draft.word_count - brief.target_words) / brief.target_words
        completeness *= max(0.0, 1.0 - min(1.0, deviation / 0.75))

    if gate_result is not None and not gate_result.ok:
        notes.append(f"draft verification failed at {gate_result.stage}")
        completeness *= 0.5

    components = {
        "correctness": correctness,
        "readability": readability,
        "structure": structure,
        "completeness": completeness,
    }
    total = sum(WEIGHTS[key] * max(0.0, min(1.0, value)) for key, value in components.items())
    return ScoreCard(total=total, components=components, notes=notes)


def decide_next(score: float, previous_score: Optional[float], *, iteration: int,
                max_iterations: int, score_target: float, min_delta: float,
                stall_window: int, stall_count: int, degraded: bool) -> tuple:
    """Return ``(decision, stop_reason, new_stall_count)``.

    ``decision`` is one of: ``stop``, ``continue``. ``stop_reason`` is always a
    concrete, logged cause (rule R8) - the loop never ends silently. A run is
    never stopped because of token spend: there is no token ceiling.
    """
    # A degraded iteration has no trustworthy score. Stopping here is how an
    # empty outline used to exit as "score target met (0.8500 >= 0.85)".
    if score >= score_target and not degraded:
        return "stop", f"score target met ({score:.4f} >= {score_target})", 0

    if iteration >= max_iterations:
        return "stop", f"max iterations reached ({max_iterations})", stall_count

    delta = 0.0 if previous_score is None else score - previous_score
    new_stall = stall_count
    if degraded:
        # A DEGRADED agent produced low-confidence output; do not let unreliable
        # readings be the reason the loop gives up.
        new_stall = stall_count
    elif previous_score is not None and delta < min_delta:
        new_stall = stall_count + 1
    else:
        new_stall = 0

    if new_stall >= stall_window:
        return ("stop",
                f"no material improvement for {new_stall} iterations (delta {delta:+.4f} < {min_delta})",
                new_stall)

    if score < 0:  # defensive; scores are clamped, but never loop on nonsense
        return "stop", "score collapsed below zero", new_stall

    return "continue", "", new_stall
