"""Anchored rubric and quartile policy for whole-article assessment.

The evaluator rates what is demonstrable from the supplied manuscript. It does not
claim to verify scientific truth, novelty against the literature, or journal fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    description: str
    anchors: Tuple[str, str, str, str, str]


DIMENSIONS: Tuple[Dimension, ...] = (
    Dimension(
        "contribution", "Contribution and focus",
        "A clear purpose, central thesis, and meaningful contribution for the intended reader.",
        ("No clear purpose or central point.", "Purpose is vague; mostly generic or repetitive.",
         "Purpose is identifiable but the contribution or focus is uneven.",
         "Clear, useful contribution with a focused through-line.",
         "Distinctive, sharply scoped contribution that consistently advances the thesis."),
    ),
    Dimension(
        "technical_soundness", "Technical soundness",
        "Accuracy and internal consistency of explanations, reasoning, methods, and conclusions.",
        ("Severe contradictions or plainly unsound reasoning.",
         "Major unsupported leaps, errors, or inconsistencies.",
         "Mostly coherent, but important reasoning gaps remain.",
         "Careful and coherent reasoning; only minor issues.",
         "Precise, rigorous reasoning with assumptions and limits made explicit."),
    ),
    Dimension(
        "evidence", "Evidence and sourcing",
        "Whether important factual claims are supported by traceable evidence in the article.",
        ("Material claims have no usable support, or sources are clearly unreliable.",
         "Many important claims lack citations, data, or adequate support.",
         "Some useful support is present, but notable claims remain weakly supported.",
         "Most important claims are supported and sources are relevant and traceable.",
         "Evidence is comprehensive, precise, well-integrated, and limitations are acknowledged."),
    ),
    Dimension(
        "organization", "Organization and completeness",
        "Logical progression, appropriate coverage, section coherence, and a useful conclusion.",
        ("Disorganized or substantially incomplete.", "Major gaps or confusing progression.",
         "Understandable structure with some gaps, imbalance, or repetition.",
         "Well organized and appropriately complete for its stated scope.",
         "Exceptionally coherent progression; every section serves the argument."),
    ),
    Dimension(
        "clarity", "Clarity and style",
        "Readability, precision, audience fit, and effective use of terminology and examples.",
        ("Frequently incomprehensible or inappropriate for the audience.",
         "Hard to follow; persistent ambiguity or distracting prose problems.",
         "Generally understandable but uneven, wordy, or insufficiently explained.",
         "Clear, precise, readable, and appropriate for the stated audience.",
         "Effortless to follow, economical, precise, and consistently engaging."),
    ),
    Dimension(
        "reproducibility", "Methods, examples, and reproducibility",
        "Whether methods, examples, data, and calculations are detailed enough to inspect or reproduce when relevant to the article type.",
        ("A central method or example is impossible to inspect where one is essential.",
         "Critical procedural or example details are missing where they are needed.",
         "Some methods or examples are inspectable, but important applicable details are missing.",
         "Methods and examples are sufficiently detailed for the article's genre and scope.",
         "Fully inspectable or reproducible where applicable, with assumptions, data, and limits clear."),
    ),
)

WEIGHTS: Dict[str, float] = {
    "contribution": 0.15,
    "technical_soundness": 0.25,
    "evidence": 0.25,
    "organization": 0.15,
    "clarity": 0.10,
    "reproducibility": 0.10,
}

QUARTILE_POLICY = {
    "Q1": "Internal band: all dimensions >= 4, no blocker, and at most one major issue. Strong draft; still requires expert review.",
    "Q2": "Internal band: all dimensions >= 3, no blocker, and at most two major issues.",
    "Q3": "Internal band: at least one dimension < 3, or more than two major issues, with no blocker.",
    "Q4": "Internal band: a blocker is present; do not rely on or submit this draft without substantial review.",
}
