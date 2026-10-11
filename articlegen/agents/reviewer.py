"""Agent 3 - Reviewer / Critic.

Readability is measured, not asserted: sentence length, passive density, filler count,
paragraph size, terminology drift and tone markers are all computed from the text and
reported with their values, so a finding can be checked by hand.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from ..llm import coerce_findings
from ..models import AgentReport, AgentStatus, Brief, Draft, Finding, Severity
from .base import BaseAgent

SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]+")
PASSIVE_RE = re.compile(
    r"\b(is|are|was|were|be|been|being)\s+(\w+ed|built|written|made|done|shown|given|taken)\b",
    re.IGNORECASE,
)
FILLER_PATTERNS: List[Tuple[str, str]] = [
    (r"\bin order to\b", "to"),
    (r"\bit is (?:important|worth) (?:to note|noting) that\s*", ""),
    (r"\bbasically\b", ""),
    (r"\bactually\b", ""),
    (r"\bvery\s+", ""),
    (r"\breally\s+", ""),
    (r"\bjust\s+", ""),
    (r"\bat the end of the day\b", ""),
]
CONTRACTION_RE = re.compile(r"\b\w+'(?:t|s|re|ve|ll|d|m)\b")
DUPLICATE_WORD_RE = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)

LONG_SENTENCE_WORDS = 30
LONG_PARAGRAPH_WORDS = 120
AVG_SENTENCE_CEILING = 24.0
PASSIVE_CEILING = 0.18
MAX_FINDINGS_PER_CATEGORY = 5


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SENTENCE_RE.findall(text) if s.strip()]


class Reviewer(BaseAgent):
    agent_id = "reviewer"
    role = "Reviewer/Critic"
    draft_required = True

    def deterministic(self, brief: Brief, draft: Draft, iteration: int,
                      peer_reports: List[AgentReport]) -> Dict[str, Any]:
        findings: List[Finding] = []
        counts: Dict[str, int] = {}

        def add(finding: Finding) -> None:
            if counts.get(finding.category, 0) >= MAX_FINDINGS_PER_CATEGORY:
                return
            counts[finding.category] = counts.get(finding.category, 0) + 1
            findings.append(finding)

        all_sentences: List[str] = []
        passive_hits = 0
        filler_hits = 0
        formal_tone = "formal" in brief.tone.lower()

        for section in draft.sections:
            body = section.body.strip()
            if not body:
                continue
            sentences = split_sentences(body)
            all_sentences.extend(sentences)

            long_sentences = [s for s in sentences if len(s.split()) > LONG_SENTENCE_WORDS]
            for sentence in long_sentences[:2]:
                add(Finding(
                    self.agent_id, Severity.MINOR, "readability", section.id,
                    f"sentence runs to {len(sentence.split())} words (ceiling {LONG_SENTENCE_WORDS})",
                    "Split the sentence so the reader does not have to re-read it.",
                    evidence=sentence[:120], auto_fixable=False, iteration=iteration,
                ))

            passive_in_section = PASSIVE_RE.findall(body)
            passive_hits += len(passive_in_section)
            if sentences and len(passive_in_section) / max(1, len(sentences)) > PASSIVE_CEILING:
                add(Finding(
                    self.agent_id, Severity.MINOR, "readability", section.id,
                    "passive constructions dominate this section",
                    "Prefer active voice where the actor matters.",
                    auto_fixable=False, iteration=iteration,
                ))

            for pattern, _replacement in FILLER_PATTERNS:
                filler_hits += len(re.findall(pattern, body, flags=re.IGNORECASE))

            for paragraph in re.split(r"\n\s*\n", body):
                words = len(paragraph.split())
                if words > LONG_PARAGRAPH_WORDS:
                    add(Finding(
                        self.agent_id, Severity.MINOR, "flow", section.id,
                        f"paragraph runs to {words} words (ceiling {LONG_PARAGRAPH_WORDS})",
                        "Break the paragraph at its logical seam.",
                        auto_fixable=False, iteration=iteration,
                    ))
                    break

            duplicates = DUPLICATE_WORD_RE.findall(body)
            if duplicates:
                add(Finding(
                    self.agent_id, Severity.MINOR, "clarity", section.id,
                    f"repeated word pair(s) detected: {', '.join(sorted(set(d.lower() for d in duplicates))[:3])}",
                    "Remove the duplicated word.",
                    evidence=duplicates[0], auto_fixable=True,
                    fix={"type": "dedupe_words", "section": section.id},
                    iteration=iteration,
                ))

            if formal_tone and CONTRACTION_RE.search(body):
                add(Finding(
                    self.agent_id, Severity.MINOR, "tone", section.id,
                    "contractions appear in prose the brief specifies as formal",
                    "Expand contractions to match the requested tone.",
                    auto_fixable=True,
                    fix={"type": "expand_contractions", "section": section.id},
                    iteration=iteration,
                ))

        # Document-level metrics.
        avg_sentence = (
            sum(len(s.split()) for s in all_sentences) / len(all_sentences)
            if all_sentences else 0.0
        )
        passive_ratio = passive_hits / len(all_sentences) if all_sentences else 0.0

        if avg_sentence > AVG_SENTENCE_CEILING:
            add(Finding(
                self.agent_id, Severity.MAJOR, "readability", "document",
                f"average sentence length is {avg_sentence:.1f} words (ceiling {AVG_SENTENCE_CEILING})",
                "Shorten the longest sentences across the document.",
                iteration=iteration,
            ))

        if filler_hits:
            add(Finding(
                self.agent_id, Severity.MINOR, "clarity", "document",
                f"{filler_hits} filler construction(s) add words without meaning",
                "Delete the filler; the sentence carries the point without it.",
                auto_fixable=True, fix={"type": "remove_fillers"},
                iteration=iteration,
            ))

        # Terminology drift: the same concept named multiple ways.
        drift = _detect_terminology_drift(draft)
        for term, variants in drift.items():
            add(Finding(
                self.agent_id, Severity.MINOR, "consistency", "document",
                f"'{term}' appears under {len(variants)} spellings: {', '.join(sorted(variants))}",
                "Pick one term and use it throughout.",
                auto_fixable=False, iteration=iteration,
            ))

        return {
            "findings": [f.to_dict() for f in findings],
            "metrics": {
                "avg_sentence_words": round(avg_sentence, 2),
                "passive_ratio": round(passive_ratio, 3),
                "filler_instances": filler_hits,
                "sentences": len(all_sentences),
            },
            "summary": (
                f"{len(findings)} readability issue(s); avg sentence {avg_sentence:.1f} words, "
                f"passive ratio {passive_ratio:.2f}, {filler_hits} filler construction(s)"
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
            summary=str(payload.get("summary") or f"{len(findings)} readability issue(s)"),
            payload={"metrics": payload.get("metrics") or {}, "summary": payload.get("summary", "")},
        )


def _detect_terminology_drift(draft: Draft) -> Dict[str, set]:
    """Flag near-identical terms used inconsistently (hyphenation / spacing variants)."""
    text = draft.to_markdown().lower()
    variants: Dict[str, set] = {}
    for spaced in re.findall(r"\b[a-z]{4,}(?:[ \-][a-z]{2,}){1,2}\b", text):
        normal = re.sub(r"[ \-]", "", spaced)
        variants.setdefault(normal, set()).add(spaced)
    return {normal: forms for normal, forms in variants.items() if len(forms) > 1}
