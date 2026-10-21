"""Agent 1 - Writer / Developer.

The Writer is LLM-backed only: it never synthesizes article prose from static
sentence pools. The deterministic hook is retained for unit-test harnesses, but
it can only apply mechanical repairs to text already present in the draft.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..llm import coerce_findings
from ..models import AgentReport, AgentStatus, Brief, Draft, Finding, Section, Severity
from .base import BaseAgent
from .payload import extract_section_list, match_section_id, section_body

WRITER_REMIT = {"readability", "clarity", "completeness", "flow", "redundancy",
                "tone", "consistency", "coverage", "balance"}
FILL_THRESHOLD = 0.8
LONG_SENTENCE_WORDS = 30
LONG_PARAGRAPH_WORDS = 120

_CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "can't": "cannot", "won't": "will not",
    "it's": "it is", "that's": "that is", "there's": "there is",
    "we're": "we are", "they're": "they are", "you're": "you are",
    "i'm": "I am", "let's": "let us", "we've": "we have", "they've": "they have",
}
FILLER_PATTERNS = [
    (r"\bin order to\b", "to"),
    (r"\bit is (?:important|worth) (?:to note|noting) that\s*", ""),
    (r"\bbasically\b", ""),
    (r"\bactually\b", ""),
    (r"\bvery\s+", ""),
    (r"\breally\s+", ""),
    (r"\bjust\s+", ""),
    (r"\bat the end of the day\b", ""),
]
_DUPLICATE_WORD_RE = re.compile(r"\b(\w+)(\s+\1)+\b", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]+")
SPLIT_CONNECTOR_RES = (
    re.compile(r";\s+"),
    re.compile(r",\s+(?=(?:which|but|so|because|or)\b)"),
)


# Mechanical repair / polish helpers are safe outside code fences and converge:
# they reorganise or clean existing text, never create new article prose.
def _split_one_long_sentence(sentence: str, max_words: int) -> str:
    """Split one overlong sentence at its central connector into sentences."""
    words = sentence.split()
    if len(words) <= max_words:
        return sentence
    candidates: List[int] = []
    for pattern in SPLIT_CONNECTOR_RES:
        # A match at index 0 cannot yield a smaller remainder: the next call
        # would cut at 0 again and recurse forever.
        candidates.extend(m.start() for m in pattern.finditer(sentence) if m.start() > 0)
    if not candidates:
        return sentence
    middle = len(sentence) // 2
    cut = min(candidates, key=lambda pos: abs(pos - middle))
    head = sentence[:cut].rstrip()
    rest = sentence[cut:].lstrip()
    # Drop the connector ("," / ";") so it is not re-matched on the remainder.
    if rest and rest[0] in ",;":
        rest = rest[1:].lstrip()
    if not head or not rest:
        return sentence
    if head[-1] not in ".!?":
        head += "."
    rest = rest[:1].upper() + rest[1:]
    if rest[-1] not in ".!?":
        rest += "."
    # Recurse only when both halves are strictly smaller than the input.
    if len(head.split()) >= len(words) or len(rest.split()) >= len(words):
        return sentence
    return " ".join([_split_one_long_sentence(head, max_words),
                     _split_one_long_sentence(rest, max_words)])


def _split_long_sentences(block: str, max_words: int = LONG_SENTENCE_WORDS) -> str:
    """Split sentences that exceed ``max_words``, preserving inter-sentence gaps."""
    pieces: List[str] = []
    last = 0
    changed = False
    for match in _SENTENCE_RE.finditer(block):
        pieces.append(block[last:match.start()])   # gap before this sentence
        split = _split_one_long_sentence(match.group(0), max_words)
        if split != match.group(0):
            changed = True
        pieces.append(split)
        last = match.end()
    pieces.append(block[last:])
    return "".join(pieces) if changed else block


def _split_long_paragraphs(block: str, max_words: int = LONG_PARAGRAPH_WORDS) -> str:
    """Break a paragraph at the sentence seam nearest the midpoint."""
    if len(block.split()) <= max_words:
        return block
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", block) if s.strip()]
    if len(sentences) < 2:
        return block
    total = len(block.split())
    acc = 0
    cut = 1
    for index, sentence in enumerate(sentences, start=1):
        acc += len(sentence.split())
        position = index
        if acc >= total - acc:
            cut = position
            break
    head = " ".join(sentences[:cut])
    rest = " ".join(sentences[cut:])
    if not rest:
        return block
    return f"{head}\n\n{rest}"


def apply_mechanical_polish(body: str) -> Tuple[str, List[str]]:
    """Deterministic cleanup the Reviewer can verify after the fact.

    Returns the polished body plus a list of the operations that changed it.
    Code fences stay byte-identical; prose keeps its meaning but sheds padding,
    and sentence/paragraph rhythm lands inside the measured ceilings.
    """
    if not body.strip():
        return body, []
    text = body
    changed: List[str] = []

    for pattern, replacement in FILLER_PATTERNS:
        new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if new_text != text:
            changed.append("fillers")
            text = new_text
    deduped = _DUPLICATE_WORD_RE.sub(r"\1", text)
    if deduped != text:
        changed.append("dedupe_words")
        text = deduped
    for contraction, expanded in _CONTRACTIONS.items():
        new_text = re.sub(rf"\b{re.escape(contraction)}\b", expanded, text, flags=re.IGNORECASE)
        if new_text != text:
            changed.append("expand_contractions")
            text = new_text
    squeeze = re.sub(r"[ ]{2,}", " ", text)
    if squeeze != text:
        changed.append("normalised_whitespace")
        text = squeeze

    # Rhythm fixes, paragraph by paragraph so code fences and existing
    # paragraph boundaries stay intact - the Polishing section body must not
    # be flattened into one giant paragraph and re-cut blindly.
    parts: List[str] = []
    for part in re.split(r"(```.*?```)", text, flags=re.DOTALL):
        if part.startswith("```"):
            parts.append(part)
            continue
        if part.strip():
            polished_paras = []
            for para in part.split("\n\n"):
                split = _split_long_sentences(para)
                if split != para:
                    changed.append("split_sentences")
                blocks = _split_long_paragraphs(split)
                if blocks != split:
                    changed.append("split_paragraphs")
                polished_paras.append(blocks)
            parts.append("\n\n".join(polished_paras))
        else:
            parts.append(part)
    text = "".join(parts).strip()

    if text != body and not changed:
        changed.append("normalised_whitespace")
    return text, sorted(set(changed))


def repair_body(body: str, categories: Iterable[str],
                section: Optional[Section] = None, brief: Optional[Brief] = None) -> Tuple[str, List[str]]:
    """Deterministically repair a section targeted by reviewer/tester findings.

    Repairs are conservative: they only reorganise what is already there. If
    nothing mechanical applies, the body is returned unchanged so the finding
    stays deferred for a future iteration instead of churning.
    """
    if not body.strip():
        return body, []
    wanted = set(categories)
    operations: List[str] = []
    text = body

    mechanical = wanted & {"readability", "flow", "clarity", "tone", "consistency", "coverage", "completeness"}
    if mechanical:
        text, applied = apply_mechanical_polish(text)
        if applied:
            operations.extend(f"polish:{op}" for op in applied)

    # Balance: trim a section that clearly overshoots its own target.
    if "balance" in wanted and section is not None and brief is not None and section.word_target:
        words = text.split()
        if len(words) > int(section.word_target * 1.3):
            sentences = re.split(r"(?<=[.!?])\s+", text)
            keep: List[str] = []
            count = 0
            for sentence in sentences:
                keep.append(sentence)
                count += len(sentence.split())
                if count >= section.word_target:
                    break
            text = " ".join(keep).strip()
            if text != body:
                operations.append("trim_to_target")

    return text, operations


class Writer(BaseAgent):
    agent_id = "writer"
    role = "Writer/Developer"
    draft_required = True

    def should_skip(self, brief: Brief, draft: Draft, peer_reports: List[AgentReport],
                    open_findings: Optional[List[Finding]] = None) -> Optional[str]:
        """Skip when all sections meet their targets and no Writer finding is open."""
        pool: List[Finding] = list(open_findings or [])
        if not pool:
            pool = [finding for report in peer_reports for finding in report.findings]
        targeted = any(
            finding.agent in ("reviewer", "tester", "architect")
            and finding.category in WRITER_REMIT
            for finding in pool
        )
        if targeted:
            return None
        for section in draft.sections:
            target = section.word_target or max(80, brief.target_words // max(1, len(draft.sections)))
            if not section.body.strip() or section.word_count < target * FILL_THRESHOLD:
                return None
        return "no section under target and no open finding inside the writer remit"

    def deterministic(self, brief: Brief, draft: Draft, iteration: int,
                      peer_reports: List[AgentReport],
                      open_findings: Optional[List[Finding]] = None) -> Dict[str, Any]:
        """Repair existing prose only; never generate or scaffold missing content."""
        pool: List[Finding] = list(open_findings or [])
        if not pool:
            pool = [finding for report in peer_reports for finding in report.findings]
        targets: Dict[str, set] = {}
        for finding in pool:
            if (finding.agent in ("reviewer", "tester", "architect")
                    and finding.category in WRITER_REMIT):
                targets.setdefault(finding.section or "-", set()).add(finding.category)

        sections: List[Dict[str, str]] = []
        notes = ["offline writing is disabled; no new article content was generated"]
        for section in draft.sections:
            if not section.body.strip():
                continue
            wanted = targets.get(section.id, set())
            if wanted:
                body, operations = repair_body(section.body, wanted, section, brief)
                if body != section.body and body.strip():
                    sections.append({"id": section.id, "body": body})
                    notes.append(f"mechanical repair applied to {section.id}: {operations}")
        return {
            "sections": sections,
            "notes": notes,
            "summary": "offline writing disabled; existing text only may be mechanically repaired",
        }

    def compose(self, *, payload: Dict[str, Any], brief: Brief, draft: Draft, iteration: int,
                status: AgentStatus, notes: List[str], engine: str, tokens_in: int,
                tokens_out: int, peer_reports: List[AgentReport],
                open_findings: Optional[List[Finding]] = None) -> AgentReport:
        findings: List[Finding] = []
        produced = extract_section_list(payload)
        usable: List[Dict[str, str]] = []

        for item in produced:
            if not isinstance(item, dict):
                continue
            raw_id = str(item.get("id") or item.get("section_id") or item.get("section")
                         or item.get("title") or "").strip()
            section_id = match_section_id(raw_id, brief)
            body = section_body(item).strip()
            if not section_id:
                if raw_id or body:
                    findings.append(Finding(
                        agent=self.agent_id, severity=Severity.MAJOR,
                        category="outline", section=raw_id or "-",
                        claim=f"writer returned section id '{raw_id}' that is not in the outline",
                        action="Drop the section or add it to the brief outline.",
                        iteration=iteration,
                    ))
                continue
            if body:
                usable.append({"id": section_id, "body": body})

        notes = notes + [str(note) for note in (payload.get("notes") or [])]
        if engine.startswith("offline"):
            status = AgentStatus.DEGRADED
            if not any("offline writing is disabled" in note for note in notes):
                notes.append("offline writing is disabled; no new article content was generated")
        findings.extend(coerce_findings(payload.get("findings"), self.agent_id, iteration))

        return self.base_report(
            iteration=iteration, status=status, notes=notes, engine=engine,
            tokens_in=tokens_in, tokens_out=tokens_out, findings=findings,
            summary=str(payload.get("summary") or f"{len(usable)} section(s) changed"),
            payload={"sections": usable},
        )
