"""Agent 1 - Writer / Developer.

Produces section content. Two engines:

* an LLM call when a provider is configured;
* a **deterministic drafting engine** otherwise.

The deterministic engine is explicit about what it is. It emits structured,
coherent, placeholder-free prose derived from the brief so that the rest of the
loop (validation, review, structure, synthesis, scoring, convergence) can be
exercised and tested without credentials. It is labelled ``offline-heuristic``
in every report and in the change log, and it never fabricates citations,
statistics or command output - because it does not make factual claims at all.

Upgrade notes (professional drafting, minimum token spend):

* the drafting engine composes paragraphs from role-classified sentence pools
  (framing, audience, method, integrity, risk, payoff, takeaway) instead of
  cycling one flat template list, and it never repeats a sentence within a
  section. A shared pool across one drafting call keeps *adjacent* sections
  from repeating sentences, so the document does not read as copies;
* every section opens on its declared purpose (framing lead) so the
  Architect's repetitive-opening check has nothing to flag;
* when a section is targeted by reviewer/tester findings, the Writer applies
  **deterministic repairs** (filler removal, word dedup, long-sentence and
  long-paragraph splitting, contraction expansion) instead of regenerating
  identical scaffold text. Repairs converge, so the loop needs fewer
  iterations - which is where real tokens are actually saved;
* ``should_skip`` lets the orchestrator drop the agent call entirely when no
  section is under target and no open finding is inside the Writer's remit.
  A skipped agent call costs zero tokens.

All mechanical helpers are shared with the Synthesizer's polish pass.
"""

from __future__ import annotations

import re
import zlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..llm import coerce_findings
from ..models import AgentReport, AgentStatus, Brief, Draft, Finding, Mode, Section, Severity
from .base import BaseAgent

# Categories the Writer is allowed to repair mechanically.
WRITER_REMIT = {"readability", "clarity", "completeness", "flow", "redundancy",
                "tone", "consistency", "coverage", "balance"}

SENTENCE_TEMPLATES: List[str] = [
    # -- framing (0-7): opens a section with its declared purpose. The first
    # four words must stay distinct across all eight so each section of a
    # document can open differently (the Architect checks repeated openings).
    # Purpose slots only appear in colon continuations, where the brief's
    # verb-phrase purposes read naturally.
    "This section does the groundwork for {title_l} so the rest of the document can build on it.",
    "Everything that follows serves one declared purpose: {purpose}.",
    "The aim here is narrow: {purpose}, handled once so later sections can rely on it.",
    "This part of the article treats {title_l} at the level of practice rather than surface description.",
    "Consider {title_l} the pivot; everything else in this part hangs from what happens here.",
    "Start from the declared job of {title_l}: {purpose}, nothing more this time.",
    "Readers who join mid-document still meet the same aim here: {purpose}, recorded once for the whole chapter.",
    "Where the document needs structure, {title_l} provides it: {purpose}, in the order shown below.",
    # -- audience (8-10): ties the section to the stated reader
    "The value below serves {audience}, and every point lands in something a reviewer can check.",
    "For {audience}, the useful question is not whether {title_l} matters, but how it changes day-to-day decisions.",
    "{audience_cap} will judge this part in two respects: accuracy of the decisions and clarity of the language.",
    # -- method (11-14): explains how the section argues
    "The discussion moves from the general shape of the problem to the decisions that fall out of it.",
    "The order follows constraint strength: what the brief fixes comes first, genuine design freedom comes second.",
    "Each claim traces back to the brief, the outline, or an open finding; nothing gets asserted from outside.",
    "The argument proceeds in steps, and each step leaves a visible seam for review to pick at.",
    # -- integrity (15-17): verification discipline
    "Figures and sources arrive from the brief and the open records; anything without support gets flagged for review.",
    "Anything the brief does not settle gets marked for verification, not smoothed over.",
    "Where a trade-off exists it gets named, because an unstated trade-off returns during review with interest.",
    # -- risk (18-20): consequences of ignoring the section
    "The main risk here is subtlety: a requirement that looks fulfilled yet stays practically uncheckable.",
    "Two failure modes dominate: doing the work twice, and doing it once without the context needed to check it.",
    "Doing this poorly does not remove the requirement; it moves the cost to whoever holds the least context.",
    # -- payoff (21-22): what the reader gains
    "Handled well, {title_l} removes a whole class of late-stage defects from the surrounding work.",
    "Payoff shows up directly: fewer surprises during review, and clearer language for the decisions that need approval.",
    # -- takeaway (23-25): closes the section
    "The section closes on its operative point: {purpose}, settled deliberately rather than by accident.",
    "The one point to carry out of this section is the constraint, not the recommendation.",
    "What follows depends on the choices recorded above, and those choices stayed explicit throughout.",
]

#: Role groups for rotation: index ranges inside ``SENTENCE_TEMPLATES``.
ROLE_RANGES: Dict[str, Tuple[int, int]] = {
    "framing": (0, 8),
    "audience": (8, 11),
    "method": (11, 15),
    "integrity": (15, 18),
    "risk": (18, 21),
    "payoff": (21, 23),
    "takeaway": (23, 26),
}

#: Paragraph composition per drafting round. Round 0 is always the framing
#: lead - every section opens on its declared purpose - and later rounds vary
#: their shape so consecutive sections do not read as copies.
PARAGRAPH_ROUNDS: List[Tuple[str, str, str]] = [
    ("framing", "method", "audience"),
    ("method", "integrity", "risk"),
    ("payoff", "method", "framing"),
    ("audience", "risk", "payoff"),
    ("integrity", "method", "risk"),
    ("risk", "payoff", "integrity"),
]

TAKEAWAY_ROUNDS = [
    "The section closes on its operative point: {purpose}, settled deliberately rather than by accident.",
    "The one point to carry out of this section is the constraint, not the recommendation.",
    "What follows depends on the choices recorded above, and those choices stayed explicit throughout.",
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

# Shared, measured limits (must match the Reviewer's ceilings) plus the
# gate's rendered-line limit, which a paragraph string must respect.
LONG_SENTENCE_WORDS = 30
LONG_PARAGRAPH_WORDS = 120
MAX_PARAGRAPH_CHARS = 360

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


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "step"


def _values(section: Section, brief: Brief) -> Dict[str, str]:
    """Format slots for the template pool.

    The audience keeps its original casing (``LLM`` stays uppercase); only the
    first character is capitalised when it starts a sentence slot.
    """
    purpose = (section.purpose or section.title).strip().rstrip(".")
    audience = brief.audience.strip()
    purpose_l = purpose[0].lower() + purpose[1:] if purpose and purpose[0].isupper() else purpose
    title = section.title.strip()
    audience_cap = audience[:1].upper() + audience[1:] if audience else audience
    return {
        "purpose": purpose_l,
        "purpose_l": purpose_l,
        "title": title,
        "title_l": title.lower(),
        "audience": audience,
        "audience_cap": audience_cap,
        "objective": brief.objective.strip().rstrip("."),
    }


def _pick(used: set, start: int, count: int, seat: int) -> int:
    """Return a template index in ``[start, start+count)`` that is unused so far.

    ``seat`` gives each section its own rotation via a stable hash, so sections
    draft from different points in the pool while remaining deterministic.
    """
    base = (start + seat) % count
    for step in range(count):
        candidate = start + (base + step) % count
        if candidate not in used:
            return candidate
    return start + (base % count)   # all used; last resort reuses the rotation start


def scaffold_body(section: Section, brief: Brief, *, include_code: bool = False,
                  rotation: int = 0, shared_used: Optional[set] = None,
                  framing_index: Optional[int] = None) -> str:
    """Deterministic section body derived from the brief. No randomness, no facts.

    ``rotation`` varies the draft across iterations (the Writer passes the
    iteration number) while keeping repeated calls with the same arguments
    byte-identical. ``shared_used`` lets one drafting call (all sections of a
    document) avoid repeating sentences between sections, and
    ``framing_index`` pins the opening sentence so each section of a document
    opens differently.
    """
    target = section.word_target or max(80, brief.target_words // max(1, len(brief.sections)))
    include_code = include_code or brief.mode == Mode.CODE

    values = _values(section, brief)
    # Stable per-section seat, shifted deterministically by the rotation.
    seat = zlib.crc32(f"{section.id}:{rotation}".encode()) % 97
    used: set = shared_used if shared_used is not None else set()

    paragraphs: List[str] = []
    buffer: List[str] = []
    words = 0
    round_index = 0
    guard = 0
    while words < target and guard < 120:
        guard += 1
        roles = PARAGRAPH_ROUNDS[0] if round_index == 0 else PARAGRAPH_ROUNDS[(round_index + seat) % len(PARAGRAPH_ROUNDS)]
        if round_index == 0 and framing_index is not None:
            f_start, f_end = ROLE_RANGES["framing"]
            if not f_start <= framing_index < f_end:
                framing_index = None   # out of range; ignore an invalid hint
        for role in roles:
            start, end = ROLE_RANGES[role]
            if role == "framing" and framing_index is not None and round_index == 0:
                index = framing_index
                framing_index = None   # consume the hint once
            else:
                index = _pick(used, start, end - start, seat + round_index)
            used.add(index)
            sentence = SENTENCE_TEMPLATES[index].format(**values)
            # Keep paragraph lines inside the gate's 400-character limit:
            # flush the buffer BEFORE a sentence would push the line over.
            projected = " ".join(buffer + [sentence])
            if buffer and len(projected) > MAX_PARAGRAPH_CHARS:
                paragraphs.append(" ".join(buffer))
                buffer = []
            buffer.append(sentence)
            words += len(sentence.split())
            if words >= target:
                break
        if len(buffer) >= 3:
            paragraphs.append(" ".join(buffer))
            buffer = []
        if words >= target:
            break
        round_index += 1
        if words > target * MAX_FILL_FACTOR:
            break
    if buffer:
        paragraphs.append(" ".join(buffer))

    # Close with a single-sentence takeaway for rhythm and a crisp exit.
    takeaway = TAKEAWAY_ROUNDS[(seat + round_index) % len(TAKEAWAY_ROUNDS)].format(**values)
    paragraphs.append(takeaway)

    if include_code:
        paragraphs.append(
            "The following listing shows the validation entry point for this step."
        )
        paragraphs.append(_CODE_TEMPLATE.format(fn=_slugify(section.title), title=section.title))

    return "\n\n".join(paragraphs).strip()


# ---------------------------------------------------------------------------
# Mechanical repair / polish helpers (shared with the Synthesizer).
# These operations are safe outside code fences and converge: once applied,
# the same finding does not reproduce.
# ---------------------------------------------------------------------------

def _split_one_long_sentence(sentence: str, max_words: int) -> str:
    """Split one overlong sentence at its central connector into sentences."""
    words = sentence.split()
    if len(words) <= max_words:
        return sentence
    candidates: List[int] = []
    for pattern in SPLIT_CONNECTOR_RES:
        candidates.extend(m.start() for m in pattern.finditer(sentence))
    if not candidates:
        return sentence
    middle = len(sentence) // 2
    cut = min(candidates, key=lambda pos: abs(pos - middle))
    head = sentence[:cut].rstrip()
    rest = sentence[cut:].lstrip()
    # The separator stays on its side; both halves keep full sentences.
    head = head.rstrip()
    if rest and rest[0] in ",":
        rest = rest[1:].lstrip()
    if head and head[-1] not in ".!?":
        head += "."
    rest = rest[:1].upper() + rest[1:]
    if rest and rest[-1] not in ".!?":
        rest += "."
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

    # -- skip rule --------------------------------------------------------
    def should_skip(self, brief: Brief, draft: Draft, peer_reports: List[AgentReport],
                    open_findings: Optional[List[Finding]] = None) -> Optional[str]:
        """Skip this agent's call when there is genuinely nothing to write."""
        pool: List[Finding] = list(open_findings or [])
        if not pool:
            pool = [f for r in peer_reports for f in r.findings]
        targeted = any(f.agent in ("reviewer", "tester", "architect") and f.category in WRITER_REMIT
                       for f in pool)
        if targeted:
            return None
        for section in draft.sections:
            target = section.word_target or max(80, brief.target_words // max(1, len(draft.sections)))
            if not section.body.strip() or section.word_count < target * FILL_THRESHOLD:
                return None
        return "no section under target and no open finding inside the writer remit"

    # -- deterministic engine ---------------------------------------------
    def deterministic(self, brief: Brief, draft: Draft, iteration: int,
                      peer_reports: List[AgentReport],
                      open_findings: Optional[List[Finding]] = None) -> Dict[str, Any]:
        pool: List[Finding] = list(open_findings or [])
        if not pool:
            pool = [f for r in peer_reports for f in r.findings]
        targets: Dict[str, set] = {}
        for finding in pool:
            if finding.agent in ("reviewer", "tester", "architect") and finding.category in WRITER_REMIT:
                targets.setdefault(finding.section or "-", set()).add(finding.category)

        sections: List[Dict[str, str]] = []
        notes: List[str] = []
        shared_used: set = set()   # no identical sentence across sections in one pass
        taken_openers: set = set()  # each section of the document opens differently
        f_start, f_end = ROLE_RANGES["framing"]
        anchor_pending = True      # quote the brief objective once, in the first drafted section

        for section in draft.sections:
            target = section.word_target or max(80, brief.target_words // max(1, len(draft.sections)))
            underfilled = (not section.body.strip()) or section.word_count < target * FILL_THRESHOLD

            if underfilled:
                # Rewrite from scratch, rotated by iteration so a section that
                # was not accepted once comes back visibly different.
                opener = _pick(taken_openers, f_start, f_end - f_start,
                               zlib.crc32(section.id.encode()) % 97)
                taken_openers.add(opener)
                body = scaffold_body(section, brief, rotation=iteration, shared_used=shared_used,
                                     framing_index=opener)
                if anchor_pending and brief.objective.strip():
                    body += f"\n\n> Brief objective: {brief.objective.strip().rstrip('.')}"
                    anchor_pending = False
                    notes.append("brief objective quoted in the first drafted section (coverage anchor)")
                sections.append({"id": section.id, "title": section.title, "body": body})
                notes.append(f"drafted {section.id} ({len(body.split())} words)")
                continue

            wanted = targets.get(section.id, set())
            if wanted:
                body, operations = repair_body(section.body, wanted, section, brief)
                if body != section.body and body.strip():
                    sections.append({"id": section.id, "body": body})
                    notes.append(
                        f"repair applied to {section.id} for {sorted(wanted)}: {operations}"
                    )
                else:
                    notes.append(
                        f"{section.id} targeted for {sorted(wanted)} but no mechanical "
                        "repair applies; deferred to the synthesizer"
                    )
            # Untouched sections stay untouched: churn wastes tokens and
            # pollutes the change log.

        if not sections:
            notes.append("no section required new content or repair this iteration")

        return {
            "sections": sections,
            "notes": notes,
            "summary": (
                f"{len(sections)} section(s) drafted or repaired; "
                f"engine=offline-heuristic, no external facts asserted"
            ),
        }

    def compose(self, *, payload: Dict[str, Any], brief: Brief, draft: Draft, iteration: int,
                status: AgentStatus, notes: List[str], engine: str, tokens_in: int,
                tokens_out: int, peer_reports: List[AgentReport],
                open_findings: Optional[List[Finding]] = None) -> AgentReport:
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
        if engine.startswith("offline") and usable:
            notes.append("content is deterministic drafting prose, not model-generated writing")

        findings.extend(coerce_findings(payload.get("findings"), self.agent_id, iteration))

        return self.base_report(
            iteration=iteration, status=status, notes=notes, engine=engine,
            tokens_in=tokens_in, tokens_out=tokens_out, findings=findings,
            summary=str(payload.get("summary") or f"{len(usable)} section(s) drafted"),
            payload={"sections": usable},
        )
