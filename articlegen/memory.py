"""Stage 4 - Token & memory management.

The rule this module enforces: **full histories are never re-passed**. Each agent
call receives a bounded ``ContextPacket`` assembled from prioritised blocks; when
the assembled size would breach the hard cap, the lowest-priority blocks are
evicted and the eviction is recorded in ``change_log.md`` so the decision is
auditable instead of silent.

Anything that leaves the working context is compressed into a rolling summary and
appended to the change log, rather than being carried in memory.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .models import AgentReport, Draft, Finding, Severity
from .tokens import TokenBudget, count_tokens, truncate_to_tokens

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SEVERITY_HINTS = {
    "blocker": 4.0,
    "major": 3.0,
    "minor": 2.0,
    "note": 1.0,
}


def summarise_text(text: str, max_tokens: int) -> str:
    """Deterministic extractive summary.

    Kept dependency-free and non-random so runs are reproducible: sentences are
    scored by severity vocabulary, key-phrase density and position, then emitted
    in their original order until the token budget is spent.
    """
    if not text:
        return ""
    if count_tokens(text) <= max_tokens:
        return text.strip()

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return truncate_to_tokens(text, max_tokens)
    if len(sentences) == 1:
        return truncate_to_tokens(sentences[0], max_tokens)

    total = len(sentences)
    scored: List[tuple] = []
    for index, sentence in enumerate(sentences):
        lowered = sentence.lower()
        score = 1.0
        for hint, weight in _SEVERITY_HINTS.items():
            if hint in lowered:
                score += weight
        score += 0.5 * len(re.findall(r"\b(must|never|always|route|evict|blocker|regression)\b", lowered))
        score += 1.5 * (1.0 - index / total)          # earlier context is more load-bearing
        score -= min(0.8, len(sentence) / 600.0)       # prefer dense, short sentences
        scored.append((score, index, sentence))

    kept: List[tuple] = []
    used = 0
    for score, index, sentence in sorted(scored, reverse=True):
        cost = count_tokens(sentence) + 1
        if used + cost > max_tokens:
            continue
        kept.append((index, sentence))
        used += cost
        if used >= max_tokens:
            break

    if not kept:
        return truncate_to_tokens(sentences[0], max_tokens)

    ordered = [sentence for _, sentence in sorted(kept)]
    summary = " ".join(ordered)
    dropped = total - len(ordered)
    if dropped > 0:
        summary += f" [compressed: {dropped} sentence(s) folded]"
    return truncate_to_tokens(summary, max_tokens)


@dataclass
class Block:
    """A named, prioritised unit of context. Lower ``priority`` wins."""

    priority: int
    key: str
    text: str
    required: bool = False

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)

    def sort_key(self) -> tuple:
        return (self.priority, self.key)


@dataclass
class AssembledContext:
    text: str
    included: List[str] = field(default_factory=list)
    evicted: List[str] = field(default_factory=list)
    truncated: List[str] = field(default_factory=list)
    tokens: int = 0
    breached_cap: bool = False

    def describe(self) -> str:
        """One-line audit string for the change log."""
        return (
            f"tokens={self.tokens} included={','.join(self.included) or 'none'} "
            f"evicted={','.join(self.evicted) or 'none'} "
            f"truncated={','.join(self.truncated) or 'none'}"
        )


def assemble(blocks: Iterable[Block], budget: TokenBudget, *, header: str = "") -> AssembledContext:
    """Pack blocks into the hard cap, evicting by priority when necessary."""
    available = max(0, budget.context_hard_cap)
    header_tokens = count_tokens(header) if header else 0
    remaining = max(0, available - header_tokens)

    ordered = sorted(blocks, key=lambda b: b.sort_key())
    included: List[str] = []
    evicted: List[str] = []
    truncated: List[str] = []
    pieces: List[str] = []
    used = 0
    breached = False

    for block in ordered:
        if not block.text.strip():
            continue
        section = f"### {block.key}\n{block.text.strip()}"
        section_cost = count_tokens(section) + 1

        if used + section_cost <= remaining:
            pieces.append(section)
            included.append(block.key)
            used += section_cost
            continue

        if block.required:
            # Required context is never dropped outright - it is truncated and
            # flagged, so the agent still sees *something* plus an explicit notice.
            room = max(0, remaining - used)
            if room > count_tokens(f"### {block.key}\n") + 8:
                body_budget = room - count_tokens(f"### {block.key}\n") - 8
                clipped = truncate_to_tokens(block.text, body_budget)
                pieces.append(f"### {block.key}\n{clipped}")
                included.append(block.key)
                truncated.append(block.key)
                used += count_tokens(f"### {block.key}\n{clipped}") + 1
                breached = True
            else:
                evicted.append(f"{block.key}(required:unfit)")
                breached = True
        else:
            evicted.append(block.key)

    text = header + ("\n\n" if header else "") + "\n\n".join(pieces)
    return AssembledContext(
        text=text,
        included=included,
        evicted=evicted,
        truncated=truncated,
        tokens=count_tokens(text),
        breached_cap=breached,
    )


class RollingMemory:
    """Fixed-size digest of everything that has already happened.

    Instead of replaying transcripts, each iteration contributes one compressed
    line. When the digest grows past its budget the oldest material is folded
    down further - so memory cost is O(budget), not O(iterations).
    """

    def __init__(self, max_tokens: int = 450, max_entries: int = 8) -> None:
        self.max_tokens = max_tokens
        self.max_entries = max_entries
        self.entries: List[str] = []

    def absorb(self, text: str) -> None:
        if not text.strip():
            return
        per_entry = max(40, self.max_tokens // max(1, self.max_entries))
        self.entries.append(summarise_text(text, per_entry))
        self._compress()

    def _compress(self) -> None:
        while len(self.entries) > self.max_entries:
            head = self.entries[: len(self.entries) - self.max_entries + 1]
            folded = summarise_text(" ".join(head), max(60, self.max_tokens // self.max_entries))
            self.entries = [f"[earlier] {folded}"] + self.entries[len(head):]
        if count_tokens(self.render()) > self.max_tokens:
            self.entries = [summarise_text(e, max(30, self.max_tokens // max(1, len(self.entries))))
                            for e in self.entries]
            while self.entries and count_tokens(self.render()) > self.max_tokens:
                self.entries = self.entries[1:]

    def render(self) -> str:
        return "\n".join(f"- {entry}" for entry in self.entries)

    def to_dict(self) -> Dict[str, Any]:
        return {"max_tokens": self.max_tokens, "max_entries": self.max_entries, "entries": self.entries}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RollingMemory":
        memory = cls(max_tokens=int(data.get("max_tokens", 450)),
                     max_entries=int(data.get("max_entries", 8)))
        memory.entries = list(data.get("entries") or [])
        return memory


def findings_block(findings: List[Finding], limit: int = 12) -> str:
    """Render findings compactly, most severe first, capped for context."""
    if not findings:
        return "none open"
    ordered = sorted(findings, key=lambda f: (-f.severity.weight, f.agent, f.category))[:limit]
    lines = []
    for finding in ordered:
        lines.append(
            f"- [{finding.severity.value}] {finding.agent}/{finding.category}"
            f"@{finding.section or '-'}: {finding.claim} -> {finding.action}"
        )
    if len(findings) > len(ordered):
        lines.append(f"- ... {len(findings) - len(ordered)} further finding(s) folded into change_log.md")
    return "\n".join(lines)


def draft_block(draft: Draft, max_tokens: int) -> str:
    """Compact rendering of the master draft for context inclusion."""
    if draft is None:
        return "no draft yet"
    pieces = []
    for section in draft.sections:
        body = section.body.strip()
        pieces.append(f"## {section.title}\n{body or '_(empty)_'}")
    text = f"# {draft.title}\n\n" + "\n\n".join(pieces)
    return truncate_to_tokens(text, max_tokens)


def working_draft_block(draft: Draft, peer_reports: List[AgentReport]) -> str:
    """Render current-iteration Writer proposals, not raw agent transcripts.

    This is an advisory view only: it never mutates or versions the master draft.
    All downstream models must see the prose they are checking/merging, including
    unchanged sections from the master when the Writer returns a delta.
    """
    sections = {section.id: {"id": section.id, "title": section.title,
                             "body": section.body} for section in draft.sections}
    for report in peer_reports:
        if report.agent_id != "writer":
            continue
        for item in report.payload.get("sections") or []:
            if not isinstance(item, dict):
                continue
            section_id, body = item.get("id"), item.get("body")
            if section_id in sections and isinstance(body, str) and body.strip():
                sections[section_id]["body"] = body
    return json.dumps({"title": draft.title, "sections": list(sections.values())},
                      ensure_ascii=False, separators=(",", ":"))


#: Writer remit mirrored from the Writer agent (kept literal to avoid a
#: circular import through the agents package).
_WRITER_REMIT = {"readability", "clarity", "completeness", "flow", "redundancy",
                 "tone", "consistency", "coverage", "balance", "narrative"}
_WRITER_FILL_RATIO = 0.8


def writer_delta_block(brief, draft: Draft, findings: List[Finding], max_tokens: int) -> str:
    """Delta rendering of the master draft for the Writer.

    The Writer may only change sections that are under target or named in an
    open finding inside its remit, so it only ever needs those sections in
    full. Everything it is forbidden to touch is reduced to a one-line
    attestation - which is what cuts the Writer's context packet from O(draft)
    to O(needed work) on every iteration after the first.
    """
    if draft is None:
        return "no draft yet"
    default_target = max(80, brief.target_words // max(1, len(draft.sections)))
    targeted = {f.section for f in findings
                if f.agent in ("reviewer", "tester", "architect")
                and f.category in _WRITER_REMIT and f.section}
    parts = []
    for section in draft.sections:
        target = section.word_target or default_target
        needs_work = (not section.body.strip()) or section.word_count < target * _WRITER_FILL_RATIO
        if needs_work or section.id in targeted:
            parts.append(f"## {section.title}\n{section.body.strip() or '_(empty)_'}")
        else:
            parts.append(f"## {section.title}\n(unchanged, {section.word_count} words "
                         "- do not resend this section in your output)")
    return truncate_to_tokens(f"# {draft.title}\n\n" + "\n\n".join(parts), max_tokens)


def outline_block(brief) -> str:
    lines = [f"title: {brief.title}", f"objective: {brief.objective}",
             f"audience: {brief.audience}", f"mode: {brief.mode.value}",
             f"tone: {brief.tone}", f"target_words: {brief.target_words}"]
    if brief.requires_citations:
        lines.append("citations: required")
    if brief.constraints:
        lines.append("constraints: " + "; ".join(brief.constraints))
    lines.append("outline:")
    for section in brief.sections:
        lines.append(f"  - {section.id} | {section.title} | target {section.word_target}w | {section.purpose}")
    return "\n".join(lines)


class MemoryStore:
    """Durable, on-disk memory. Nothing important lives only in RAM."""

    def __init__(self, root: str, task_id: str) -> None:
        self.root = os.path.join(root, task_id)
        self.task_id = task_id
        self.iterations_dir = os.path.join(self.root, "iterations")
        os.makedirs(self.iterations_dir, exist_ok=True)
        self.state_path = os.path.join(self.root, "state.json")
        self.rolling = RollingMemory()
        self.open_findings: List[Finding] = []
        self.history: List[Dict[str, Any]] = []
        self._load()

    # -- persistence ------------------------------------------------------
    def _load(self) -> None:
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        self.rolling = RollingMemory.from_dict(state.get("rolling", {}))
        self.open_findings = [Finding.from_dict(f) for f in (state.get("open_findings") or [])]
        self.history = list(state.get("history") or [])

    def save_state(self) -> None:
        state = {
            "task_id": self.task_id,
            "rolling": self.rolling.to_dict(),
            "open_findings": [f.to_dict() for f in self.open_findings],
            "history": self.history,
        }
        _atomic_write_json(self.state_path, state)

    # -- artefacts --------------------------------------------------------
    def iteration_dir(self, iteration: int) -> str:
        path = os.path.join(self.iterations_dir, f"iter-{iteration:02d}")
        os.makedirs(path, exist_ok=True)
        return path

    def write_text(self, iteration: int, name: str, content: str) -> str:
        path = os.path.join(self.iteration_dir(iteration), name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def write_json(self, iteration: int, name: str, payload: Dict[str, Any]) -> str:
        path = os.path.join(self.iteration_dir(iteration), name)
        _atomic_write_json(path, payload)
        return path

    def write_master_draft(self, draft: Draft) -> str:
        path = os.path.join(self.root, "master_draft.md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(draft.to_markdown())
        return path

    def record_iteration(self, record: Dict[str, Any]) -> None:
        self.history.append(record)
        self.save_state()

    # -- findings bookkeeping --------------------------------------------
    def resolve(self, resolved_keys: Iterable[str]) -> int:
        resolved = set(resolved_keys)
        before = len(self.open_findings)
        self.open_findings = [f for f in self.open_findings if f.key not in resolved]
        return before - len(self.open_findings)

    def absorb_findings(self, findings: List[Finding]) -> None:
        known = {f.key for f in self.open_findings}
        for finding in findings:
            if finding.key in known:
                continue
            self.open_findings.append(finding)
            known.add(finding.key)
        # Never let the open list itself become unbounded context.
        if len(self.open_findings) > 40:
            self.open_findings = sorted(self.open_findings,
                                        key=lambda f: (-f.severity.weight, f.iteration))[:40]

    def drop_notes(self) -> None:
        self.open_findings = [f for f in self.open_findings if f.severity != Severity.NOTE]

    # -- context packet ---------------------------------------------------
    def build_packet(self, *, agent_id: str, brief, draft: Draft, rules_digest: str,
                     budget: TokenBudget, iteration: int,
                     draft_required: bool,
                     peer_reports: Optional[List[AgentReport]] = None) -> AssembledContext:
        """Assemble the bounded context for one agent call."""
        agent_budget = budget.agent(agent_id)
        draft_key = "master_draft"
        if agent_id != "writer" and peer_reports:
            # Keep the actual prose once, rather than duplicating Writer output
            # alongside an obsolete master. Let assemble enforce the hard cap;
            # pre-truncating JSON would silently hide the final sections.
            draft_key = "working_draft"
            draft_text_fn = lambda: working_draft_block(draft, peer_reports)
        elif agent_id == "writer":
            # Delta packet: full text only for sections the Writer may change.
            draft_text_fn = lambda: writer_delta_block(brief, draft, self.open_findings,
                                                       max_tokens=agent_budget.max_input_tokens // 2)
        else:
            draft_text_fn = lambda: draft_block(draft, max_tokens=agent_budget.max_input_tokens // 2)
        header = (
            f"agent: {agent_id} | iteration: {iteration} | task: {brief.task_id}\n"
            f"context_hard_cap: {budget.context_hard_cap} tokens | "
            f"agent_input_cap: {agent_budget.max_input_tokens} tokens\n"
            "Rule: work only from the material below. Do not assume unseen history.\n"
            "reasoning_effort: low. Put the JSON answer in message content, "
            "not only in a thinking trace."
        )
        if draft_key == "working_draft":
            header += (
                "\nworking_draft is the master with this iteration's Writer revisions "
                "overlaid. Review/merge these bodies, not the old empty outline. "
                "It is advisory; only the Synthesizer may publish a new master."
            )
        blocks = [
            Block(1, "system_rules_digest", rules_digest, required=True),
            Block(2, "task_brief", outline_block(brief), required=True),
            Block(3, draft_key, draft_text_fn(),
                  required=draft_required),
            Block(4, "open_findings", findings_block(self.open_findings), required=False),
            Block(5, "rolling_history", self.rolling.render() or "no prior iterations",
                  required=False),
        ]
        return assemble(blocks, budget, header=header)


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write JSON atomically so a crash mid-write cannot corrupt run state."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory,
                                         delete=False, suffix=".tmp")
    try:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)
