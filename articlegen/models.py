"""Core data structures shared by the gate, agents, memory and orchestrator.

Every structure is JSON-serialisable: the orchestrator persists them to disk so a
run can be resumed or audited without re-reading full agent transcripts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional

TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,48}$")


def _text_value(value: Any, field_name: str, default: str = "") -> str:
    """Read a text field without turning JSON objects/lists into misleading prose."""
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value.strip()


def _integer_value(value: Any, field_name: str, default: int = 0) -> int:
    """Read an integer field without silently truncating floats or accepting booleans."""
    if value is None or value == "":
        return default
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{field_name} must be an integer")
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{field_name} must be an integer") from None


class Severity(str, Enum):
    """Ordered finding severity. `rank` drives both scoring and gate routing."""

    BLOCKER = "blocker"
    MAJOR = "major"
    MINOR = "minor"
    NOTE = "note"

    @property
    def weight(self) -> float:
        return {"blocker": 1.0, "major": 0.5, "minor": 0.2, "note": 0.05}[self.value]

    @classmethod
    def parse(cls, value: str, default: "Severity" = None) -> "Severity":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except (ValueError, AttributeError):
            return default if default is not None else cls.MINOR


class Mode(str, Enum):
    ARTICLE = "article"
    CODE = "code"

    @classmethod
    def parse(cls, value: str, default: "Mode" = None) -> "Mode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except (ValueError, AttributeError):
            return default if default is not None else cls.ARTICLE


class AgentStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"   # ran, but produced a partial/low-confidence result
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class Section:
    id: str
    title: str
    purpose: str = ""
    body: str = ""
    word_target: int = 0

    @property
    def word_count(self) -> int:
        return len(self.body.split())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Section":
        return cls(
            id=str(data.get("id", "")).strip(),
            title=str(data.get("title", "")).strip(),
            purpose=str(data.get("purpose", "")).strip(),
            body=str(data.get("body", "") or "").strip(),
            word_target=int(data.get("word_target", 0) or 0),
        )


@dataclass
class Brief:
    """Validated task input. Produced from a JSON/YAML-free dict by the gate."""

    task_id: str
    title: str
    objective: str
    audience: str
    mode: Mode = Mode.ARTICLE
    tone: str = "neutral, precise, professional"
    target_words: int = 1200
    sections: List[Section] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    requires_citations: bool = False
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        data["sections"] = [s.to_dict() for s in self.sections]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Brief":
        """Parse a JSON brief and reject malformed shapes with field-level errors.

        This is a schema boundary: a string such as ``"false"`` must not silently
        become ``True`` for ``requires_citations``, and a string in ``sections``
        must not be iterated character by character before the gate can run.
        """
        if not isinstance(data, dict):
            raise ValueError("brief must be a JSON object")

        raw_sections = data.get("sections")
        if raw_sections is None:
            raw_sections = []
        if not isinstance(raw_sections, list):
            raise ValueError("sections must be an array of objects")

        sections: List[Section] = []
        for index, raw_section in enumerate(raw_sections):
            prefix = f"sections[{index}]"
            if not isinstance(raw_section, dict):
                raise ValueError(f"{prefix} must be an object")
            sections.append(Section(
                id=_text_value(raw_section.get("id", ""), f"{prefix}.id"),
                title=_text_value(raw_section.get("title", ""), f"{prefix}.title"),
                purpose=_text_value(raw_section.get("purpose", ""), f"{prefix}.purpose"),
                body=_text_value(raw_section.get("body", ""), f"{prefix}.body"),
                word_target=_integer_value(raw_section.get("word_target", 0),
                                           f"{prefix}.word_target"),
            ))

        raw_constraints = data.get("constraints")
        if raw_constraints is None:
            raw_constraints = []
        if not isinstance(raw_constraints, list):
            raise ValueError("constraints must be an array of strings")
        constraints = [
            _text_value(item, f"constraints[{index}]")
            for index, item in enumerate(raw_constraints)
        ]

        raw_mode = data.get("mode", "article")
        if raw_mode is None or raw_mode == "":
            raw_mode = "article"
        if isinstance(raw_mode, Mode):
            mode = raw_mode
        elif isinstance(raw_mode, str):
            try:
                mode = Mode(raw_mode.strip().lower())
            except ValueError:
                raise ValueError("mode must be either 'article' or 'code'") from None
        else:
            raise ValueError("mode must be either 'article' or 'code'")

        requires_citations = data.get("requires_citations", False)
        if requires_citations is None:
            requires_citations = False
        if not isinstance(requires_citations, bool):
            raise ValueError("requires_citations must be a boolean")

        tone = _text_value(data.get("tone", ""), "tone") or "neutral, precise, professional"
        language = _text_value(data.get("language", "en"), "language") or "en"

        return cls(
            task_id=_text_value(data.get("task_id", ""), "task_id"),
            title=_text_value(data.get("title", ""), "title"),
            objective=_text_value(data.get("objective", ""), "objective"),
            audience=_text_value(data.get("audience", ""), "audience"),
            mode=mode,
            tone=tone,
            target_words=_integer_value(data.get("target_words", 0), "target_words"),
            sections=sections,
            constraints=constraints,
            requires_citations=requires_citations,
            language=language,
        )

    @property
    def section_ids(self) -> List[str]:
        return [s.id for s in self.sections]


@dataclass
class Finding:
    """A single actionable observation raised by a sub-agent."""

    agent: str
    severity: Severity
    category: str
    section: str
    claim: str
    action: str
    evidence: str = ""
    auto_fixable: bool = False
    fix: Optional[Dict[str, Any]] = None
    iteration: int = 0

    @property
    def key(self) -> str:
        return f"{self.agent}:{self.category}:{self.section}:{self.claim[:60]}"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Finding":
        return cls(
            agent=str(data.get("agent", "")),
            severity=Severity.parse(data.get("severity", "minor")),
            category=str(data.get("category", "general")),
            section=str(data.get("section", "")),
            claim=str(data.get("claim", "")),
            action=str(data.get("action", "")),
            evidence=str(data.get("evidence", "")),
            auto_fixable=bool(data.get("auto_fixable", False)),
            fix=data.get("fix"),
            iteration=int(data.get("iteration", 0) or 0),
        )


@dataclass
class AgentReport:
    agent_id: str
    role: str
    iteration: int
    status: AgentStatus = AgentStatus.OK
    findings: List[Finding] = field(default_factory=list)
    summary: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    engine: str = "heuristic"
    notes: List[str] = field(default_factory=list)

    def findings_by_severity(self, severity: Severity) -> List[Finding]:
        return [f for f in self.findings if f.severity == severity]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "iteration": self.iteration,
            "status": self.status.value,
            "engine": self.engine,
            "summary": self.summary,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "notes": self.notes,
            "findings": [f.to_dict() for f in self.findings],
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentReport":
        return cls(
            agent_id=data["agent_id"],
            role=data.get("role", ""),
            iteration=int(data.get("iteration", 0)),
            status=AgentStatus(data.get("status", "ok")),
            findings=[Finding.from_dict(f) for f in (data.get("findings") or [])],
            summary=data.get("summary", ""),
            payload=data.get("payload") or {},
            tokens_in=int(data.get("tokens_in", 0)),
            tokens_out=int(data.get("tokens_out", 0)),
            engine=data.get("engine", "heuristic"),
            notes=list(data.get("notes") or []),
        )


@dataclass
class Draft:
    """The master draft. Only the Synthesizer may produce a new version."""

    task_id: str
    version: int
    title: str
    sections: List[Section] = field(default_factory=list)
    engine: str = "heuristic"

    @property
    def word_count(self) -> int:
        return sum(s.word_count for s in self.sections)

    def section_map(self) -> Dict[str, Section]:
        return {s.id: s for s in self.sections}

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.body.strip() or "_(empty)_")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "version": self.version,
            "title": self.title,
            "engine": self.engine,
            "sections": [s.to_dict() for s in self.sections],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Draft":
        return cls(
            task_id=data["task_id"],
            version=int(data.get("version", 0)),
            title=data.get("title", ""),
            engine=data.get("engine", "heuristic"),
            sections=[Section.from_dict(s) for s in (data.get("sections") or [])],
        )

    @classmethod
    def from_brief(cls, brief: Brief) -> "Draft":
        """Seed the draft with the outline and preserve any supplied section prose.

        Empty or underfilled sections are left for the Writer; full seed sections
        are reviewed as-is rather than being silently discarded.
        """
        return cls(
            task_id=brief.task_id,
            version=0,
            title=brief.title,
            engine="seed",
            sections=[
                Section(id=s.id, title=s.title, purpose=s.purpose, body=s.body,
                        word_target=s.word_target)
                for s in brief.sections
            ],
        )


@dataclass
class GateResult:
    """Decision-gate verdict. `routed_back=True` blocks the pipeline."""

    ok: bool
    stage: str
    issues: List[Finding] = field(default_factory=list)
    checks_run: List[str] = field(default_factory=list)
    routed_back: bool = False
    reason: str = ""

    @property
    def blockers(self) -> List[Finding]:
        return [i for i in self.issues if i.severity == Severity.BLOCKER]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "stage": self.stage,
            "routed_back": self.routed_back,
            "reason": self.reason,
            "checks_run": self.checks_run,
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class IterationRecord:
    iteration: int
    score: float
    previous_score: Optional[float]
    delta: float
    reports: List[AgentReport] = field(default_factory=list)
    evicted: List[str] = field(default_factory=list)
    tokens_total: int = 0
    decision: str = "continue"
    applied: int = 0
    deferred: int = 0

    @property
    def finding_count(self) -> int:
        return sum(len(r.findings) for r in self.reports)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "score": round(self.score, 4),
            "previous_score": None if self.previous_score is None else round(self.previous_score, 4),
            "delta": round(self.delta, 4),
            "decision": self.decision,
            "tokens_total": self.tokens_total,
            "evicted": self.evicted,
            "applied": self.applied,
            "deferred": self.deferred,
            "agents": [
                {
                    "agent_id": r.agent_id,
                    "status": r.status.value,
                    "engine": r.engine,
                    "findings": len(r.findings),
                    "tokens_in": r.tokens_in,
                    "tokens_out": r.tokens_out,
                }
                for r in self.reports
            ],
        }


@dataclass
class RunResult:
    task_id: str
    status: str
    iterations: List[IterationRecord] = field(default_factory=list)
    gate: Optional[GateResult] = None
    final_score: float = 0.0
    final_draft: Optional[Draft] = None
    workspace: str = ""
    tokens_total: int = 0
    stop_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "final_score": round(self.final_score, 4),
            "tokens_total": self.tokens_total,
            "workspace": self.workspace,
            "gate": self.gate.to_dict() if self.gate else None,
            "iterations": [i.to_dict() for i in self.iterations],
            "draft": self.final_draft.to_dict() if self.final_draft else None,
        }
