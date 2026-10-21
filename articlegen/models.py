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


class ArticleType(str, Enum):
    """What kind of article is being produced.

    The type selects the research-plan shape, the required sections and the
    validation rules. A general/educational article is not forced into the
    empirical-paper checklist; a computational paper must supply executed
    experiments, a correctness argument and complexity analysis.
    """

    GENERAL = "general"
    EDUCATIONAL = "educational"
    CONCEPTUAL = "conceptual"
    EMPIRICAL = "empirical"
    COMPUTATIONAL = "computational"
    THEORETICAL = "theoretical"
    SYSTEMATIC_REVIEW = "systematic_review"
    NARRATIVE_REVIEW = "narrative_review"
    METHODS = "methods"

    @classmethod
    def parse(cls, value: Any, default: "ArticleType" = None) -> "ArticleType":
        fallback = default if default is not None else cls.GENERAL
        if isinstance(value, cls):
            return value
        if value is None or value == "":
            return fallback
        token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "review": "narrative_review",
            "literature_review": "narrative_review",
            "systematic": "systematic_review",
            "meta_analysis": "systematic_review",
            "algorithmic": "computational",
            "computer_science": "computational",
            "theory": "theoretical",
            "mathematics": "theoretical",
            "software": "methods",
            "method": "methods",
            "tutorial": "educational",
            "conceptual_paper": "conceptual",
            "experimental": "empirical",
        }
        token = aliases.get(token, token)
        try:
            return cls(token)
        except ValueError:
            return fallback

    @property
    def is_research(self) -> bool:
        """True when the article makes a research contribution requiring evidence."""
        return self not in (ArticleType.GENERAL, ArticleType.EDUCATIONAL)

    @property
    def wants_sources(self) -> bool:
        return self.is_research

    @property
    def wants_experiments(self) -> bool:
        return self in (ArticleType.EMPIRICAL, ArticleType.COMPUTATIONAL,
                        ArticleType.METHODS, ArticleType.SYSTEMATIC_REVIEW)

    @property
    def wants_correctness_argument(self) -> bool:
        return self in (ArticleType.COMPUTATIONAL, ArticleType.THEORETICAL)


class CitationStyle(str, Enum):
    """Supported bibliography formats. Vendors add their own later."""

    APA = "apa"
    IEEE = "ieee"
    VANCOUVER = "vancouver"
    PLAIN = "plain"

    @classmethod
    def parse(cls, value: Any, default: "CitationStyle" = None) -> "CitationStyle":
        fallback = default if default is not None else cls.APA
        if isinstance(value, cls):
            return value
        if value is None or value == "":
            return fallback
        token = str(value).strip().lower()
        if token.startswith("apa") or "american psychological" in token:
            return cls.APA
        if token.startswith("ieee") or "electrical" in token:
            return cls.IEEE
        if token.startswith("vancouver") or token.startswith("icmje"):
            return cls.VANCOUVER
        if token in ("plain", "numeric", "custom", "journal"):
            return cls.PLAIN
        return fallback


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
    # -- research configuration (all optional; defaults keep general articles) --
    article_type: ArticleType = ArticleType.GENERAL
    venue: str = ""
    citation_style: CitationStyle = CitationStyle.APA
    research_question: str = ""
    contribution_type: str = ""
    requires_experiments: bool = False
    execute_experiments: bool = True
    source_retrieval: str = "auto"
    max_sources: int = 20
    review_iterations: int = 1
    experiment_timeout: int = 30
    novelty_claim: bool = False
    #: User-supplied source records. They are stored verbatim and marked
    #: user-provided; they are never silently treated as index-verified.
    sources: List[Dict[str, Any]] = field(default_factory=list)
    #: Benchmark descriptors supplied by the user (hardware, repetitions, ...).
    benchmark: Dict[str, Any] = field(default_factory=dict)

    @property
    def research_enabled(self) -> bool:
        """Research mode is explicit, never inferred from prose style."""
        return bool(self.article_type.is_research or self.research_question
                    or self.contribution_type or self.requires_experiments
                    or self.citation_style != CitationStyle.APA)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["mode"] = self.mode.value
        data["article_type"] = self.article_type.value
        data["citation_style"] = self.citation_style.value
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

        raw_sources = data.get("sources")
        if raw_sources is None:
            raw_sources = []
        if not isinstance(raw_sources, list):
            raise ValueError("sources must be an array of source records")
        sources: List[Dict[str, Any]] = []
        for index, item in enumerate(raw_sources):
            if not isinstance(item, dict):
                raise ValueError(f"sources[{index}] must be an object")
            sources.append(dict(item))

        raw_benchmark = data.get("benchmark")
        if raw_benchmark is None:
            raw_benchmark = {}
        if not isinstance(raw_benchmark, dict):
            raise ValueError("benchmark must be an object")

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

        article_type = ArticleType.parse(data.get("article_type", ""))
        citation_style = CitationStyle.parse(data.get("citation_style", ""))

        requires_experiments = data.get("requires_experiments", False)
        if requires_experiments is None:
            requires_experiments = False
        if not isinstance(requires_experiments, bool):
            raise ValueError("requires_experiments must be a boolean")

        execute_experiments = data.get("execute_experiments", True)
        if execute_experiments is None:
            execute_experiments = True
        if not isinstance(execute_experiments, bool):
            raise ValueError("execute_experiments must be a boolean")

        novelty_claim = data.get("novelty_claim", False)
        if novelty_claim is None:
            novelty_claim = False
        if not isinstance(novelty_claim, bool):
            raise ValueError("novelty_claim must be a boolean")

        source_retrieval = _text_value(data.get("source_retrieval", ""),
                                       "source_retrieval").lower() or "auto"

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
            article_type=article_type,
            venue=_text_value(data.get("venue", ""), "venue"),
            citation_style=citation_style,
            research_question=_text_value(data.get("research_question", ""),
                                          "research_question"),
            contribution_type=_text_value(data.get("contribution_type", ""),
                                         "contribution_type"),
            requires_experiments=requires_experiments,
            execute_experiments=execute_experiments,
            source_retrieval=source_retrieval,
            max_sources=_integer_value(data.get("max_sources", 20), "max_sources", 20),
            review_iterations=_integer_value(data.get("review_iterations", 1),
                                             "review_iterations", 1),
            experiment_timeout=_integer_value(data.get("experiment_timeout", 30),
                                              "experiment_timeout", 30),
            novelty_claim=novelty_claim,
            sources=sources,
            benchmark=dict(raw_benchmark),
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
    #: Structured research artefacts (plan, sources, experiments, validation,
    #: review, quality gate). Empty for a general non-research article.
    research: Dict[str, Any] = field(default_factory=dict)
    #: READY_FOR_REVIEW / NEEDS_REVISION / INCOMPLETE_OR_UNVERIFIED, or "" when
    #: the research pipeline did not run.
    quality_status: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "final_score": round(self.final_score, 4),
            "tokens_total": self.tokens_total,
            "workspace": self.workspace,
            "quality_status": self.quality_status,
            "research": self.research,
            "gate": self.gate.to_dict() if self.gate else None,
            "iterations": [i.to_dict() for i in self.iterations],
            "draft": self.final_draft.to_dict() if self.final_draft else None,
        }
