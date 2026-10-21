"""Typed research records.

Everything in the research layer is a record: plans, sources, experiments, runs,
results, claims, benchmarks, review findings, validation checks and the gate verdict.
The types live here so that the planner, the executor, the validators and the report
writer all describe the same object with the same fields, and so that a record can be
serialised next to the draft it supports.

Two rules are encoded in these types rather than in prose:

* a record carries its own *provenance* (retrieved via, executed by, computed from), so
  a claim can always be traced back to something that exists;
* "unknown" is a first-class value. Nothing here has a default that invents content.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple

from ..models import ArticleType, CitationStyle

__all__ = [
    "ArticleTypeProfile",
    "BenchmarkRecord",
    "CheckStatus",
    "ClaimRecord",
    "ContributionType",
    "EvidenceKind",
    "EvidenceLedger",
    "ExperimentRun",
    "ExperimentSpec",
    "ExperimentStatus",
    "GateStatus",
    "Measurement",
    "QualityGateResult",
    "ResearchPlan",
    "ResultRecord",
    "ReviewIssue",
    "ReviewReport",
    "ReviewSeverity",
    "RevisionSummary",
    "SourceRecord",
    "TestOutcome",
    "ValidationCheck",
    "ValidationReport",
    "WorkflowStates",
    "classify_section_role",
    "normalise_doi",
    "profile_for",
    "utc_now",
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _coerce(value: Any) -> Any:
    """Make a value JSON-serialisable without dropping information."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _coerce(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [part.strip() for part in re.split(r"(?:\n|\s*;\s*)", text) if part.strip()]
    return [value]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value if item)
    return str(value).strip()


DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.I)
ARXIV_RE = re.compile(r"\barxiv:\s*(\d{4}\.\d{4,5})(v\d+)?\b", re.I)


def normalise_doi(value: Any) -> str:
    """Return a bare DOI, or ``""`` when the string is not a DOI.

    This never invents one: a URL prefix is stripped, anything that does not match the
    DOI grammar is rejected, and the canonical ``10.xxxx/...`` form is returned.
    """
    text = _as_text(value)
    if not text:
        return ""
    text = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", text.strip(), flags=re.I)
    text = re.sub(r"^doi:\s*", "", text, flags=re.I).strip()
    match = DOI_RE.search(text)
    if not match:
        return ""
    doi = match.group(0).rstrip(".,;)")
    return doi.lower()


def _year(value: Any) -> Optional[int]:
    text = _as_text(value)
    match = re.search(r"(1[5-9]\d{2}|20\d{2}|21\d{2})", text)
    return int(match.group(1)) if match else None


def _fingerprint(*parts: Any) -> str:
    digest = hashlib.sha1("|".join(_as_text(p).lower() for p in parts).encode("utf-8"))
    return digest.hexdigest()[:12]


# ---------------------------------------------------------------------------
# enums
# ---------------------------------------------------------------------------


class ReviewSeverity(str, Enum):
    """Severity of a finding, of a failed check, or of the thing it measures.

    ``blocking`` is not a synonym for "serious": it decides whether the *gate* may
    still call a manuscript ready. Only blockers and majors block; a moderate or minor
    finding is reported and can be deferred by a human reader.
    """

    CRITICAL = "critical"
    MAJOR = "major"
    MODERATE = "moderate"
    MINOR = "minor"

    @classmethod
    def parse(cls, value: Any) -> "ReviewSeverity":
        if isinstance(value, ReviewSeverity):
            return value
        text = _as_text(value).lower().strip()
        aliases = {
            "blocker": cls.CRITICAL, "fatal": cls.CRITICAL, "high": cls.MAJOR,
            "medium": cls.MODERATE, "low": cls.MINOR, "note": cls.MINOR,
            "info": cls.MINOR,
        }
        if text in aliases:
            return aliases[text]
        for severity in cls:
            if text == severity.value:
                return severity
        return cls.MODERATE

    @property
    def weight(self) -> int:
        """Higher is more severe, so callers can sort with ``-severity.weight``."""
        return {ReviewSeverity.CRITICAL: 3, ReviewSeverity.MAJOR: 2,
                ReviewSeverity.MODERATE: 1, ReviewSeverity.MINOR: 0}[self]

    @property
    def blocking(self) -> bool:
        return self in (ReviewSeverity.CRITICAL, ReviewSeverity.MAJOR)

    def at_least(self, other: "ReviewSeverity") -> bool:
        return self.weight >= other.weight


class EvidenceKind(str, Enum):
    """How a sentence is substantiated."""

    ESTABLISHED_FACT = "established_fact"      # supported by a verified source
    STUDY_RESULT = "study_result"              # a result this study recorded
    EXTERNAL_FINDING = "external_finding"      # a result attributed to a source
    HYPOTHESIS = "hypothesis"                  # explicitly proposed, not shown
    INTERPRETATION = "interpretation"          # reasoning over recorded results
    ASSUMPTION = "assumption"                  # explicitly assumed
    FUTURE_WORK = "future_work"                # explicitly not done
    UNVERIFIED = "unverified"                  # no support identified


#: Kinds that require evidence to exist for the claim to count as supported.
EVIDENCE_REQUIRED_KINDS = (
    EvidenceKind.ESTABLISHED_FACT,
    EvidenceKind.STUDY_RESULT,
    EvidenceKind.EXTERNAL_FINDING,
)


class ContributionType(str, Enum):
    NOVEL_THEORY = "novel_theory"
    NOVEL_EMPIRICAL = "novel_empirical"
    NOVEL_METHOD = "novel_method"
    NOVEL_ALGORITHM = "novel_algorithm"
    SYNTHESIS = "synthesis"
    REPLICATION = "replication"
    ROUTINE_IMPLEMENTATION = "routine_implementation"
    EDUCATIONAL = "educational"
    CONCEPTUAL = "conceptual"
    EMPIRICAL_STUDY = "empirical_study"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ")

    @property
    def is_novel(self) -> bool:
        return self in (ContributionType.NOVEL_THEORY, ContributionType.NOVEL_EMPIRICAL,
                        ContributionType.NOVEL_METHOD, ContributionType.NOVEL_ALGORITHM)

    @property
    def claims_novelty(self) -> bool:
        return self in (ContributionType.NOVEL_THEORY, ContributionType.NOVEL_EMPIRICAL,
                        ContributionType.NOVEL_METHOD,
                        ContributionType.NOVEL_ALGORITHM)

    @classmethod
    def parse(cls, value: Any) -> "ContributionType":
        if isinstance(value, ContributionType):
            return value
        text = _as_text(value).lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "theory": cls.NOVEL_THEORY, "novel": cls.NOVEL_THEORY,
            "empirical": cls.EMPIRICAL_STUDY, "experiment": cls.NOVEL_EMPIRICAL,
            "method": cls.NOVEL_METHOD, "algorithm": cls.NOVEL_ALGORITHM,
            "survey": cls.SYNTHESIS, "review": cls.SYNTHESIS,
            "implementation": cls.ROUTINE_IMPLEMENTATION,
            "tutorial": cls.EDUCATIONAL, "conceptual": cls.CONCEPTUAL,
        }
        if text in aliases:
            return aliases[text]
        for item in cls:
            if text == item.value:
                return item
        return cls.CONCEPTUAL


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNVERIFIED = "unverified"
    NOT_APPLICABLE = "not_applicable"

    @property
    def ok(self) -> bool:
        return self in (CheckStatus.PASS, CheckStatus.NOT_APPLICABLE)


class ExperimentStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    NOT_RUN = "not_run"

    @property
    def executed(self) -> bool:
        """The code actually ran (passing or failing is a separate question)."""
        return self in (ExperimentStatus.OK, ExperimentStatus.FAILED)

    @property
    def ran(self) -> bool:
        return self is ExperimentStatus.OK

    @property
    def succeeded(self) -> bool:
        return self is ExperimentStatus.OK

    @property
    def successful(self) -> bool:
        return self is ExperimentStatus.OK

    @classmethod
    def parse(cls, value: Any) -> "ExperimentStatus":
        if isinstance(value, ExperimentStatus):
            return value
        text = _as_text(value).lower()
        for item in cls:
            if text == item.value:
                return item
        return cls.ERROR


# ---------------------------------------------------------------------------
# article-type profiles and section roles
# ---------------------------------------------------------------------------

#: Canonical section roles. The planner fills required ones, the validators map the
#: draft's headings onto them and the reviewer is told which roles to expect.
SECTION_ROLES: Tuple[str, ...] = (
    "introduction", "background", "methods", "algorithm", "results",
    "complexity", "discussion", "limitations", "conclusion", "references", "other",
)

_ROLE_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("references", ("reference", "bibliograph", "works cited")),
    ("limitations", ("limitation", "threat", "validity", "caveat", "constraint")),
    ("conclusion", ("conclusion", "concluding", "summary of findings", "closing")),
    ("complexity", ("complexity", "asymptotic", "cost analysis")),
    ("algorithm", ("algorithm", "implementation", "pseudocode", "listing", "code")),
    ("methods", ("method", "methodology", "experimental design", "procedure",
                 "materials", "setup", "protocol", "approach")),
    ("results", ("result", "finding", "evaluation", "measurement", "benchmark",
                 "experiment result", "outcome")),
    ("background", ("background", "related work", "literature", "state of the art",
                    "prior work", "survey")),
    ("introduction", ("introduction", "motivation", "overview", "problem",
                      "research question")),
    ("discussion", ("discussion", "interpretation", "implication", "future work",
                    "analysis")),
)


def classify_section_role(section_id: str = "", title: str = "") -> str:
    """Map a heading onto a canonical role.

    The heading is authoritative (``"Related Work"`` under id ``s3`` is a background
    section); the id is only used when the title says nothing.
    """
    for text in (_as_text(title).lower(), _as_text(section_id).lower().replace("_", " ")):
        if not text:
            continue
        for role, keywords in _ROLE_KEYWORDS:
            if any(keyword in text for keyword in keywords):
                return role
    return "other"


@dataclass(frozen=True)
class ArticleTypeProfile:
    """What a given article type must contain to be a valid instance of that type.

    A proof-based theory paper needs a correctness argument and no experiments; an
    empirical study needs measurements and statistics; an educational article needs
    worked examples but no novelty claim. Encoding this here keeps the validators from
    applying one generic checklist to every article.
    """

    article_type: ArticleType
    label: str
    description: str
    required_roles: Tuple[str, ...] = ("introduction",)
    recommended_roles: Tuple[str, ...] = ()
    requires_experiments: bool = False
    requires_correctness_argument: bool = False
    requires_proof_rigour: bool = False
    requires_statistics: bool = False
    requires_source_citations: bool = True
    allows_novelty_claims: bool = True
    requires_benchmark_metadata: bool = False
    requires_search_strategy: bool = False
    requires_complexity_analysis: bool = False
    min_words: int = 600
    #: True when the type makes a research contribution requiring evidence. A
    #: general or educational article is not forced through the research checklist.
    research: bool = True

    @property
    def requires_sources(self) -> bool:
        return self.requires_source_citations

    def missing_roles(self, roles: Iterable[str]) -> List[str]:
        present = set(roles)
        return [role for role in self.required_roles if role not in present]


def _profile(article_type: ArticleType, label: str, description: str, **kwargs: Any
             ) -> ArticleTypeProfile:
    return ArticleTypeProfile(article_type=article_type, label=label,
                              description=description, **kwargs)


_ARTICLE_PROFILES: Dict[ArticleType, ArticleTypeProfile] = {
    ArticleType.GENERAL: _profile(
        ArticleType.GENERAL, "General article", "Explanatory prose for a broad audience.",
        required_roles=("introduction", "conclusion"),
        recommended_roles=("background", "discussion"),
        requires_source_citations=True, min_words=400, research=False),
    ArticleType.EDUCATIONAL: _profile(
        ArticleType.EDUCATIONAL, "Educational article",
        "Pedagogical explanation with worked examples.",
        required_roles=("introduction", "methods", "conclusion"),
        recommended_roles=("discussion",),
        requires_experiments=False, allows_novelty_claims=False, research=False),
    ArticleType.CONCEPTUAL: _profile(
        ArticleType.CONCEPTUAL, "Conceptual article",
        "Argument-led piece that develops ideas rather than measurements.",
        required_roles=("introduction", "background", "discussion", "conclusion"),
        requires_correctness_argument=False, requires_experiments=False),
    ArticleType.EMPIRICAL: _profile(
        ArticleType.EMPIRICAL, "Empirical study",
        "Measured study with statistics and reported variability.",
        required_roles=("introduction", "methods", "results", "discussion",
                        "limitations", "conclusion"),
        requires_experiments=True, requires_correctness_argument=True,
        requires_statistics=True, requires_benchmark_metadata=True,
        requires_search_strategy=True, requires_complexity_analysis=True),
    ArticleType.COMPUTATIONAL: _profile(
        ArticleType.COMPUTATIONAL, "Computational article",
        "Executable implementation with a recorded run and complexity analysis.",
        required_roles=("introduction", "methods", "algorithm", "results",
                        "discussion", "limitations", "conclusion"),
        requires_experiments=True, requires_correctness_argument=True,
        requires_benchmark_metadata=True, requires_complexity_analysis=True),
    ArticleType.THEORETICAL: _profile(
        ArticleType.THEORETICAL, "Theory article",
        "Definitions, theorems and proofs; no experiment is expected.",
        required_roles=("introduction", "background", "methods", "discussion",
                        "conclusion"),
        recommended_roles=("limitations",),
        requires_experiments=False, requires_correctness_argument=True,
        requires_proof_rigour=True),
    ArticleType.SYSTEMATIC_REVIEW: _profile(
        ArticleType.SYSTEMATIC_REVIEW, "Systematic review",
        "Protocol-led, reproducible synthesis of the literature.",
        required_roles=("introduction", "background", "methods", "results",
                        "discussion", "limitations", "conclusion"),
        requires_experiments=False, requires_correctness_argument=True,
        requires_source_citations=True, requires_search_strategy=True),
    ArticleType.NARRATIVE_REVIEW: _profile(
        ArticleType.NARRATIVE_REVIEW, "Narrative review",
        "Argument-led literature review.",
        required_roles=("introduction", "background", "discussion", "conclusion"),
        recommended_roles=("limitations",),
        requires_source_citations=True),
    ArticleType.METHODS: _profile(
        ArticleType.METHODS, "Methods article",
        "A scoped technical note describing a procedure or tool.",
        required_roles=("introduction", "methods", "conclusion"),
        recommended_roles=("limitations", "results"),
        requires_correctness_argument=True),
}


def profile_for(article_type: Any) -> ArticleTypeProfile:
    """Profile for an article type, tolerant of strings and unknown values."""
    parsed = article_type if isinstance(article_type, ArticleType) else None
    if parsed is None:
        text = _as_text(article_type).lower().replace(" ", "_").replace("-", "_")
        for candidate in ArticleType:
            if text == candidate.value or text == candidate.name.lower():
                parsed = candidate
                break
    return _ARTICLE_PROFILES.get(parsed or ArticleType.GENERAL,
                                 _ARTICLE_PROFILES[ArticleType.GENERAL])


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

VERIFICATION_STATES: Tuple[str, ...] = (
    "verified-index",      # matched against a scholarly index by this run
    "user-provided",       # supplied by the author; metadata is theirs, not checked
    "unverified",          # could not be checked
    "metadata-incomplete",  # missing the fields a citation needs
)

SOURCE_TYPES: Tuple[str, ...] = (
    "journal", "conference", "book", "preprint", "report", "dataset",
    "documentation", "web", "other",
)


@dataclass
class SourceRecord:
    """One citable record. ``verification`` is never set optimistically."""

    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    doi: str = ""
    url: str = ""
    abstract: str = ""
    source_type: str = "journal"
    publisher: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    isbn: str = ""
    arxiv_id: str = ""
    retrieved_via: str = ""
    verification: str = "unverified"
    verification_note: str = ""
    match_score: Optional[float] = None
    retrieved_from_query: str = ""
    verified_at: str = ""
    retrieved_at: str = field(default_factory=utc_now)
    source_id: str = ""
    notes: List[str] = field(default_factory=list)
    #: Set by the evidence ledger when this record substantiates a claim.
    supports: List[str] = field(default_factory=list)
    supported_claims: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.title = _as_text(self.title)
        self.authors = [_as_text(author) for author in _as_list(self.authors) if _as_text(author)]
        self.year = _year(self.year) if not isinstance(self.year, int) else self.year
        self.venue = _as_text(self.venue)
        self.doi = normalise_doi(self.doi)
        self.url = _as_text(self.url)
        self.publisher = _as_text(self.publisher)
        self.isbn = _as_text(self.isbn)
        arxiv = ARXIV_RE.search(_as_text(self.arxiv_id) or self.url or self.doi)
        if arxiv:
            self.arxiv_id = arxiv.group(1)
        if not self.source_id:
            self.source_id = "src-" + _fingerprint(self.title, self.authors[:1], self.year,
                                                   self.doi)
        if not self.verification:
            self.verification = "unverified"

    # -- quality --------------------------------------------------------
    def metadata_issues(self) -> List[str]:
        issues: List[str] = []
        if not self.title or len(self.title) < 4:
            issues.append("missing or unusable title")
        if not self.authors:
            issues.append("missing authors")
        if not self.year:
            issues.append("missing year")
        if not self.venue and self.source_type not in ("dataset", "documentation", "web"):
            issues.append("missing venue")
        if not (self.doi or self.url or self.arxiv_id or self.isbn):
            issues.append("missing DOI or stable URL")
        if self.year and self.year > datetime.now(timezone.utc).year:
            issues.append(f"year {self.year} is in the future")
        return issues

    @property
    def metadata_complete(self) -> bool:
        return not self.metadata_issues()

    @property
    def is_verified(self) -> bool:
        return self.verification == "verified-index"

    @property
    def is_usable(self) -> bool:
        """A record may only enter a bibliography if its metadata is complete.

        An unverified record can still be cited if the author supplied it - the gate
        then reports the manuscript as not fully verified. What is forbidden is citing a
        record whose metadata cannot form a real reference.
        """
        return self.metadata_complete

    @property
    def identifier(self) -> str:
        if self.doi:
            return f"doi:{self.doi}"
        if self.arxiv_id:
            return f"arXiv:{self.arxiv_id}"
        if self.isbn:
            return f"isbn:{self.isbn}"
        return self.url

    @property
    def citation_key(self) -> str:
        """Author-year key, e.g. ``erdos1981``; stable for sorting and cross-linking."""
        surname = ""
        if self.authors:
            surname = re.sub(r"[^a-z]", "", self.authors[0].split()[-1].lower())
        return f"{surname}{self.year or ''}" or self.source_id

    def short(self) -> str:
        authors = ", ".join(self.authors[:3]) + (" et al." if len(self.authors) > 3 else "")
        return f"{authors} ({self.year or 'n.d.'}) {self.title}"

    def mark_verified(self, provider: str, note: str = "") -> None:
        self.verification = "verified-index"
        self.retrieved_via = provider or self.retrieved_via
        self.verification_note = note
        self.verified_at = utc_now()

    def to_dict(self) -> Dict[str, Any]:
        return {key: _coerce(value) for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceRecord":
        known = {field_name for field_name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {key: value for key, value in (data or {}).items() if key in known}
        return cls(**payload)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

#: The ten elements a research plan must state before drafting may proceed.
PLAN_ELEMENTS: Tuple[str, ...] = (
    "problem", "research_question", "background", "gap", "contribution",
    "methodology", "evaluation_criteria", "limitations", "failure_modes", "outline",
)

PLAN_ELEMENT_LABELS: Dict[str, str] = {
    "problem": "the concrete problem",
    "research_question": "a research question",
    "background": "relevant background",
    "gap": "the gap with evidence from the literature",
    "contribution": "the expected contribution",
    "methodology": "a justified methodology",
    "evaluation_criteria": "success/evaluation criteria",
    "limitations": "limitations",
    "failure_modes": "failure modes",
    "outline": "a section outline matching the target article type",
}


@dataclass
class ResearchPlan:
    """The pre-drafting contract. A plan is a record, not a promise."""

    plan_id: str = ""
    task_id: str = ""
    article_type: ArticleType = ArticleType.GENERAL
    title: str = ""
    problem: str = ""
    research_question: str = ""
    background: str = ""
    gap: str = ""
    gap_evidence: List[str] = field(default_factory=list)
    contribution: str = ""
    contribution_type: ContributionType = ContributionType.CONCEPTUAL
    novelty_claims: List[str] = field(default_factory=list)
    methodology: str = ""
    methodology_justification: str = ""
    required_materials: List[str] = field(default_factory=list)
    required_experiments: List[str] = field(default_factory=list)
    evaluation_criteria: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    outline: List[Dict[str, Any]] = field(default_factory=list)
    source_queries: List[str] = field(default_factory=list)
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    origin: str = "deterministic"
    decision: str = "accept"
    decision_reasons: List[str] = field(default_factory=list)
    needs_user_direction: bool = False
    validated: bool = False
    validation_issues: List[Dict[str, Any]] = field(default_factory=list)
    #: Deterministic checks produced by ``validate_plan``; empty until it has run.
    issues: List[Any] = field(default_factory=list)
    #: The planner payload exactly as received, kept for audit rather than parsing.
    raw: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not isinstance(self.article_type, ArticleType):
            self.article_type = profile_for(self.article_type).article_type
        if not isinstance(self.contribution_type, ContributionType):
            self.contribution_type = ContributionType.parse(self.contribution_type)
        for name in ("gap_evidence", "novelty_claims", "required_materials",
                     "required_experiments", "evaluation_criteria", "limitations",
                     "failure_modes", "source_queries", "assumptions",
                     "decision_reasons", "validation_issues"):
            setattr(self, name, _as_list(getattr(self, name)))
        for name in ("outline", "experiments"):
            setattr(self, name, [item for item in _as_list(getattr(self, name))
                                 if isinstance(item, dict)])
        if not self.plan_id:
            self.plan_id = "plan-" + _fingerprint(self.task_id, self.title,
                                                  self.research_question)

    def section_ids(self) -> List[str]:
        return [_as_text(item.get("id")) for item in self.outline if item.get("id")]

    # -- completeness ---------------------------------------------------
    def element(self, name: str) -> Any:
        return getattr(self, name, None)

    def has_element(self, name: str) -> bool:
        value = self.element(name)
        if name == "research_question":
            return len(_as_text(value)) >= 20
        if name == "gap":
            return bool(_as_text(value)) and bool(self.gap_evidence)
        if isinstance(value, str):
            return len(value.strip()) >= 30
        return bool(value)

    def missing_elements(self) -> List[str]:
        return [name for name in PLAN_ELEMENTS if not self.has_element(name)]

    def outline_roles(self) -> List[str]:
        roles: List[str] = []
        for item in self.outline:
            role = _as_text(item.get("role")) or classify_section_role(
                _as_text(item.get("id")), _as_text(item.get("title")))
            roles.append(role)
        return roles

    @property
    def is_routine(self) -> bool:
        return self.contribution_type in (ContributionType.ROUTINE_IMPLEMENTATION,
                                          ContributionType.REPLICATION,
                                          ContributionType.EDUCATIONAL)

    def profile(self) -> ArticleTypeProfile:
        return profile_for(self.article_type)

    def summary_lines(self) -> List[str]:
        lines = [
            f"Research question: {self.research_question or '(none stated)'}",
            f"Article type: {self.profile().label}",
            f"Contribution ({self.contribution_type.label}): "
            f"{self.contribution or '(none stated)'}",
            f"Gap: {self.gap or '(none established)'}",
        ]
        if self.limitations:
            lines.append("Limitations: " + "; ".join(self.limitations[:3]))
        if self.missing_elements():
            lines.append("Missing plan elements: " + ", ".join(self.missing_elements()))
        return lines

    def to_dict(self) -> Dict[str, Any]:
        return {key: _coerce(value) for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchPlan":
        known = {field_name for field_name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {key: value for key, value in (data or {}).items() if key in known}
        return cls(**payload)


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------


@dataclass
class ExperimentSpec:
    """An executable experiment: code, tests, metrics and the parameters it ran with."""

    experiment_id: str = "exp-1"
    title: str = ""
    description: str = ""
    language: str = "python"
    code: str = ""
    tests: str = ""
    metrics: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    repetitions: int = 1
    dataset: str = ""
    benchmark: bool = False
    hardware: str = ""
    runtime_version: str = ""
    timeout_seconds: int = 60
    success_criteria: List[str] = field(default_factory=list)
    section_id: str = "results"
    source_path: str = ""

    def __post_init__(self) -> None:
        # Code and tests are kept byte-for-byte: the hash of the executed text is what
        # the manuscript's displayed listing is compared against.
        self.code = "" if self.code is None else str(self.code)
        self.tests = "" if self.tests is None else str(self.tests)
        self.metrics = [_as_text(m) for m in _as_list(self.metrics) if _as_text(m)]
        self.parameters = dict(self.parameters or {})
        try:
            self.repetitions = max(1, int(self.repetitions or 1))
        except (TypeError, ValueError):
            self.repetitions = 1
        self.benchmark = bool(self.benchmark)

    @property
    def has_tests(self) -> bool:
        return bool(self.tests.strip())

    @property
    def runnable(self) -> bool:
        return bool(self.code.strip()) and self.language.lower() in ("python", "python3")

    @property
    def code_hash(self) -> str:
        return hashlib.sha256(self.code.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {key: _coerce(value) for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentSpec":
        known = {field_name for field_name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {key: value for key, value in (data or {}).items() if key in known}
        return cls(**payload)


@dataclass
class TestOutcome:
    name: str = ""
    status: str = "not_run"       # passed / failed / error / skipped / not_run
    detail: str = ""
    duration_seconds: float = 0.0

    @property
    def message(self) -> str:
        return self.detail

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> Dict[str, Any]:
        return _coerce(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestOutcome":
        known = {field_name for field_name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class ExperimentRun:
    """What actually happened when the code was executed. No result, no claim."""

    experiment_id: str = ""
    run_id: str = ""
    status: ExperimentStatus = ExperimentStatus.NOT_RUN
    exit_code: Optional[int] = None
    duration_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""
    tests: List[TestOutcome] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metric_units: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    code_sha256: str = ""
    #: Produced files and their contents (name -> content hash or text).
    artifacts: Dict[str, Any] = field(default_factory=dict)
    repetition_metrics: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    error_type: str = ""
    skip_reason: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    repetitions: int = 1
    started_at: str = ""
    finished_at: str = ""
    command: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, ExperimentStatus):
            self.status = ExperimentStatus.parse(self.status)
        self.tests = [test if isinstance(test, TestOutcome) else TestOutcome.from_dict(test)
                      for test in _as_list(self.tests)]
        self.metrics = dict(self.metrics or {})
        self.metric_units = dict(self.metric_units or {})
        self.environment = dict(self.environment or {})
        self.parameters = dict(self.parameters or {})
        self.repetition_metrics = [dict(item) for item in _as_list(self.repetition_metrics)
                                  if isinstance(item, dict)]
        self.artifacts = dict(self.artifacts or {})
        if not self.run_id:
            self.run_id = "run-" + _fingerprint(self.experiment_id, self.code_sha256,
                                                self.started_at or utc_now())

    # -- outcome --------------------------------------------------------
    @property
    def ran(self) -> bool:
        return self.status.executed

    @property
    def succeeded(self) -> bool:
        return self.status == ExperimentStatus.OK

    @property
    def tests_run(self) -> int:
        return len([test for test in self.tests if test.status != "not_run"])

    @property
    def tests_passed(self) -> int:
        return len([test for test in self.tests if test.passed])

    @property
    def tests_failed(self) -> int:
        return len([test for test in self.tests
                    if test.status in ("failed", "error")])

    @property
    def all_tests_passed(self) -> bool:
        return self.tests_run > 0 and self.tests_failed == 0

    def metric(self, name: str) -> Any:
        if name in self.metrics:
            return self.metrics[name]
        for key, value in self.metrics.items():
            if key.lower() == name.lower():
                return value
        return None

    def summary_line(self) -> str:
        parts = [self.experiment_id, self.status.value]
        if self.tests_run:
            parts.append(f"tests {self.tests_passed}/{self.tests_run}")
        if self.metrics:
            parts.append(", ".join(f"{key}={value}" for key, value in
                                   list(self.metrics.items())[:4]))
        if self.error:
            parts.append(self.error[:80])
        return " · ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {key: _coerce(value) for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRun":
        known = {field_name for field_name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {key: value for key, value in (data or {}).items() if key in known}
        return cls(**payload)


@dataclass
class ResultRecord:
    """A recorded measurement, traceable to the run that produced it."""

    result_id: str = ""
    experiment_id: str = ""
    run_id: str = ""
    metric: str = ""
    value: Any = None
    unit: str = ""
    description: str = ""
    section_id: str = ""
    #: Size of the input the measurement was taken on, when the run recorded one.
    n: Any = None
    repetition: Optional[int] = None
    provenance: str = "executed"
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.result_id:
            self.result_id = "res-" + _fingerprint(self.run_id, self.metric,
                                                   str(self.value))
        if not self.unit:
            text = _as_text(self.value)
            match = re.search(r"[a-zA-Z%]+", text)
            if match:
                self.unit = match.group(0)

    @property
    def numeric(self) -> Optional[float]:
        if isinstance(self.value, bool):
            return None
        if isinstance(self.value, (int, float)):
            return float(self.value)
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", _as_text(self.value))
        return float(match.group(0)) if match else None

    def display(self) -> str:
        return f"{self.value}{(' ' + self.unit) if self.unit else ''}"

    def to_dict(self) -> Dict[str, Any]:
        return {key: _coerce(value) for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResultRecord":
        known = {field_name for field_name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class Measurement:
    name: str = ""
    value: Any = None
    unit: str = ""

    def display(self) -> str:
        return f"{self.value}{(' ' + self.unit) if self.unit else ''}"

    def to_dict(self) -> Dict[str, Any]:
        return _coerce(self.__dict__)


@dataclass
class BenchmarkRecord:
    """Everything a reader needs to interpret a number (§7 of the brief)."""

    benchmark_id: str = ""
    experiment_id: str = ""
    run_ids: List[str] = field(default_factory=list)
    description: str = ""
    input_sizes: List[Any] = field(default_factory=list)
    input_characteristics: str = ""
    dataset: str = ""
    dataset_ids: List[str] = field(default_factory=list)
    benchmark: bool = False
    hardware: str = ""
    operating_system: str = ""
    language: str = ""
    runtime_version: str = ""
    dependencies: List[str] = field(default_factory=list)
    repetitions: int = 1
    seed: Optional[int] = None
    timing_method: str = ""
    memory_method: str = ""
    metrics: Dict[str, Measurement] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    failed_runs: int = 0
    timeouts: int = 0
    excluded_runs: int = 0
    exclusions: List[str] = field(default_factory=list)
    variability: str = ""
    section_id: str = "results"

    def __post_init__(self) -> None:
        self.metrics = {name: value if isinstance(value, Measurement)
                        else Measurement.from_dict({"name": name, **value})
                        if isinstance(value, dict) else Measurement(name=name, value=value)
                        for name, value in (self.metrics or {}).items()}
        self.run_ids = [str(item) for item in _as_list(self.run_ids)]
        self.dataset_ids = [str(item) for item in _as_list(self.dataset_ids)]
        self.exclusions = [_as_text(item) for item in _as_list(self.exclusions)]
        if not self.benchmark_id:
            self.benchmark_id = "bench-" + _fingerprint(self.experiment_id, self.dataset,
                                                        *self.run_ids)

    def metadata_issues(self) -> List[str]:
        issues: List[str] = []
        if not self.hardware:
            issues.append("hardware is not stated")
        if not self.language:
            issues.append("implementation language is not stated")
        if not self.runtime_version:
            issues.append("runtime/compiler version is not stated")
        if not self.input_sizes:
            issues.append("input sizes are not stated")
        if not self.run_ids:
            issues.append("no executed run is attached")
        if not self.seed and self.seed != 0:
            issues.append("random seed is not stated")
        if not self.timing_method:
            issues.append("timing method is not stated")
        if self.repetitions < 2:
            issues.append("only one repetition: run-to-run variability is unknown")
        if self.failed_runs:
            issues.append(f"{self.failed_runs} run(s) failed and must be reported")
        return issues

    @property
    def complete(self) -> bool:
        return not self.metadata_issues()

    def to_dict(self) -> Dict[str, Any]:
        payload = {key: _coerce(value) for key, value in self.__dict__.items()}
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkRecord":
        known = {field_name for field_name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in (data or {}).items() if k in known}
        payload["metrics"] = {name: Measurement.from_dict(value) if isinstance(value, dict)
                              else value
                              for name, value in (payload.get("metrics") or {}).items()}
        return cls(**payload)


# ---------------------------------------------------------------------------
# claims and the evidence ledger
# ---------------------------------------------------------------------------


@dataclass
class ClaimRecord:
    """One checkable sentence, with what (if anything) substantiates it."""

    claim_id: str = ""
    text: str = ""
    kind: EvidenceKind = EvidenceKind.UNVERIFIED
    section_id: str = ""
    role: str = "other"
    source_ids: List[str] = field(default_factory=list)
    result_ids: List[str] = field(default_factory=list)
    experiment_ids: List[str] = field(default_factory=list)
    support_status: str = "unverified"   # verified / unsupported / unverified / not_required
    numbers: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EvidenceKind):
            text = _as_text(self.kind).lower()
            self.kind = next((kind for kind in EvidenceKind if kind.value == text),
                             EvidenceKind.UNVERIFIED)
        for name in ("source_ids", "result_ids", "experiment_ids", "notes", "numbers"):
            setattr(self, name, [_as_text(item) for item in _as_list(getattr(self, name))])

    @property
    def requires_evidence(self) -> bool:
        return self.kind in EVIDENCE_REQUIRED_KINDS

    @property
    def supported(self) -> bool:
        return self.support_status == "verified"

    @property
    def unverified(self) -> bool:
        return self.requires_evidence and self.support_status in ("unsupported",
                                                                 "unverified")

    def to_dict(self) -> Dict[str, Any]:
        return {key: _coerce(value) for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimRecord":
        known = {field_name for field_name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class EvidenceLedger:
    """The link between prose and provenance.

    ``support_status`` is only ``verified`` when the claim points at a verified source or
    at a result an executed run actually recorded. A claim with no such link stays
    ``unverified`` - the ledger never upgrades it by inspection.
    """

    claims: List[ClaimRecord] = field(default_factory=list)
    results: List[ResultRecord] = field(default_factory=list)
    runs: List[ExperimentRun] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    _counter: int = 0

    # -- population -----------------------------------------------------
    def add_claim(self, claim: ClaimRecord) -> ClaimRecord:
        self._counter += 1
        if not claim.claim_id:
            claim.claim_id = f"C{self._counter}"
        self.claims.append(claim)
        return claim

    def add_result(self, result: ResultRecord) -> ResultRecord:
        if not result.result_id:
            result.result_id = f"R{len(self.results) + 1}"
        self.results.append(result)
        return result

    def add_run(self, run: ExperimentRun) -> ExperimentRun:
        self.runs.append(run)
        return run

    def result_metric_index(self) -> Dict[str, List[ResultRecord]]:
        index: Dict[str, List[ResultRecord]] = {}
        for result in self.results:
            key = re.sub(r"[^a-z0-9]+", " ", result.metric.lower()).strip()
            if key:
                index.setdefault(key, []).append(result)
        return index

    def links_for_claim(self, claim_id: str) -> List[ResultRecord]:
        claim = next((c for c in self.claims if c.claim_id == claim_id), None)
        wanted = set(claim.result_ids if claim else [])
        return [result for result in self.results if result.result_id in wanted]

    # -- queries --------------------------------------------------------
    def executed_results(self) -> List[ResultRecord]:
        return [result for result in self.results if result.provenance == "executed"]

    def unsupported(self) -> List[ClaimRecord]:
        return [claim for claim in self.claims if claim.unverified]

    def by_kind(self, kind: EvidenceKind) -> List[ClaimRecord]:
        return [claim for claim in self.claims if claim.kind == kind]

    def claim_counts(self) -> Dict[str, int]:
        counts = {kind.value: 0 for kind in EvidenceKind}
        for claim in self.claims:
            counts[claim.kind.value] += 1
        return counts

    def summary(self) -> Dict[str, Any]:
        return {
            "claims": len(self.claims),
            "kinds": self.claim_counts(),
            "unsupported": len(self.unsupported()),
            "supported": len([c for c in self.claims if c.supported]),
            "results": len(self.results),
            "executed_results": len(self.executed_results()),
            "runs": len(self.runs),
            "successful_runs": len([r for r in self.runs if r.status == ExperimentStatus.OK]),
            "notes": list(self.notes),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claims": [claim.to_dict() for claim in self.claims],
            "results": [result.to_dict() for result in self.results],
            "runs": [run.to_dict() for run in self.runs],
            "notes": list(self.notes),
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceLedger":
        ledger = cls()
        for item in data.get("claims") or []:
            ledger.claims.append(ClaimRecord.from_dict(item))
        for item in data.get("results") or []:
            ledger.results.append(ResultRecord.from_dict(item))
        for item in data.get("runs") or []:
            ledger.runs.append(ExperimentRun.from_dict(item))
        ledger.notes = list(data.get("notes") or [])
        ledger._counter = len(ledger.claims)
        return ledger


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------

REVIEW_DIMENSIONS: Tuple[str, ...] = (
    "originality", "novelty", "contribution", "methodology", "correctness",
    "evidence", "citations", "experiments", "code", "statistics", "reproducibility",
    "clarity", "structure", "limitations", "ethics",
)

REVIEW_STATUSES: Tuple[str, ...] = ("open", "addressed", "fixed", "deferred",
                                    "unresolved", "rejected")


@dataclass
class ReviewIssue:
    issue_id: str = ""
    dimension: str = "correctness"
    severity: ReviewSeverity = ReviewSeverity.MODERATE
    section: str = ""
    claim: str = ""
    explanation: str = ""
    evidence: str = ""
    correction: str = ""
    auto_fixable: bool = False
    requires_user_input: bool = False
    status: str = "open"
    source: str = "model"          # model / deterministic
    iteration: int = 0
    check_id: str = ""
    corrected_claim: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.severity, ReviewSeverity):
            self.severity = ReviewSeverity.parse(self.severity)
        if not self.issue_id:
            self.issue_id = "I-" + _fingerprint(self.dimension, self.severity.value,
                                                self.section, self.claim,
                                                self.correction)
        if self.status not in REVIEW_STATUSES:
            self.status = "open"
        if self.dimension not in REVIEW_DIMENSIONS:
            text = _as_text(self.dimension).lower()
            self.dimension = text if text in REVIEW_DIMENSIONS else "correctness"

    @property
    def resolved(self) -> bool:
        return self.status in ("addressed", "fixed", "rejected")

    @property
    def is_blocking(self) -> bool:
        return (self.severity.blocking and not self.resolved
                and self.status in ("open", "unresolved"))

    @property
    def needs_author(self) -> bool:
        return self.requires_user_input or not self.auto_fixable

    def to_dict(self) -> Dict[str, Any]:
        return {key: _coerce(value) for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewIssue":
        known = {field_name for field_name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class RevisionSummary:
    iteration: int = 0
    changed_sections: List[str] = field(default_factory=list)
    applied: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    deferred: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _coerce(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RevisionSummary":
        known = {field_name for field_name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class ReviewReport:
    issues: List[ReviewIssue] = field(default_factory=list)
    revisions: List[RevisionSummary] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    iterations_run: int = 0
    model_review_ran: bool = False
    model_review_error: str = ""
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.issues = [issue if isinstance(issue, ReviewIssue) else ReviewIssue.from_dict(issue)
                       for issue in _as_list(self.issues)]
        self.revisions = [item if isinstance(item, RevisionSummary)
                          else RevisionSummary.from_dict(item)
                          for item in _as_list(self.revisions)]

    def add(self, issue: ReviewIssue) -> ReviewIssue:
        self.issues.append(issue)
        return issue

    def by_severity(self, severity: Optional[Any] = None) -> Any:
        """Open issues of one severity, or severity counts when called without one."""
        if severity is None:
            return self.severity_counts()
        wanted = ReviewSeverity.parse(severity)
        return [issue for issue in self.issues
                if issue.severity == wanted and not issue.resolved]

    def severity_counts(self) -> Dict[str, int]:
        counts = {severity.value: 0 for severity in ReviewSeverity}
        for issue in self.issues:
            if issue.resolved:
                continue
            counts[issue.severity.value] += 1
        return counts

    def blocking(self) -> List[ReviewIssue]:
        return [issue for issue in self.issues if issue.is_blocking]

    def unresolved(self) -> List[ReviewIssue]:
        return [issue for issue in self.issues if not issue.resolved]

    def dimensions_covered(self) -> List[str]:
        return sorted({issue.dimension for issue in self.issues})

    @property
    def can_be_ready(self) -> bool:
        return bool(self.model_review_ran) and not self.blocking()

    def summary(self) -> Dict[str, Any]:
        return {
            "issues": len(self.issues),
            "open": len(self.unresolved()),
            "blocking": len(self.blocking()),
            "by_severity": self.severity_counts(),
            "dimensions": self.dimensions_covered(),
            "iterations_run": self.iterations_run,
            "model_review_ran": self.model_review_ran,
            "model_review_error": self.model_review_error,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issues": [issue.to_dict() for issue in self.issues],
            "revisions": [rev.to_dict() for rev in self.revisions],
            "notes": list(self.notes),
            "iterations_run": self.iterations_run,
            "model_review_ran": self.model_review_ran,
            "model_review_error": self.model_review_error,
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewReport":
        report = cls()
        for item in data.get("issues") or []:
            report.issues.append(ReviewIssue.from_dict(item))
        for item in data.get("revisions") or []:
            report.revisions.append(RevisionSummary.from_dict(item))
        report.notes = list(data.get("notes") or [])
        report.iterations_run = int(data.get("iterations_run") or 0)
        report.model_review_ran = bool(data.get("model_review_ran"))
        report.model_review_error = _as_text(data.get("model_review_error"))
        return report


# ---------------------------------------------------------------------------
# validation checks
# ---------------------------------------------------------------------------


@dataclass
class ValidationCheck:
    """One deterministic check and its verdict.

    There is deliberately no score here. A check either passed, failed, could not be
    performed, or does not apply, and each failed check names the gate effect it has.
    """

    check_id: str = ""
    category: str = "general"
    description: str = ""
    status: CheckStatus = CheckStatus.UNVERIFIED
    severity: ReviewSeverity = ReviewSeverity.MODERATE
    details: str = ""
    evidence: List[str] = field(default_factory=list)
    section: str = ""
    gate_effect: str = "needs_revision"   # none / needs_revision / incomplete
    #: Article types the check applies to; empty means every type.
    article_scope: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, CheckStatus):
            text = _as_text(self.status).lower()
            aliases = {"ok": CheckStatus.PASS, "passed": CheckStatus.PASS,
                       "failed": CheckStatus.FAIL, "skipped": CheckStatus.NOT_APPLICABLE,
                       "n/a": CheckStatus.NOT_APPLICABLE, "na": CheckStatus.NOT_APPLICABLE}
            self.status = aliases.get(text) or next(
                (item for item in CheckStatus if item.value == text),
                CheckStatus.UNVERIFIED)
        if not isinstance(self.severity, ReviewSeverity):
            self.severity = ReviewSeverity.parse(self.severity)
        self.evidence = [_as_text(item) for item in _as_list(self.evidence)]
        self.article_scope = tuple(_as_text(item) for item in _as_list(self.article_scope))

    @property
    def blocking(self) -> bool:
        return (self.status in (CheckStatus.FAIL, CheckStatus.UNVERIFIED)
                and self.gate_effect != "none"
                and self.severity.at_least(ReviewSeverity.MODERATE))

    def to_dict(self) -> Dict[str, Any]:
        return {key: _coerce(value) for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationCheck":
        known = {field_name for field_name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class ValidationReport:
    """The deterministic verdict set. ``summary()`` is counts, not a score."""

    article_type: ArticleType = ArticleType.GENERAL
    checks: List[ValidationCheck] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    validated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.checks = [check if isinstance(check, ValidationCheck)
                       else ValidationCheck.from_dict(check)
                       for check in _as_list(self.checks)]

    def add(self, check: ValidationCheck) -> ValidationCheck:
        self.checks.append(check)
        return check

    def get(self, check_id: str) -> Optional[ValidationCheck]:
        return next((check for check in self.checks if check.check_id == check_id), None)

    def by_status(self, status: CheckStatus) -> List[ValidationCheck]:
        return [check for check in self.checks if check.status == status]

    def failing(self) -> List[ValidationCheck]:
        return [check for check in self.checks if check.status == CheckStatus.FAIL
                or check.status == CheckStatus.UNVERIFIED]

    def problems(self) -> List[ValidationCheck]:
        """Failed or unverified checks, ordered by severity (most severe first)."""
        return sorted(self.failing(),
                      key=lambda check: (-check.severity.weight, check.check_id))

    def unverified_checks(self) -> List[ValidationCheck]:
        return [check for check in self.checks
                if check.status == CheckStatus.UNVERIFIED]

    def blocking(self) -> List[ValidationCheck]:
        return [check for check in self.checks if check.blocking]

    def category(self, name: str) -> List[ValidationCheck]:
        return [check for check in self.checks if check.category == name]

    def summary(self) -> Dict[str, int]:
        counts = {status.value: 0 for status in CheckStatus}
        for check in self.checks:
            counts[check.status.value] += 1
        counts["total"] = len(self.checks)
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_type": self.article_type.value,
            "checks": [check.to_dict() for check in self.checks],
            "notes": list(self.notes),
            "summary": self.summary(),
            "validated_at": self.validated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationReport":
        profile = profile_for(data.get("article_type"))
        report = cls(article_type=profile.article_type,
                     checks=[ValidationCheck.from_dict(item)
                             for item in data.get("checks") or []],
                     notes=list(data.get("notes") or []))
        return report


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


class GateStatus(str, Enum):
    READY_FOR_REVIEW = "ready_for_final_review"
    NEEDS_REVISION = "needs_revision"
    INCOMPLETE_OR_UNVERIFIED = "incomplete_or_unverified"

    @property
    def label(self) -> str:
        return {
            GateStatus.READY_FOR_REVIEW: "Ready for final human review",
            GateStatus.NEEDS_REVISION: "Needs revision",
            GateStatus.INCOMPLETE_OR_UNVERIFIED: "Incomplete or unverified",
        }[self]

    @property
    def permits_final_review(self) -> bool:
        return self == GateStatus.READY_FOR_REVIEW


@dataclass
class WorkflowStates:
    """Each stage a study goes through, tracked separately.

    "Generated" is not "verified": the fields are deliberately not collapsed into one
    status, because a manuscript can be beautifully written and still have fabricated
    citations or an experiment that never ran.
    """

    draft_generated: bool = False
    plan_validated: bool = False
    sources_retrieved: bool = False
    sources_verified: bool = False
    experiments_executed: bool = False
    tests_passed: bool = False
    technical_checks_completed: bool = False
    peer_review_completed: bool = False
    revision_completed: bool = False
    ready_for_human_review: bool = False
    #: Whether the article type (or the brief) requires these stages at all, so a UI
    #: can distinguish "not required" from "required but not done".
    required_sources: bool = False
    required_experiments: bool = False

    STATE_LABELS: ClassVar[Dict[str, str]] = {
        "draft_generated": "Draft generated",
        "plan_validated": "Research plan validated",
        "sources_retrieved": "Sources retrieved",
        "sources_verified": "Sources verified",
        "experiments_executed": "Experiments executed",
        "tests_passed": "Tests passed",
        "technical_checks_completed": "Technical checks completed",
        "peer_review_completed": "Peer review completed",
        "revision_completed": "Revision completed",
        "ready_for_human_review": "Ready for final human review",
    }

    def as_rows(self) -> List[Tuple[str, str, str, bool]]:
        required = {
            "sources_retrieved": self.required_sources,
            "sources_verified": self.required_sources,
            "experiments_executed": self.required_experiments,
            "tests_passed": self.required_experiments,
        }
        rows = []
        for key, label in self.STATE_LABELS.items():
            is_required = required.get(key, True)
            rows.append((key, label, "required" if is_required else "not required",
                         bool(getattr(self, key))))
        return rows

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key in self.STATE_LABELS:
            value = getattr(self, key)
            payload[key] = bool(value) if isinstance(value, bool) else value
        payload["required_sources"] = bool(self.required_sources)
        payload["required_experiments"] = bool(self.required_experiments)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowStates":
        states = cls()
        for key in cls.STATE_LABELS:
            if key in (data or {}):
                setattr(states, key, bool(data[key]))
        for key in ("required_sources", "required_experiments"):
            if key in (data or {}):
                setattr(states, key, bool(data[key]))
        return states


@dataclass
class QualityGateResult:
    """The three-state verdict, with the reasons that produced it."""

    status: GateStatus = GateStatus.INCOMPLETE_OR_UNVERIFIED
    reasons: List[str] = field(default_factory=list)
    critical_failures: List[str] = field(default_factory=list)
    major_failures: List[str] = field(default_factory=list)
    unchecked: List[str] = field(default_factory=list)
    states: WorkflowStates = field(default_factory=WorkflowStates)
    counts: Dict[str, int] = field(default_factory=dict)
    incomplete_reasons: List[str] = field(default_factory=list)
    revision_reasons: List[str] = field(default_factory=list)
    unresolved_issues: List[str] = field(default_factory=list)
    validated: bool = False
    article_type: ArticleType = ArticleType.GENERAL
    evaluated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not isinstance(self.status, GateStatus):
            text = _as_text(self.status).lower()
            self.status = next((item for item in GateStatus if item.value == text),
                               GateStatus.INCOMPLETE_OR_UNVERIFIED)
        if isinstance(self.states, dict):
            self.states = WorkflowStates.from_dict(self.states)
        self.reasons = [_as_text(reason) for reason in _as_list(self.reasons)]
        self.incomplete_reasons = [_as_text(item) for item in
                                   _as_list(self.incomplete_reasons)]
        self.revision_reasons = [_as_text(item) for item in _as_list(self.revision_reasons)]
        self.unresolved_issues = [_as_text(item) for item in _as_list(self.unresolved_issues)]

    @property
    def permits_final_review(self) -> bool:
        return self.status.permits_final_review

    @property
    def label(self) -> str:
        return self.status.label

    def reason_block(self, limit: int = 12) -> str:
        if not self.reasons:
            return "no blocking findings"
        lines = [f"- {reason}" for reason in self.reasons[:limit]]
        if len(self.reasons) > limit:
            lines.append(f"- ... {len(self.reasons) - limit} more")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "label": self.label,
            "reasons": list(self.reasons),
            "critical_failures": list(self.critical_failures),
            "major_failures": list(self.major_failures),
            "unchecked": list(self.unchecked),
            "incomplete_reasons": list(self.incomplete_reasons),
            "revision_reasons": list(self.revision_reasons),
            "unresolved_issues": list(self.unresolved_issues),
            "validated": bool(self.validated),
            "states": self.states.to_dict(),
            "counts": dict(self.counts),
            "article_type": self.article_type.value,
            "evaluated_at": self.evaluated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityGateResult":
        return cls(
            status=data.get("status", GateStatus.INCOMPLETE_OR_UNVERIFIED),
            reasons=list(data.get("reasons") or []),
            critical_failures=list(data.get("critical_failures") or []),
            major_failures=list(data.get("major_failures") or []),
            unchecked=list(data.get("unchecked") or []),
            states=WorkflowStates.from_dict(data.get("states") or {}),
            counts=dict(data.get("counts") or {}),
            article_type=profile_for(data.get("article_type")).article_type,
        )


def dedupe_issues(issues: Sequence[ReviewIssue]) -> List[ReviewIssue]:
    """Collapse findings that describe the same problem.

    Model reviewers repeat themselves across sections and iterations; deterministic
    findings can duplicate a model finding. Two issues are the same when they share a
    check id, or share a dimension plus a near-identical claim. The most severe copy
    wins, and an addressed copy never masks an open one.
    """
    merged: List[ReviewIssue] = []
    seen: Dict[Tuple[str, str, str], ReviewIssue] = {}
    for issue in issues:
        claim_key = re.sub(r"[^a-z0-9 ]+", " ", (issue.claim or "").lower())
        claim_key = re.sub(r"\s+", " ", claim_key).strip()[:90]
        keys = [("check", issue.check_id, "")] if issue.check_id else []
        keys.append(("claim", issue.dimension, claim_key))
        existing = next((seen[key] for key in keys if key in seen), None)
        if existing is None:
            merged.append(issue)
            for key in keys:
                seen[key] = issue
            continue
        # Keep the more severe / more actionable record, but never let a resolved
        # copy overwrite an open one.
        if existing.resolved and not issue.resolved:
            index = merged.index(existing)
            merged[index] = issue
            for key in keys:
                seen[key] = issue
            continue
        if issue.severity.weight < existing.severity.weight and not issue.resolved:
            existing.severity = issue.severity
        if issue.correction and not existing.correction:
            existing.correction = issue.correction
        if issue.evidence and not existing.evidence:
            existing.evidence = issue.evidence
        if issue.requires_user_input:
            existing.requires_user_input = True
        for key in keys:
            seen.setdefault(key, existing)
    return merged
