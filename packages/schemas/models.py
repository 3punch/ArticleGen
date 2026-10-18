"""Pydantic contracts shared by the ProjectForge API and orchestration packages.

These are deliberately data-only models: they validate persisted and exchanged
state, but contain no agent behavior or database code.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)


AgentName = Literal[
    "requirement",
    "planner",
    "coder",
    "tester",
    "reviewer",
    "debugger",
    "packager",
]
ProjectRuntime = Literal["python", "node"]
TaskStatus = Literal[
    "pending",
    "ready",
    "running",
    "blocked",
    "completed",
    "failed",
    "cancelled",
]
RubricType = Literal["automatable", "human-judgment"]


class WorkflowState(str, Enum):
    INTAKE = "INTAKE"
    SPEC = "SPEC"
    PLAN = "PLAN"
    AWAITING_PLAN_APPROVAL = "AWAITING_PLAN_APPROVAL"
    AWAITING_CLARIFICATION = "AWAITING_CLARIFICATION"
    BUILD_LOOP = "BUILD_LOOP"
    INTEGRATION = "INTEGRATION"
    REVIEW = "REVIEW"
    AWAITING_FINAL_APPROVAL = "AWAITING_FINAL_APPROVAL"
    PACKAGE = "PACKAGE"
    DONE = "DONE"
    FAILED = "FAILED"
    ESCALATED = "ESCALATED"


class ContractModel(BaseModel):
    """Common strictness for external and persisted data."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RubricItem(ContractModel):
    criterion: str = Field(min_length=1, max_length=2000)
    type: RubricType
    check_method: str = Field(min_length=1, max_length=2000)


class ProjectSpec(ContractModel):
    """Normalized assignment requirements and grading rubric."""

    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=20_000)
    deliverables: list[str] = Field(min_length=1, max_length=100)
    constraints: list[str] = Field(default_factory=list, max_length=200)
    tech_stack_hints: list[str] = Field(default_factory=list, max_length=100)
    rubric_items: list[RubricItem] = Field(min_length=1, max_length=200)

    @field_validator("deliverables", "constraints", "tech_stack_hints")
    @classmethod
    def nonempty_list_items(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("list items must not be empty")
        return cleaned

    @model_validator(mode="after")
    def unique_rubric_criteria(self) -> "ProjectSpec":
        normalized = [item.criterion.casefold() for item in self.rubric_items]
        if len(normalized) != len(set(normalized)):
            raise ValueError("rubric criteria must be unique")
        return self


class RetrievedContextChunk(ContractModel):
    """A bounded ProjectSpec excerpt selected by pgvector for one agent task."""

    chunk_id: UUID
    project_id: UUID
    chunk_kind: Literal[
        "title", "description", "deliverable", "constraint", "tech_stack_hint", "rubric_item",
    ]
    chunk_key: str = Field(min_length=1, max_length=160)
    ordinal: StrictInt = Field(ge=0, le=1000)
    content: str = Field(min_length=1, max_length=8000)
    similarity: float = Field(ge=-1.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=40)


class ClarifyingQuestion(ContractModel):
    question_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    question: str = Field(min_length=8, max_length=2000)
    why_it_matters: str = Field(min_length=8, max_length=2000)


class RequirementAgentResponse(ContractModel):
    status: Literal["completed", "needs_clarification"]
    confidence: float = Field(ge=0.0, le=1.0)
    project_spec: Optional[ProjectSpec] = None
    questions: list[ClarifyingQuestion] = Field(default_factory=list, max_length=20)
    uncertainties: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_response_shape(self) -> "RequirementAgentResponse":
        if self.status == "needs_clarification":
            if self.project_spec is not None:
                raise ValueError("a clarification response must not guess a ProjectSpec")
            if not self.questions:
                raise ValueError("a clarification response must include questions")
        elif self.project_spec is None:
            raise ValueError("a completed response must include a ProjectSpec")
        question_ids = [question.question_id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("clarification question ids must be unique")
        return self


class TaskFeedback(ContractModel):
    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    feedback: str = Field(min_length=10, max_length=2000)


class TaskNode(ContractModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    description: str = Field(min_length=1, max_length=10_000)
    dependencies: list[str] = Field(default_factory=list, max_length=200)
    status: TaskStatus = "pending"
    acceptance_criteria: list[str] = Field(min_length=1, max_length=100)
    retries: StrictInt = Field(default=0, ge=0, le=100)
    assigned_agent: AgentName
    # Empty read/write declarations are treated conservatively as an exclusive
    # workspace claim by the deterministic scheduler.
    read_set: list[str] = Field(default_factory=list, max_length=100)
    write_set: list[str] = Field(default_factory=list, max_length=100)
    runtime: ProjectRuntime | None = None
    timeout_seconds: StrictInt = Field(default=300, ge=1, le=3600)
    review_feedback: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("review_feedback")
    @classmethod
    def validate_review_feedback(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value or len(value) > 2000 for value in cleaned):
            raise ValueError("task review feedback must be non-empty and at most 2000 characters")
        return cleaned

    @field_validator("dependencies", "acceptance_criteria")
    @classmethod
    def validate_string_lists(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("list items must not be empty")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("list items must be unique")
        return cleaned

    @field_validator("read_set", "write_set")
    @classmethod
    def validate_resource_paths(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip().rstrip("/") for value in values]
        if any(
            not value
            or value.startswith("/")
            or "\\" in value
            or ":" in value
            or "\x00" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
            for value in cleaned
        ):
            raise ValueError("resource declarations must be safe relative paths")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("resource declarations must be unique")
        return cleaned

    @model_validator(mode="after")
    def no_self_dependency(self) -> "TaskNode":
        if self.id in self.dependencies:
            raise ValueError("a task cannot depend on itself")
        return self


class AgentFileChange(ContractModel):
    """One explicit, auditable file write or delete requested by a worker agent."""

    path: str = Field(min_length=1, max_length=240)
    action: Literal["write", "delete"] = "write"
    content: str | None = Field(default=None, max_length=1_000_000)

    @field_validator("path")
    @classmethod
    def validate_file_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        forbidden_names = {
            ".git", ".env", ".netrc", ".npmrc", ".pypirc", ".ssh", ".aws",
            ".gnupg", ".git-credentials", "credentials", "id_rsa", "id_ed25519",
            "id_ecdsa",
        }
        if (
            value != value.strip()
            or path.is_absolute()
            or "\\" in value
            or ":" in value
            or "\x00" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or any(part.casefold() in forbidden_names for part in path.parts)
            or path.suffix.casefold() in {".pem", ".key", ".p12", ".pfx"}
        ):
            raise ValueError("file path must be a safe relative path without credentials or VCS data")
        return value

    @model_validator(mode="after")
    def validate_content_action(self) -> "AgentFileChange":
        if self.action == "write" and self.content is None:
            raise ValueError("write actions require file content")
        if self.action == "delete" and self.content is not None:
            raise ValueError("delete actions must not include file content")
        return self


class TaskEdge(ContractModel):
    source: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    target: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

    @model_validator(mode="after")
    def no_self_edge(self) -> "TaskEdge":
        if self.source == self.target:
            raise ValueError("an edge cannot point to itself")
        return self


class TaskGraph(ContractModel):
    nodes: list[TaskNode] = Field(min_length=1, max_length=500)
    edges: list[TaskEdge] = Field(default_factory=list, max_length=2000)

    @model_validator(mode="before")
    @classmethod
    def derive_edges_from_dependencies(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "edges" not in data or data["edges"] is None:
            nodes = data.get("nodes") or []
            derived_edges = []
            for node in nodes:
                if isinstance(node, dict):
                    node_id = node.get("id")
                    dependencies = node.get("dependencies") or []
                else:
                    node_id = getattr(node, "id", None)
                    dependencies = getattr(node, "dependencies", []) or []
                derived_edges.extend(
                    {"source": dependency, "target": node_id}
                    for dependency in dependencies
                )
            data["edges"] = derived_edges
        return data

    @model_validator(mode="after")
    def validate_graph(self) -> "TaskGraph":
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("task node ids must be unique")
        known = set(node_ids)

        expected_edges: set[tuple[str, str]] = set()
        for node in self.nodes:
            for dependency in node.dependencies:
                if dependency not in known:
                    raise ValueError(f"task '{node.id}' depends on unknown task '{dependency}'")
                expected_edges.add((dependency, node.id))

        actual_edges: set[tuple[str, str]] = set()
        adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(f"edge '{edge.source}' -> '{edge.target}' references an unknown task")
            key = (edge.source, edge.target)
            if key in actual_edges:
                raise ValueError(f"duplicate edge '{edge.source}' -> '{edge.target}'")
            actual_edges.add(key)
            adjacency[edge.source].append(edge.target)

        if actual_edges != expected_edges:
            raise ValueError("edges must exactly match the nodes' dependencies")

        indegree = {node_id: 0 for node_id in node_ids}
        for source in node_ids:
            for target in adjacency[source]:
                indegree[target] += 1
        ready = [node_id for node_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            source = ready.pop()
            visited += 1
            for target in adjacency[source]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if visited != len(node_ids):
            raise ValueError("task graph must be acyclic")
        return self


class PlannerAgentResponse(ContractModel):
    plan_summary: str = Field(min_length=10, max_length=5000)
    task_graph: TaskGraph
    evidence: list[str] = Field(min_length=1, max_length=100)

    @field_validator("evidence")
    @classmethod
    def evidence_items_must_not_be_empty(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("evidence items must not be empty")
        return cleaned


class CoderAgentResponse(ContractModel):
    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    summary: str = Field(min_length=10, max_length=5000)
    changes: list[AgentFileChange] = Field(min_length=1, max_length=20)
    evidence: list[str] = Field(min_length=1, max_length=100)

    @field_validator("evidence")
    @classmethod
    def evidence_items_must_not_be_empty(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("evidence items must not be empty")
        return cleaned

    @model_validator(mode="after")
    def changes_must_be_unique(self) -> "CoderAgentResponse":
        paths = [change.path for change in self.changes]
        if len(paths) != len(set(paths)):
            raise ValueError("Coder file change paths must be unique")
        return self


class TesterAgentResponse(ContractModel):
    summary: str = Field(min_length=10, max_length=5000)
    test_files: list[AgentFileChange] = Field(min_length=1, max_length=20)
    evidence: list[str] = Field(min_length=1, max_length=100)

    @field_validator("evidence")
    @classmethod
    def evidence_items_must_not_be_empty(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("evidence items must not be empty")
        return cleaned

    @model_validator(mode="after")
    def test_files_must_be_unique_writes(self) -> "TesterAgentResponse":
        paths = [change.path for change in self.test_files]
        if len(paths) != len(set(paths)):
            raise ValueError("Tester file paths must be unique")
        if any(change.action != "write" for change in self.test_files):
            raise ValueError("Tester may write tests but must not delete repository files")
        if any(not change.path.startswith("tests/") for change in self.test_files):
            raise ValueError("Tester-generated files must be located under tests/")
        return self


class DebuggerAgentResponse(ContractModel):
    diagnosis: str = Field(min_length=10, max_length=5000)
    fix_instructions: list[str] = Field(min_length=1, max_length=30)
    evidence: list[str] = Field(min_length=1, max_length=100)

    @field_validator("fix_instructions", "evidence")
    @classmethod
    def validate_debugger_lists(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("Debugger instructions and evidence must not be empty")
        return cleaned


class HumanReviewRecommendation(ContractModel):
    criterion: str = Field(min_length=1, max_length=2000)
    recommendation: Literal["approve", "request_changes"]
    rationale: str = Field(min_length=10, max_length=5000)
    evidence: list[str] = Field(min_length=1, max_length=100)

    @field_validator("evidence")
    @classmethod
    def human_evidence_must_not_be_empty(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("human review evidence must not be empty")
        return cleaned


class ReviewerHumanResponse(ContractModel):
    recommendations: list[HumanReviewRecommendation] = Field(max_length=200)
    summary: str = Field(min_length=10, max_length=5000)

    @model_validator(mode="after")
    def recommendations_must_be_unique(self) -> "ReviewerHumanResponse":
        criteria = [item.criterion.casefold() for item in self.recommendations]
        if len(criteria) != len(set(criteria)):
            raise ValueError("human review recommendations must be unique")
        return self


class ReviewCriterionAssessment(ContractModel):
    rubric_item: RubricItem
    status: Literal["passed", "failed", "not_verified", "human_review_required"]
    evidence: list[str] = Field(min_length=1, max_length=200)
    recommendation: Literal["approve", "request_changes"] | None = None
    rationale: str = Field(default="", max_length=5000)

    @model_validator(mode="after")
    def keep_human_judgment_human(self) -> "ReviewCriterionAssessment":
        if self.rubric_item.type == "human-judgment":
            if self.status != "human_review_required" or self.recommendation is None:
                raise ValueError(
                    "human-judgment rubric items must remain pending human review and include a recommendation"
                )
        elif self.status == "human_review_required" or self.recommendation is not None:
            raise ValueError("automatable rubric items cannot use a human recommendation")
        return self


class ReviewerAgentResponse(ContractModel):
    summary: str = Field(min_length=10, max_length=10_000)
    criteria: list[ReviewCriterionAssessment] = Field(min_length=1, max_length=200)
    automatable_checks_passed: StrictBool
    human_review_required: StrictBool
    evidence: list[str] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def verify_summary_flags_and_unique_criteria(self) -> "ReviewerAgentResponse":
        names = [item.rubric_item.criterion.casefold() for item in self.criteria]
        if len(names) != len(set(names)):
            raise ValueError("review criteria must be unique")
        automated = [item for item in self.criteria if item.rubric_item.type == "automatable"]
        human = [item for item in self.criteria if item.rubric_item.type == "human-judgment"]
        if self.automatable_checks_passed != all(item.status == "passed" for item in automated):
            raise ValueError("automatable_checks_passed must reflect every automatable criterion")
        if self.human_review_required != bool(human):
            raise ValueError("human_review_required must reflect the rubric items")
        return self


class PackagerAgentResponse(ContractModel):
    readme_markdown: str = Field(min_length=20, max_length=100_000)
    report_markdown: str = Field(min_length=20, max_length=100_000)
    summary: str = Field(min_length=10, max_length=5000)
    evidence: list[str] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_markdown_documents(self) -> "PackagerAgentResponse":
        if not self.readme_markdown.lstrip().startswith("#"):
            raise ValueError("README must start with a Markdown heading")
        if not self.report_markdown.lstrip().startswith("#"):
            raise ValueError("report must start with a Markdown heading")
        return self


class AgentRun(ContractModel):
    agent_name: AgentName
    input: dict[str, Any]
    output: dict[str, Any]
    tokens_used: StrictInt = Field(ge=0)
    cost: Decimal = Field(ge=Decimal("0"), max_digits=16, decimal_places=8)
    timestamp: datetime
    project_id: UUID
    workflow_id: Optional[UUID] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    budget_reservation_id: Optional[UUID] = None
    idempotency_key: Optional[str] = None

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value


class ReviewResult(ContractModel):
    rubric_item: RubricItem
    passed: StrictBool | None = None
    status: Literal["passed", "failed", "not_verified", "human_review_required"] | None = None
    evidence: list[str] = Field(default_factory=list, max_length=200)
    reviewer_notes: str = Field(default="", max_length=10_000)
    recommendation: Literal["approve", "request_changes"] | None = None
    workflow_id: UUID | None = None

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_status(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("status") is None:
            data = dict(value)
            if data.get("passed") is True:
                data["status"] = "passed"
            elif data.get("passed") is False:
                data["status"] = "failed"
            return data
        return value

    @field_validator("evidence")
    @classmethod
    def evidence_items_must_not_be_empty(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("evidence items must not be empty")
        return cleaned

    @model_validator(mode="after")
    def human_judgment_is_not_a_machine_pass(self) -> "ReviewResult":
        if self.status in {"passed", "failed"}:
            expected = self.status == "passed"
            if self.passed is not expected:
                raise ValueError("passed must agree with the review status")
        elif self.passed is not None:
            raise ValueError("unverified and human-review results must not claim pass or fail")
        if self.rubric_item.type == "human-judgment":
            if self.status != "human_review_required" or self.passed is not None:
                raise ValueError("human-judgment review results must remain pending human review")
            if self.recommendation is None:
                raise ValueError("human-judgment review results require a recommendation")
        elif self.status == "human_review_required" or self.recommendation is not None:
            raise ValueError("automatable review results cannot use human-review fields")
        if self.status is None:
            raise ValueError("review status is required")
        return self


class BudgetLedger(ContractModel):
    project_id: UUID
    token_cap: StrictInt = Field(ge=0)
    spend_cap: Decimal = Field(ge=Decimal("0"), max_digits=16, decimal_places=8)
    used_tokens: StrictInt = Field(default=0, ge=0)
    used_spend: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), max_digits=16,
                                decimal_places=8)
    reserved_tokens: StrictInt = Field(default=0, ge=0)
    reserved_spend: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), max_digits=16,
                                    decimal_places=8)

    # Usage may exceed a cap because actual provider charges must still be recorded.
    # The execution controller is responsible for pausing/escalating on a breach.


class ProjectCreate(ContractModel):
    name: str = Field(min_length=1, max_length=200)
    spec: Optional[ProjectSpec] = None
    raw_brief_text: Optional[str] = Field(default=None, min_length=20, max_length=20_000)
    rubric_text: Optional[str] = Field(default=None, max_length=20_000)
    token_cap: StrictInt = Field(default=100_000, ge=0)
    spend_cap: Decimal = Field(default=Decimal("10.00"), ge=Decimal("0"), max_digits=16,
                                decimal_places=8)

    @model_validator(mode="after")
    def require_spec_or_raw_brief(self) -> "ProjectCreate":
        if self.spec is None and self.raw_brief_text is None:
            raise ValueError("provide either a ProjectSpec or raw_brief_text")
        return self


class ProjectUpdate(ContractModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    spec: Optional[ProjectSpec] = None
    raw_brief_text: Optional[str] = Field(default=None, min_length=20, max_length=20_000)
    rubric_text: Optional[str] = Field(default=None, max_length=20_000)
    token_cap: Optional[StrictInt] = Field(default=None, ge=0)
    spend_cap: Optional[Decimal] = Field(default=None, ge=Decimal("0"), max_digits=16,
                                         decimal_places=8)

    @model_validator(mode="after")
    def require_at_least_one_change(self) -> "ProjectUpdate":
        if not self.model_fields_set:
            raise ValueError("at least one field must be supplied")
        return self


class ProjectRecord(ContractModel):
    id: UUID
    name: str
    spec: Optional[ProjectSpec] = None
    raw_brief_text: Optional[str] = None
    rubric_text: Optional[str] = None
    budget: BudgetLedger
    created_at: datetime
    updated_at: datetime


class TaskGraphRecord(ContractModel):
    id: UUID
    project_id: UUID
    version: StrictInt = Field(ge=1)
    graph: TaskGraph
    created_at: datetime


class WorkflowStartRequest(ContractModel):
    # ``task_graph_id`` is an optional seed graph for migrations/tests; the Planner
    # still runs and its validated output becomes the workflow's authoritative graph.
    task_graph_id: Optional[UUID] = None
    raw_brief_text: Optional[str] = Field(default=None, min_length=20, max_length=20_000)
    rubric_text: Optional[str] = Field(default=None, max_length=20_000)
    max_retries: StrictInt = Field(default=3, ge=0, le=3)
    timeout_seconds: StrictInt = Field(default=3600, ge=1, le=86_400)


class WorkflowClarificationRequest(ContractModel):
    answers: dict[str, str] = Field(min_length=1, max_length=20)

    @field_validator("answers")
    @classmethod
    def answers_must_be_nonempty(cls, values: dict[str, str]) -> dict[str, str]:
        cleaned = {key.strip(): value.strip() for key, value in values.items()}
        if any(not key or not value or len(value) > 2000 for key, value in cleaned.items()):
            raise ValueError("every clarification answer must be non-empty and at most 2000 characters")
        return cleaned


class WorkflowApprovalRequest(ContractModel):
    stage: Literal["plan", "final"]
    approved: StrictBool | None = None
    decision: Literal["approve", "reject", "request_changes"] | None = None
    comment: str = Field(default="", max_length=2000)
    task_feedback: list[TaskFeedback] = Field(default_factory=list, max_length=100)

    @model_validator(mode="before")
    @classmethod
    def normalize_decision(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        decision = data.get("decision")
        approved = data.get("approved")
        if decision is None and isinstance(approved, bool):
            data["decision"] = "approve" if approved else "reject"
        elif approved is None and decision in {"approve", "reject", "request_changes"}:
            data["approved"] = decision == "approve"
        return data

    @model_validator(mode="after")
    def validate_decision_and_feedback(self) -> "WorkflowApprovalRequest":
        if self.decision is None or self.approved is None:
            raise ValueError("an explicit approval decision is required")
        if self.approved != (self.decision == "approve"):
            raise ValueError("approved must agree with decision")
        feedback_ids = [item.task_id for item in self.task_feedback]
        if len(feedback_ids) != len(set(feedback_ids)):
            raise ValueError("task feedback must not repeat a task id")
        if self.stage == "plan" and self.task_feedback:
            raise ValueError("task-specific feedback is only valid for final review decisions")
        if self.stage == "final" and self.decision != "approve" and not self.task_feedback:
            raise ValueError("final rejection or change requests must identify affected tasks and feedback")
        if self.decision == "approve" and self.task_feedback:
            raise ValueError("approval cannot include task changes")
        return self


class WorkflowReviewBundle(ContractModel):
    workflow_id: UUID
    project_id: UUID
    stage: Literal["plan", "final"]
    state: Literal[WorkflowState.AWAITING_PLAN_APPROVAL, WorkflowState.AWAITING_FINAL_APPROVAL]
    plan_summary: str | None = None
    task_graph: TaskGraph | None = None
    code_diff: str = Field(default="", max_length=500_000)
    changed_files: list[str] = Field(default_factory=list, max_length=2000)
    diff_truncated: StrictBool = False
    review: ReviewerAgentResponse | None = None
    reviewer_summary: str | None = None


class WorkflowPackageExport(ContractModel):
    workflow_id: UUID
    project_id: UUID
    filename: str = Field(min_length=1, max_length=255)
    download_url: str = Field(min_length=1, max_length=500)
    size_bytes: StrictInt = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: list[str] = Field(min_length=1, max_length=2000)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def export_timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("export timestamp must include a timezone")
        return value


class WorkflowCancelRequest(ContractModel):
    reason: str = Field(default="Cancellation requested by user", min_length=1, max_length=2000)


class AgentUsageActivity(ContractModel):
    audit_run_id: UUID
    agent_name: AgentName
    action: str = Field(min_length=1, max_length=2000)
    timestamp: datetime
    provider: str | None = None
    model: str | None = None
    task_id: str | None = None
    tokens_used: StrictInt = Field(ge=0)
    spend: Decimal = Field(ge=Decimal("0"), max_digits=16, decimal_places=8)

    @field_validator("timestamp")
    @classmethod
    def activity_timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("activity timestamp must include a timezone")
        return value


class AIUsageDisclosure(ContractModel):
    project_id: UUID
    workflow_id: UUID
    agent_calls: StrictInt = Field(ge=0)
    stub_agent_calls: StrictInt = Field(default=0, ge=0)
    agent_names: list[AgentName] = Field(default_factory=list)
    audit_run_ids: list[UUID] = Field(default_factory=list)
    tokens_used: StrictInt = Field(ge=0)
    spend: Decimal = Field(ge=Decimal("0"), max_digits=16, decimal_places=8)
    llm_calls: StrictInt = Field(default=0, ge=0)
    code_execution_count: StrictInt = Field(default=0, ge=0)
    statement: str = Field(min_length=1, max_length=2000)
    generated_at: datetime
    activities: list[AgentUsageActivity] = Field(default_factory=list, max_length=10_000)

    @field_validator("generated_at")
    @classmethod
    def disclosure_timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value


class WorkflowRunRecord(ContractModel):
    id: UUID
    project_id: UUID
    task_graph_id: Optional[UUID] = None
    state: WorkflowState
    max_retries: StrictInt = Field(ge=0, le=3)
    timeout_seconds: StrictInt = Field(default=3600, ge=1, le=86_400)
    failure_reason: Optional[str] = None
    result_payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def run_timestamps_must_be_timezone_aware(self) -> "WorkflowRunRecord":
        if any(value.tzinfo is None or value.utcoffset() is None
               for value in (self.created_at, self.updated_at)):
            raise ValueError("workflow timestamps must include a timezone")
        return self


class WorkflowEventRecord(ContractModel):
    workflow_id: UUID
    sequence: StrictInt = Field(ge=1)
    from_state: Optional[WorkflowState] = None
    to_state: WorkflowState
    reason: str = Field(min_length=1, max_length=2000)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def event_timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value
