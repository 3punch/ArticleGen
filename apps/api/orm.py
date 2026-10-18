"""SQLAlchemy persistence models for ProjectForge control and workflow state."""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import UserDefinedType


JSON_DOCUMENT = JSON().with_variant(JSONB(), "postgresql")
VECTOR_DIMENSIONS = 384


class Vector384(UserDefinedType[list[float]]):
    """Native pgvector(384) with a JSON-backed SQLite test representation."""

    cache_ok = True

    def get_col_spec(self, **kw: Any) -> str:
        return f"vector({VECTOR_DIMENSIONS})"

    def bind_processor(self, dialect: Any):
        def process(value: Any) -> str | None:
            if value is None:
                return None
            vector = [float(item) for item in value]
            if len(vector) != VECTOR_DIMENSIONS or any(not math.isfinite(item) for item in vector):
                raise ValueError(f"embedding must contain {VECTOR_DIMENSIONS} finite values")
            return json.dumps(vector, separators=(",", ":"))
        return process

    def result_processor(self, dialect: Any, coltype: Any):
        def process(value: Any) -> list[float] | None:
            if value is None:
                return None
            decoded = json.loads(value) if isinstance(value, str) else value
            vector = [float(item) for item in decoded]
            if len(vector) != VECTOR_DIMENSIONS:
                raise ValueError(f"stored embedding must contain {VECTOR_DIMENSIONS} values")
            return vector
        return process


@compiles(Vector384, "sqlite")
def _compile_vector_sqlite(type_: Vector384, compiler: Any, **kw: Any) -> str:
    return "JSON"


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True,
                                           default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  server_default=func.now(), onupdate=func.now(),
                                                  nullable=False)
    raw_brief_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    rubric_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    spec: Mapped[Optional["ProjectSpecRow"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", uselist=False,
        passive_deletes=True,
    )
    budget: Mapped["BudgetLedgerRow"] = relationship(
        back_populates="project", cascade="all, delete-orphan", uselist=False,
        passive_deletes=True,
    )
    task_graphs: Mapped[list["TaskGraphRow"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True,
    )
    agent_runs: Mapped[list["AgentRunRow"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True,
    )
    review_results: Mapped[list["ReviewResultRow"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True,
    )
    workflow_runs: Mapped[list["WorkflowRunRow"]] = relationship(
        back_populates="project", passive_deletes="all",
    )
    context_chunks: Mapped[list["ProjectContextChunkRow"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True,
    )


class ProjectSpecRow(Base):
    __tablename__ = "project_specs"

    project_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True,
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    deliverables: Mapped[list[str]] = mapped_column(JSON_DOCUMENT, nullable=False)
    constraints: Mapped[list[str]] = mapped_column(JSON_DOCUMENT, nullable=False)
    tech_stack_hints: Mapped[list[str]] = mapped_column(JSON_DOCUMENT, nullable=False)
    rubric_items: Mapped[list[dict[str, Any]]] = mapped_column(JSON_DOCUMENT, nullable=False)

    project: Mapped[Project] = relationship(back_populates="spec")


class ProjectContextChunkRow(Base):
    """Retrieval-ready excerpts of a normalized spec, embedded in pgvector."""

    __tablename__ = "project_context_chunks"
    __table_args__ = (
        UniqueConstraint(
            "project_id", "chunk_kind", "chunk_key", "ordinal",
            name="uq_project_context_chunk_source",
        ),
        CheckConstraint("ordinal >= 0", name="ck_project_context_chunk_ordinal_nonnegative"),
        CheckConstraint(
            "chunk_kind IN ('title', 'description', 'deliverable', 'constraint', "
            "'tech_stack_hint', 'rubric_item')",
            name="ck_project_context_chunk_kind",
        ),
        Index("ix_project_context_chunks_project", "project_id", "chunk_kind"),
        Index(
            "ix_project_context_chunks_embedding_hnsw", "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True,
                                           default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False,
    )
    chunk_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    chunk_key: Mapped[str] = mapped_column(String(160), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector384(), nullable=False)
    metadata_payload: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON_DOCUMENT, nullable=False, default=dict, server_default=text("'{}'"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  server_default=func.now(), nullable=False)

    project: Mapped[Project] = relationship(back_populates="context_chunks")


class BudgetLedgerRow(Base):
    __tablename__ = "budget_ledgers"
    __table_args__ = (
        CheckConstraint("token_cap >= 0", name="ck_budget_token_cap_nonnegative"),
        CheckConstraint("used_tokens >= 0", name="ck_budget_used_tokens_nonnegative"),
        CheckConstraint("reserved_tokens >= 0", name="ck_budget_reserved_tokens_nonnegative"),
        CheckConstraint("spend_cap >= 0", name="ck_budget_spend_cap_nonnegative"),
        CheckConstraint("used_spend >= 0", name="ck_budget_used_spend_nonnegative"),
        CheckConstraint("reserved_spend >= 0", name="ck_budget_reserved_spend_nonnegative"),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True,
    )
    token_cap: Mapped[int] = mapped_column(BigInteger, nullable=False)
    spend_cap: Mapped[Decimal] = mapped_column(Numeric(16, 8), nullable=False)
    used_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0,
                                              server_default="0")
    used_spend: Mapped[Decimal] = mapped_column(Numeric(16, 8), nullable=False,
                                                 default=Decimal("0"), server_default="0")
    reserved_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0,
                                                  server_default="0")
    reserved_spend: Mapped[Decimal] = mapped_column(Numeric(16, 8), nullable=False,
                                                     default=Decimal("0"), server_default="0")

    project: Mapped[Project] = relationship(back_populates="budget")


class TaskGraphRow(Base):
    __tablename__ = "task_graphs"
    __table_args__ = (
        UniqueConstraint("project_id", "version", name="uq_task_graph_project_version"),
        CheckConstraint("version >= 1", name="ck_task_graphs_version_positive"),
        Index("ix_task_graphs_project_created", "project_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True,
                                           default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  server_default=func.now(), nullable=False)

    project: Mapped[Project] = relationship(back_populates="task_graphs")
    nodes: Mapped[list["TaskNodeRow"]] = relationship(
        back_populates="graph", cascade="all, delete-orphan", passive_deletes=True,
        order_by="TaskNodeRow.id",
    )
    edges: Mapped[list["TaskEdgeRow"]] = relationship(
        back_populates="graph", cascade="all, delete-orphan", passive_deletes=True,
    )


class TaskNodeRow(Base):
    __tablename__ = "task_nodes"
    __table_args__ = (
        CheckConstraint("retries >= 0", name="ck_task_nodes_retries_nonnegative"),
        CheckConstraint("timeout_seconds >= 1", name="ck_task_nodes_timeout_positive"),
        CheckConstraint("runtime IS NULL OR runtime IN ('python', 'node')",
                        name="ck_task_nodes_runtime"),
        CheckConstraint(
            "status IN ('pending', 'ready', 'running', 'blocked', 'completed', 'failed', 'cancelled')",
            name="ck_task_nodes_status",
        ),
        CheckConstraint(
            "assigned_agent IN ('requirement', 'planner', 'coder', 'tester', 'reviewer', 'debugger', 'packager')",
            name="ck_task_nodes_assigned_agent",
        ),
        ForeignKeyConstraint(["graph_id"], ["task_graphs.id"], ondelete="CASCADE"),
        Index("ix_task_nodes_graph_status", "graph_id", "status"),
    )

    graph_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    dependencies: Mapped[list[str]] = mapped_column(JSON_DOCUMENT, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending",
                                        server_default="pending")
    acceptance_criteria: Mapped[list[str]] = mapped_column(JSON_DOCUMENT, nullable=False)
    retries: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    assigned_agent: Mapped[str] = mapped_column(String(32), nullable=False)
    read_set: Mapped[list[str]] = mapped_column(JSON_DOCUMENT, nullable=False, default=list,
                                                server_default=text("'[]'"))
    write_set: Mapped[list[str]] = mapped_column(JSON_DOCUMENT, nullable=False, default=list,
                                                 server_default=text("'[]'"))
    runtime: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=300,
                                                  server_default="300")
    review_feedback: Mapped[list[str]] = mapped_column(
        JSON_DOCUMENT, nullable=False, default=list, server_default=text("'[]'"),
    )

    graph: Mapped[TaskGraphRow] = relationship(back_populates="nodes")


class TaskEdgeRow(Base):
    __tablename__ = "task_edges"
    __table_args__ = (
        CheckConstraint("source_id <> target_id", name="ck_task_edges_not_self_referential"),
        ForeignKeyConstraint(["graph_id"], ["task_graphs.id"], ondelete="CASCADE"),
        ForeignKeyConstraint(
            ["graph_id", "source_id"], ["task_nodes.graph_id", "task_nodes.id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["graph_id", "target_id"], ["task_nodes.graph_id", "task_nodes.id"],
            ondelete="CASCADE",
        ),
    )

    graph_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    target_id: Mapped[str] = mapped_column(String(64), primary_key=True)

    graph: Mapped[TaskGraphRow] = relationship(back_populates="edges")


class AgentRunRow(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        CheckConstraint("tokens_used >= 0", name="ck_agent_runs_tokens_nonnegative"),
        CheckConstraint("cost >= 0", name="ck_agent_runs_cost_nonnegative"),
        CheckConstraint(
            "agent_name IN ('requirement', 'planner', 'coder', 'tester', 'reviewer', 'debugger', 'packager')",
            name="ck_agent_runs_agent_name",
        ),
        Index("ix_agent_runs_project_timestamp", "project_id", "timestamp"),
        Index("ix_agent_runs_workflow_timestamp", "workflow_id", "timestamp"),
        UniqueConstraint("idempotency_key", name="uq_agent_runs_idempotency_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True,
                                           default=uuid.uuid4)
    agent_name: Mapped[str] = mapped_column(String(32), nullable=False)
    input_payload: Mapped[dict[str, Any]] = mapped_column("input", JSON_DOCUMENT, nullable=False)
    output_payload: Mapped[dict[str, Any]] = mapped_column("output", JSON_DOCUMENT, nullable=False)
    tokens_used: Mapped[int] = mapped_column(BigInteger, nullable=False)
    cost: Mapped[Decimal] = mapped_column(Numeric(16, 8), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    provider: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    project_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False,
    )
    workflow_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("workflow_runs.id", ondelete="SET NULL"), nullable=True,
    )
    budget_reservation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("budget_reservations.id", ondelete="SET NULL"), nullable=True,
    )

    project: Mapped[Project] = relationship(back_populates="agent_runs")


class ReviewResultRow(Base):
    __tablename__ = "review_results"
    __table_args__ = (
        CheckConstraint(
            "rubric_type IN ('automatable', 'human-judgment')",
            name="ck_review_results_rubric_type",
        ),
        CheckConstraint(
            "status IN ('passed', 'failed', 'not_verified', 'human_review_required')",
            name="ck_review_results_status",
        ),
        CheckConstraint(
            "(status = 'passed' AND rubric_type = 'automatable' AND passed IS TRUE) OR "
            "(status = 'failed' AND rubric_type = 'automatable' AND passed IS FALSE) OR "
            "(status = 'not_verified' AND passed IS NULL) OR "
            "(status = 'human_review_required' AND rubric_type = 'human-judgment' AND passed IS NULL)",
            name="ck_review_results_verdict_consistency",
        ),
        Index("ix_review_results_project", "project_id"),
        Index("ix_review_results_workflow", "workflow_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True,
                                           default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False,
    )
    workflow_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("workflow_runs.id", ondelete="SET NULL"), nullable=True,
    )
    rubric_item: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    rubric_type: Mapped[str] = mapped_column(String(24), nullable=False,
                                            default="automatable", server_default="automatable")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="not_verified",
                                        server_default="not_verified")
    passed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    recommendation: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    evidence: Mapped[list[str]] = mapped_column(JSON_DOCUMENT, nullable=False)
    reviewer_notes: Mapped[str] = mapped_column(Text, nullable=False, default="",
                                                 server_default="")

    project: Mapped[Project] = relationship(back_populates="review_results")


class WorkflowRunRow(Base):
    __tablename__ = "workflow_runs"
    __table_args__ = (
        CheckConstraint(
            "state IN ('INTAKE', 'SPEC', 'PLAN', 'AWAITING_PLAN_APPROVAL', 'AWAITING_CLARIFICATION', "
            "'BUILD_LOOP', 'INTEGRATION', 'REVIEW', 'AWAITING_FINAL_APPROVAL', 'PACKAGE', 'DONE', 'FAILED', 'ESCALATED')",
            name="ck_workflow_runs_state",
        ),
        CheckConstraint("max_retries BETWEEN 0 AND 3", name="ck_workflow_runs_max_retries"),
        CheckConstraint("timeout_seconds >= 1", name="ck_workflow_runs_timeout_positive"),
        UniqueConstraint("task_graph_id", name="uq_workflow_run_task_graph"),
        Index("ix_workflow_runs_project_created", "project_id", "created_at"),
        Index("ix_workflow_runs_state_updated", "state", "updated_at"),
        Index("ix_workflow_runs_lease_expiry", "lease_expires_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True,
                                           default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("projects.id", ondelete="RESTRICT"), nullable=False,
    )
    task_graph_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("task_graphs.id", ondelete="RESTRICT"), nullable=True,
    )
    state: Mapped[str] = mapped_column(String(32), nullable=False, default="INTAKE",
                                        server_default="INTAKE")
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=3,
                                              server_default="3")
    timeout_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=3600,
                                                  server_default="3600")
    failure_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_payload: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False,
                                                           default=dict,
                                                           server_default=text("'{}'"))
    cancel_requested_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True),
                                                                    nullable=True)
    lease_owner: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True),
                                                                  nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  server_default=func.now(), onupdate=func.now(),
                                                  nullable=False)

    project: Mapped[Project] = relationship(back_populates="workflow_runs")
    events: Mapped[list["WorkflowEventRow"]] = relationship(
        back_populates="workflow", cascade="all, delete-orphan", passive_deletes=True,
        order_by="WorkflowEventRow.sequence",
    )


class BudgetReservationRow(Base):
    """Durable pre-call budget hold, settled or released after the provider response."""

    __tablename__ = "budget_reservations"
    __table_args__ = (
        CheckConstraint("reserved_tokens >= 0", name="ck_budget_reservations_tokens_nonnegative"),
        CheckConstraint("reserved_spend >= 0", name="ck_budget_reservations_spend_nonnegative"),
        CheckConstraint(
            "state IN ('reserved', 'settled', 'released', 'expired')",
            name="ck_budget_reservations_state",
        ),
        Index("ix_budget_reservations_project_state_expiry",
              "project_id", "state", "expires_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True,
                                           default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("budget_ledgers.project_id", ondelete="CASCADE"),
        nullable=False,
    )
    workflow_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("workflow_runs.id", ondelete="SET NULL"), nullable=True,
    )
    reserved_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False)
    reserved_spend: Mapped[Decimal] = mapped_column(Numeric(16, 8), nullable=False)
    actual_tokens: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    actual_spend: Mapped[Optional[Decimal]] = mapped_column(Numeric(16, 8), nullable=True)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="reserved",
                                        server_default="reserved")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  server_default=func.now(), nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class WorkflowEventRow(Base):
    """Append-only audit event recording every workflow transition/approval/cancel."""

    __tablename__ = "workflow_events"
    __table_args__ = (
        CheckConstraint("sequence >= 1", name="ck_workflow_events_sequence_positive"),
        CheckConstraint(
            "to_state IN ('INTAKE', 'SPEC', 'PLAN', 'AWAITING_PLAN_APPROVAL', 'AWAITING_CLARIFICATION', "
            "'BUILD_LOOP', 'INTEGRATION', 'REVIEW', 'AWAITING_FINAL_APPROVAL', 'PACKAGE', 'DONE', 'FAILED', 'ESCALATED')",
            name="ck_workflow_events_to_state",
        ),
        Index("ix_workflow_events_created", "workflow_id", "created_at"),
    )

    workflow_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("workflow_runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_state: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    to_state: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str] = mapped_column(String(2000), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False,
                                                    default=dict,
                                                    server_default=text("'{}'"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                  server_default=func.now(), nullable=False)

    workflow: Mapped[WorkflowRunRow] = relationship(back_populates="events")
