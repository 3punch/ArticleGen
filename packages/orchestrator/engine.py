"""Durable, deterministic FSM orchestration backed by the ProjectForge database.

Control flow is code-only. Requirement and Planner may call a configured model, but
clarification, approvals, retries, DAG scheduling, budgets, and state transitions are
validated and persisted deterministically.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable
from uuid import UUID, uuid4
import time

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from projectforge_agents.config import ModelConfigurationError
from projectforge_agents.errors import AgentOutputValidationError
from projectforge_agents.litellm_client import PromptTokenLimitExceeded
from projectforge_agents.rag import PgVectorContextRetriever
from projectforge_agents.results import AgentInvocationResult
from projectforge_agents.stubs import STUB_AGENT_SCOPES, StubAgentSuite
from projectforge_agents.suite import ProjectForgeAgentSuite
from projectforge_api.orm import (
    AgentRunRow,
    Project,
    ProjectSpecRow,
    ReviewResultRow,
    TaskEdgeRow,
    TaskGraphRow,
    TaskNodeRow,
    WorkflowEventRow,
    WorkflowRunRow,
)
from projectforge_schemas.models import (
    AgentFileChange,
    AIUsageDisclosure,
    AgentUsageActivity,
    PlannerAgentResponse,
    ProjectSpec,
    ReportGroundednessResult,
    RequirementAgentResponse,
    ReviewerAgentResponse,
    TaskFeedback,
    TaskGraph,
    WorkflowEventRecord,
    WorkflowPackageExport,
    WorkflowReviewBundle,
    WorkflowRunRecord,
    WorkflowState,
)

from .budget import BudgetExceeded, BudgetGuard
from .scheduler import TaskScheduler


TERMINAL_STATES = {WorkflowState.DONE, WorkflowState.FAILED, WorkflowState.ESCALATED}
WAITING_STATES = {
    WorkflowState.AWAITING_PLAN_APPROVAL,
    WorkflowState.AWAITING_FINAL_APPROVAL,
    WorkflowState.AWAITING_CLARIFICATION,
}
ALLOWED_TRANSITIONS: dict[WorkflowState, set[WorkflowState]] = {
    WorkflowState.INTAKE: {WorkflowState.SPEC},
    WorkflowState.SPEC: {WorkflowState.PLAN, WorkflowState.AWAITING_CLARIFICATION},
    WorkflowState.PLAN: {WorkflowState.AWAITING_PLAN_APPROVAL},
    WorkflowState.AWAITING_CLARIFICATION: {WorkflowState.SPEC},
    WorkflowState.AWAITING_PLAN_APPROVAL: {WorkflowState.BUILD_LOOP, WorkflowState.PLAN},
    WorkflowState.BUILD_LOOP: {WorkflowState.INTEGRATION},
    WorkflowState.INTEGRATION: {WorkflowState.REVIEW},
    WorkflowState.REVIEW: {WorkflowState.AWAITING_FINAL_APPROVAL},
    WorkflowState.AWAITING_FINAL_APPROVAL: {WorkflowState.PACKAGE, WorkflowState.BUILD_LOOP},
    WorkflowState.PACKAGE: {WorkflowState.DONE},
    WorkflowState.DONE: set(),
    WorkflowState.FAILED: set(),
    WorkflowState.ESCALATED: set(),
}


class WorkflowError(RuntimeError):
    pass


class AgentOutputValidationFailure(WorkflowError):
    """A model response failed its schema/grounding gate and may be regenerated."""


class InvalidWorkflowTransition(WorkflowError):
    pass


class RetryLimitExceeded(WorkflowError):
    pass


class WorkflowTimedOut(WorkflowError):
    pass


class WorkflowCancelled(WorkflowError):
    pass


class WorkflowLeaseLost(WorkflowError):
    pass


@runtime_checkable
class WorkflowEngine(Protocol):
    """Swappable workflow engine contract (Postgres FSM now; Temporal later)."""

    def start(
        self,
        project_id: UUID,
        task_graph_id: UUID | None = None,
        *,
        raw_brief_text: str | None = None,
        rubric_text: str | None = None,
        max_retries: int = 3,
        timeout_seconds: int = 3600,
    ) -> WorkflowRunRecord: ...

    def submit_clarification(
        self, workflow_id: UUID, *, answers: Mapping[str, str],
    ) -> WorkflowRunRecord: ...

    def drive(self, workflow_id: UUID) -> WorkflowRunRecord: ...

    def get(self, workflow_id: UUID) -> WorkflowRunRecord: ...

    def list_events(self, workflow_id: UUID) -> list[WorkflowEventRecord]: ...

    def approve(
        self,
        workflow_id: UUID,
        *,
        stage: Literal["plan", "final"],
        approved: bool | None = None,
        decision: Literal["approve", "reject", "request_changes"] | None = None,
        comment: str = "",
        task_feedback: Sequence[TaskFeedback | Mapping[str, str]] = (),
    ) -> WorkflowRunRecord: ...

    def review_bundle(self, workflow_id: UUID) -> WorkflowReviewBundle: ...

    def cancel(self, workflow_id: UUID, *, reason: str) -> WorkflowRunRecord: ...


class PostgresFSMEngine:
    """Database-backed deterministic state machine with audited transitions."""

    _lease_seconds = 60

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        agents: Any | None = None,
        max_parallel_tasks: int = 4,
    ) -> None:
        self._session_factory = session_factory
        # The production suite gets the same session factory for pgvector retrieval
        # and project persistence. Tests that want a fully offline FSM inject stubs.
        self.agents = agents if agents is not None else ProjectForgeAgentSuite(
            session_factory=session_factory,
        )
        self.workspace_manager = getattr(self.agents, "workspace_manager", None)
        self.package_manager = getattr(self.agents, "package_manager", None)
        self.context_retriever = PgVectorContextRetriever(session_factory)
        configured_model_timeout = int(getattr(getattr(self.agents, "config", None), "timeout_seconds", 60))
        self._lease_seconds = max(120, configured_model_timeout + 30)
        self.budget_guard = BudgetGuard(session_factory)
        self.scheduler = TaskScheduler(max_parallel_tasks=max_parallel_tasks)

    def start(
        self,
        project_id: UUID,
        task_graph_id: UUID | None = None,
        *,
        raw_brief_text: str | None = None,
        rubric_text: str | None = None,
        max_retries: int = 3,
        timeout_seconds: int = 3600,
    ) -> WorkflowRunRecord:
        if not 0 <= max_retries <= 3:
            raise ValueError("max_retries must be between 0 and 3")
        if not 1 <= timeout_seconds <= 86_400:
            raise ValueError("timeout_seconds must be between 1 and 86400")
        now = self._now()
        with self._session_factory() as session, session.begin():
            project = session.scalar(
                select(Project).where(Project.id == project_id).with_for_update()
            )
            if project is None:
                raise LookupError("project not found")
            if task_graph_id is not None:
                graph_row = session.scalar(
                    select(TaskGraphRow).where(TaskGraphRow.id == task_graph_id).with_for_update()
                )
                if graph_row is None or graph_row.project_id != project_id:
                    raise LookupError("seed task graph not found for project")
            active_run = session.scalar(
                select(WorkflowRunRow.id).where(
                    WorkflowRunRow.project_id == project_id,
                    WorkflowRunRow.state.not_in([state.value for state in TERMINAL_STATES]),
                ).limit(1)
            )
            if active_run is not None:
                raise InvalidWorkflowTransition("project already has an active workflow")

            spec_row = session.get(ProjectSpecRow, project_id)
            source_spec = None
            if spec_row is not None:
                source_spec = {
                    "title": spec_row.title,
                    "description": spec_row.description,
                    "deliverables": spec_row.deliverables,
                    "constraints": spec_row.constraints,
                    "tech_stack_hints": spec_row.tech_stack_hints,
                    "rubric_items": spec_row.rubric_items,
                }
                ProjectSpec.model_validate(source_spec)
            brief = (raw_brief_text or project.raw_brief_text or "").strip()
            if not brief:
                if source_spec is None:
                    raise ValueError("workflow requires raw brief text or a persisted ProjectSpec")
                # Compatibility for existing projects created before raw brief intake.
                # New requests should supply/store the original assignment text.
                brief = (
                    "Legacy normalized ProjectSpec supplied as the intake source:\n"
                    + json.dumps(source_spec, ensure_ascii=False, sort_keys=True)
                )
            if len(brief) < 20:
                raise ValueError("raw brief text must contain at least 20 characters")
            rubric = rubric_text if rubric_text is not None else project.rubric_text
            input_snapshot = {
                "raw_brief_text": brief,
                "rubric_text": rubric,
                "source_project_spec": source_spec,
                "clarification_answers": {},
                "requirement_revision": 0,
                "seed_task_graph_id": str(task_graph_id) if task_graph_id else None,
            }
            run = WorkflowRunRow(
                id=uuid4(),
                project_id=project_id,
                task_graph_id=None,
                state=WorkflowState.INTAKE.value,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
                result_payload={"input_snapshot": input_snapshot},
                created_at=now,
                updated_at=now,
            )
            session.add(run)
            session.flush()
            session.add(WorkflowEventRow(
                workflow_id=run.id,
                sequence=1,
                from_state=None,
                to_state=WorkflowState.INTAKE.value,
                reason="Workflow created; raw brief persisted for Requirement-agent processing.",
                payload={
                    "max_retries": max_retries,
                    "timeout_seconds": timeout_seconds,
                    "input_mode": "raw_brief" if raw_brief_text or project.raw_brief_text else "legacy_spec",
                    "seed_task_graph_id": str(task_graph_id) if task_graph_id else None,
                },
                created_at=now,
            ))
        return self.get(UUID(str(run.id)))

    def submit_clarification(
        self, workflow_id: UUID, *, answers: Mapping[str, str],
    ) -> WorkflowRunRecord:
        cleaned = {str(key).strip(): str(value).strip() for key, value in answers.items()}
        if not cleaned or any(not key or not value or len(value) > 2000 for key, value in cleaned.items()):
            raise ValueError("every clarification answer must be non-empty and at most 2000 characters")
        if len(cleaned) > 20:
            raise ValueError("at most 20 clarification answers may be submitted at once")
        now = self._now()
        with self._session_factory() as session, session.begin():
            row = session.scalar(
                select(WorkflowRunRow).where(WorkflowRunRow.id == workflow_id).with_for_update()
            )
            if row is None:
                raise LookupError("workflow not found")
            if WorkflowState(row.state) is not WorkflowState.AWAITING_CLARIFICATION:
                raise InvalidWorkflowTransition("workflow is not awaiting clarification")
            if row.cancel_requested_at is not None:
                raise InvalidWorkflowTransition("clarification cannot resume a cancelled workflow")
            payload = dict(row.result_payload or {})
            requirement_output = payload.get("requirement_response") or {}
            questions = requirement_output.get("questions") or payload.get("pending_questions") or []
            expected_ids = {
                str(question.get("question_id"))
                for question in questions if isinstance(question, dict)
            }
            if not expected_ids:
                raise WorkflowError("clarification questions are missing from workflow state")
            if set(cleaned) != expected_ids:
                missing = sorted(expected_ids - set(cleaned))
                unexpected = sorted(set(cleaned) - expected_ids)
                details = []
                if missing:
                    details.append(f"missing answers for: {', '.join(missing)}")
                if unexpected:
                    details.append(f"unexpected question ids: {', '.join(unexpected)}")
                raise ValueError("; ".join(details))
            input_snapshot = dict(payload.get("input_snapshot") or {})
            all_answers = dict(input_snapshot.get("clarification_answers") or {})
            all_answers.update(cleaned)
            input_snapshot["clarification_answers"] = all_answers
            input_snapshot["requirement_revision"] = int(input_snapshot.get("requirement_revision", 0)) + 1
            history = list(payload.get("clarification_history") or [])
            history.append({"question_ids": sorted(expected_ids), "answers": cleaned})
            row.result_payload = {
                **payload,
                "input_snapshot": input_snapshot,
                "clarification_history": history,
                "pending_questions": [],
            }
            row.state = WorkflowState.SPEC.value
            row.updated_at = now
            self._append_event(
                session,
                row,
                from_state=WorkflowState.AWAITING_CLARIFICATION,
                to_state=WorkflowState.SPEC,
                reason="User supplied the requested clarification; Requirement agent may resume.",
                payload={"answer_ids": sorted(cleaned),
                         "requirement_revision": input_snapshot["requirement_revision"]},
                created_at=now,
            )
        return self.get(workflow_id)

    def get(self, workflow_id: UUID) -> WorkflowRunRecord:
        with self._session_factory() as session:
            row = session.get(WorkflowRunRow, workflow_id)
            if row is None:
                raise LookupError("workflow not found")
            return self._workflow_record(row)

    def list_events(self, workflow_id: UUID) -> list[WorkflowEventRecord]:
        with self._session_factory() as session:
            if session.get(WorkflowRunRow, workflow_id) is None:
                raise LookupError("workflow not found")
            rows = session.scalars(
                select(WorkflowEventRow)
                .where(WorkflowEventRow.workflow_id == workflow_id)
                .order_by(WorkflowEventRow.sequence)
            ).all()
            return [self._event_record(row) for row in rows]

    def approve(
        self,
        workflow_id: UUID,
        *,
        stage: Literal["plan", "final"],
        approved: bool | None = None,
        decision: Literal["approve", "reject", "request_changes"] | None = None,
        comment: str = "",
        task_feedback: Sequence[TaskFeedback | Mapping[str, str]] = (),
    ) -> WorkflowRunRecord:
        if decision is None and isinstance(approved, bool):
            decision = "approve" if approved else "reject"
        if decision not in {"approve", "reject", "request_changes"}:
            raise ValueError("an explicit approval decision is required")
        expected_approved = decision == "approve"
        if approved is not None and approved is not expected_approved:
            raise ValueError("approved must agree with decision")
        if decision == "approve" and task_feedback:
            raise ValueError("approval cannot include task-specific feedback")
        if stage == "plan" and task_feedback:
            raise ValueError("task-specific feedback is only valid for final review decisions")
        normalized_feedback = [
            item if isinstance(item, TaskFeedback) else TaskFeedback.model_validate(item)
            for item in task_feedback
        ]
        feedback_ids = [item.task_id for item in normalized_feedback]
        if len(feedback_ids) != len(set(feedback_ids)):
            raise ValueError("task feedback must not repeat a task id")
        if stage == "final" and decision != "approve" and not normalized_feedback:
            raise ValueError("final rejection or change requests must identify affected tasks and feedback")

        expected_state = (
            WorkflowState.AWAITING_PLAN_APPROVAL
            if stage == "plan"
            else WorkflowState.AWAITING_FINAL_APPROVAL
        )
        should_cancel = False
        with self._session_factory() as session, session.begin():
            run = session.scalar(
                select(WorkflowRunRow).where(WorkflowRunRow.id == workflow_id).with_for_update()
            )
            if run is None:
                raise LookupError("workflow not found")
            current = WorkflowState(run.state)
            if current is not expected_state:
                raise InvalidWorkflowTransition(
                    f"expected {expected_state.value}, found {current.value}"
                )
            now = self._now()
            result_payload = dict(run.result_payload or {})
            reason: str
            failure_reason: str | None = None
            if stage == "plan":
                if decision == "approve":
                    next_state = WorkflowState.BUILD_LOOP
                    reason = "Human approved the plan gate; the deterministic build loop may begin."
                else:
                    next_state = WorkflowState.PLAN
                    reason = f"Human requested plan changes ({decision}); code remains gated."
                    result_payload["plan_revision"] = int(result_payload.get("plan_revision", 0)) + 1
                    feedback_history = list(result_payload.get("plan_feedback", []))
                    if comment.strip():
                        feedback_history.append(comment.strip())
                    result_payload["plan_feedback"] = feedback_history[-20:]
                result_payload["plan_gate_decision"] = {
                    "decision": decision,
                    "comment": comment.strip(),
                    "timestamp": now.isoformat(),
                }
            elif decision == "approve":
                next_state = WorkflowState.PACKAGE
                reason = "Human approved final review; the mandatory packaging gate is satisfied."
                result_payload["final_gate_decision"] = {
                    "decision": decision,
                    "approved": True,
                    "comment": comment.strip(),
                    "timestamp": now.isoformat(),
                    "task_feedback": [],
                }
                final_history = list(result_payload.get("final_gate_history", []))
                final_history.append(result_payload["final_gate_decision"])
                result_payload["final_gate_history"] = final_history[-20:]
            else:
                if run.task_graph_id is None:
                    raise WorkflowError("final review rejection requires a persisted task graph")
                node_rows = session.scalars(
                    select(TaskNodeRow).where(TaskNodeRow.graph_id == run.task_graph_id)
                    .with_for_update()
                ).all()
                rows_by_id = {node.id: node for node in node_rows}
                unknown = sorted(set(feedback_ids) - set(rows_by_id))
                if unknown:
                    raise ValueError(f"task feedback references unknown task ids: {', '.join(unknown)}")
                incomplete = sorted(
                    node_id for node_id in feedback_ids
                    if rows_by_id[node_id].status != "completed"
                )
                if incomplete:
                    raise InvalidWorkflowTransition(
                        "final review feedback can only target completed tasks: "
                        + ", ".join(incomplete)
                    )
                descendants: dict[str, set[str]] = {node_id: set() for node_id in rows_by_id}
                edge_rows = session.scalars(
                    select(TaskEdgeRow).where(TaskEdgeRow.graph_id == run.task_graph_id)
                ).all()
                for edge in edge_rows:
                    descendants.setdefault(edge.source_id, set()).add(edge.target_id)
                affected = set(feedback_ids)
                frontier = list(feedback_ids)
                while frontier:
                    source = frontier.pop()
                    for target in sorted(descendants.get(source, set())):
                        if target not in affected:
                            affected.add(target)
                            frontier.append(target)
                affected_rows = [rows_by_id[node_id] for node_id in sorted(affected)]
                exhausted = [node.id for node in affected_rows if node.retries >= run.max_retries]
                feedback_by_id = {item.task_id: item.feedback for item in normalized_feedback}
                for node in affected_rows:
                    if node.id in feedback_by_id:
                        note = f"Human review: {feedback_by_id[node.id]}"
                    else:
                        parents = sorted(
                            source for source, targets in descendants.items()
                            if node.id in targets and source in feedback_ids
                        )
                        note = (
                            "Re-run because upstream review rework changed dependencies: "
                            + ", ".join(parents or feedback_ids)
                        )
                    notes = list(node.review_feedback or [])
                    notes.append(note[:2000])
                    node.review_feedback = notes[-20:]
                    if not exhausted:
                        node.retries += 1
                        node.status = "pending"
                result_payload["review_revision"] = int(result_payload.get("review_revision", 0)) + 1
                final_entry = {
                    "decision": decision,
                    "approved": False,
                    "comment": comment.strip(),
                    "timestamp": now.isoformat(),
                    "task_feedback": [item.model_dump(mode="json") for item in normalized_feedback],
                    "affected_task_ids": sorted(affected),
                }
                final_history = list(result_payload.get("final_gate_history", []))
                final_history.append(final_entry)
                result_payload["final_gate_history"] = final_history[-20:]
                result_payload["final_gate_decision"] = final_entry
                if exhausted:
                    next_state = WorkflowState.ESCALATED
                    reason = (
                        "Human review rework exceeded the configured retry budget for task(s): "
                        + ", ".join(exhausted)
                    )
                    failure_reason = reason
                    should_cancel = True
                else:
                    next_state = WorkflowState.BUILD_LOOP
                    reason = (
                        f"Human requested changes to {len(feedback_ids)} task(s); affected tasks "
                        "and their dependents were reset for the bounded build loop."
                    )

            transition_allowed = (
                current not in TERMINAL_STATES
                if next_state in TERMINAL_STATES
                else next_state in ALLOWED_TRANSITIONS[current]
            )
            if not transition_allowed:
                raise InvalidWorkflowTransition(
                    f"transition {current.value} -> {next_state.value} is not allowed"
                )
            run.state = next_state.value
            run.updated_at = now
            run.result_payload = result_payload
            run.failure_reason = failure_reason
            self._append_event(
                session,
                run,
                from_state=current,
                to_state=next_state,
                reason=reason,
                payload={
                    "approval_stage": stage,
                    "decision": decision,
                    "approved": expected_approved,
                    "comment": comment.strip(),
                    "task_feedback": [item.model_dump(mode="json") for item in normalized_feedback],
                    "affected_task_ids": sorted(affected) if stage == "final" and decision != "approve" else [],
                },
                created_at=now,
            )
        result = self.get(workflow_id)
        if should_cancel:
            self._cancel_incomplete_tasks(result.task_graph_id)
        return result

    def review_bundle(self, workflow_id: UUID) -> WorkflowReviewBundle:
        run = self.get(workflow_id)
        if run.state not in {WorkflowState.AWAITING_PLAN_APPROVAL, WorkflowState.AWAITING_FINAL_APPROVAL}:
            raise InvalidWorkflowTransition("workflow is not waiting at a human review gate")
        stage: Literal["plan", "final"] = (
            "plan" if run.state is WorkflowState.AWAITING_PLAN_APPROVAL else "final"
        )
        graph = None
        if run.task_graph_id is not None:
            graph = self._load_project_graph(run.project_id, run.task_graph_id)
        diff, changed_files, truncated = ("", [], False)
        if stage == "final" and self.workspace_manager is not None:
            diff, changed_files, truncated = self.workspace_manager.review_diff(run.project_id)
        payload = run.result_payload or {}
        reviewer = None
        reviewer_summary = None
        if stage == "final":
            raw_review = payload.get("review")
            if isinstance(raw_review, Mapping):
                try:
                    reviewer = ReviewerAgentResponse.model_validate({
                        key: raw_review[key]
                        for key in (
                            "summary", "criteria", "automatable_checks_passed",
                            "human_review_required", "evidence",
                        )
                        if key in raw_review
                    })
                    reviewer_summary = reviewer.summary
                except ValueError:
                    reviewer_summary = str(raw_review.get("summary", "Legacy review output is not schema-shaped."))
        grounding_payload = payload.get("report_groundedness")
        report_groundedness = (
            ReportGroundednessResult.model_validate(grounding_payload)
            if stage == "final" and isinstance(grounding_payload, Mapping)
            else None
        )
        return WorkflowReviewBundle(
            workflow_id=workflow_id,
            project_id=run.project_id,
            stage=stage,
            state=run.state,
            plan_summary=payload.get("plan_summary"),
            task_graph=graph,
            code_diff=diff,
            changed_files=changed_files,
            diff_truncated=truncated,
            review=reviewer,
            reviewer_summary=reviewer_summary,
            report_groundedness=report_groundedness,
        )

    def cancel(self, workflow_id: UUID, *, reason: str) -> WorkflowRunRecord:
        with self._session_factory() as session, session.begin():
            row = session.scalar(
                select(WorkflowRunRow).where(WorkflowRunRow.id == workflow_id).with_for_update()
            )
            if row is None:
                raise LookupError("workflow not found")
            state = WorkflowState(row.state)
            if state in TERMINAL_STATES:
                raise InvalidWorkflowTransition("terminal workflows cannot be cancelled")
            now = self._now()
            row.cancel_requested_at = now
            row.updated_at = now
            self._append_event(
                session,
                row,
                from_state=state,
                to_state=state,
                reason="Cancellation requested.",
                payload={"reason": reason},
                created_at=now,
            )
        return self.get(workflow_id)

    def drive(self, workflow_id: UUID) -> WorkflowRunRecord:
        owner = str(uuid4())
        if not self._claim(workflow_id, owner):
            return self.get(workflow_id)
        try:
            while True:
                self._renew_lease(workflow_id, owner)
                row = self._load_workflow(workflow_id)
                state = WorkflowState(row.state)
                if state in TERMINAL_STATES:
                    return self.get(workflow_id)
                if state in WAITING_STATES:
                    if row.cancel_requested_at is not None:
                        self._terminalize(
                            workflow_id, WorkflowState.FAILED,
                            reason=(
                                "workflow cancellation requested while awaiting clarification"
                                if state is WorkflowState.AWAITING_CLARIFICATION
                                else "workflow cancellation requested while awaiting approval"
                            ),
                            owner=owner,
                        )
                    return self.get(workflow_id)
                try:
                    self._check_cancelled(workflow_id)
                    self._check_workflow_timeout(row)
                    if state is WorkflowState.INTAKE:
                        input_snapshot = self._workflow_input(row)
                        if len(str(input_snapshot.get("raw_brief_text", "")).strip()) < 20:
                            raise WorkflowError("persisted raw brief is missing or too short")
                        self._transition(
                            workflow_id, WorkflowState.SPEC,
                            reason="Raw brief and optional rubric snapshot validated for Requirement processing.",
                            expected_state=WorkflowState.INTAKE,
                            owner=owner,
                        )
                    elif state is WorkflowState.SPEC:
                        input_snapshot = self._workflow_input(row)
                        revision = int(input_snapshot.get("requirement_revision", 0))
                        result = self._call_agent(
                            workflow_id,
                            row.project_id,
                            "requirement",
                            {
                                "raw_brief_text": input_snapshot.get("raw_brief_text", ""),
                                "rubric_text": input_snapshot.get("rubric_text"),
                                "clarification_answers": input_snapshot.get("clarification_answers", {}),
                                "source_project_spec": input_snapshot.get("source_project_spec"),
                            },
                            call_key=f"{workflow_id}:requirement:{revision}",
                        )
                        response = self._requirement_response(result.output)
                        if response.status == "needs_clarification":
                            response_data = response.model_dump(mode="json")
                            self._transition(
                                workflow_id, WorkflowState.AWAITING_CLARIFICATION,
                                reason="Requirement agent identified critical ambiguity; awaiting user answers.",
                                expected_state=WorkflowState.SPEC,
                                owner=owner,
                                payload={"question_ids": [q.question_id for q in response.questions]},
                                result_updates={
                                    "requirement_response": response_data,
                                    "pending_questions": [q.model_dump(mode="json") for q in response.questions],
                                },
                            )
                        else:
                            project_spec = response.project_spec
                            if project_spec is None:
                                raise WorkflowError("completed Requirement response has no ProjectSpec")
                            self._transition(
                                workflow_id, WorkflowState.PLAN,
                                reason="Requirement agent produced a schema-validated ProjectSpec.",
                                expected_state=WorkflowState.SPEC,
                                owner=owner,
                                result_updates={
                                    "normalized_spec": project_spec.model_dump(mode="json"),
                                    "requirement_response": result.output,
                                },
                                persist_project_spec=project_spec,
                            )
                    elif state is WorkflowState.PLAN:
                        project_spec = self._workflow_spec(row)
                        planner_feedback = (row.result_payload or {}).get("plan_feedback", [])
                        planner_input: dict[str, Any] = {
                            "spec": project_spec.model_dump(mode="json"),
                            "plan_feedback": [str(item)[:2000] for item in planner_feedback[-20:]],
                        }
                        input_snapshot = self._workflow_input(row)
                        seed_graph_id = input_snapshot.get("seed_task_graph_id")
                        if seed_graph_id and isinstance(self.agents, StubAgentSuite):
                            seed_graph = self._load_project_graph(row.project_id, UUID(seed_graph_id))
                            planner_input["seed_graph"] = seed_graph.model_dump(mode="json")
                        result = self._call_agent(
                            workflow_id,
                            row.project_id,
                            "planner",
                            planner_input,
                            call_key=(
                                f"{workflow_id}:planner:v{int((row.result_payload or {}).get('plan_revision', 0))}"
                            ),
                        )
                        plan = self._planner_response(result.output)
                        self._persist_planned_graph(
                            workflow_id,
                            plan,
                            planner_output=result.output,
                            owner=owner,
                        )
                    elif state is WorkflowState.BUILD_LOOP:
                        self._run_build_loop(row, workflow_id, owner)
                        self._transition(
                            workflow_id, WorkflowState.INTEGRATION,
                            reason="All dependency-ready tasks completed their CODE -> TEST loops.",
                            expected_state=WorkflowState.BUILD_LOOP,
                            owner=owner,
                        )
                    elif state is WorkflowState.INTEGRATION:
                        result = self._call_agent(
                            workflow_id,
                            row.project_id,
                            "tester",
                            {"phase": "INTEGRATION", "task_graph_id": str(row.task_graph_id)},
                        )
                        if not result.output.get("passed", False):
                            raise WorkflowError("integration stub reported a failure")
                        self._transition(
                            workflow_id, WorkflowState.REVIEW,
                            reason="Deterministic integration checks passed (no code executed).",
                            expected_state=WorkflowState.INTEGRATION,
                            owner=owner,
                            result_updates={"integration": result.output},
                        )
                    elif state is WorkflowState.REVIEW:
                        project_spec = self._workflow_spec(row)
                        if row.task_graph_id is None:
                            raise WorkflowError("review cannot run without the persisted task graph")
                        review_revision = int((row.result_payload or {}).get("review_revision", 0))
                        task_graph = self._load_project_graph(row.project_id, row.task_graph_id)
                        test_runs = self._workflow_test_runs(workflow_id)
                        result = self._call_agent(
                            workflow_id,
                            row.project_id,
                            "reviewer",
                            {
                                "project_spec": project_spec.model_dump(mode="json"),
                                "task_graph_id": str(row.task_graph_id),
                                "task_graph": task_graph.model_dump(mode="json"),
                                "test_runs": test_runs,
                            },
                            call_key=f"{workflow_id}:review:v{review_revision}",
                        )
                        review_response = self._persist_review_results(
                            workflow_id, row.project_id, project_spec, result,
                        )
                        review_passed = (
                            review_response.automatable_checks_passed
                            if review_response is not None
                            else result.output.get("passed") is True
                        )
                        if not review_passed:
                            self._transition(
                                workflow_id, WorkflowState.ESCALATED,
                                reason=(
                                    "Automatable ProjectSpec review checks failed or could not be verified; "
                                    "the workflow is escalated instead of silently passing the rubric."
                                ),
                                expected_state=WorkflowState.REVIEW,
                                owner=owner,
                                failure_reason="Automatable review checks failed or remain unverified.",
                                result_updates={"review": result.output},
                            )
                            self._cancel_incomplete_tasks(row.task_graph_id)
                        else:
                            review_updates: dict[str, Any] = {
                                "review": result.output,
                                "review_revision": review_revision,
                            }
                            event_payload: dict[str, Any] = {}
                            if not result.is_stub:
                                agent_runs, review_results = self._workflow_packager_evidence(workflow_id)
                                package_result, groundedness = self._generate_grounded_report(
                                    workflow_id=workflow_id,
                                    project_id=row.project_id,
                                    review_revision=review_revision,
                                    project_spec=project_spec,
                                    task_graph=task_graph,
                                    review_output=result.output,
                                    test_runs=test_runs,
                                    agent_runs=agent_runs,
                                    review_results=review_results,
                                )
                                review_updates["package_documents"] = package_result.output
                                review_updates["report_groundedness"] = groundedness.model_dump(mode="json")
                                event_payload["report_groundedness"] = groundedness.model_dump(mode="json")
                            self._transition(
                                workflow_id, WorkflowState.AWAITING_FINAL_APPROVAL,
                                reason=(
                                    "Automatable review checks passed and the project report passed its "
                                    "deterministic groundedness gate; code, README, and report are ready "
                                    "for the mandatory final human checkpoint."
                                    if not result.is_stub else
                                    "Automatable review checks passed; human-judgment evidence and the "
                                    "code diff are ready for the mandatory final human gate."
                                ),
                                expected_state=WorkflowState.REVIEW,
                                owner=owner,
                                payload=event_payload,
                                result_updates=review_updates,
                            )
                    elif state is WorkflowState.PACKAGE:
                        if row.task_graph_id is None:
                            raise WorkflowError("packaging cannot run without the persisted task graph")
                        test_runs = self._workflow_test_runs(workflow_id)
                        review_output = (row.result_payload or {}).get("review", {})
                        gate_decision = (row.result_payload or {}).get("final_gate_decision", {})
                        package_documents = (row.result_payload or {}).get("package_documents")
                        if isinstance(package_documents, Mapping):
                            package = dict(package_documents)
                            if self.package_manager is None:
                                raise WorkflowError(
                                    "production package documents require a configured package export manager"
                                )
                            package["final_gate_documentation"] = self._finalize_report_with_human_gate(
                                row.project_id, workflow_id, gate_decision,
                            )
                        else:
                            if self.package_manager is not None:
                                raise WorkflowError(
                                    "grounded package documents are missing at final packaging"
                                )
                            # The deterministic Phase 2 stub has no generated docs or
                            # export manager. Keep it after human approval and visibly
                            # mark it as test-only; production reports are generated and
                            # checked before the human sees the final review bundle.
                            result = self._call_agent(
                                workflow_id,
                                row.project_id,
                                "packager",
                                {
                                    "workflow_id": str(workflow_id),
                                    "project_spec": self._workflow_spec(row).model_dump(mode="json"),
                                    "task_graph": self._load_project_graph(
                                        row.project_id, row.task_graph_id,
                                    ).model_dump(mode="json"),
                                    "review": review_output,
                                    "test_runs": test_runs,
                                    "final_gate_decision": gate_decision,
                                },
                                call_key=f"{workflow_id}:packager:stub:v0",
                            )
                            if not result.is_stub:
                                raise WorkflowError(
                                    "production package documents were not generated before the final human checkpoint"
                                )
                            package = dict(result.output)

                        disclosure = self._build_disclosure(workflow_id, row.project_id)
                        package["disclosure"] = disclosure.model_dump(mode="json")
                        package_export = None
                        if self.package_manager is not None:
                            record, _ = self.package_manager.export(
                                project_id=row.project_id,
                                workflow_id=workflow_id,
                                disclosure=disclosure,
                                review=review_output,
                                test_runs=test_runs,
                            )
                            package_export = record.model_dump(mode="json")
                            package["export"] = package_export
                        self._transition(
                            workflow_id, WorkflowState.DONE,
                            reason=(
                                "The pre-reviewed, grounded README/report were finalized with the explicit "
                                "human decision; disclosure and downloadable bundle were built from audit data."
                                if package_export is not None else
                                "Test-only stub package metadata completed after final human approval."
                            ),
                            expected_state=WorkflowState.PACKAGE,
                            owner=owner,
                            result_updates={
                                "package": package,
                                "disclosure": disclosure.model_dump(mode="json"),
                                **({"package_export": package_export} if package_export is not None else {}),
                            },
                        )
                    else:
                        raise InvalidWorkflowTransition(f"unsupported workflow state: {state.value}")
                except BudgetExceeded as exc:
                    self._terminalize(
                        workflow_id, WorkflowState.ESCALATED,
                        reason=str(exc), owner=owner,
                    )
                except RetryLimitExceeded as exc:
                    self._terminalize(
                        workflow_id, WorkflowState.ESCALATED,
                        reason=str(exc), owner=owner,
                    )
                except WorkflowTimedOut as exc:
                    self._terminalize(
                        workflow_id, WorkflowState.ESCALATED,
                        reason=str(exc), owner=owner,
                    )
                except WorkflowCancelled as exc:
                    self._terminalize(
                        workflow_id, WorkflowState.FAILED,
                        reason=str(exc), owner=owner,
                    )
                except (InvalidWorkflowTransition, WorkflowLeaseLost):
                    raise
                except WorkflowError as exc:
                    self._terminalize(
                        workflow_id, WorkflowState.FAILED,
                        reason=str(exc), owner=owner,
                    )
                except Exception as exc:
                    self._terminalize(
                        workflow_id, WorkflowState.FAILED,
                        reason=f"Workflow execution error: {type(exc).__name__}: {exc}",
                        owner=owner,
                    )
        finally:
            self._release(workflow_id, owner)

    def _run_build_loop(self, run: WorkflowRunRow, workflow_id: UUID, owner: str) -> None:
        project_id = run.project_id
        graph_id = run.task_graph_id
        if graph_id is None:
            raise WorkflowError("workflow entered BUILD_LOOP without a persisted task graph")
        while True:
            self._renew_lease(workflow_id, owner)
            self._check_cancelled(workflow_id)
            self._check_workflow_timeout(self._load_workflow(workflow_id))
            graph = self._load_project_graph(project_id, graph_id)
            with self._session_factory() as session:
                node_rows = session.scalars(
                    select(TaskNodeRow).where(TaskNodeRow.graph_id == graph_id)
                ).all()
                states = {node.id: node.status for node in node_rows}
            failed = sorted(node_id for node_id, status in states.items() if status == "failed")
            if failed:
                raise RetryLimitExceeded(f"task retry limit exhausted: {', '.join(failed)}")
            unfinished = {
                node_id for node_id, status in states.items()
                if status not in {"completed", "cancelled"}
            }
            if not unfinished:
                if any(status == "cancelled" for status in states.values()):
                    raise WorkflowCancelled("one or more tasks were cancelled")
                return
            completed = {node_id for node_id, status in states.items() if status == "completed"}
            batch = self.scheduler.next_batch(
                graph, completed_ids=completed, pending_ids=unfinished,
            )
            if not batch:
                raise WorkflowError("no dependency-ready task is available")
            with self._session_factory() as session, session.begin():
                for node in batch:
                    row = session.get(TaskNodeRow, (graph_id, node.id))
                    if row is not None and row.status in {"pending", "ready"}:
                        row.status = "running"
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {
                    node.id: executor.submit(
                        self._execute_task,
                        workflow_id,
                        project_id,
                        graph_id,
                        node,
                        run.max_retries,
                    )
                    for node in batch
                }
                # Resolve in stable task-id order. Work runs concurrently, while the
                # control-flow decision and surfaced failure remain deterministic.
                failures: list[Exception] = []
                for task_id in sorted(futures):
                    try:
                        futures[task_id].result()
                    except Exception as exc:  # wait for the entire independent batch
                        failures.append(exc)
                if failures:
                    raise failures[0]

    def _execute_task(
        self,
        workflow_id: UUID,
        project_id: UUID,
        graph_id: UUID,
        task: Any,
        max_retries: int,
    ) -> None:
        deadline = time.monotonic() + task.timeout_seconds
        debugger_feedback: dict[str, Any] | None = None
        while True:
            self._check_cancelled(workflow_id)
            if time.monotonic() > deadline:
                raise WorkflowTimedOut(f"task '{task.id}' exceeded {task.timeout_seconds}s timeout")
            with self._session_factory() as session:
                row = session.get(TaskNodeRow, (graph_id, task.id))
                if row is None:
                    raise WorkflowError(f"task '{task.id}' disappeared from persisted graph")
                retries = row.retries
                if row.status == "completed":
                    return
                if row.status == "cancelled":
                    raise WorkflowCancelled(f"task '{task.id}' was cancelled")

            attempt = retries + 1
            task_node = task.model_dump(mode="json")
            task_node.update({"status": "running", "retries": retries})
            base_input = {
                "task_id": task.id,
                "task_node": task_node,
                "description": task.description,
                "acceptance_criteria": task.acceptance_criteria,
                "read_set": task.read_set,
                "write_set": task.write_set,
                "runtime": task.runtime,
                "attempt": attempt,
                "timeout_seconds": task.timeout_seconds,
                "code_execution_performed": False,
            }
            self._call_agent(
                workflow_id,
                project_id,
                "coder",
                {**base_input, "phase": "CODE", "debugger_feedback": debugger_feedback},
                call_key=f"{workflow_id}:task:{task.id}:coder:{attempt}",
            )
            self._check_cancelled(workflow_id)
            if time.monotonic() > deadline:
                raise WorkflowTimedOut(f"task '{task.id}' exceeded {task.timeout_seconds}s timeout")
            test_result = self._call_agent(
                workflow_id,
                project_id,
                "tester",
                {**base_input, "phase": "TEST"},
                call_key=f"{workflow_id}:task:{task.id}:tester:{attempt}",
            )
            if time.monotonic() > deadline:
                raise WorkflowTimedOut(f"task '{task.id}' exceeded {task.timeout_seconds}s timeout")
            if test_result.output.get("passed") is True:
                with self._session_factory() as session, session.begin():
                    row = session.get(TaskNodeRow, (graph_id, task.id))
                    if row is not None:
                        row.status = "completed"
                return

            exhausted = False
            with self._session_factory() as session, session.begin():
                row = session.get(TaskNodeRow, (graph_id, task.id))
                if row is None:
                    raise WorkflowError(f"task '{task.id}' disappeared from persisted graph")
                retries = row.retries
                if retries >= max_retries:
                    row.status = "failed"
                    exhausted = True
                else:
                    row.retries = retries + 1
                    row.status = "running"
                    retries = row.retries
            if exhausted:
                raise RetryLimitExceeded(
                    f"task '{task.id}' failed after {max_retries} retries"
                )

            output = test_result.output
            test_failure = {
                "exit_code": output.get("exit_code"),
                "timed_out": output.get("timed_out", False),
                "runtime": output.get("runtime", task.runtime),
                "command": output.get("command", []),
                "tested_files": output.get("tested_files", []),
                "stdout": str(output.get("stdout", ""))[:24_000],
                "stderr": str(output.get("stderr", ""))[:24_000],
                "evidence": output.get("evidence", []),
            }
            debugger_result = self._call_agent(
                workflow_id,
                project_id,
                "debugger",
                {
                    **base_input,
                    "phase": "DEBUG",
                    "retry_number": retries,
                    "test_failure": test_failure,
                },
                call_key=f"{workflow_id}:task:{task.id}:debugger:{retries}",
            )
            debugger_feedback = {
                key: debugger_result.output[key]
                for key in ("diagnosis", "fix_instructions", "evidence", "retry_number")
                if key in debugger_result.output
            }

    def _call_agent(
        self,
        workflow_id: UUID,
        project_id: UUID,
        agent_name: str,
        payload: Mapping[str, Any],
        *,
        call_key: str | None = None,
    ) -> AgentInvocationResult:
        self._check_cancelled(workflow_id)
        workflow_row = self._load_workflow(workflow_id)
        self._check_workflow_timeout(workflow_row)
        created_at = self._as_utc(workflow_row.created_at)
        if created_at is None:
            raise WorkflowError("workflow creation timestamp is missing")
        deadline_seconds = workflow_row.timeout_seconds - (self._now() - created_at).total_seconds()
        scoped_payload = {
            **dict(payload),
            "project_id": str(project_id),
            "agent_scope": list(STUB_AGENT_SCOPES[agent_name]),
        }
        if agent_name in {"requirement", "planner", "coder", "tester", "debugger", "reviewer", "packager"}:
            task_timeout = scoped_payload.get("timeout_seconds")
            bounded_timeout = deadline_seconds
            if isinstance(task_timeout, int) and not isinstance(task_timeout, bool):
                bounded_timeout = min(bounded_timeout, task_timeout)
            scoped_payload["agent_timeout_seconds"] = max(1, int(bounded_timeout))
        prepare_payload = getattr(self.agents, "prepare_payload", None)
        if callable(prepare_payload):
            scoped_payload = prepare_payload(agent_name, scoped_payload)
        billable_checker = getattr(self.agents, "is_billable_call", None)
        if callable(billable_checker):
            billable = bool(billable_checker(agent_name, scoped_payload))
        else:
            billable = agent_name in {"requirement", "planner"}
        billable = (
            billable
            and hasattr(self.agents, "reservation_tokens")
            and hasattr(self.agents, "reservation_spend")
        )
        if call_key:
            with self._session_factory() as session:
                previous = session.scalar(
                    select(AgentRunRow).where(AgentRunRow.idempotency_key == call_key)
                )
            if previous is not None:
                previous_output = dict(previous.output_payload or {})
                if previous_output.get("output_validation_failed") and agent_name == "packager":
                    raise AgentOutputValidationFailure(
                        str(previous_output.get("error", "previous Packager output failed validation"))
                    )
                if (previous_output.get("provider_error")
                        or previous_output.get("output_validation_failed")
                        or previous_output.get("preflight_error")):
                    raise WorkflowError(str(previous_output.get("error", "previous agent call failed")))
                result = AgentInvocationResult(
                    output=previous_output,
                    tokens_used=previous.tokens_used,
                    cost=previous.cost,
                    provider=previous.provider,
                    model=previous.model,
                    llm_calls=int(previous_output.get("llm_calls", 0)),
                    is_stub=previous_output.get("stub") is True,
                    usage_known=True,
                )
                self._check_cancelled(workflow_id)
                return result

        reservation_id: UUID | None = None
        if billable:
            token_ceiling = int(self.agents.reservation_tokens)
            estimator = getattr(self.agents, "estimate_call_tokens", None)
            estimated_tokens = int(estimator(agent_name, scoped_payload)) if callable(estimator) else token_ceiling
            reserved_tokens = min(token_ceiling, max(1, estimated_tokens))
            reserved_spend = Decimal(str(self.agents.reservation_spend))
            reservation_id = self.budget_guard.reserve(
                project_id,
                workflow_id,
                tokens=reserved_tokens,
                spend=reserved_spend,
                ttl_seconds=int(getattr(self.agents, "reservation_ttl_seconds", 900)),
            )
        else:
            # Non-billable worker roles still consult the persisted ledger before running.
            self.budget_guard.check(project_id, reserve_tokens=0, reserve_spend=Decimal("0"))

        try:
            result = self.agents.invoke(agent_name, scoped_payload)
        except PromptTokenLimitExceeded as exc:
            if reservation_id is not None:
                self.budget_guard.release(reservation_id)
                self.budget_guard.record_failed_provider_call(agent_run_values={
                    "id": uuid4(), "agent_name": agent_name,
                    "input_payload": {"workflow_id": str(workflow_id), **scoped_payload},
                    "output_payload": {
                        "error": str(exc), "preflight_error": True,
                        "llm_calls": 0, "stub": False,
                        "reserved_tokens": reserved_tokens,
                        "reserved_spend": str(reserved_spend),
                        "scope": list(STUB_AGENT_SCOPES[agent_name]),
                    },
                    "tokens_used": 0, "cost": Decimal("0"),
                    "timestamp": self._now(), "project_id": project_id,
                    "workflow_id": workflow_id,
                    "provider": getattr(getattr(self.agents, "config", None), "provider", None),
                    "model": None, "idempotency_key": call_key,
                    "budget_reservation_id": reservation_id,
                })
            raise BudgetExceeded(project_id, str(exc)) from exc
        except ModelConfigurationError as exc:
            if reservation_id is not None:
                self.budget_guard.release(reservation_id)
                config = getattr(self.agents, "config", None)
                self.budget_guard.record_failed_provider_call(agent_run_values={
                    "id": uuid4(), "agent_name": agent_name,
                    "input_payload": {"workflow_id": str(workflow_id), **scoped_payload},
                    "output_payload": {
                        "error": str(exc), "preflight_error": True,
                        "llm_calls": 0, "stub": False,
                        "scope": list(STUB_AGENT_SCOPES[agent_name]),
                    },
                    "tokens_used": 0, "cost": Decimal("0"),
                    "timestamp": self._now(), "project_id": project_id,
                    "workflow_id": workflow_id,
                    "provider": getattr(config, "provider", None), "model": None,
                    "idempotency_key": call_key,
                    "budget_reservation_id": reservation_id,
                })
            raise WorkflowError(str(exc)) from exc
        except AgentOutputValidationError as exc:
            if reservation_id is not None:
                completion = exc.completion
                tokens = completion.tokens_used if completion.tokens_used is not None else reserved_tokens
                spend = completion.cost if completion.cost is not None else reserved_spend
                failed_output = {
                    "error": str(exc),
                    "output_validation_failed": True,
                    "llm_calls": 1,
                    "stub": False,
                    "provider": completion.provider,
                    "model": completion.model,
                    "scope": list(STUB_AGENT_SCOPES[agent_name]),
                }
                self.budget_guard.settle_provider_call(
                    reservation_id,
                    tokens=tokens,
                    spend=spend,
                    agent_run_values={
                        "id": uuid4(), "agent_name": agent_name,
                        "input_payload": {"workflow_id": str(workflow_id), **scoped_payload},
                        "output_payload": failed_output,
                        "timestamp": self._now(), "project_id": project_id,
                        "workflow_id": workflow_id, "provider": completion.provider,
                        "model": completion.model, "idempotency_key": call_key,
                    },
                )
            raise AgentOutputValidationFailure(str(exc)) from exc
        except Exception as exc:
            if reservation_id is not None:
                config = getattr(self.agents, "config", None)
                provider = getattr(config, "provider", None)
                model = None
                if config is not None:
                    try:
                        model = config.model_for(agent_name)
                    except Exception:
                        model = None
                # A transport error may arrive after the remote endpoint billed the
                # request. Charge the full pre-call ceiling rather than release a
                # potentially consumed budget hold.
                failed_output = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "provider_error": True,
                    "usage_unknown": True,
                    "usage_estimated_from_reservation": True,
                    "reserved_tokens": reserved_tokens,
                    "reserved_spend": str(reserved_spend),
                    "llm_calls": 1,
                    "stub": False,
                    "provider": provider,
                    "model": model,
                    "scope": list(STUB_AGENT_SCOPES[agent_name]),
                }
                self.budget_guard.settle_provider_call(
                    reservation_id,
                    tokens=reserved_tokens,
                    spend=reserved_spend,
                    agent_run_values={
                        "id": uuid4(), "agent_name": agent_name,
                        "input_payload": {"workflow_id": str(workflow_id), **scoped_payload},
                        "output_payload": failed_output,
                        "timestamp": self._now(), "project_id": project_id,
                        "workflow_id": workflow_id, "provider": provider, "model": model,
                        "idempotency_key": call_key,
                    },
                )
                raise WorkflowError(
                    f"{agent_name} provider call failed ({type(exc).__name__}); "
                    "the configured maximum-call budget was conservatively charged"
                ) from exc
            raise

        output = dict(result.output)
        output.setdefault("llm_calls", result.llm_calls)
        output.setdefault("stub", result.is_stub)
        if result.provider:
            output.setdefault("provider", result.provider)
        if result.model:
            output.setdefault("model", result.model)
        input_payload = {"workflow_id": str(workflow_id), **scoped_payload}
        if reservation_id is not None:
            actual_tokens = result.tokens_used if result.tokens_used is not None else reserved_tokens
            actual_spend = result.cost if result.cost is not None else reserved_spend
            output.setdefault("usage_estimated_from_reservation", not result.usage_known)
            self.budget_guard.settle_provider_call(
                reservation_id,
                tokens=actual_tokens,
                spend=actual_spend,
                agent_run_values={
                    "id": uuid4(), "agent_name": agent_name,
                    "input_payload": input_payload,
                    "output_payload": output,
                    "timestamp": self._now(), "project_id": project_id,
                    "workflow_id": workflow_id, "provider": result.provider,
                    "model": result.model, "idempotency_key": call_key,
                },
            )
            result = AgentInvocationResult(
                output=output, tokens_used=actual_tokens, cost=actual_spend,
                provider=result.provider, model=result.model, llm_calls=result.llm_calls,
                is_stub=result.is_stub, usage_known=result.usage_known,
            )
        else:
            tokens = int(result.tokens_used or 0)
            spend = Decimal(str(result.cost or Decimal("0")))
            with self._session_factory() as session, session.begin():
                session.add(AgentRunRow(
                    id=uuid4(),
                    agent_name=agent_name,
                    input_payload=input_payload,
                    output_payload=output,
                    tokens_used=tokens,
                    cost=spend,
                    timestamp=self._now(),
                    project_id=project_id,
                    workflow_id=workflow_id,
                    provider=result.provider,
                    model=result.model,
                    idempotency_key=call_key,
                ))
            self.budget_guard.record_usage(project_id, tokens=tokens, spend=spend)
            result = AgentInvocationResult(
                output=output, tokens_used=tokens, cost=spend,
                provider=result.provider, model=result.model, llm_calls=result.llm_calls,
                is_stub=result.is_stub, usage_known=result.usage_known,
            )
        self._check_cancelled(workflow_id)
        self._check_workflow_timeout(self._load_workflow(workflow_id))
        return result

    @staticmethod
    def _workflow_input(run: WorkflowRunRow) -> dict[str, Any]:
        snapshot = (run.result_payload or {}).get("input_snapshot")
        if not isinstance(snapshot, dict):
            raise WorkflowError("workflow intake snapshot is missing")
        return snapshot

    @staticmethod
    def _workflow_spec(run: WorkflowRunRow) -> ProjectSpec:
        payload = run.result_payload or {}
        spec_snapshot = payload.get("normalized_spec") or payload.get("spec_snapshot")
        if not isinstance(spec_snapshot, dict):
            raise WorkflowError("workflow specification snapshot is missing")
        return ProjectSpec.model_validate(spec_snapshot)

    @staticmethod
    def _requirement_response(output: Mapping[str, Any]) -> RequirementAgentResponse:
        fields = {"status", "confidence", "project_spec", "questions", "uncertainties"}
        return RequirementAgentResponse.model_validate({
            key: output[key] for key in fields if key in output
        })

    @staticmethod
    def _planner_response(output: Mapping[str, Any]) -> PlannerAgentResponse:
        fields = {"plan_summary", "task_graph", "evidence"}
        return PlannerAgentResponse.model_validate({
            key: output[key] for key in fields if key in output
        })

    def _persist_planned_graph(
        self,
        workflow_id: UUID,
        plan: PlannerAgentResponse,
        *,
        planner_output: Mapping[str, Any],
        owner: str,
    ) -> WorkflowRunRecord:
        now = self._now()
        with self._session_factory() as session, session.begin():
            run = session.scalar(
                select(WorkflowRunRow).where(WorkflowRunRow.id == workflow_id).with_for_update()
            )
            if run is None:
                raise LookupError("workflow not found")
            if WorkflowState(run.state) is not WorkflowState.PLAN:
                raise InvalidWorkflowTransition("workflow left PLAN before graph persistence")
            if run.lease_owner != owner:
                raise WorkflowLeaseLost("workflow execution lease was lost")
            if run.cancel_requested_at is not None:
                raise WorkflowCancelled("workflow cancellation requested")
            project = session.scalar(
                select(Project).where(Project.id == run.project_id).with_for_update()
            )
            if project is None:
                raise WorkflowError("planner project disappeared")
            version = session.scalar(
                select(func.coalesce(func.max(TaskGraphRow.version), 0))
                .where(TaskGraphRow.project_id == run.project_id)
            ) + 1
            graph_id = uuid4()
            graph_row = TaskGraphRow(
                id=graph_id, project_id=run.project_id, version=version, created_at=now,
            )
            session.add(graph_row)
            session.flush()
            for node in plan.task_graph.nodes:
                session.add(TaskNodeRow(
                    graph_id=graph_id,
                    id=node.id,
                    description=node.description,
                    dependencies=node.dependencies,
                    status="pending",
                    acceptance_criteria=node.acceptance_criteria,
                    retries=0,
                    assigned_agent=node.assigned_agent,
                    read_set=node.read_set,
                    write_set=node.write_set,
                    runtime=node.runtime,
                    timeout_seconds=node.timeout_seconds,
                    review_feedback=node.review_feedback,
                ))
            session.flush()
            for edge in plan.task_graph.edges:
                session.add(TaskEdgeRow(
                    graph_id=graph_id, source_id=edge.source, target_id=edge.target,
                ))
            run.task_graph_id = graph_id
            current = WorkflowState(run.state)
            run.state = WorkflowState.AWAITING_PLAN_APPROVAL.value
            run.updated_at = now
            payload = dict(run.result_payload or {})
            payload.update({
                "plan": dict(planner_output),
                "plan_summary": plan.plan_summary,
                "planned_graph_id": str(graph_id),
            })
            run.result_payload = payload
            self._append_event(
                session,
                run,
                from_state=current,
                to_state=WorkflowState.AWAITING_PLAN_APPROVAL,
                reason="Planner produced a validated DAG; code remains gated on human approval.",
                payload={
                    "task_graph_id": str(graph_id),
                    "task_graph_version": version,
                    "task_count": len(plan.task_graph.nodes),
                },
                created_at=now,
            )
        return self.get(workflow_id)

    def _load_project_graph(self, project_id: UUID, graph_id: UUID) -> TaskGraph:
        with self._session_factory() as session:
            graph_row = session.get(TaskGraphRow, graph_id)
            if graph_row is None or graph_row.project_id != project_id:
                raise WorkflowError("persisted task graph is missing or belongs to another project")
            node_rows = session.scalars(
                select(TaskNodeRow).where(TaskNodeRow.graph_id == graph_id)
                .order_by(TaskNodeRow.id)
            ).all()
            edge_rows = session.scalars(
                select(TaskEdgeRow).where(TaskEdgeRow.graph_id == graph_id)
                .order_by(TaskEdgeRow.source_id, TaskEdgeRow.target_id)
            ).all()
            return TaskGraph.model_validate({
                "nodes": [{
                    "id": node.id,
                    "description": node.description,
                    "dependencies": node.dependencies,
                    "status": node.status,
                    "acceptance_criteria": node.acceptance_criteria,
                    "retries": node.retries,
                    "assigned_agent": node.assigned_agent,
                    "read_set": node.read_set,
                    "write_set": node.write_set,
                    "runtime": node.runtime,
                    "timeout_seconds": node.timeout_seconds,
                    "review_feedback": node.review_feedback or [],
                } for node in node_rows],
                "edges": [{"source": edge.source_id, "target": edge.target_id}
                          for edge in edge_rows],
            })

    def _load_workflow(self, workflow_id: UUID) -> WorkflowRunRow:
        with self._session_factory() as session:
            row = session.get(WorkflowRunRow, workflow_id)
            if row is None:
                raise LookupError("workflow not found")
            return row

    def _persist_review_results(
        self,
        workflow_id: UUID,
        project_id: UUID,
        spec: ProjectSpec,
        result: AgentInvocationResult,
    ) -> ReviewerAgentResponse | None:
        response: ReviewerAgentResponse | None = None
        if isinstance(result.output.get("criteria"), list):
            response = ReviewerAgentResponse.model_validate({
                key: result.output[key]
                for key in (
                    "summary", "criteria", "automatable_checks_passed",
                    "human_review_required", "evidence",
                )
                if key in result.output
            })
            expected = {item.criterion.casefold(): item for item in spec.rubric_items}
            actual = {item.rubric_item.criterion.casefold(): item for item in response.criteria}
            if set(actual) != set(expected):
                raise WorkflowError("Reviewer output does not cover each ProjectSpec rubric item exactly once")
            if any(
                actual[key].rubric_item.model_dump(mode="json")
                != expected[key].model_dump(mode="json")
                for key in expected
            ):
                raise WorkflowError("Reviewer changed a rubric item's ProjectSpec content")
        elif result.output.get("stub") is not True:
            raise WorkflowError("Reviewer returned no itemized, schema-validated rubric assessments")

        evidence_default = result.output.get("evidence", [])
        with self._session_factory() as session, session.begin():
            session.execute(delete(ReviewResultRow).where(ReviewResultRow.workflow_id == workflow_id))
            if response is not None:
                assessments = response.criteria
            else:
                # Compatibility for tests explicitly using StubAgentSuite: its output
                # is archival scaffolding, not an assertion that a rubric passed.
                assessments = []
            if response is not None:
                for assessment in assessments:
                    status = assessment.status
                    session.add(ReviewResultRow(
                        id=uuid4(),
                        project_id=project_id,
                        workflow_id=workflow_id,
                        rubric_item=assessment.rubric_item.model_dump(mode="json"),
                        rubric_type=assessment.rubric_item.type,
                        status=status,
                        passed=(True if status == "passed" else False if status == "failed" else None),
                        recommendation=assessment.recommendation,
                        evidence=assessment.evidence,
                        reviewer_notes=assessment.rationale,
                    ))
            else:
                for rubric_item in spec.rubric_items:
                    human = rubric_item.type == "human-judgment"
                    session.add(ReviewResultRow(
                        id=uuid4(),
                        project_id=project_id,
                        workflow_id=workflow_id,
                        rubric_item=rubric_item.model_dump(mode="json"),
                        rubric_type=rubric_item.type,
                        status="human_review_required" if human else "not_verified",
                        passed=None,
                        recommendation=None,
                        evidence=evidence_default or [
                            "Stub reviewer does not produce a verified criterion-level verdict."
                        ],
                        reviewer_notes=(
                            "Test-only stub output; no automated result or human-judgment pass is claimed."
                        ),
                    ))
        return response

    def _workflow_test_runs(self, workflow_id: UUID) -> list[dict[str, Any]]:
        """Return the last persisted sandbox TEST verdict for each task, in stable order."""
        with self._session_factory() as session:
            rows = session.scalars(
                select(AgentRunRow).where(
                    AgentRunRow.workflow_id == workflow_id,
                    AgentRunRow.agent_name == "tester",
                ).order_by(AgentRunRow.timestamp, AgentRunRow.id)
            ).all()
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            output = row.output_payload or {}
            if str(output.get("phase", "")).upper() != "TEST":
                continue
            task_id = str(output.get("task_id") or "unknown")
            latest[task_id] = {
                "task_id": task_id,
                "phase": "TEST",
                "passed": output.get("passed") is True,
                "runtime": output.get("runtime"),
                "command": output.get("command", []),
                "tested_files": output.get("tested_files", []),
                "exit_code": output.get("exit_code"),
                "timed_out": output.get("timed_out", False),
                "duration_seconds": output.get("duration_seconds"),
                "stdout": output.get("stdout", ""),
                "stderr": output.get("stderr", ""),
                "evidence": output.get("evidence", []),
            }
        return [latest[task_id] for task_id in sorted(latest)]

    def _workflow_packager_evidence(
        self,
        workflow_id: UUID,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Load compact, persisted AgentRun and ReviewResult evidence for reporting."""
        with self._session_factory() as session:
            agent_rows = session.scalars(
                select(AgentRunRow).where(
                    AgentRunRow.workflow_id == workflow_id,
                    AgentRunRow.agent_name != "packager",
                ).order_by(AgentRunRow.timestamp, AgentRunRow.id)
            ).all()
            review_rows = session.scalars(
                select(ReviewResultRow).where(
                    ReviewResultRow.workflow_id == workflow_id,
                ).order_by(ReviewResultRow.id)
            ).all()

        agent_runs: list[dict[str, Any]] = []
        for row in agent_rows:
            input_payload = row.input_payload or {}
            output = row.output_payload or {}
            task_node = input_payload.get("task_node") or {}
            evidence = output.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = [str(evidence)] if evidence else []
            record: dict[str, Any] = {
                "agent_run_id": str(row.id),
                "agent_name": row.agent_name,
                "timestamp": self._as_utc(row.timestamp).isoformat(),
                "phase": str(output.get("phase") or input_payload.get("phase") or ""),
                "task_id": str(
                    output.get("task_id") or input_payload.get("task_id")
                    or (task_node.get("id") if isinstance(task_node, Mapping) else "")
                    or ""
                ),
                "status": str(output.get("status") or ""),
                "summary": str(
                    output.get("summary") or output.get("diagnosis")
                    or output.get("plan_summary") or ""
                ),
                "evidence": [str(item) for item in evidence[:8]],
            }
            for key in ("passed", "exit_code", "timed_out"):
                if key in output:
                    record[key] = output[key]
            for key in ("changed_paths", "files_written", "tested_files"):
                value = output.get(key)
                if isinstance(value, list):
                    record[key] = value[:30]
            agent_runs.append(record)

        review_results = [
            {
                "review_result_id": str(row.id),
                "rubric_item": dict(row.rubric_item or {}),
                "status": row.status,
                "passed": row.passed,
                "recommendation": row.recommendation,
                "evidence": [str(item) for item in (row.evidence or [])[:10]],
                "reviewer_notes": row.reviewer_notes or "",
            }
            for row in review_rows
        ]
        return agent_runs, review_results

    def _generate_grounded_report(
        self,
        *,
        workflow_id: UUID,
        project_id: UUID,
        review_revision: int,
        project_spec: ProjectSpec,
        task_graph: TaskGraph,
        review_output: Mapping[str, Any],
        test_runs: Sequence[Mapping[str, Any]],
        agent_runs: Sequence[Mapping[str, Any]],
        review_results: Sequence[Mapping[str, Any]],
    ) -> tuple[AgentInvocationResult, ReportGroundednessResult]:
        """Generate once and automatically regenerate once after a deterministic rejection."""
        feedback = ""
        last_issue = ""
        for attempt in range(2):
            try:
                result = self._call_agent(
                    workflow_id,
                    project_id,
                    "packager",
                    {
                        "workflow_id": str(workflow_id),
                        "project_spec": project_spec.model_dump(mode="json"),
                        "task_graph": task_graph.model_dump(mode="json"),
                        "review": dict(review_output),
                        "test_runs": list(test_runs),
                        "agent_runs": list(agent_runs),
                        "review_results": list(review_results),
                        "final_gate_decision": {"decision": "pending"},
                        "regeneration_feedback": feedback,
                    },
                    call_key=(
                        f"{workflow_id}:packager:review:v{review_revision}:attempt:{attempt}"
                    ),
                )
            except AgentOutputValidationFailure as exc:
                last_issue = str(exc)[:3500]
                feedback = "Previous report was rejected by deterministic validation: " + last_issue
                continue
            grounding_data = result.output.get("groundedness_validation")
            if not isinstance(grounding_data, Mapping):
                raise WorkflowError(
                    "Packager returned no deterministic groundedness validation record"
                )
            grounding_payload = dict(grounding_data)
            grounding_payload["regeneration_attempts"] = attempt
            groundedness = ReportGroundednessResult.model_validate(grounding_payload)
            if not groundedness.passed:
                raise WorkflowError(
                    "Packager reported a failed groundedness result after response validation"
                )
            return result, groundedness
        raise RetryLimitExceeded(
            "Packager report failed deterministic groundedness validation after one "
            f"automatic regeneration: {last_issue or feedback}"
        )

    def _finalize_report_with_human_gate(
        self,
        project_id: UUID,
        workflow_id: UUID,
        decision: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Append the actual final approval after the human has seen the report draft."""
        if decision.get("decision") != "approve":
            raise WorkflowError("only an approved final gate may enter package finalization")
        if self.workspace_manager is None:
            raise WorkflowError("report finalization requires the project Git workspace")
        report_path = "reports/PROJECT_REPORT.md"
        files = self.workspace_manager.read_context(
            project_id,
            read_scope=(report_path,),
            max_files=1,
            max_total_bytes=100_000,
            max_file_bytes=100_000,
        )
        report = files.get(report_path)
        if report is None:
            raise WorkflowError("the reviewed project report is missing at final packaging")
        marker = "## Final human checkpoint"
        if marker in report:
            report = report.split(marker, 1)[0].rstrip()
        comment = self._escape_markdown_inline(decision.get("comment") or "none supplied")
        timestamp = self._escape_markdown_inline(decision.get("timestamp") or "not recorded")
        finalized = (
            report.rstrip()
            + f"\n\n{marker}\n\n"
            + "- Decision: `approved`.\n"
            + f"- Recorded at: `{timestamp}`.\n"
            + f"- Comment: {comment}\n"
        )
        safe_suffix = "".join(character for character in str(workflow_id) if character.isalnum())[:16]
        commit = self.workspace_manager.apply_changes_and_commit(
            project_id,
            task_id=f"finalize-{safe_suffix or 'approved'}",
            agent_name="packager",
            changes=[AgentFileChange(path=report_path, content=finalized)],
            write_scope=("reports",),
            summary="Record the explicit final human approval in the reviewed report",
        )
        return {
            "commit_hash": commit.commit_hash,
            "changed_paths": list(commit.changed_paths),
            "report_path": report_path,
        }

    @staticmethod
    def _escape_markdown_inline(value: Any) -> str:
        cleaned = " ".join(str(value).split())[:1000]
        return (cleaned.replace("\\", "\\\\").replace("`", "\\`")
                .replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;"))

    def _build_disclosure(self, workflow_id: UUID, project_id: UUID) -> AIUsageDisclosure:
        with self._session_factory() as session:
            rows = session.scalars(
                select(AgentRunRow).where(
                    AgentRunRow.project_id == project_id,
                    AgentRunRow.workflow_id == workflow_id,
                ).order_by(AgentRunRow.timestamp, AgentRunRow.id)
            ).all()
            calls = len(rows)
            names = list(dict.fromkeys(row.agent_name for row in rows))
            tokens = sum(row.tokens_used for row in rows)
            spend = sum((row.cost for row in rows), Decimal("0"))
            run_ids = [row.id for row in rows]
            activities: list[AgentUsageActivity] = []
            stub_calls = llm_calls = code_execution_count = 0
            for row in rows:
                output = row.output_payload or {}
                input_payload = row.input_payload or {}
                is_stub = output.get("stub") is True
                stub_calls += int(is_stub)
                recorded_llm_calls = output.get("llm_calls")
                if isinstance(recorded_llm_calls, int) and not isinstance(recorded_llm_calls, bool):
                    llm_calls += max(0, recorded_llm_calls)
                else:
                    llm_calls += 0 if is_stub else 1
                recorded_executions = output.get("code_execution_count")
                if isinstance(recorded_executions, int) and not isinstance(recorded_executions, bool):
                    executions = max(0, recorded_executions)
                else:
                    executions = int(
                        output.get("code_executed") is True
                        or output.get("code_execution_performed") is True
                    )
                code_execution_count += executions
                task_node = input_payload.get("task_node") or {}
                task_id = (
                    output.get("task_id") or input_payload.get("task_id")
                    or task_node.get("id") if isinstance(task_node, Mapping) else
                    output.get("task_id") or input_payload.get("task_id")
                )
                phase = str(output.get("phase") or input_payload.get("phase") or "")
                if row.agent_name == "tester" and executions:
                    action = (
                        f"Executed {phase or 'TEST'} checks in the isolated {output.get('runtime', 'configured')} sandbox"
                    )
                    if output.get("test_generation_performed"):
                        action += " and generated tests"
                elif row.agent_name == "coder":
                    action = f"Implemented task {task_id or 'unknown'} and submitted scoped Git changes"
                elif row.agent_name == "debugger":
                    action = f"Diagnosed task {task_id or 'unknown'} test failure and returned repair guidance"
                elif row.agent_name == "reviewer":
                    action = "Compared final project code and test evidence against the ProjectSpec rubric"
                elif row.agent_name == "packager":
                    action = "Generated package documentation and committed README/report artifacts"
                elif row.agent_name == "planner":
                    action = "Produced a validated task DAG from the normalized ProjectSpec"
                elif row.agent_name == "requirement":
                    action = "Normalized the submitted brief into a validated ProjectSpec"
                else:
                    action = f"Performed {row.agent_name} agent action"
                if not executions and row.agent_name == "tester":
                    action = "Generated or inspected tests; no project code execution was recorded"
                activities.append(AgentUsageActivity(
                    audit_run_id=row.id,
                    agent_name=row.agent_name,
                    action=action,
                    timestamp=self._as_utc(row.timestamp),
                    provider=row.provider,
                    model=row.model,
                    task_id=str(task_id) if task_id else None,
                    tokens_used=row.tokens_used,
                    spend=row.cost,
                ))
        return AIUsageDisclosure(
            project_id=project_id,
            workflow_id=workflow_id,
            agent_calls=calls,
            stub_agent_calls=stub_calls,
            agent_names=names,
            audit_run_ids=run_ids,
            tokens_used=tokens,
            spend=spend,
            llm_calls=llm_calls,
            code_execution_count=code_execution_count,
            statement=(
                f"Workflow audit trail records {calls} agent actions ({stub_calls} stub actions), "
                f"{llm_calls} LLM calls, and {code_execution_count} sandbox code executions. "
                "Each activity, timestamp, provider/model, token count, and cost is derived from "
                "persisted AgentRun records; missing model data is not inferred."
            ),
            generated_at=self._now(),
            activities=activities,
        )

    def _transition(
        self,
        workflow_id: UUID,
        to_state: WorkflowState,
        *,
        reason: str,
        expected_state: WorkflowState | None = None,
        payload: Mapping[str, Any] | None = None,
        result_updates: Mapping[str, Any] | None = None,
        failure_reason: str | None = None,
        persist_project_spec: ProjectSpec | None = None,
        owner: str | None = None,
    ) -> WorkflowRunRecord:
        with self._session_factory() as session, session.begin():
            row = session.scalar(
                select(WorkflowRunRow).where(WorkflowRunRow.id == workflow_id).with_for_update()
            )
            if row is None:
                raise LookupError("workflow not found")
            current = WorkflowState(row.state)
            if expected_state is not None and current is not expected_state:
                raise InvalidWorkflowTransition(
                    f"expected {expected_state.value}, found {current.value}"
                )
            if owner is not None and row.lease_owner != owner:
                raise WorkflowLeaseLost("workflow execution lease was lost")
            if row.cancel_requested_at is not None and to_state not in TERMINAL_STATES:
                raise WorkflowCancelled("workflow cancellation requested")
            if to_state in {WorkflowState.FAILED, WorkflowState.ESCALATED}:
                allowed = current not in TERMINAL_STATES
            else:
                allowed = to_state in ALLOWED_TRANSITIONS[current]
            if not allowed:
                raise InvalidWorkflowTransition(
                    f"transition {current.value} -> {to_state.value} is not allowed"
                )
            if persist_project_spec is not None:
                spec_data = persist_project_spec.model_dump(mode="json")
                spec_row = session.scalar(
                    select(ProjectSpecRow)
                    .where(ProjectSpecRow.project_id == row.project_id)
                    .with_for_update()
                )
                if spec_row is None:
                    spec_row = ProjectSpecRow(project_id=row.project_id, **spec_data)
                    session.add(spec_row)
                else:
                    for field_name, field_value in spec_data.items():
                        setattr(spec_row, field_name, field_value)
                self.context_retriever.replace_project_spec(
                    session, row.project_id, persist_project_spec,
                )
            now = self._now()
            row.state = to_state.value
            row.updated_at = now
            if failure_reason is not None:
                row.failure_reason = failure_reason
            if result_updates:
                merged = dict(row.result_payload or {})
                merged.update(result_updates)
                row.result_payload = merged
            self._append_event(
                session, row,
                from_state=current,
                to_state=to_state,
                reason=reason,
                payload=payload or {},
                created_at=now,
            )
        return self.get(workflow_id)

    def _terminalize(
        self,
        workflow_id: UUID,
        state: WorkflowState,
        *,
        reason: str,
        owner: str,
    ) -> None:
        current = self._load_workflow(workflow_id)
        if WorkflowState(current.state) in TERMINAL_STATES:
            return
        try:
            self._transition(
                workflow_id,
                state,
                reason=reason[:2000],
                expected_state=WorkflowState(current.state),
                payload={"terminal_reason": reason[:2000]},
                failure_reason=reason[:2000],
                owner=owner,
            )
        finally:
            self._cancel_incomplete_tasks(current.task_graph_id)

    def _cancel_incomplete_tasks(self, graph_id: UUID | None) -> None:
        if graph_id is None:
            return
        with self._session_factory() as session, session.begin():
            rows = session.scalars(
                select(TaskNodeRow).where(
                    TaskNodeRow.graph_id == graph_id,
                    TaskNodeRow.status.in_(["pending", "ready", "running"]),
                )
            ).all()
            for row in rows:
                row.status = "cancelled"

    def _append_event(
        self,
        session: Session,
        run: WorkflowRunRow,
        *,
        from_state: WorkflowState | None,
        to_state: WorkflowState,
        reason: str,
        payload: Mapping[str, Any],
        created_at: datetime,
    ) -> None:
        current_max = session.scalar(
            select(func.coalesce(func.max(WorkflowEventRow.sequence), 0))
            .where(WorkflowEventRow.workflow_id == run.id)
        ) or 0
        session.add(WorkflowEventRow(
            workflow_id=run.id,
            sequence=current_max + 1,
            from_state=from_state.value if from_state else None,
            to_state=to_state.value,
            reason=reason[:2000],
            payload=dict(payload),
            created_at=created_at,
        ))

    def _claim(self, workflow_id: UUID, owner: str) -> bool:
        now = self._now()
        with self._session_factory() as session, session.begin():
            row = session.scalar(
                select(WorkflowRunRow).where(WorkflowRunRow.id == workflow_id).with_for_update()
            )
            if row is None:
                raise LookupError("workflow not found")
            state = WorkflowState(row.state)
            if state in TERMINAL_STATES:
                return False
            if state in WAITING_STATES and row.cancel_requested_at is None:
                return False
            lease_until = self._as_utc(row.lease_expires_at)
            if row.lease_owner and lease_until and lease_until > now:
                return False
            row.lease_owner = owner
            row.lease_expires_at = now + timedelta(seconds=self._lease_seconds)
            row.updated_at = now
            return True

    def _renew_lease(self, workflow_id: UUID, owner: str) -> None:
        now = self._now()
        with self._session_factory() as session, session.begin():
            row = session.scalar(
                select(WorkflowRunRow).where(WorkflowRunRow.id == workflow_id).with_for_update()
            )
            if row is None or row.lease_owner != owner:
                raise WorkflowLeaseLost("workflow execution lease was lost")
            row.lease_expires_at = now + timedelta(seconds=self._lease_seconds)

    def _release(self, workflow_id: UUID, owner: str) -> None:
        with self._session_factory() as session, session.begin():
            row = session.scalar(
                select(WorkflowRunRow).where(WorkflowRunRow.id == workflow_id).with_for_update()
            )
            if row is not None and row.lease_owner == owner:
                row.lease_owner = None
                row.lease_expires_at = None

    def _check_cancelled(self, workflow_id: UUID) -> None:
        with self._session_factory() as session:
            row = session.get(WorkflowRunRow, workflow_id)
            if row is None:
                raise LookupError("workflow not found")
            if row.cancel_requested_at is not None:
                raise WorkflowCancelled("workflow cancellation requested")

    def _check_workflow_timeout(self, run: WorkflowRunRow) -> None:
        created = self._as_utc(run.created_at)
        if created is None:
            raise WorkflowError("workflow creation timestamp is missing")
        if self._now() > created + timedelta(seconds=run.timeout_seconds):
            raise WorkflowTimedOut(f"workflow exceeded {run.timeout_seconds}s timeout")

    @staticmethod
    def _as_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _workflow_record(row: WorkflowRunRow) -> WorkflowRunRecord:
        return WorkflowRunRecord(
            id=row.id,
            project_id=row.project_id,
            task_graph_id=row.task_graph_id,
            state=WorkflowState(row.state),
            max_retries=row.max_retries,
            timeout_seconds=row.timeout_seconds,
            failure_reason=row.failure_reason,
            result_payload=row.result_payload or {},
            created_at=PostgresFSMEngine._as_utc(row.created_at),
            updated_at=PostgresFSMEngine._as_utc(row.updated_at),
        )

    @staticmethod
    def _event_record(row: WorkflowEventRow) -> WorkflowEventRecord:
        return WorkflowEventRecord(
            workflow_id=row.workflow_id,
            sequence=row.sequence,
            from_state=WorkflowState(row.from_state) if row.from_state else None,
            to_state=WorkflowState(row.to_state),
            reason=row.reason,
            payload=row.payload or {},
            created_at=PostgresFSMEngine._as_utc(row.created_at),
        )

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
