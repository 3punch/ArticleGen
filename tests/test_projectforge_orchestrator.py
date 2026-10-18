from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import select

from projectforge_agents.stubs import StubAgentSuite
from projectforge_api.db import build_engine, make_session_factory
from projectforge_api.orm import (
    AgentRunRow,
    Base,
    BudgetLedgerRow,
    Project,
    ProjectSpecRow,
    ReviewResultRow,
    TaskEdgeRow,
    TaskGraphRow,
    TaskNodeRow,
    WorkflowEventRow,
    WorkflowRunRow,
)
from projectforge_orchestrator.engine import PostgresFSMEngine
from projectforge_orchestrator.scheduler import TaskScheduler
from projectforge_schemas.models import TaskGraph, TaskNode, WorkflowState


SPEC_DATA = {
    "title": "Workflow integration fixture",
    "description": "Exercise the deterministic workflow FSM.",
    "deliverables": ["Report", "Source"],
    "constraints": ["Do not execute code in Phase 2"],
    "tech_stack_hints": ["Python"],
    "rubric_items": [
        {"criterion": "Evidence exists", "type": "automatable",
         "check_method": "Stub evidence record"},
        {"criterion": "Clear explanation", "type": "human-judgment",
         "check_method": "Human approval gate"},
    ],
}


def _node(task_id: str, *, dependencies: list[str] | None = None,
          read_set: list[str] | None = None, write_set: list[str] | None = None) -> dict:
    return {
        "id": task_id,
        "description": f"Build {task_id}",
        "dependencies": dependencies or [],
        "acceptance_criteria": [f"{task_id} is represented in the audit output"],
        "assigned_agent": "coder",
        "read_set": read_set or [],
        "write_set": write_set or [],
    }


def _database(tmp_path, *, token_cap: int = 1000, used_tokens: int = 0,
              spend_cap: Decimal = Decimal("10.00"), used_spend: Decimal = Decimal("0"),
              tasks: list[dict] | None = None):
    db_path = tmp_path / f"projectforge-{uuid4().hex}.sqlite"
    engine = build_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    project_id, graph_id = uuid4(), uuid4()
    tasks = tasks or [_node("only-task")]
    graph = TaskGraph.model_validate({"nodes": tasks})
    now = datetime.now(timezone.utc)
    with factory() as session, session.begin():
        session.add(Project(id=project_id, name="Workflow fixture"))
        session.flush()
        session.add(ProjectSpecRow(
            project_id=project_id,
            title=SPEC_DATA["title"],
            description=SPEC_DATA["description"],
            deliverables=SPEC_DATA["deliverables"],
            constraints=SPEC_DATA["constraints"],
            tech_stack_hints=SPEC_DATA["tech_stack_hints"],
            rubric_items=SPEC_DATA["rubric_items"],
        ))
        session.add(BudgetLedgerRow(
            project_id=project_id,
            token_cap=token_cap,
            spend_cap=spend_cap,
            used_tokens=used_tokens,
            used_spend=used_spend,
        ))
        graph_row = TaskGraphRow(id=graph_id, project_id=project_id, version=1, created_at=now)
        session.add(graph_row)
        session.flush()
        for node in graph.nodes:
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
            ))
        session.flush()
        for edge in graph.edges:
            session.add(TaskEdgeRow(
                graph_id=graph_id, source_id=edge.source, target_id=edge.target,
            ))
    return engine, factory, project_id, graph_id


def _advance_to_plan_approval(engine: PostgresFSMEngine, project_id, graph_id):
    run = engine.start(project_id, graph_id)
    waiting = engine.drive(run.id)
    assert waiting.state is WorkflowState.AWAITING_PLAN_APPROVAL
    return waiting


def test_scheduler_runs_topologically_and_only_batches_resource_safe_tasks():
    graph = TaskGraph.model_validate({
        "nodes": [
            _node("a", write_set=["src/a"]),
            _node("b", write_set=["src/b"]),
            _node("c", read_set=["src/a"]),
            _node("d", dependencies=["a"], write_set=["tests"]),
        ]
    })
    batches = TaskScheduler(max_parallel_tasks=4).batches(graph)
    assert [[node.id for node in batch] for batch in batches] == [["a", "b"], ["c", "d"]]
    assert TaskScheduler.resource_conflict(
        TaskNode.model_validate(_node("x", write_set=["src/common/file.py"])),
        TaskNode.model_validate(_node("y", read_set=["src/common"])),
    )
    assert TaskScheduler.resource_conflict(
        TaskNode.model_validate(_node("x")),
        TaskNode.model_validate(_node("y", write_set=["src/other"])),
    )


def test_full_workflow_waits_for_both_approvals_and_emits_audit_derived_disclosure(tmp_path):
    engine, factory, project_id, graph_id = _database(
        tmp_path,
        tasks=[
            _node("a", write_set=["src/a"]),
            _node("b", write_set=["src/b"]),
            _node("c", dependencies=["a", "b"], write_set=["tests"]),
        ],
    )
    workflow = PostgresFSMEngine(
        factory, agents=StubAgentSuite(), max_parallel_tasks=3,
    )
    created = workflow.start(project_id, graph_id)
    # The persisted spec is snapshotted at workflow creation for reproducibility.
    with factory() as session, session.begin():
        session.get(ProjectSpecRow, project_id).title = "Changed after run creation"
    plan_gate = workflow.drive(created.id)
    assert plan_gate.state is WorkflowState.AWAITING_PLAN_APPROVAL
    assert plan_gate.result_payload["normalized_spec"]["title"] == SPEC_DATA["title"]

    # The build loop cannot be entered until the first explicit human decision.
    approved_plan = workflow.approve(
        plan_gate.id, stage="plan", approved=True, comment="Plan reviewed.",
    )
    assert approved_plan.state is WorkflowState.BUILD_LOOP
    final_gate = workflow.drive(plan_gate.id)
    assert final_gate.state is WorkflowState.AWAITING_FINAL_APPROVAL
    assert final_gate.result_payload["integration"]["code_executed"] is False

    assert plan_gate.task_graph_id is not None
    planned_graph_id = plan_gate.task_graph_id
    with factory() as session:
        statuses = dict(session.execute(
            select(TaskNodeRow.id, TaskNodeRow.status)
            .where(TaskNodeRow.graph_id == planned_graph_id)
        ).all())
        assert statuses == {"a": "completed", "b": "completed", "c": "completed"}
        assert len(session.scalars(select(ReviewResultRow)).all()) == 2

    approved_final = workflow.approve(
        final_gate.id, stage="final", approved=True, comment="Review approved.",
    )
    assert approved_final.state is WorkflowState.PACKAGE
    done = workflow.drive(final_gate.id)
    assert done.state is WorkflowState.DONE
    disclosure = done.result_payload["disclosure"]
    assert disclosure["llm_calls"] == 0
    assert disclosure["code_execution_count"] == 0
    assert disclosure["agent_calls"] == 11
    assert disclosure["stub_agent_calls"] == 11
    assert disclosure["project_id"] == str(project_id)
    assert len(disclosure["audit_run_ids"]) == disclosure["agent_calls"]
    assert "persisted AgentRun records" in disclosure["statement"]
    assert done.result_payload["package"]["disclosure"] == disclosure

    events = workflow.list_events(done.id)
    states = [event.to_state for event in events if event.from_state != event.to_state]
    assert states == [
        WorkflowState.INTAKE,
        WorkflowState.SPEC,
        WorkflowState.PLAN,
        WorkflowState.AWAITING_PLAN_APPROVAL,
        WorkflowState.BUILD_LOOP,
        WorkflowState.INTEGRATION,
        WorkflowState.REVIEW,
        WorkflowState.AWAITING_FINAL_APPROVAL,
        WorkflowState.PACKAGE,
        WorkflowState.DONE,
    ]
    with factory() as session:
        agent_runs = session.scalars(
            select(AgentRunRow).where(AgentRunRow.workflow_id == done.id)
        ).all()
        assert len(agent_runs) == disclosure["agent_calls"]
        assert all(run.tokens_used == 0 and run.cost == Decimal("0") for run in agent_runs)
        assert all(run.output_payload.get("stub") is True for run in agent_runs)
        assert all(run.output_payload.get("evidence") for run in agent_runs)
        assert all(run.output_payload.get("scope") == run.input_payload["agent_scope"]
                   for run in agent_runs)
    engine.dispose()


@pytest.mark.parametrize(
    "budget,expected_reason",
    [
        ({"token_cap": 2, "used_tokens": 3}, "token budget exceeded"),
        ({"spend_cap": Decimal("0.01"), "used_spend": Decimal("0.02")},
         "spend budget exceeded"),
    ],
)
def test_budget_breach_escalates_before_any_stub_agent_call(tmp_path, budget, expected_reason):
    engine, factory, project_id, graph_id = _database(tmp_path, **budget)
    workflow = PostgresFSMEngine(factory, agents=StubAgentSuite())
    run = workflow.start(project_id, graph_id)
    escalated = workflow.drive(run.id)

    assert escalated.state is WorkflowState.ESCALATED
    assert expected_reason in escalated.failure_reason
    with factory() as session:
        assert session.scalar(
            select(AgentRunRow).where(AgentRunRow.workflow_id == run.id)
        ) is None
        states = [event.to_state for event in session.scalars(
            select(WorkflowEventRow).where(WorkflowEventRow.workflow_id == run.id)
            .order_by(WorkflowEventRow.sequence)
        ).all()]
        assert states[-1] == WorkflowState.ESCALATED.value
    engine.dispose()


def test_plan_rejection_returns_to_plan_with_feedback_and_replans_without_calling_coder(tmp_path):
    engine, factory, project_id, graph_id = _database(tmp_path)
    workflow = PostgresFSMEngine(factory, agents=StubAgentSuite())
    waiting = _advance_to_plan_approval(workflow, project_id, graph_id)
    rejected = workflow.approve(
        waiting.id, stage="plan", decision="request_changes",
        comment="Add a dedicated verification task before the final gate.",
    )
    assert rejected.state is WorkflowState.PLAN
    assert rejected.result_payload["plan_feedback"] == [
        "Add a dedicated verification task before the final gate."
    ]
    replanned = workflow.drive(waiting.id)
    assert replanned.state is WorkflowState.AWAITING_PLAN_APPROVAL
    with factory() as session:
        agents = session.scalars(
            select(AgentRunRow).where(AgentRunRow.workflow_id == waiting.id)
        ).all()
        assert [row.agent_name for row in agents].count("planner") == 2
        assert "coder" not in {row.agent_name for row in agents}
        assert session.get(TaskNodeRow, (waiting.task_graph_id, "only-task")).status == "pending"
        assert replanned.task_graph_id != waiting.task_graph_id
    engine.dispose()


def test_final_review_changes_are_task_scoped_propagated_and_retry_bounded(tmp_path):
    engine, factory, project_id, graph_id = _database(tmp_path)
    workflow = PostgresFSMEngine(factory, agents=StubAgentSuite())
    plan_gate = workflow.start(project_id, graph_id, max_retries=1)
    plan_gate = workflow.drive(plan_gate.id)
    workflow.approve(plan_gate.id, stage="plan", decision="approve")
    final_gate = workflow.drive(plan_gate.id)
    assert final_gate.state is WorkflowState.AWAITING_FINAL_APPROVAL

    changed = workflow.approve(
        final_gate.id,
        stage="final",
        decision="request_changes",
        comment="Fix the source task before accepting the final package.",
        task_feedback=[{
            "task_id": "only-task",
            "feedback": "Include the required source row identifier in the output.",
        }],
    )
    assert changed.state is WorkflowState.BUILD_LOOP
    with factory() as session:
        row = session.get(TaskNodeRow, (changed.task_graph_id, "only-task"))
        assert row.status == "pending"
        assert row.retries == 1
        assert "required source row identifier" in row.review_feedback[-1]

    second_gate = workflow.drive(changed.id)
    assert second_gate.state is WorkflowState.AWAITING_FINAL_APPROVAL
    with factory() as session:
        coder_runs = session.scalars(
            select(AgentRunRow).where(
                AgentRunRow.workflow_id == changed.id,
                AgentRunRow.agent_name == "coder",
            ).order_by(AgentRunRow.timestamp, AgentRunRow.id)
        ).all()
        assert len(coder_runs) == 2
        assert coder_runs[-1].input_payload["task_node"]["review_feedback"]

    exhausted = workflow.approve(
        second_gate.id,
        stage="final",
        decision="reject",
        comment="The same task still needs correction.",
        task_feedback=[{
            "task_id": "only-task",
            "feedback": "Correct the source row identifier before publication.",
        }],
    )
    assert exhausted.state is WorkflowState.ESCALATED
    assert "retry budget" in exhausted.failure_reason
    engine.dispose()


def test_cancellation_is_persisted_and_stops_a_waiting_workflow(tmp_path):
    engine, factory, project_id, graph_id = _database(tmp_path)
    workflow = PostgresFSMEngine(factory, agents=StubAgentSuite())
    waiting = _advance_to_plan_approval(workflow, project_id, graph_id)
    requested = workflow.cancel(waiting.id, reason="Requester stopped the run")
    assert requested.state is WorkflowState.AWAITING_PLAN_APPROVAL

    stopped = workflow.drive(waiting.id)
    assert stopped.state is WorkflowState.FAILED
    assert stopped.failure_reason == "workflow cancellation requested while awaiting approval"
    events = workflow.list_events(waiting.id)
    assert any(event.reason == "Cancellation requested." for event in events)
    with factory() as session:
        task = session.get(TaskNodeRow, (waiting.task_graph_id, "only-task"))
        assert task.status == "cancelled"
    engine.dispose()


def test_maximum_three_debug_retries_escalates_and_persists_retry_count(tmp_path):
    engine, factory, project_id, graph_id = _database(tmp_path)
    agents = StubAgentSuite(test_failures={"only-task": 4})
    workflow = PostgresFSMEngine(factory, agents=agents)
    plan_gate = _advance_to_plan_approval(workflow, project_id, graph_id)
    workflow.approve(plan_gate.id, stage="plan", approved=True)
    escalated = workflow.drive(plan_gate.id)

    assert escalated.state is WorkflowState.ESCALATED
    assert "failed after 3 retries" in escalated.failure_reason
    with factory() as session:
        task = session.get(TaskNodeRow, (plan_gate.task_graph_id, "only-task"))
        assert task.status == "failed"
        assert task.retries == 3
        calls = session.scalars(
            select(AgentRunRow).where(AgentRunRow.workflow_id == plan_gate.id)
        ).all()
        phases = [call.input_payload.get("phase") for call in calls]
        assert phases.count("CODE") == 4
        assert phases.count("TEST") == 4
        assert phases.count("DEBUG") == 3
        assert all(call.output_payload.get("stub") is True for call in calls)
    engine.dispose()
