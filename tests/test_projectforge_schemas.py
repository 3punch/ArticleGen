from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from projectforge_schemas.models import (
    AIUsageDisclosure,
    AgentFileChange,
    AgentRun,
    AgentUsageActivity,
    CoderAgentResponse,
    HumanReviewRecommendation,
    PackagerAgentResponse,
    ReviewCriterionAssessment,
    ReviewerAgentResponse,
    DebuggerAgentResponse,
    BudgetLedger,
    ProjectSpec,
    ProjectUpdate,
    ReviewResult,
    RubricItem,
    RetrievedContextChunk,
    TaskGraph,
    TaskNode,
    TesterAgentResponse as BuildTesterAgentResponse,
    WorkflowApprovalRequest,
    WorkflowEventRecord,
    WorkflowRunRecord,
    WorkflowStartRequest,
    WorkflowState,
)


PROJECT_SPEC = {
    "title": "Statistical Analysis Project",
    "description": "Analyze the supplied dataset and document reproducible results.",
    "deliverables": ["Report", "Source code"],
    "constraints": ["Use Python"],
    "tech_stack_hints": ["Python", "pandas"],
    "rubric_items": [
        {"criterion": "Reproducibility", "type": "automatable",
         "check_method": "Run the provided test suite"},
        {"criterion": "Discussion quality", "type": "human-judgment",
         "check_method": "Instructor review"},
    ],
}


def task(task_id: str, *, deps: list[str] | None = None, agent: str = "coder") -> dict:
    return {
        "id": task_id,
        "description": f"Complete {task_id}",
        "dependencies": deps or [],
        "acceptance_criteria": [f"{task_id} is checked"],
        "assigned_agent": agent,
    }


def test_project_spec_accepts_both_rubric_types_and_normalizes_strings():
    spec = ProjectSpec.model_validate(PROJECT_SPEC)
    assert spec.rubric_items[0].type == "automatable"
    assert spec.rubric_items[1].type == "human-judgment"
    assert spec.deliverables == ["Report", "Source code"]


@pytest.mark.parametrize(
    "update",
    [
        {"deliverables": []},
        {"deliverables": ["  "]},
        {"rubric_items": [{"criterion": "x", "type": "automatic", "check_method": "test"}]},
        {"rubric_items": [{"criterion": "x", "type": "automatable", "check_method": " "}]},
        {"rubric_items": [
            {"criterion": "Same", "type": "automatable", "check_method": "test"},
            {"criterion": "same", "type": "human-judgment", "check_method": "review"},
        ]},
        {"unexpected": "field"},
    ],
)
def test_project_spec_rejects_invalid_requirements(update):
    payload = {**PROJECT_SPEC, **update}
    with pytest.raises(ValidationError):
        ProjectSpec.model_validate(payload)


def test_task_node_rejects_invalid_retry_count_agent_and_dependencies():
    with pytest.raises(ValidationError):
        TaskNode.model_validate({**task("code"), "retries": -1})
    with pytest.raises(ValidationError):
        TaskNode.model_validate({**task("code"), "assigned_agent": "orchestrator"})
    with pytest.raises(ValidationError):
        TaskNode.model_validate(task("code", deps=["code"]))
    with pytest.raises(ValidationError):
        TaskNode.model_validate({**task("code"), "dependencies": ["a", "a"]})


def test_task_node_write_set_requires_unique_safe_relative_paths():
    node = TaskNode.model_validate({**task("code"), "write_set": ["src/app.py", "tests/"]})
    assert node.write_set == ["src/app.py", "tests"]
    for unsafe in ("/etc/passwd", "../outside", "src/../outside", "C:\\\\tmp", ""):
        with pytest.raises(ValidationError):
            TaskNode.model_validate({**task("code"), "write_set": [unsafe]})
    with pytest.raises(ValidationError):
        TaskNode.model_validate({**task("code"), "write_set": ["src/app.py", "src/app.py"]})


def test_task_node_carries_a_supported_runtime_without_breaking_older_graphs():
    assert TaskNode.model_validate(task("legacy")).runtime is None
    assert TaskNode.model_validate({**task("python"), "runtime": "python"}).runtime == "python"
    assert TaskNode.model_validate({**task("node"), "runtime": "node"}).runtime == "node"
    with pytest.raises(ValidationError):
        TaskNode.model_validate({**task("invalid"), "runtime": "bash"})


def test_agent_file_change_contract_blocks_escape_secrets_and_invalid_actions():
    assert AgentFileChange(path="src/main.py", content="print('ok')").action == "write"
    assert AgentFileChange(path="src/old.py", action="delete").content is None
    for path in ("../outside.py", "/etc/passwd", r"src\main.py", "src/bad\x00.py",
                 ".git/config", ".env", "secrets/id_rsa", "cert.pem"):
        with pytest.raises(ValidationError):
            AgentFileChange(path=path, content="unsafe")
    with pytest.raises(ValidationError, match="require file content"):
        AgentFileChange(path="src/empty.py")
    with pytest.raises(ValidationError, match="must not include file content"):
        AgentFileChange(path="src/old.py", action="delete", content="not a deletion")


def test_build_agent_response_contracts_are_structured_and_evidence_bearing():
    coder = CoderAgentResponse.model_validate({
        "task_id": "implement-cli",
        "summary": "Implement the CLI entry point and its validation helpers.",
        "changes": [{"path": "src/main.py", "content": "def main(): pass\\n"}],
        "evidence": ["The change is scoped to the declared source directory."],
    })
    assert coder.changes[0].path == "src/main.py"

    tester = BuildTesterAgentResponse.model_validate({
        "summary": "Add regression tests for required columns and malformed rows.",
        "test_files": [{"path": "tests/test_cli.py", "content": "def test_cli(): pass\\n"}],
        "evidence": ["Each test maps to an acceptance criterion."],
    })
    assert tester.test_files[0].path.startswith("tests/")

    debugger = DebuggerAgentResponse.model_validate({
        "diagnosis": "The parser accepts a row with a negative count, violating the input constraint.",
        "fix_instructions": ["Reject negative counts before building the summary."],
        "evidence": ["The failing test names the negative-count case."],
    })
    assert debugger.fix_instructions

    with pytest.raises(ValidationError):
        CoderAgentResponse.model_validate({
            "task_id": "implement-cli", "summary": "A valid enough summary.",
            "changes": [], "evidence": ["evidence"],
        })
    with pytest.raises(ValidationError):
        BuildTesterAgentResponse.model_validate({
            "summary": "A valid enough summary.",
            "test_files": [{"path": "src/not-a-test.py", "content": "pass\\n"}],
            "evidence": ["evidence"],
        })


def test_retrieved_context_chunk_is_bounded_and_project_scoped():
    chunk = RetrievedContextChunk.model_validate({
        "chunk_id": uuid4(), "project_id": uuid4(), "chunk_kind": "constraint",
        "chunk_key": "constraint:0", "ordinal": 0,
        "content": "The CLI must not require network access.", "similarity": 0.84,
        "metadata": {"source": "project_spec"},
    })
    assert 0 <= chunk.similarity <= 1
    with pytest.raises(ValidationError):
        RetrievedContextChunk.model_validate({
            **chunk.model_dump(mode="json"), "similarity": 1.1,
        })


def test_task_graph_derives_edges_and_round_trips_dependencies():
    graph = TaskGraph.model_validate({"nodes": [task("research"), task("code", deps=["research"]) ]})
    assert [(edge.source, edge.target) for edge in graph.edges] == [("research", "code")]
    assert graph.model_dump(mode="json")["nodes"][1]["dependencies"] == ["research"]


def test_task_graph_accepts_tasknode_instances_when_deriving_edges():
    graph = TaskGraph(nodes=[
        TaskNode.model_validate(task("research")),
        TaskNode.model_validate(task("code", deps=["research"])),
    ])
    assert [(edge.source, edge.target) for edge in graph.edges] == [("research", "code")]


@pytest.mark.parametrize(
    "nodes,edges",
    [
        ([task("a"), task("a")], []),
        ([task("a", deps=["missing"])], None),
        ([task("a", deps=["b"]), task("b", deps=["a"])], None),
        ([task("a"), task("b", deps=["a"])], []),
        ([task("a"), task("b")], [{"source": "a", "target": "missing"}]),
    ],
)
def test_task_graph_rejects_duplicate_unknown_or_inconsistent_edges(nodes, edges):
    payload = {"nodes": nodes}
    if edges is not None:
        payload["edges"] = edges
    with pytest.raises(ValidationError):
        TaskGraph.model_validate(payload)


def test_agent_run_requires_nonnegative_usage_and_timezone_aware_timestamp():
    common = {
        "agent_name": "coder",
        "input": {"task": "implement"},
        "output": {"status": "completed"},
        "tokens_used": 12,
        "cost": "0.0012",
        "timestamp": datetime.now(timezone.utc),
        "project_id": uuid4(),
    }
    record = AgentRun.model_validate(common)
    assert record.tokens_used == 12
    assert record.cost == Decimal("0.0012")

    with pytest.raises(ValidationError):
        AgentRun.model_validate({**common, "tokens_used": -1})
    with pytest.raises(ValidationError):
        AgentRun.model_validate({**common, "cost": "-0.01"})
    with pytest.raises(ValidationError):
        AgentRun.model_validate({**common, "timestamp": datetime.now()})
    with pytest.raises(ValidationError):
        AgentRun.model_validate({**common, "agent_name": "writer"})


def test_review_result_requires_strict_boolean_and_nonempty_evidence_items():
    rubric = RubricItem.model_validate(PROJECT_SPEC["rubric_items"][0])
    review = ReviewResult(rubric_item=rubric, passed=True, evidence=["test run #12"])
    assert review.passed is True
    assert review.status == "passed"
    with pytest.raises(ValidationError):
        ReviewResult(rubric_item=rubric, passed="yes", evidence=[])
    with pytest.raises(ValidationError):
        ReviewResult(rubric_item=rubric, passed=False, evidence=[" "])


def test_reviewer_contract_never_marks_human_judgment_as_machine_passed():
    human_rubric = RubricItem.model_validate(PROJECT_SPEC["rubric_items"][1])
    recommendation = HumanReviewRecommendation(
        criterion=human_rubric.criterion,
        recommendation="approve",
        rationale="The deliverables appear clear and the evidence supports the claim.",
        evidence=["The generated report contains the requested discussion section."],
    )
    assessment = ReviewCriterionAssessment(
        rubric_item=human_rubric,
        status="human_review_required",
        recommendation=recommendation.recommendation,
        rationale=recommendation.rationale,
        evidence=recommendation.evidence,
    )
    response = ReviewerAgentResponse(
        summary="Automated checks passed; one subjective rubric item requires a human decision.",
        criteria=[assessment],
        automatable_checks_passed=True,
        human_review_required=True,
        evidence=["Human-review item remains explicitly pending."],
    )
    assert response.criteria[0].status == "human_review_required"
    with pytest.raises(ValidationError):
        ReviewCriterionAssessment(
            rubric_item=human_rubric,
            status="passed",
            evidence=["The model liked it."],
            recommendation="approve",
        )
    with pytest.raises(ValidationError):
        ReviewResult(
            rubric_item=human_rubric,
            passed=True,
            status="passed",
            evidence=["Model output"],
            recommendation="approve",
        )


def test_reviewer_and_packager_outputs_require_evidence_and_markdown_documents():
    automated_rubric = RubricItem.model_validate(PROJECT_SPEC["rubric_items"][0])
    assessment = ReviewCriterionAssessment(
        rubric_item=automated_rubric,
        status="not_verified",
        evidence=["Coverage tooling was not available in the sandbox."],
    )
    response = ReviewerAgentResponse(
        summary="One automated rubric item could not be verified and needs follow-up.",
        criteria=[assessment],
        automatable_checks_passed=False,
        human_review_required=False,
        evidence=["Static source checks were completed."],
    )
    assert response.criteria[0].status == "not_verified"
    with pytest.raises(ValidationError):
        ReviewerAgentResponse(
            summary="An invalid result would have silently marked an unverified criterion as passed.",
            criteria=[assessment],
            automatable_checks_passed=True,
            human_review_required=False,
            evidence=["Conflicting summary flag."],
        )
    package = PackagerAgentResponse(
        readme_markdown="# Project\n\nRun the test suite.",
        report_markdown="# Review report\n\nTests passed.",
        summary="The package contains documented source and verified test evidence.",
        evidence=["README and Markdown report satisfy the package contract."],
    )
    assert package.report_markdown.startswith("#")
    with pytest.raises(ValidationError):
        PackagerAgentResponse(
            readme_markdown="README without a heading and enough descriptive content.",
            report_markdown="# Report\n\nDetails.",
            summary="This must fail because the README is not a Markdown document.",
            evidence=["bad"],
        )


def test_final_gate_rejections_require_task_scoped_actionable_feedback():
    plan_rejection = WorkflowApprovalRequest(
        stage="plan", decision="request_changes", comment="Split the integration task."
    )
    assert plan_rejection.approved is False
    assert plan_rejection.decision == "request_changes"

    final_rejection = WorkflowApprovalRequest(
        stage="final", decision="reject", comment="The output misses its input validation.",
        task_feedback=[{"task_id": "implement-parser",
                        "feedback": "Reject malformed rows before producing any output."}],
    )
    assert final_rejection.task_feedback[0].task_id == "implement-parser"
    with pytest.raises(ValidationError):
        WorkflowApprovalRequest(stage="final", decision="reject")
    with pytest.raises(ValidationError):
        WorkflowApprovalRequest(
            stage="plan", approved=True, decision="reject",
        )


def test_usage_activities_are_timestamped_and_reference_the_audit_run():
    activity = AgentUsageActivity(
        audit_run_id=uuid4(), agent_name="reviewer", action="Static review and rubric assessment",
        timestamp=datetime.now(timezone.utc), provider="dahl", model="review-model",
        tokens_used=42, spend="0.001",
    )
    assert activity.model == "review-model"
    with pytest.raises(ValidationError):
        AgentUsageActivity(
            audit_run_id=uuid4(), agent_name="packager", action="Generated package",
            timestamp=datetime.now(), tokens_used=0, spend="0",
        )


def test_budget_ledger_rejects_negative_values_but_can_record_a_breach():
    project_id = uuid4()
    ledger = BudgetLedger(
        project_id=project_id, token_cap=100, spend_cap=Decimal("1.00"),
        used_tokens=120, used_spend=Decimal("1.20"),
    )
    assert ledger.used_tokens > ledger.token_cap
    assert ledger.used_spend > ledger.spend_cap
    with pytest.raises(ValidationError):
        BudgetLedger(project_id=project_id, token_cap=-1, spend_cap=Decimal("1"))
    with pytest.raises(ValidationError):
        BudgetLedger(project_id=project_id, token_cap=1, spend_cap=Decimal("-0.01"))


def test_workflow_contracts_cover_required_states_and_audit_events():
    assert [state.value for state in WorkflowState] == [
        "INTAKE", "SPEC", "PLAN", "AWAITING_PLAN_APPROVAL", "AWAITING_CLARIFICATION", "BUILD_LOOP",
        "INTEGRATION", "REVIEW", "AWAITING_FINAL_APPROVAL", "PACKAGE", "DONE",
        "FAILED", "ESCALATED",
    ]
    now = datetime.now(timezone.utc)
    run = WorkflowRunRecord(
        id=uuid4(), project_id=uuid4(), task_graph_id=uuid4(),
        state=WorkflowState.AWAITING_PLAN_APPROVAL, max_retries=3,
        created_at=now, updated_at=now,
    )
    assert run.state is WorkflowState.AWAITING_PLAN_APPROVAL
    event = WorkflowEventRecord(
        workflow_id=run.id, sequence=1, from_state=WorkflowState.INTAKE,
        to_state=WorkflowState.SPEC, reason="intake accepted", created_at=now,
    )
    assert event.sequence == 1
    with pytest.raises(ValidationError):
        WorkflowEventRecord(
            workflow_id=run.id, sequence=0, to_state=WorkflowState.SPEC,
            reason="invalid sequence", created_at=now,
        )
    with pytest.raises(ValidationError):
        WorkflowRunRecord(
            id=uuid4(), project_id=uuid4(), task_graph_id=uuid4(),
            state="not-a-state", max_retries=3,
            created_at=now, updated_at=now,
        )


def test_workflow_request_caps_retries_and_disclosure_is_typed():
    project_id, workflow_id = uuid4(), uuid4()
    start = WorkflowStartRequest(task_graph_id=uuid4())
    assert start.max_retries == 3
    with pytest.raises(ValidationError):
        WorkflowStartRequest(task_graph_id=uuid4(), max_retries=4)
    approval = WorkflowApprovalRequest(stage="plan", approved=True, comment="Reviewed")
    assert approval.approved is True
    with pytest.raises(ValidationError):
        WorkflowApprovalRequest(stage="plan", approved="true")
    disclosure = AIUsageDisclosure(
        project_id=project_id, workflow_id=workflow_id, agent_calls=1,
        agent_names=["coder"], audit_run_ids=[uuid4()], tokens_used=0,
        spend=Decimal("0"), statement="No LLM usage.",
        generated_at=datetime.now(timezone.utc),
    )
    assert disclosure.llm_calls == 0
    assert disclosure.code_execution_count == 0


def test_project_update_must_change_at_least_one_field():
    with pytest.raises(ValidationError):
        ProjectUpdate.model_validate({})
    assert ProjectUpdate.model_validate({"name": "Updated"}).name == "Updated"
