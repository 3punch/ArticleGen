from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import httpx

from projectforge_agents.config import ModelConfig
from projectforge_agents.litellm_client import ModelCompletion
from projectforge_agents.suite import ProjectForgeAgentSuite
from projectforge_api.main import create_app
from projectforge_api.orm import (
    AgentRunRow,
    BudgetLedgerRow,
    BudgetReservationRow,
    ProjectSpecRow,
    TaskGraphRow,
    TaskNodeRow,
)
from projectforge_schemas.models import WorkflowState


SPEC = {
    "title": "Workflow API test",
    "description": "Exercise human approval and RQ dispatch boundaries.",
    "deliverables": ["A report"],
    "constraints": [],
    "tech_stack_hints": ["Python"],
    "rubric_items": [
        {"criterion": "Evidence", "type": "automatable", "check_method": "Persist audit"}
    ],
}


class RecordingDispatcher:
    def __init__(self):
        self.jobs: list[tuple[UUID, int]] = []

    def enqueue_workflow(self, workflow_id: UUID, *, timeout_seconds: int) -> str:
        self.jobs.append((workflow_id, timeout_seconds))
        return f"job-{len(self.jobs)}"


class SyncASGIClient:
    def __init__(self, app):
        self.app = app
        self.transport = httpx.ASGITransport(app=app)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.app.state.engine is not None:
            self.app.state.engine.dispose()

    def request(self, method: str, url: str, **kwargs):
        async def _request():
            async with httpx.AsyncClient(
                transport=self.transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, url, **kwargs)
        return asyncio.run(_request())

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs):
        return self.request("DELETE", url, **kwargs)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "projectforge"


class ClarificationSequenceClient:
    def __init__(self):
        fixture = json.loads((FIXTURE_DIR / "simple_script.json").read_text(encoding="utf-8"))
        self.requirement_responses = [
            {
                "status": "needs_clarification",
                "confidence": 0.51,
                "questions": [{
                    "question_id": "input_format",
                    "question": "Which column names should the input CSV use?",
                    "why_it_matters": "The parser and automated validation depend on these exact input fields.",
                }],
                "uncertainties": ["The input CSV column names were not specified."],
            },
            fixture["requirement_response"],
        ]
        self.planner_response = fixture["planner_response"]
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str,
                 agent_name: str, timeout_seconds: int | None = None) -> ModelCompletion:
        self.calls.append((agent_name, user_prompt))
        if agent_name == "requirement":
            response = self.requirement_responses.pop(0)
        elif agent_name == "planner":
            response = self.planner_response
        else:
            raise AssertionError(f"unexpected model-backed agent: {agent_name}")
        return ModelCompletion(
            content=json.dumps(response, ensure_ascii=False),
            tokens_used=100,
            cost=Decimal("0.00100000"),
            provider="mock-dahl",
            model=f"openai/offline-{agent_name}-model",
        )


def _eval_config() -> ModelConfig:
    return ModelConfig.from_env({
        "PROJECTFORGE_LLM_PROVIDER": "dahl",
        "PROJECTFORGE_LLM_TIER": "strong",
        "PROJECTFORGE_DAHL_STRONG_MODEL": "offline-strong-model",
        "DAHL_API_KEY": "not-used-by-mock",
        "PROJECTFORGE_LLM_RESERVATION_TOKENS": "1000",
        "PROJECTFORGE_LLM_MAX_OUTPUT_TOKENS": "512",
        "PROJECTFORGE_LLM_RESERVATION_SPEND": "0.05000000",
        "PROJECTFORGE_LLM_RESERVATION_TTL_SECONDS": "600",
    })


def test_workflow_api_persists_raw_brief_pauses_for_clarification_then_persists_planner_dag():
    dispatcher = RecordingDispatcher()
    model_client = ClarificationSequenceClient()
    suite = ProjectForgeAgentSuite(config=_eval_config(), client=model_client)
    app = create_app(
        database_url="sqlite:///:memory:",
        create_schema_for_tests=True,
        queue_dispatcher=dispatcher,
        agent_suite=suite,
    )
    raw_brief = (
        "Build a Python CLI that reads a campus bicycle-count CSV, validates its fields, "
        "computes summary statistics, and writes a deterministic JSON report with tests."
    )
    with SyncASGIClient(app) as client:
        project_response = client.post("/api/projects", json={
            "name": "Raw brief intake",
            "raw_brief_text": raw_brief,
            "rubric_text": "Correct summary statistics; clear documentation.",
            "token_cap": 10000,
            "spend_cap": "2.00",
        })
        assert project_response.status_code == 201, project_response.text
        project_data = project_response.json()
        assert project_data["spec"] is None
        project_id = project_data["id"]

        created = client.post(f"/api/projects/{project_id}/workflows", json={
            "max_retries": 3,
            "timeout_seconds": 300,
        })
        assert created.status_code == 202, created.text
        workflow_id = UUID(created.json()["id"])
        assert created.json()["task_graph_id"] is None
        assert dispatcher.jobs == [(workflow_id, 300)]

        clarification_gate = app.state.workflow_engine.drive(workflow_id)
        assert clarification_gate.state is WorkflowState.AWAITING_CLARIFICATION
        assert clarification_gate.task_graph_id is None
        question = clarification_gate.result_payload["pending_questions"][0]
        assert question["question_id"] == "input_format"
        assert clarification_gate.result_payload["requirement_response"]["project_spec"] is None

        resumed = client.post(f"/api/workflows/{workflow_id}/clarifications", json={
            "answers": {"input_format": "The CSV columns are date and count."},
        })
        assert resumed.status_code == 202, resumed.text
        assert resumed.json()["state"] == WorkflowState.SPEC.value
        assert len(dispatcher.jobs) == 2

        plan_gate = app.state.workflow_engine.drive(workflow_id)
        assert plan_gate.state is WorkflowState.AWAITING_PLAN_APPROVAL
        assert plan_gate.task_graph_id is not None
        assert plan_gate.result_payload["input_snapshot"]["clarification_answers"] == {
            "input_format": "The CSV columns are date and count."
        }
        assert plan_gate.result_payload["input_snapshot"]["requirement_revision"] == 1
        graph_id = plan_gate.task_graph_id
        graph_response = client.get(
            f"/api/projects/{project_id}/task-graphs/{graph_id}"
        )
        assert graph_response.status_code == 200, graph_response.text
        assert 3 <= len(graph_response.json()["graph"]["nodes"]) <= 6

        assert [name for name, _ in model_client.calls] == [
            "requirement", "requirement", "planner",
        ]
        assert "The CSV columns are date and count." in model_client.calls[1][1]
        with app.state.session_factory() as session:
            stored_spec = session.get(ProjectSpecRow, UUID(project_id))
            assert stored_spec is not None
            assert stored_spec.title == "Campus Bicycle Count Summary CLI"
            runs = session.query(AgentRunRow).filter(AgentRunRow.workflow_id == workflow_id).all()
            assert len(runs) == 3
            assert all(run.output_payload["llm_calls"] == 1 for run in runs)
            assert all(run.output_payload["stub"] is False for run in runs)
            versions = [run.output_payload["prompt_version"] for run in runs]
            assert versions.count("requirement_v1") == 2
            assert versions.count("planner_v1") == 1
            assert all(run.budget_reservation_id is not None for run in runs)
            reservations = session.query(BudgetReservationRow).all()
            assert len(reservations) == 3
            assert all(row.state == "settled" for row in reservations)
            ledger = session.get(BudgetLedgerRow, UUID(project_id))
            assert ledger.used_tokens == 300
            assert ledger.used_spend == Decimal("0.00300000")
            assert ledger.reserved_tokens == 0
            assert ledger.reserved_spend == Decimal("0")
            planned_graph = session.get(TaskGraphRow, graph_id)
            assert planned_graph is not None
            assert session.query(TaskNodeRow).filter(TaskNodeRow.graph_id == graph_id).count() == 4
