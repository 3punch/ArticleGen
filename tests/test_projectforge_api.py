from __future__ import annotations

import asyncio
from uuid import UUID

import httpx

from projectforge_agents.git_workspace import GitWorkspaceManager
from projectforge_api.main import create_app
from projectforge_api.orm import TaskNodeRow
from projectforge_schemas.models import WorkflowState


SPEC = {
    "title": "Research Methods Assignment",
    "description": "Build and document a reproducible project that meets the rubric.",
    "deliverables": ["Project files", "Usage guide"],
    "constraints": ["Keep the data reproducible"],
    "tech_stack_hints": ["Python"],
    "rubric_items": [
        {"criterion": "Correctness", "type": "automatable", "check_method": "Run tests"},
        {"criterion": "Clarity", "type": "human-judgment", "check_method": "Review report"},
    ],
}

GRAPH = {
    "nodes": [
        {"id": "requirements", "description": "Extract requirements",
         "acceptance_criteria": ["Requirements are listed"], "assigned_agent": "requirement"},
        {"id": "implementation", "description": "Implement the approved plan",
         "dependencies": ["requirements"],
         "acceptance_criteria": ["Implementation meets the spec"], "assigned_agent": "coder"},
    ]
}


class RecordingDispatcher:
    def __init__(self):
        self.jobs = []

    def enqueue_workflow(self, workflow_id, *, timeout_seconds: int):
        self.jobs.append((workflow_id, timeout_seconds))
        return f"job-{len(self.jobs)}"


class SyncASGIClient:
    """Small sync wrapper around HTTPX ASGITransport (avoids Starlette TestClient)."""

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

    def patch(self, url: str, **kwargs):
        return self.request("PATCH", url, **kwargs)

    def put(self, url: str, **kwargs):
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs):
        return self.request("DELETE", url, **kwargs)


def make_client(monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.delenv("PROJECTFORGE_DATABASE_URL", raising=False)
    app = create_app(database_url="sqlite:///:memory:", create_schema_for_tests=True)
    return SyncASGIClient(app)


def create_project(client: SyncASGIClient) -> dict:
    response = client.post("/api/projects", json={
        "name": "Campus Research Tool",
        "spec": SPEC,
        "token_cap": 25_000,
        "spend_cap": "4.50",
    })
    assert response.status_code == 201, response.text
    return response.json()


def test_project_crud_endpoints():
    with make_client() as client:
        assert client.get("/health").json() == {"status": "ok", "database": "reachable"}
        created = create_project(client)
        project_id = created["id"]
        assert created["name"] == "Campus Research Tool"
        assert created["spec"]["rubric_items"][1]["type"] == "human-judgment"
        assert created["budget"]["token_cap"] == 25_000
        assert created["budget"]["spend_cap"].startswith("4.5")

        fetched = client.get(f"/api/projects/{project_id}")
        assert fetched.status_code == 200
        assert fetched.json()["id"] == project_id

        listed = client.get("/api/projects")
        assert listed.status_code == 200
        assert [item["id"] for item in listed.json()] == [project_id]

        updated = client.patch(f"/api/projects/{project_id}", json={
            "name": "Updated Campus Tool", "token_cap": 30_000,
        })
        assert updated.status_code == 200
        assert updated.json()["name"] == "Updated Campus Tool"
        assert updated.json()["budget"]["token_cap"] == 30_000

        deleted = client.delete(f"/api/projects/{project_id}")
        assert deleted.status_code == 204
        assert client.get(f"/api/projects/{project_id}").status_code == 404


def test_project_create_rejects_invalid_rubric_and_update_rejects_empty_patch():
    with make_client() as client:
        invalid = {**SPEC, "rubric_items": [
            {"criterion": "Invalid type", "type": "automatic", "check_method": "Run tests"}
        ]}
        response = client.post("/api/projects", json={"name": "Invalid", "spec": invalid})
        assert response.status_code == 422

        project = create_project(client)
        patch = client.patch(f"/api/projects/{project['id']}", json={})
        assert patch.status_code == 422


def test_task_graph_crud_and_versioned_replacement():
    with make_client() as client:
        project = create_project(client)
        project_id = project["id"]

        created = client.post(f"/api/projects/{project_id}/task-graphs", json=GRAPH)
        assert created.status_code == 201, created.text
        graph = created.json()
        graph_id = graph["id"]
        assert graph["version"] == 1
        assert graph["graph"]["edges"] == [{"source": "requirements", "target": "implementation"}]

        fetched = client.get(f"/api/projects/{project_id}/task-graphs/{graph_id}")
        assert fetched.status_code == 200
        assert fetched.json()["graph"]["nodes"][0]["assigned_agent"] in {"coder", "requirement"}

        listed = client.get(f"/api/projects/{project_id}/task-graphs")
        assert listed.status_code == 200
        assert len(listed.json()) == 1

        replacement = {
            "nodes": [
                {"id": "research", "description": "Review the supplied references",
                 "acceptance_criteria": ["References summarized"], "assigned_agent": "requirement"},
                {"id": "review", "description": "Review the resulting project",
                 "dependencies": ["research"],
                 "acceptance_criteria": ["Review evidence recorded"], "assigned_agent": "reviewer"},
            ]
        }
        replaced = client.put(
            f"/api/projects/{project_id}/task-graphs/{graph_id}", json=replacement
        )
        assert replaced.status_code == 200, replaced.text
        assert replaced.json()["version"] == 2
        assert {node["id"] for node in replaced.json()["graph"]["nodes"]} == {"research", "review"}

        removed = client.delete(f"/api/projects/{project_id}/task-graphs/{graph_id}")
        assert removed.status_code == 204
        assert client.get(f"/api/projects/{project_id}/task-graphs/{graph_id}").status_code == 404


def test_task_graph_rejects_cycle_and_unknown_dependencies():
    with make_client() as client:
        project = create_project(client)
        cyclic = {
            "nodes": [
                {"id": "a", "description": "A", "dependencies": ["b"],
                 "acceptance_criteria": ["done"], "assigned_agent": "coder"},
                {"id": "b", "description": "B", "dependencies": ["a"],
                 "acceptance_criteria": ["done"], "assigned_agent": "tester"},
            ]
        }
        response = client.post(f"/api/projects/{project['id']}/task-graphs", json=cyclic)
        assert response.status_code == 422

        unknown = {
            "nodes": [
                {"id": "a", "description": "A", "dependencies": ["missing"],
                 "acceptance_criteria": ["done"], "assigned_agent": "coder"}
            ]
        }
        response = client.post(f"/api/projects/{project['id']}/task-graphs", json=unknown)
        assert response.status_code == 422


def test_task_graph_endpoints_return_404_for_unknown_parent_or_graph():
    with make_client() as client:
        missing_project = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        assert client.get(f"/api/projects/{missing_project}/task-graphs").status_code == 404
        project = create_project(client)
        missing_graph = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
        assert client.get(
            f"/api/projects/{project['id']}/task-graphs/{missing_graph}"
        ).status_code == 404


def test_api_reports_missing_database_configuration(monkeypatch):
    monkeypatch.delenv("PROJECTFORGE_DATABASE_URL", raising=False)
    app = create_app()
    with SyncASGIClient(app) as client:
        assert client.get("/health").json() == {
            "status": "not-ready", "database": "not-configured"
        }
        assert client.get("/api/projects").status_code == 503


def test_review_api_exposes_plan_and_diff_and_routes_rejections_back_to_work(tmp_path):
    dispatcher = RecordingDispatcher()
    app = create_app(
        database_url="sqlite:///:memory:",
        create_schema_for_tests=True,
        queue_dispatcher=dispatcher,
    )
    with SyncASGIClient(app) as client:
        project = create_project(client)
        started = client.post(
            f"/api/projects/{project['id']}/workflows",
            json={"max_retries": 2, "timeout_seconds": 300},
        )
        assert started.status_code == 202, started.text
        workflow_id = UUID(started.json()["id"])
        plan_gate = app.state.workflow_engine.drive(workflow_id)
        assert plan_gate.state is WorkflowState.AWAITING_PLAN_APPROVAL

        review = client.get(f"/api/workflows/{workflow_id}/review")
        assert review.status_code == 200, review.text
        assert review.json()["stage"] == "plan"
        assert review.json()["plan_summary"]
        assert review.json()["task_graph"]["nodes"]
        assert review.json()["code_diff"] == ""

        rejected = client.post(f"/api/workflows/{workflow_id}/approvals", json={
            "stage": "plan",
            "decision": "request_changes",
            "comment": "Add a dedicated task for documenting observable verification evidence.",
        })
        assert rejected.status_code == 200, rejected.text
        assert rejected.json()["state"] == WorkflowState.PLAN.value
        assert len(dispatcher.jobs) == 2

        replanned = app.state.workflow_engine.drive(workflow_id)
        assert replanned.state is WorkflowState.AWAITING_PLAN_APPROVAL
        approved_plan = client.post(f"/api/workflows/{workflow_id}/approvals", json={
            "stage": "plan", "decision": "approve", "comment": "The revised plan is clear.",
        })
        assert approved_plan.status_code == 200, approved_plan.text
        final_gate = app.state.workflow_engine.drive(workflow_id)
        assert final_gate.state is WorkflowState.AWAITING_FINAL_APPROVAL
        final_review = client.get(f"/api/workflows/{workflow_id}/review")
        assert final_review.status_code == 200, final_review.text
        assert final_review.json()["stage"] == "final"
        assert final_review.json()["task_graph"]["nodes"]
        assert final_review.json()["review"] is None  # explicit test stub has no itemized verdict

        task_id = final_gate.result_payload["plan"]["task_graph"]["nodes"][0]["id"]
        task_logs = client.get(f"/api/workflows/{workflow_id}/tasks/{task_id}/logs")
        assert task_logs.status_code == 200, task_logs.text
        assert {record["agent_name"] for record in task_logs.json()} == {"coder", "tester"}
        assert all(record["task_id"] == task_id for record in task_logs.json())
        assert all(record["timestamp"].endswith("Z") or "+00:00" in record["timestamp"]
                   for record in task_logs.json())
        assert client.get(f"/api/workflows/{workflow_id}/tasks/missing/logs").status_code == 404

        changes_requested = client.post(f"/api/workflows/{workflow_id}/approvals", json={
            "stage": "final",
            "decision": "request_changes",
            "comment": "The first deliverable is incomplete.",
            "task_feedback": [{
                "task_id": task_id,
                "feedback": "Include an observable verification result for this deliverable.",
            }],
        })
        assert changes_requested.status_code == 200, changes_requested.text
        assert changes_requested.json()["state"] == WorkflowState.BUILD_LOOP.value
        assert len(dispatcher.jobs) == 4
        with app.state.session_factory() as session:
            task_row = session.get(TaskNodeRow, (final_gate.task_graph_id, task_id))
            assert task_row.status == "pending"
            assert "observable verification result" in task_row.review_feedback[-1]
        assert client.get(f"/api/workflows/{workflow_id}/review").status_code == 409


def test_project_creation_initializes_its_git_workspace(tmp_path):
    workspaces = GitWorkspaceManager(tmp_path / "repos")
    app = create_app(
        database_url="sqlite:///:memory:",
        create_schema_for_tests=True,
        workspace_manager=workspaces,
    )
    with SyncASGIClient(app) as client:
        project = create_project(client)
        repo = workspaces.project_path(project["id"])
        assert repo.is_dir()
        assert (repo / ".git").is_dir()
