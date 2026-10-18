from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
import subprocess
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from projectforge_agents.config import ModelConfig
from projectforge_agents.git_workspace import AgentWriteScopeError, GitWorkspaceManager
from projectforge_agents.rag import PgVectorContextRetriever, chunk_project_spec, embed_text
from projectforge_agents.suite import ProjectForgeAgentSuite
from projectforge_api.db import build_engine, make_session_factory
from projectforge_api.orm import (
    AgentRunRow,
    Base,
    BudgetLedgerRow,
    Project,
    ProjectContextChunkRow,
    ProjectSpecRow,
    TaskNodeRow,
)
from projectforge_orchestrator.engine import PostgresFSMEngine
from projectforge_sandbox import SandboxManager, SandboxResult
from projectforge_schemas.models import AgentFileChange, ProjectSpec, WorkflowState


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "projectforge"
FIXTURE = json.loads((FIXTURE_DIR / "simple_script.json").read_text(encoding="utf-8"))


CLI_SOURCE = '''from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any


def summarize_csv(input_path: str | Path) -> dict[str, Any]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        missing = [name for name in ("date", "count") if name not in headers]
        if missing:
            raise ValueError(f"missing required CSV header(s): {', '.join(missing)}")
        counts: list[int] = []
        for line_number, row in enumerate(reader, start=2):
            date_value = (row.get("date") or "").strip()
            count_value = (row.get("count") or "").strip()
            try:
                date.fromisoformat(date_value)
                count = int(count_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid date or count on row {line_number}") from exc
            if count < 0:
                raise ValueError(f"count must not be negative on row {line_number}")
            counts.append(count)
    if not counts:
        raise ValueError("CSV contains no data rows")
    total = sum(counts)
    return {
        "row_count": len(counts),
        "total_count": total,
        "mean_count": total / len(counts),
        "maximum_daily_count": max(counts),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize daily campus bicycle counts.")
    parser.add_argument("input_csv", help="UTF-8 CSV containing date and count columns")
    parser.add_argument("--output", required=True, help="path for the deterministic JSON summary")
    args = parser.parse_args(argv)
    try:
        summary = summarize_csv(args.input_csv)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

TEST_SOURCE = '''import json
from pathlib import Path

import pytest

from src.bicycle_summary import main, summarize_csv


FIXTURE = "date,count\\n2026-01-01,4\\n2026-01-02,8\\n2026-01-03,12\\n"


def test_fixture_metrics_and_cli_json_are_deterministic(tmp_path):
    source = tmp_path / "counts.csv"
    source.write_text(FIXTURE, encoding="utf-8")
    assert summarize_csv(source) == {
        "row_count": 3,
        "total_count": 24,
        "mean_count": 8,
        "maximum_daily_count": 12,
    }
    output = tmp_path / "reports" / "summary.json"
    assert main([str(source), "--output", str(output)]) == 0
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert set(parsed) == {"row_count", "total_count", "mean_count", "maximum_daily_count"}
    assert parsed["row_count"] == 3
    assert parsed["total_count"] == 24
    assert parsed["mean_count"] == 8
    assert parsed["maximum_daily_count"] == 12


def test_missing_required_header_is_named(tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text("day,count\\n2026-01-01,4\\n", encoding="utf-8")
    with pytest.raises(ValueError, match="date"):
        summarize_csv(source)


def test_malformed_date_and_negative_count_are_rejected_with_row_number(tmp_path):
    malformed = tmp_path / "bad-date.csv"
    malformed.write_text("date,count\\nnot-a-date,4\\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 2"):
        summarize_csv(malformed)
    negative = tmp_path / "negative.csv"
    negative.write_text("date,count\\n2026-01-01,-1\\n", encoding="utf-8")
    with pytest.raises(ValueError, match="negative.*row 2"):
        summarize_csv(negative)
'''

README_SOURCE = """# Campus Bicycle Count Summary CLI\n\nRun `python -m src.bicycle_summary counts.csv --output reports/summary.json`. The UTF-8 CSV must contain `date` and `count` columns. Dates use ISO format, counts are non-negative integers, and malformed rows are reported with their row number. The JSON output contains `row_count`, `total_count`, `mean_count`, and `maximum_daily_count` in deterministic key order. The program has no network dependency.\n"""


class OfflineBuildModelClient:
    """Fixture-backed model client exercising every real worker without network access."""

    def __init__(self, *, fail_first_test_cycle: bool = False) -> None:
        self.fail_first_test_cycle = fail_first_test_cycle
        self.calls: list[tuple[str, str, str]] = []
        self.coder_calls: dict[str, int] = {}
        self.requirement = FIXTURE["requirement_response"]
        self.planner = FIXTURE["planner_response"]
        self.tests = {
            "summary": "Add deterministic CLI and malformed-input regression tests.",
            "test_files": [{"path": "tests/test_bicycle_summary.py", "content": TEST_SOURCE}],
            "evidence": ["The fixture, all metrics, missing columns, malformed rows, and CLI JSON are covered."],
        }
        self.debugger = {
            "diagnosis": "The initial execution exposed an implementation mismatch in the CSV validation path.",
            "fix_instructions": [
                "Make the parser reject invalid or negative counts with the row number while preserving the required summary fields."
            ],
            "evidence": ["The sandbox test run reported a failing acceptance test."],
        }
        self.correct_source = CLI_SOURCE
        self.faulty_source = CLI_SOURCE.replace(
            "if count < 0:\n                raise ValueError(f\"count must not be negative on row {line_number}\")",
            "if count < 0:\n                pass",
        )

    def complete(self, *, system_prompt: str, user_prompt: str,
                 agent_name: str, timeout_seconds: int | None = None):
        self.calls.append((agent_name, system_prompt, user_prompt))
        if agent_name == "requirement":
            output = self.requirement
        elif agent_name == "planner":
            output = self.planner
        elif agent_name == "coder":
            task = json.loads(user_prompt)["task"]
            task_id = task["id"]
            count = self.coder_calls.get(task_id, 0) + 1
            self.coder_calls[task_id] = count
            if task_id == "tests-and-readme":
                changes = [{"path": "README.md", "content": README_SOURCE}]
                summary = "Document the invocation, CSV assumptions, and malformed-row behavior."
            else:
                source = (
                    self.faulty_source
                    if self.fail_first_test_cycle and task_id == "validate-csv" and count == 1
                    else self.correct_source
                )
                changes = [
                    {"path": "src/__init__.py", "content": ""},
                    {"path": "src/bicycle_summary.py", "content": source},
                ]
                summary = "Implement CSV validation and deterministic bicycle-count summary output."
            output = {
                "task_id": task_id,
                "summary": summary,
                "changes": changes,
                "evidence": ["The proposed files implement the declared acceptance behavior."],
            }
        elif agent_name == "tester":
            output = self.tests
        elif agent_name == "debugger":
            output = self.debugger
        elif agent_name == "reviewer":
            payload = json.loads(user_prompt)
            human_items = payload["human_judgment_criteria"]
            output = {
                "summary": "The README is clear enough for a human to inspect the documented input and errors.",
                "recommendations": [{
                    "criterion": item["criterion"],
                    "recommendation": "approve",
                    "rationale": "README examples and the stated validation behavior are present; a human must confirm clarity.",
                    "evidence": ["README.md: documents the command, CSV fields, and malformed-row behavior."],
                } for item in human_items],
            }
        else:
            raise AssertionError(f"unexpected model-backed role: {agent_name}")
        from projectforge_agents.litellm_client import ModelCompletion

        return ModelCompletion(
            content=json.dumps(output, ensure_ascii=False),
            tokens_used=180,
            cost=Decimal("0.00100000"),
            provider="offline-fixture",
            model=f"fixture-{agent_name}",
        )


class RecordingSandbox:
    def __init__(self, *, first_fails: bool = False) -> None:
        self.first_fails = first_fails
        self.calls: list[dict[str, Any]] = []

    def run(self, runtime, command, *, project_dir, timeout_seconds, network_allowlist):
        self.calls.append({
            "runtime": runtime,
            "command": list(command),
            "project_dir": Path(project_dir),
            "timeout_seconds": timeout_seconds,
            "network_allowlist": tuple(network_allowlist),
        })
        fail = self.first_fails and len(self.calls) == 1
        return SandboxResult(
            runtime=runtime,
            stdout="FAILED test_malformed_date_and_negative_count_are_rejected_with_row_number\n" if fail else "5 passed in 0.03s\n",
            stderr="assert count >= 0\n" if fail else "",
            exit_code=1 if fail else 0,
            timed_out=False,
            duration_seconds=0.03,
            container_id=f"offline-container-{len(self.calls)}",
        )


def _config() -> ModelConfig:
    return ModelConfig.from_env({
        "PROJECTFORGE_LLM_PROVIDER": "dahl",
        "PROJECTFORGE_LLM_TIER": "strong",
        "PROJECTFORGE_DAHL_STRONG_MODEL": "offline-build-fixture",
        "PROJECTFORGE_LLM_RESERVATION_TOKENS": "20000",
        "PROJECTFORGE_LLM_MAX_OUTPUT_TOKENS": "2048",
        "PROJECTFORGE_LLM_RESERVATION_SPEND": "0.05000000",
        "PROJECTFORGE_LLM_RESERVATION_TTL_SECONDS": "600",
        "PROJECTFORGE_LLM_TIMEOUT_SECONDS": "30",
    })


def _project(tmp_path: Path, *, client: OfflineBuildModelClient,
             sandbox: Any) -> tuple[Any, Any, UUID, ProjectForgeAgentSuite, PostgresFSMEngine]:
    database_path = tmp_path / f"projectforge-{uuid4().hex}.sqlite"
    engine = build_engine(f"sqlite:///{database_path}")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    project_id = uuid4()
    with factory() as session, session.begin():
        session.add(Project(
            id=project_id,
            name=FIXTURE["name"],
            raw_brief_text=FIXTURE["raw_brief_text"],
            rubric_text=FIXTURE["rubric_text"],
        ))
        session.add(BudgetLedgerRow(
            project_id=project_id,
            token_cap=100_000,
            spend_cap=Decimal("5.00"),
            used_tokens=0,
            used_spend=Decimal("0"),
        ))
    suite = ProjectForgeAgentSuite(
        config=_config(),
        client=client,
        session_factory=factory,
        workspace_manager=GitWorkspaceManager(tmp_path / "repos"),
        sandbox_manager=sandbox,
    )
    workflow = PostgresFSMEngine(factory, agents=suite, max_parallel_tasks=2)
    return engine, factory, project_id, suite, workflow


def test_pgvector_context_index_retrieval_is_project_scoped_and_bounded(tmp_path):
    spec = ProjectSpec.model_validate(FIXTURE["requirement_response"]["project_spec"])
    project_id, other_id = uuid4(), uuid4()
    chunks = chunk_project_spec(project_id, spec)
    assert len(chunks) > 5
    assert len(embed_text("daily bicycle counts")) == 384
    assert sum(value * value for value in embed_text("daily bicycle counts")) == pytest.approx(1.0)

    engine = build_engine(f"sqlite:///{tmp_path / 'rag.sqlite'}")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    with factory() as session, session.begin():
        session.add_all([
            Project(id=project_id, name="bicycle"),
            Project(id=other_id, name="other"),
        ])
        session.flush()
        retriever = PgVectorContextRetriever(factory, default_top_k=3)
        assert retriever.replace_project_spec(session, project_id, spec) == len(chunks)
        other_spec = ProjectSpec.model_validate({
            **FIXTURE["requirement_response"]["project_spec"],
            "title": "Unrelated project",
            "description": "Design a sculpture from reclaimed wood.",
            "deliverables": ["Sculpture"],
        })
        retriever.replace_project_spec(session, other_id, other_spec)
    retrieved = PgVectorContextRetriever(factory).retrieve(
        project_id, "validate daily bicycle count CSV and malformed rows", top_k=3,
    )
    assert 1 <= len(retrieved) <= 3
    assert all(chunk.project_id == project_id for chunk in retrieved)
    assert any("count" in chunk.content.casefold() for chunk in retrieved)
    with factory() as session:
        assert session.scalar(
            select(ProjectContextChunkRow.id).where(ProjectContextChunkRow.project_id == other_id)
        ) is not None
    engine.dispose()


def test_git_workspace_is_per_project_scoped_and_commits_task_referenced_changes(tmp_path):
    manager = GitWorkspaceManager(tmp_path / "repos")
    project_id, other_project_id = uuid4(), uuid4()
    repo = manager.ensure_repository(project_id)
    other_repo = manager.ensure_repository(other_project_id)
    assert repo == tmp_path / "repos" / str(project_id)
    assert repo != other_repo
    assert (repo / ".git").is_dir()
    assert manager.detect_runtime(
        project_id, path_hints=["server"],
        tech_stack_hints=["Python HTTP API", "HTML/CSS/JavaScript", "pytest"],
    ) == "python"
    assert manager.detect_runtime(
        project_id, path_hints=["web/app.js"],
        tech_stack_hints=["Python HTTP API", "JavaScript"],
    ) == "python"
    assert manager.detect_runtime(
        project_id, explicit_runtime="node", path_hints=["web/app.js"],
        tech_stack_hints=["Python HTTP API", "JavaScript"],
    ) == "node"

    change = AgentFileChange(path="src/main.py", content="print('hello')\n")
    commit = manager.apply_changes_and_commit(
        project_id,
        task_id="build-cli",
        agent_name="coder",
        changes=[change],
        write_scope=["src"],
        summary="create the CLI entry point",
    )
    assert commit.commit_hash
    assert commit.changed_paths == ("src/main.py",)
    assert "task build-cli" in commit.message.casefold()
    logged = subprocess.run(
        ["git", "log", "-1", "--format=%s"], cwd=repo,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert "build-cli" in logged
    assert manager.read_context(project_id, write_scope=["src"]) == {
        "src/main.py": "print('hello')"
    }
    with pytest.raises(AgentWriteScopeError):
        manager.apply_changes_and_commit(
            project_id,
            task_id="build-cli",
            agent_name="coder",
            changes=[AgentFileChange(path="README.md", content="out of scope")],
            write_scope=["src"],
            summary="escape declared write paths",
        )
    assert not (repo / "README.md").exists()


def _drive_to_final_approval(tmp_path: Path, *, fail_first_test_cycle: bool,
                             sandbox: Any):
    client = OfflineBuildModelClient(fail_first_test_cycle=fail_first_test_cycle)
    engine, factory, project_id, suite, workflow = _project(
        tmp_path, client=client, sandbox=sandbox,
    )
    created = workflow.start(project_id, max_retries=3, timeout_seconds=1800)
    plan_gate = workflow.drive(created.id)
    assert plan_gate.state is WorkflowState.AWAITING_PLAN_APPROVAL
    assert len(plan_gate.result_payload["plan"]["task_graph"]["nodes"]) == 4
    workflow.approve(plan_gate.id, stage="plan", approved=True, comment="Fixture plan approved")
    final_gate = workflow.drive(plan_gate.id)
    return engine, factory, project_id, suite, workflow, client, final_gate


def test_build_loop_propagates_debugger_feedback_and_audits_commits_and_sandbox_results(tmp_path):
    sandbox = RecordingSandbox(first_fails=True)
    engine, factory, project_id, suite, workflow, client, final_gate = _drive_to_final_approval(
        tmp_path, fail_first_test_cycle=True, sandbox=sandbox,
    )
    assert final_gate.state is WorkflowState.AWAITING_FINAL_APPROVAL
    assert len(sandbox.calls) == 5
    assert all(call["runtime"] == "python" for call in sandbox.calls)
    assert all(call["network_allowlist"] == () for call in sandbox.calls)

    with factory() as session:
        tasks = session.scalars(
            select(TaskNodeRow).where(TaskNodeRow.graph_id == final_gate.task_graph_id)
        ).all()
        assert {task.status for task in tasks} == {"completed"}
        assert next(task for task in tasks if task.id == "validate-csv").retries == 1
        runs = session.scalars(
            select(AgentRunRow).where(AgentRunRow.workflow_id == final_gate.id)
            .order_by(AgentRunRow.timestamp, AgentRunRow.id)
        ).all()
        coder_runs = [run for run in runs if run.agent_name == "coder"]
        tester_runs = [run for run in runs if run.agent_name == "tester"
                       and run.input_payload.get("phase") == "TEST"]
        debugger_runs = [run for run in runs if run.agent_name == "debugger"]
        assert len(coder_runs) == 5
        assert len(tester_runs) == 5
        assert len(debugger_runs) == 1
        debug_input = debugger_runs[0].input_payload
        assert "src/bicycle_summary.py" in debug_input["repository_files"]
        assert "FAILED" in debug_input["test_failure"]["stdout"]
        assert tester_runs[0].output_payload["passed"] is False
        assert tester_runs[-1].output_payload["passed"] is True
        assert tester_runs[-1].output_payload["exit_code"] == 0
        assert tester_runs[-1].output_payload["code_execution_count"] == 1
        assert tester_runs[0].output_payload["test_generation_performed"] is True
        assert all(run.output_payload["stub"] is False for run in coder_runs + tester_runs + debugger_runs)
        retry = next(
            run for run in coder_runs
            if run.input_payload.get("task_id") == "validate-csv"
            and run.input_payload.get("attempt") == 2
        )
        assert retry.input_payload["debugger_feedback"]["diagnosis"] == client.debugger["diagnosis"]
        assert retry.input_payload["debugger_feedback"]["fix_instructions"] == client.debugger["fix_instructions"]
        assert retry.output_payload["commit_hash"]

    coder_prompts = [json.loads(user) for name, _, user in client.calls if name == "coder"]
    first_context = coder_prompts[0]["relevant_project_spec_context"]
    assert len(first_context) <= 3
    assert "tech_stack_hints" not in coder_prompts[0]
    assert "rubric_items" not in coder_prompts[0]
    assert len(first_context) < len(chunk_project_spec(
        project_id, ProjectSpec.model_validate(FIXTURE["requirement_response"]["project_spec"]),
    ))
    repo = suite.workspace_manager.project_path(project_id)
    assert (repo / "src" / "bicycle_summary.py").is_file()
    assert (repo / "tests" / "test_bicycle_summary.py").is_file()
    assert (repo / "README.md").is_file()
    logs = subprocess.run(
        ["git", "log", "--format=%s"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout
    assert "validate-csv" in logs
    assert "tests-and-readme" in logs
    engine.dispose()


def test_simple_script_fixture_runs_plan_and_build_loop_in_docker(tmp_path):
    require_docker = os.environ.get("PROJECTFORGE_REQUIRE_DOCKER") == "1"
    try:
        import docker
    except ImportError as exc:
        if require_docker:
            pytest.fail(f"Docker SDK is required in this test run: {exc}")
        pytest.skip("Docker SDK optional extra is not installed")
    try:
        docker_client = docker.from_env(timeout=5)
        docker_client.ping()
    except Exception as exc:
        if require_docker:
            pytest.fail(f"Docker daemon is required in this test run: {exc}")
        pytest.skip(f"Docker daemon is unavailable: {exc}")
    sandbox = SandboxManager(client=docker_client, default_timeout_seconds=60)
    try:
        docker_client.images.get(sandbox.images["python"])
    except Exception as exc:
        if require_docker:
            pytest.fail(f"required prebuilt image {sandbox.images['python']!r} is unavailable: {exc}")
        pytest.skip(f"required prebuilt image {sandbox.images['python']!r} is unavailable")

    engine, factory, project_id, suite, workflow, client, final_gate = _drive_to_final_approval(
        tmp_path, fail_first_test_cycle=False, sandbox=sandbox,
    )
    assert final_gate.state is WorkflowState.AWAITING_FINAL_APPROVAL
    with factory() as session:
        all_tester_runs = session.scalars(
            select(AgentRunRow).where(
                AgentRunRow.workflow_id == final_gate.id,
                AgentRunRow.agent_name == "tester",
            ).order_by(AgentRunRow.timestamp, AgentRunRow.id)
        ).all()
        test_runs = [run for run in all_tester_runs if run.input_payload.get("phase") == "TEST"]
        assert len(test_runs) == 4
        assert all(run.output_payload["passed"] is True for run in test_runs)
        assert all(run.output_payload["exit_code"] == 0 for run in test_runs)
        assert all(run.output_payload["code_execution_count"] == 1 for run in test_runs)
        assert sum(run.output_payload["test_generation_performed"] for run in test_runs) == 1
    repo = suite.workspace_manager.project_path(project_id)
    final = sandbox.run(
        "python", ["pytest", "-q"], project_dir=repo,
        timeout_seconds=60, network_allowlist=(),
    )
    assert final.exit_code == 0, f"stdout:\\n{final.stdout}\\nstderr:\\n{final.stderr}"
    assert "3 passed" in final.stdout
    engine.dispose()
