from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from uuid import uuid4
import zipfile

import pytest
from pydantic import ValidationError

from projectforge_agents.git_workspace import GitWorkspaceManager
from projectforge_agents.litellm_client import ModelCompletion
from projectforge_agents.package_export import PackageExportError, ProjectPackageManager
from projectforge_agents.reviewer import ReviewerAgent
from projectforge_schemas.models import (
    AIUsageDisclosure,
    AgentFileChange,
    AgentUsageActivity,
    ProjectSpec,
    ReviewCriterionAssessment,
    ReviewerAgentResponse,
    RubricItem,
)


SPEC = ProjectSpec.model_validate({
    "title": "Rainfall report",
    "description": "Analyze a fixed rainfall fixture and document the method.",
    "deliverables": ["Offline analysis", "Methods report"],
    "constraints": ["Do not use network access"],
    "tech_stack_hints": ["Python", "pytest"],
    "rubric_items": [
        {
            "criterion": "Per-station summaries and wettest station-month match fixture values.",
            "type": "automatable",
            "check_method": "Run the fixed tests for station statistics and the wettest month.",
        },
        {
            "criterion": "The methods section explains statistical choices accurately.",
            "type": "human-judgment",
            "check_method": "A reviewer checks the sample standard deviation and data limitations.",
        },
    ],
})


class ReviewerClient:
    def __init__(self, *, recommendation: str = "approve") -> None:
        self.recommendation = recommendation
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str,
                 agent_name: str, timeout_seconds: int | None = None):
        self.calls.append((agent_name, user_prompt))
        assert agent_name == "reviewer"
        payload = json.loads(user_prompt)
        criterion = payload["human_judgment_criteria"][0]["criterion"]
        return ModelCompletion(
            content=json.dumps({
                "summary": "The report provides evidence but its explanation needs a human decision.",
                "recommendations": [{
                    "criterion": criterion,
                    "recommendation": self.recommendation,
                    "rationale": "The report names the statistical method and its limitations, subject to human confirmation.",
                    "evidence": ["README.md: describes sample standard deviation and invalid-row policy."],
                }],
            }),
            tokens_used=80,
            cost=Decimal("0.00080000"),
            provider="offline-review",
            model="fixture-reviewer-v1",
        )


def test_reviewer_combines_static_and_passing_test_evidence_without_passing_human_items():
    files = {
        "analysis/rainfall.py": (
            "def summarize_station(rows):\n"
            "    return {'count': 4, 'mean': 7.5, 'median': 6.0, 'stddev': 5.0}\n"
        ),
        "tests/test_rainfall.py": (
            "def test_station_summaries_and_wettest_station_month():\n"
            "    station_summaries = {'count': 4, 'mean': 7.5, 'median': 6.0, 'stddev': 5.0}\n"
            "    wettest_station_month = ('North', '2025-06', 19.0)\n"
            "    assert station_summaries['count'] == 4\n"
            "    assert wettest_station_month[0] == 'North'\n"
        ),
        "README.md": "# Methods\n\nSample standard deviation and invalid-row exclusions are explained.\n",
    }
    client = ReviewerClient()
    response, completion = ReviewerAgent(client=client).run(
        SPEC,
        repository_files=files,
        test_runs=[{
            "task_id": "station-statistics", "phase": "TEST", "passed": True,
            "exit_code": 0, "timed_out": False,
            "tested_files": ["tests/test_rainfall.py"],
        }],
    )
    assert completion is not None and completion.model == "fixture-reviewer-v1"
    assert response.automatable_checks_passed is True
    automated, human = response.criteria
    assert automated.status == "passed"
    assert any("AST static parse passed" in item for item in automated.evidence)
    assert any("station" in item.casefold() for item in automated.evidence)
    assert human.status == "human_review_required"
    assert human.recommendation == "approve"
    assert any("Human inspection is mandatory" in item for item in human.evidence)
    assert response.human_review_required is True


def test_reviewer_marks_unverifiable_automation_not_verified_and_python_syntax_failure_failed():
    no_tests = ProjectSpec.model_validate({
        **SPEC.model_dump(mode="python"),
        "rubric_items": [SPEC.rubric_items[0].model_dump(mode="python")],
    })
    response, completion = ReviewerAgent().run(
        no_tests,
        repository_files={"analysis/rainfall.py": "def summarize(\n"},
        test_runs=[],
    )
    assert completion is None
    assert response.criteria[0].status == "failed"
    assert any("Python static parse failed" in item for item in response.criteria[0].evidence)

    valid_source, _ = ReviewerAgent().run(
        no_tests,
        repository_files={"analysis/rainfall.py": "def summarize():\n    return {}\n"},
        test_runs=[],
    )
    assert valid_source.criteria[0].status == "not_verified"
    assert valid_source.automatable_checks_passed is False


def test_packager_contract_and_zip_export_include_audit_disclosure_and_test_results(tmp_path):
    project_id, workflow_id = uuid4(), uuid4()
    workspace = GitWorkspaceManager(tmp_path / "repos")
    workspace.ensure_repository(project_id)
    workspace.apply_changes_and_commit(
        project_id,
        task_id="implement-rainfall",
        agent_name="coder",
        changes=[
            AgentFileChange(path="analysis/rainfall.py", content="def run():\n    return 1\n"),
            AgentFileChange(path="tests/test_rainfall.py", content="def test_run():\n    assert True\n"),
        ],
        write_scope=["analysis", "tests"],
        summary="Add rainfall analysis and passing regression test.",
    )
    workspace.apply_changes_and_commit(
        project_id,
        task_id="package-docs",
        agent_name="packager",
        changes=[
            AgentFileChange(path="README.md", content="# Rainfall analysis\n\nRun the offline analysis.\n"),
            AgentFileChange(path="reports/PROJECT_REPORT.md", content="# Project report\n\nThe fixture test passed.\n"),
        ],
        write_scope=["README.md", "reports"],
        summary="Document the analysis and review evidence.",
    )

    reviewer = ReviewerAgentResponse(
        summary="Automation passed and the methods section requires explicit human review.",
        criteria=[
            ReviewCriterionAssessment(
                rubric_item=SPEC.rubric_items[0], status="passed",
                evidence=["tests/test_rainfall.py passed in the sandbox."],
            ),
            ReviewCriterionAssessment(
                rubric_item=SPEC.rubric_items[1], status="human_review_required",
                recommendation="approve",
                rationale="The report is plausible, but its statistical explanation requires a human decision.",
                evidence=["README.md explains the chosen standard deviation."],
            ),
        ],
        automatable_checks_passed=True,
        human_review_required=True,
        evidence=["Automated test and explicit human-review evidence are retained."],
    )
    audit_id = uuid4()
    disclosure = AIUsageDisclosure(
        project_id=project_id,
        workflow_id=workflow_id,
        agent_calls=1,
        agent_names=["reviewer"],
        audit_run_ids=[audit_id],
        tokens_used=80,
        spend=Decimal("0.00080000"),
        llm_calls=1,
        code_execution_count=1,
        statement="The workflow audit trail recorded one reviewer action.",
        generated_at=datetime.now(timezone.utc),
        activities=[AgentUsageActivity(
            audit_run_id=audit_id,
            agent_name="reviewer",
            action="Compared code and test evidence against the rubric",
            timestamp=datetime.now(timezone.utc),
            provider="offline-review",
            model="fixture-reviewer-v1",
            tokens_used=80,
            spend=Decimal("0.00080000"),
        )],
    )
    manager = ProjectPackageManager(workspace, export_root=tmp_path / "exports")
    record, path = manager.export(
        project_id=project_id,
        workflow_id=workflow_id,
        disclosure=disclosure,
        review=reviewer,
        test_runs=[{
            "task_id": "implement-rainfall", "phase": "TEST", "passed": True,
            "command": ["python", "-m", "pytest", "-q"], "exit_code": 0,
            "tested_files": ["tests/test_rainfall.py"], "evidence": ["1 passed"],
        }],
    )
    assert record.download_url == f"/api/workflows/{workflow_id}/package"
    assert record.sha256
    assert path.stat().st_size == record.size_bytes
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        assert {"README.md", "PROJECT_REPORT.md", "AI_USAGE_DISCLOSURE.md",
                "audit/AI_USAGE_DISCLOSURE.json", "reports/TEST_RESULTS.json",
                "tests/test_rainfall.py"} <= names
        assert not any(name.startswith(".git/") for name in names)
        disclosure_text = archive.read("AI_USAGE_DISCLOSURE.md").decode("utf-8")
        assert "fixture-reviewer-v1" in disclosure_text
        assert "Compared code and test evidence" in disclosure_text
        test_payload = json.loads(archive.read("reports/TEST_RESULTS.json"))
        assert test_payload["all_passed"] is True
        assert test_payload["runs"][0]["exit_code"] == 0
        manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
        assert manifest["workflow_id"] == str(workflow_id)

    with pytest.raises(PackageExportError, match="passing sandbox test run"):
        manager.export(
            project_id=project_id,
            workflow_id=workflow_id,
            disclosure=disclosure,
            review=reviewer,
            test_runs=[{"task_id": "broken", "phase": "TEST", "passed": False}],
        )
