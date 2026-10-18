from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from projectforge_agents.config import ModelConfig
from projectforge_agents.grounding import (
    extract_experiment_output,
    extract_key_terms,
    has_repeated_sentences,
    inspect_report_groundedness,
)
from projectforge_agents.packager import PackagerAgent, PackagerInputError
from projectforge_schemas.models import (
    ProjectSpec,
    ReviewCriterionAssessment,
    ReviewerAgentResponse,
    RubricItem,
    TaskGraph,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "projectforge" / "erdos_gyarfas.json"
ERDOS_SPEC = ProjectSpec.model_validate(
    json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["requirement_response"]["project_spec"]
)


SOURCE = '''from collections.abc import Mapping, Sequence


def validate_simple_graph(adjacency: Mapping[str, Sequence[str]]):
    return adjacency


def enumerate_simple_cycles(adjacency):
    return []


def power_of_two_cycle_lengths(adjacency):
    return []
'''
TEST_SOURCE = '''def test_petersen_cycle_lengths():
    assert [5, 6, 8, 9] == [5, 6, 8, 9]
'''
RESULTS = json.dumps({
    "graph": "Petersen graph",
    "vertex_count": 10,
    "edge_count": 15,
    "minimum_degree": 3,
    "cycle_count": 57,
    "cycle_lengths": [5, 6, 8, 9],
    "power_of_two_cycle_lengths": [8],
})

REPORT = """# Erdős–Gyárfás Cycles in Cubic Graphs

## Implementation

`graphs/cycles.py` validates finite undirected simple graphs and provides `validate_simple_graph`, `enumerate_simple_cycles`, and `power_of_two_cycle_lengths`. It canonicalizes cycles under rotation and reversal. The Petersen graph has 10 vertices, 15 edges, and minimum degree 3.

## Observed computation

`results/petersen_cycle_summary.json` records 57 simple cycles with lengths 5, 6, 8, and 9; the power-of-two cycle length is 8. `tests/test_erdos_gyarfas.py` recorded 3 passed in the sandbox.

## Limits and complexity

This finite computation concerns one Petersen graph and is not a proof of the Erdős–Gyárfás conjecture. The depth-first search explores simple paths, has factorial worst-case time in the number of vertices, and may need output-sized storage.
"""


RUBRIC = RubricItem(
    criterion="The Petersen graph result contains cycle lengths 5, 6, 8, and 9, including the power-of-two length 8.",
    type="automatable",
    check_method="Run the fixed Petersen graph regression test.",
)
REVIEW = ReviewerAgentResponse(
    summary="The graph implementation and finite-example evidence are ready for human review.",
    criteria=[ReviewCriterionAssessment(
        rubric_item=RUBRIC,
        status="passed",
        evidence=["tests/test_erdos_gyarfas.py passed against the fixed graph result."],
    )],
    automatable_checks_passed=True,
    human_review_required=False,
    evidence=["ReviewerResult is persisted for the final report context."],
)
GRAPH = TaskGraph.model_validate({
    "nodes": [{
        "id": "analyze-petersen",
        "description": "Enumerate the Petersen graph's simple cycles and report power-of-two lengths.",
        "dependencies": [],
        "acceptance_criteria": ["The result lists cycle lengths 5, 6, 8, and 9."],
        "assigned_agent": "coder",
        "read_set": ["graphs"],
        "write_set": ["results"],
    }],
})
REPOSITORY_FILES = {
    "graphs/cycles.py": SOURCE,
    "results/petersen_cycle_summary.json": RESULTS,
    "tests/test_erdos_gyarfas.py": TEST_SOURCE,
}
TEST_RUNS = [{
    "task_id": "analyze-petersen",
    "phase": "TEST",
    "passed": True,
    "exit_code": 0,
    "timed_out": False,
    "tested_files": ["tests/test_erdos_gyarfas.py"],
    "stdout": "3 passed in 0.02s\n",
    "stderr": "",
    "evidence": ["The fixed cycle-length and invalid-graph cases passed."],
}]
AGENT_RUNS = [{
    "agent_run_id": "coder-run-01",
    "agent_name": "coder",
    "phase": "CODE",
    "task_id": "analyze-petersen",
    "status": "completed",
    "summary": "Implemented finite graph validation and Petersen cycle analysis.",
    "evidence": ["The source and result artifact were committed."],
}]
REVIEW_RESULTS = [{
    "review_result_id": "review-result-01",
    "rubric_item": RUBRIC.model_dump(mode="json"),
    "status": "passed",
    "passed": True,
    "evidence": ["The Petersen result matches the criterion."],
    "reviewer_notes": "Automated result was checked.",
}]


class FailingModelClient:
    def complete(self, *, system_prompt: str, user_prompt: str,
                 agent_name: str, timeout_seconds: int | None = None):
        raise RuntimeError("simulated report-provider transport failure")


def _packager_inputs() -> dict[str, Any]:
    return {
        "repository_files": REPOSITORY_FILES,
        "test_runs": TEST_RUNS,
        "agent_runs": AGENT_RUNS,
        "review_results": REVIEW_RESULTS,
        "final_gate_decision": {"decision": "pending"},
    }


def test_erdos_gyarfas_report_grounding_matches_hyphenated_accented_terms_and_artifacts():
    result = inspect_report_groundedness(
        REPORT,
        ERDOS_SPEC,
        repository_files=REPOSITORY_FILES,
        test_runs=TEST_RUNS,
        agent_runs=AGENT_RUNS,
        review_results=REVIEW_RESULTS,
    )
    assert result.passed, result.issues
    assert result.matched_terms == result.required_terms
    assert "Erdős–Gyárfás" in extract_key_terms(ERDOS_SPEC)
    assert "Erdős–Gyárfás" in result.matched_terms
    assert "results/petersen_cycle_summary.json" in result.artifact_paths
    assert "enumerate_simple_cycles" in result.matched_code_anchors
    assert "tests/test_erdos_gyarfas.py" in result.matched_test_anchors
    artifact = json.loads(REPOSITORY_FILES["results/petersen_cycle_summary.json"])
    assert artifact["vertex_count"] == 10 and str(artifact["vertex_count"]) in REPORT
    assert artifact["edge_count"] == 15 and str(artifact["edge_count"]) in REPORT
    assert artifact["minimum_degree"] == 3 and str(artifact["minimum_degree"]) in REPORT
    assert artifact["cycle_count"] == 57 and str(artifact["cycle_count"]) in REPORT
    assert artifact["cycle_lengths"] == [5, 6, 8, 9]
    assert all(str(length) in REPORT for length in artifact["cycle_lengths"])
    assert artifact["power_of_two_cycle_lengths"] == [8]
    assert "not a proof" in REPORT


def test_experiment_output_selector_includes_textual_svg_charts():
    result = extract_experiment_output({
        "outputs/rainfall_comparison.svg": (
            '<svg><title>Monthly rainfall comparison by station</title>'
            '<desc>North, South, East, and West station measurements.</desc></svg>'
        ),
    })
    assert result["included_artifact_count"] == 1
    assert result["artifacts"][0]["path"] == "outputs/rainfall_comparison.svg"
    assert "Monthly rainfall comparison by station" in result["artifacts"][0]["content_excerpt"]


def test_report_grounding_flags_duplicate_sentences_section_echo_and_meta_commentary():
    first = "The Petersen computation is finite evidence about one cubic graph with minimum degree three."
    near_duplicate = "The Petersen computation is finite evidence for one cubic graph with minimum degree three."
    assert has_repeated_sentences(f"{first}\n{near_duplicate}")

    invalid_report = (
        "# Erdős–Gyárfás Cycles in Cubic Graphs\n\n"
        "The core concept is important for this graph report and supports the analysis.\n"
        "The core concept remains useful for this graph report and supports the analysis.\n"
        "The core concept is central to the graph report and supports the analysis.\n"
        "Reviewers will judge this section based on the writing process.\n"
        + first + "\n" + near_duplicate + "\n"
    )
    result = inspect_report_groundedness(
        invalid_report,
        ERDOS_SPEC,
        repository_files=REPOSITORY_FILES,
        test_runs=TEST_RUNS,
        agent_runs=AGENT_RUNS,
        review_results=REVIEW_RESULTS,
    )
    assert any("section-name echo" in issue for issue in result.issues)
    assert any("meta-commentary" in issue for issue in result.issues)
    assert any("Repeated near-identical sentences" in issue for issue in result.issues)
    assert any("concrete result" in issue for issue in result.issues)


def test_packager_prompt_preserves_source_test_output_audit_and_result_artifact():
    agent = PackagerAgent(config=ModelConfig.from_env({}))
    _, user_prompt = agent.build_prompts(
        ERDOS_SPEC,
        GRAPH,
        REVIEW,
        **_packager_inputs(),
    )
    payload = json.loads(user_prompt)
    assert "enumerate_simple_cycles" in payload["code_summary"]["files"][0]["symbols"]
    assert payload["sandbox_test_results"][0]["stdout"] == "3 passed in 0.02s\n"
    assert payload["agent_run_records"][0]["agent_run_id"] == "coder-run-01"
    assert payload["review_result_records"][0]["review_result_id"] == "review-result-01"
    assert payload["experiment_output"]["artifacts"][0]["path"] == (
        "results/petersen_cycle_summary.json"
    )
    assert '"cycle_lengths": [5, 6, 8, 9]' in payload["experiment_output"]["artifacts"][0]["content_excerpt"]


def test_packager_rejects_missing_audit_inputs_and_propagates_provider_errors():
    agent = PackagerAgent(client=FailingModelClient(), config=ModelConfig.from_env({}))
    with pytest.raises(PackagerInputError, match="sandbox TEST results"):
        agent.build_prompts(
            ERDOS_SPEC,
            GRAPH,
            REVIEW,
            **{**_packager_inputs(), "test_runs": []},
        )
    with pytest.raises(PackagerInputError, match="AgentRun evidence"):
        agent.build_prompts(
            ERDOS_SPEC,
            GRAPH,
            REVIEW,
            **{**_packager_inputs(), "agent_runs": []},
        )
    with pytest.raises(RuntimeError, match="simulated report-provider transport failure"):
        agent.run(
            ERDOS_SPEC,
            GRAPH,
            REVIEW,
            **_packager_inputs(),
        )
