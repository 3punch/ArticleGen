"""Model-backed README and grounded project-report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from projectforge_schemas.models import (
    PackagerAgentResponse,
    ProjectSpec,
    ReviewerAgentResponse,
    TaskGraph,
)

from .config import ModelConfig
from .errors import AgentOutputValidationError
from .grounding import (
    build_code_summary,
    extract_experiment_output,
    extract_key_terms,
    inspect_report_groundedness,
)
from .langgraph_runner import LangGraphModelCall
from .litellm_client import ChatModelClient, LiteLLMChatClient, ModelCompletion


_PROMPT = Path(__file__).with_name("prompts").joinpath("packager_v2_grounded.txt").read_text(
    encoding="utf-8",
)
# Keep the evidence payload bounded. The model adapter separately estimates the
# serialized prompt and enforces the configured token reservation.
MAX_USER_PROMPT_BYTES = 40_000


class PackagerInputError(ValueError):
    """Required project evidence is absent or unsafe to fit into the report prompt."""


class PackagerAgent:
    """Generate README/report text from source, test, review, and audit artifacts."""

    agent_name = "packager"
    prompt_version = "packager_v2_grounded"

    def __init__(self, *, client: ChatModelClient | None = None,
                 config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_env()
        self.client = client or LiteLLMChatClient(self.config)
        self.model_call = LangGraphModelCall(self.client)

    def build_prompts(
        self,
        project_spec: ProjectSpec | Mapping[str, Any],
        task_graph: TaskGraph | Mapping[str, Any],
        review: ReviewerAgentResponse | Mapping[str, Any],
        *,
        repository_files: Mapping[str, str],
        test_runs: Sequence[Mapping[str, Any]],
        final_gate_decision: Mapping[str, Any],
        agent_runs: Sequence[Mapping[str, Any]] = (),
        review_results: Sequence[Mapping[str, Any]] = (),
        regeneration_feedback: str = "",
    ) -> tuple[str, str]:
        spec = ProjectSpec.model_validate(project_spec)
        graph = task_graph if isinstance(task_graph, TaskGraph) else TaskGraph.model_validate(task_graph)
        if isinstance(review, ReviewerAgentResponse):
            reviewer = review
        else:
            reviewer_fields = (
                "summary", "criteria", "automatable_checks_passed",
                "human_review_required", "evidence",
            )
            reviewer = ReviewerAgentResponse.model_validate({
                key: review[key] for key in reviewer_fields if key in review
            })

        code_summary = build_code_summary(repository_files)
        self._validate_evidence_inputs(
            repository_files=repository_files,
            code_summary=code_summary,
            test_runs=test_runs,
            agent_runs=agent_runs,
            review_results=review_results,
        )
        schema = json.dumps(PackagerAgentResponse.model_json_schema(), ensure_ascii=False)
        payload = {
            "project_spec": self._bounded_project_spec(spec),
            "required_domain_terms": extract_key_terms(spec),
            "task_graph": self._bounded_task_graph(graph),
            "review": self._bounded_review(reviewer),
            "review_result_records": [
                self._bounded_review_result(record) for record in review_results[:30]
            ],
            "agent_run_records": [self._bounded_agent_run(record) for record in agent_runs[-30:]],
            "code_summary": code_summary,
            "repository_file_inventory": self._bounded_inventory(repository_files),
            "experiment_output": extract_experiment_output(repository_files),
            "sandbox_test_results": [self._bounded_test_run(run) for run in test_runs[-20:]],
            "final_human_gate_decision": self._bounded_gate(final_gate_decision),
            "regeneration_feedback": str(regeneration_feedback).strip()[:4000],
        }
        user_prompt = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        prompt_bytes = len(user_prompt.encode("utf-8"))
        if prompt_bytes > MAX_USER_PROMPT_BYTES:
            raise PackagerInputError(
                "Grounding artifacts exceed the Packager prompt limit "
                f"({prompt_bytes} > {MAX_USER_PROMPT_BYTES} bytes); reduce or partition the evidence."
            )
        return (
            f"{_PROMPT}\n\nExpected JSON Schema:\n{schema}",
            user_prompt,
        )

    def run(
        self,
        project_spec: ProjectSpec | Mapping[str, Any],
        task_graph: TaskGraph | Mapping[str, Any],
        review: ReviewerAgentResponse | Mapping[str, Any],
        *,
        repository_files: Mapping[str, str],
        test_runs: Sequence[Mapping[str, Any]],
        final_gate_decision: Mapping[str, Any],
        agent_runs: Sequence[Mapping[str, Any]] = (),
        review_results: Sequence[Mapping[str, Any]] = (),
        regeneration_feedback: str = "",
        timeout_seconds: int | None = None,
    ) -> tuple[PackagerAgentResponse, ModelCompletion]:
        spec = ProjectSpec.model_validate(project_spec)
        system_prompt, user_prompt = self.build_prompts(
            spec,
            task_graph,
            review,
            repository_files=repository_files,
            test_runs=test_runs,
            final_gate_decision=final_gate_decision,
            agent_runs=agent_runs,
            review_results=review_results,
            regeneration_feedback=regeneration_feedback,
        )
        completion = self.model_call.invoke(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name=self.agent_name,
            timeout_seconds=timeout_seconds,
        )
        try:
            response = PackagerAgentResponse.model_validate(json.loads(completion.content))
        except (json.JSONDecodeError, ValueError) as exc:
            raise AgentOutputValidationError(
                f"Packager agent returned invalid structured output: {exc}",
                completion=completion,
            ) from exc

        grounding = inspect_report_groundedness(
            response.report_markdown,
            spec,
            repository_files=repository_files,
            test_runs=test_runs,
            agent_runs=agent_runs,
            review_results=review_results,
        )
        if not grounding.passed:
            details = "; ".join(grounding.issues[:20])
            raise AgentOutputValidationError(
                "Packager report failed deterministic groundedness validation: " + details,
                completion=completion,
            )
        response.evidence.append(
            "Deterministic groundedness gate passed against ProjectSpec terms, "
            "implementation symbols, persisted sandbox results, AgentRun records, and ReviewResult records."
        )
        return response, completion

    @staticmethod
    def _validate_evidence_inputs(
        *,
        repository_files: Mapping[str, str],
        code_summary: Mapping[str, Any],
        test_runs: Sequence[Mapping[str, Any]],
        agent_runs: Sequence[Mapping[str, Any]],
        review_results: Sequence[Mapping[str, Any]],
    ) -> None:
        if not repository_files:
            raise PackagerInputError("Packager cannot generate a report without repository artifacts.")
        if not code_summary.get("source_file_count"):
            raise PackagerInputError("Packager cannot generate a report without implementation source files.")
        if not test_runs:
            raise PackagerInputError("Packager cannot generate a report without persisted sandbox TEST results.")
        failed = [
            str(run.get("task_id", "unknown")) for run in test_runs
            if (str(run.get("phase", "TEST")).upper() != "TEST"
                or run.get("passed") is not True
                or run.get("exit_code") != 0
                or bool(run.get("timed_out", False)))
        ]
        if failed:
            raise PackagerInputError(
                "Packager requires passing sandbox TEST records with exit code zero; "
                "invalid task(s): " + ", ".join(failed[:20])
            )
        if not agent_runs:
            raise PackagerInputError("Packager cannot generate a report without persisted AgentRun evidence.")
        if not review_results:
            raise PackagerInputError("Packager cannot generate a report without persisted ReviewResult evidence.")

    @staticmethod
    def _bounded_project_spec(spec: ProjectSpec) -> dict[str, Any]:
        payload = spec.model_dump(mode="json")
        payload["description"] = payload["description"][:5000]
        for name, item_limit, item_chars in (
            ("deliverables", 20, 400),
            ("constraints", 20, 400),
            ("tech_stack_hints", 20, 160),
        ):
            values = payload[name]
            payload[name] = [str(item)[:item_chars] for item in values[:item_limit]]
            payload[f"{name}_omitted_count"] = max(0, len(values) - item_limit)
        rubric_items = payload["rubric_items"]
        payload["rubric_items"] = [
            {
                "criterion": item["criterion"][:400],
                "type": item["type"],
                "check_method": item["check_method"][:400],
            }
            for item in rubric_items[:25]
        ]
        payload["rubric_items_omitted_count"] = max(0, len(rubric_items) - 25)
        return payload

    @staticmethod
    def _bounded_task_graph(graph: TaskGraph) -> dict[str, Any]:
        nodes = graph.nodes
        edges = graph.edges
        return {
            "task_count": len(nodes),
            "nodes": [
                {
                    "id": node.id,
                    "description": node.description[:400],
                    "dependencies": node.dependencies[:20],
                    "status": node.status,
                    "acceptance_criteria": [item[:300] for item in node.acceptance_criteria[:5]],
                    "read_set": node.read_set[:20],
                    "write_set": node.write_set[:20],
                }
                for node in nodes[:30]
            ],
            "omitted_node_count": max(0, len(nodes) - 30),
            "edges": [edge.model_dump(mode="json") for edge in edges[:50]],
            "omitted_edge_count": max(0, len(edges) - 50),
        }

    @staticmethod
    def _bounded_review(review: ReviewerAgentResponse) -> dict[str, Any]:
        data = review.model_dump(mode="json")
        data["summary"] = data["summary"][:1800]
        data["criteria"] = [
            {
                **item,
                "rubric_item": {
                    "criterion": item["rubric_item"]["criterion"][:400],
                    "type": item["rubric_item"]["type"],
                    "check_method": item["rubric_item"]["check_method"][:400],
                },
                "evidence": [str(value)[:350] for value in item["evidence"][:5]],
                "rationale": item["rationale"][:700],
            }
            for item in data["criteria"][:30]
        ]
        data["criteria_omitted_count"] = max(0, len(review.criteria) - 30)
        data["evidence"] = [str(value)[:350] for value in data["evidence"][:10]]
        return data

    @staticmethod
    def _bounded_review_result(record: Mapping[str, Any]) -> dict[str, Any]:
        rubric = record.get("rubric_item", {})
        if not isinstance(rubric, Mapping):
            rubric = {}
        return {
            "review_result_id": str(record.get("review_result_id", ""))[:80],
            "rubric_item": {
                "criterion": str(rubric.get("criterion", ""))[:400],
                "type": str(rubric.get("type", ""))[:40],
                "check_method": str(rubric.get("check_method", ""))[:400],
            },
            "status": str(record.get("status", ""))[:40],
            "passed": record.get("passed"),
            "recommendation": record.get("recommendation"),
            "evidence": [str(item)[:300] for item in record.get("evidence", [])[:5]],
            "reviewer_notes": str(record.get("reviewer_notes", ""))[:500],
        }

    @staticmethod
    def _bounded_agent_run(record: Mapping[str, Any]) -> dict[str, Any]:
        bounded = {
            "agent_run_id": str(record.get("agent_run_id", ""))[:80],
            "agent_name": str(record.get("agent_name", ""))[:32],
            "phase": str(record.get("phase", ""))[:40],
            "task_id": str(record.get("task_id", ""))[:80],
            "status": str(record.get("status", ""))[:80],
            "summary": str(record.get("summary", ""))[:350],
            "evidence": [str(item)[:240] for item in record.get("evidence", [])[:4]],
        }
        for key in ("passed", "exit_code", "timed_out"):
            if key in record:
                bounded[key] = record[key]
        for key in ("changed_paths", "files_written", "tested_files"):
            value = record.get(key)
            if isinstance(value, (list, tuple)):
                bounded[key] = [str(item)[:180] for item in value[:15]]
        return bounded

    @staticmethod
    def _bounded_inventory(repository_files: Mapping[str, str]) -> list[dict[str, Any]]:
        return [
            {"path": path, "size_chars": len(content)}
            for path, content in sorted(repository_files.items())[:80]
        ]

    @staticmethod
    def _bounded_gate(decision: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in ("decision", "approved", "timestamp"):
            if key in decision:
                result[key] = decision[key]
        if "comment" in decision:
            result["comment"] = " ".join(str(decision["comment"]).split())[:1000]
        result.setdefault("decision", "pending")
        return result

    @staticmethod
    def _bounded_test_run(run: Mapping[str, Any]) -> dict[str, Any]:
        keys = (
            "task_id", "phase", "passed", "runtime", "command", "tested_files",
            "exit_code", "timed_out", "duration_seconds", "evidence", "stdout", "stderr",
        )
        bounded: dict[str, Any] = {}
        for key in keys:
            if key not in run:
                continue
            value = run[key]
            if key in {"stdout", "stderr"} and isinstance(value, str):
                bounded[key] = value[:800 if key == "stdout" else 500]
            elif key == "evidence" and isinstance(value, list):
                bounded[key] = [str(item)[:250] for item in value[:5]]
            elif isinstance(value, list):
                bounded[key] = [str(item)[:180] for item in value[:20]]
            elif isinstance(value, str):
                bounded[key] = value[:500]
            else:
                bounded[key] = value
        return bounded
