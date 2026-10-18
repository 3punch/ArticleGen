"""Model-backed README and project-report generation for final packaging."""

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
from .langgraph_runner import LangGraphModelCall
from .litellm_client import ChatModelClient, LiteLLMChatClient, ModelCompletion


_PROMPT = Path(__file__).with_name("prompts").joinpath("packager_v1.txt").read_text(
    encoding="utf-8",
)


class PackagerAgent:
    """Generate docs only; deterministic host code commits and exports the files."""

    agent_name = "packager"
    prompt_version = "packager_v1"

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
        schema = json.dumps(PackagerAgentResponse.model_json_schema(), ensure_ascii=False)
        payload = {
            "project_spec": spec.model_dump(mode="json"),
            "task_graph": graph.model_dump(mode="json"),
            "review": reviewer.model_dump(mode="json"),
            "repository_file_inventory": [
                {"path": path, "size_chars": len(content),
                 "excerpt": content[:1200] if Path(path).name.casefold() in {"readme.md", "pyproject.toml", "package.json"} else None}
                for path, content in sorted(repository_files.items())
            ],
            "sandbox_test_runs": [self._bounded_test_run(run) for run in test_runs],
            "final_human_gate_decision": dict(final_gate_decision),
        }
        return (
            f"{_PROMPT}\n\nExpected JSON Schema:\n{schema}",
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
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
        timeout_seconds: int | None = None,
    ) -> tuple[PackagerAgentResponse, ModelCompletion]:
        system_prompt, user_prompt = self.build_prompts(
            project_spec,
            task_graph,
            review,
            repository_files=repository_files,
            test_runs=test_runs,
            final_gate_decision=final_gate_decision,
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
        return response, completion

    @staticmethod
    def _bounded_test_run(run: Mapping[str, Any]) -> dict[str, Any]:
        keys = (
            "task_id", "phase", "passed", "runtime", "command", "tested_files",
            "exit_code", "timed_out", "duration_seconds", "evidence",
        )
        bounded: dict[str, Any] = {}
        for key in keys:
            if key not in run:
                continue
            value = run[key]
            if isinstance(value, list):
                bounded[key] = value[:50]
            elif isinstance(value, str):
                bounded[key] = value[:4000]
            else:
                bounded[key] = value
        return bounded
