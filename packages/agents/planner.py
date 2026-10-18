"""Model-backed TaskGraph planning with schema, specificity, and DAG validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from projectforge_schemas.models import PlannerAgentResponse, ProjectSpec

from .config import ModelConfig
from .errors import AgentOutputValidationError
from .langgraph_runner import LangGraphModelCall
from .litellm_client import ChatModelClient, LiteLLMChatClient, ModelCompletion


_PROMPT = Path(__file__).with_name("prompts").joinpath("planner_v1.txt").read_text(
    encoding="utf-8",
)
_VAGUE_CRITERIA = {
    "done", "works", "works correctly", "is complete", "implementation complete",
    "looks good", "functionality works", "all requirements are met",
}


class PlannerAgent:
    """Create a testable, acyclic implementation plan from a validated ProjectSpec."""

    def __init__(self, *, client: ChatModelClient | None = None,
                 config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_env()
        self.client = client or LiteLLMChatClient(self.config)
        self.model_call = LangGraphModelCall(self.client)

    def build_prompts(
        self,
        project_spec: ProjectSpec | Mapping[str, Any],
        *,
        plan_feedback: Sequence[str] = (),
    ) -> tuple[str, str]:
        spec = ProjectSpec.model_validate(project_spec)
        schema = json.dumps(PlannerAgentResponse.model_json_schema(), ensure_ascii=False)
        payload = {
            "project_spec": spec.model_dump(mode="json"),
            "human_plan_feedback": [str(item)[:2000] for item in plan_feedback[:20]],
        }
        return (
            f"{_PROMPT}\n\nExpected JSON Schema:\n{schema}",
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )

    def run(
        self,
        project_spec: ProjectSpec | Mapping[str, Any],
        *,
        plan_feedback: Sequence[str] = (),
        timeout_seconds: int | None = None,
    ) -> tuple[PlannerAgentResponse, ModelCompletion]:
        system_prompt, user_prompt = self.build_prompts(
            project_spec, plan_feedback=plan_feedback,
        )
        completion = self.model_call.invoke(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name="planner",
            timeout_seconds=timeout_seconds,
        )
        try:
            decoded = json.loads(completion.content)
            plan = PlannerAgentResponse.model_validate(decoded)
            self._validate_plan(plan)
        except (json.JSONDecodeError, ValueError) as exc:
            raise AgentOutputValidationError(
                f"Planner agent returned invalid structured output: {exc}",
                completion=completion,
            ) from exc
        return plan, completion

    @staticmethod
    def _validate_plan(plan: PlannerAgentResponse) -> None:
        graph = plan.task_graph
        if not 1 <= len(graph.nodes) <= 100:
            raise ValueError("Planner output must contain between 1 and 100 independently testable tasks")
        for node in graph.nodes:
            if node.assigned_agent != "coder":
                raise ValueError(
                    f"task '{node.id}' must be assigned to coder until role-specific task execution is enabled"
                )
            if not node.write_set:
                raise ValueError(f"task '{node.id}' must declare a non-empty write_set for scoped code changes")
            for criterion in node.acceptance_criteria:
                normalized = " ".join(criterion.casefold().split()).strip(" .!?")
                if normalized in _VAGUE_CRITERIA or len(normalized) < 15:
                    raise ValueError(
                        f"task '{node.id}' has a vague or non-concrete acceptance criterion"
                    )
