"""Model-backed Coder, Tester, and Debugger workers for deterministic orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from projectforge_schemas.models import (
    CoderAgentResponse,
    DebuggerAgentResponse,
    RetrievedContextChunk,
    TaskNode,
    TesterAgentResponse,
)

from .config import ModelConfig
from .errors import AgentOutputValidationError
from .langgraph_runner import LangGraphModelCall
from .litellm_client import ChatModelClient, LiteLLMChatClient, ModelCompletion


_PROMPTS = {
    "coder": Path(__file__).with_name("prompts").joinpath("coder_v1.txt").read_text(
        encoding="utf-8",
    ),
    "tester": Path(__file__).with_name("prompts").joinpath("tester_v1.txt").read_text(
        encoding="utf-8",
    ),
    "debugger": Path(__file__).with_name("prompts").joinpath("debugger_v1.txt").read_text(
        encoding="utf-8",
    ),
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _task(value: TaskNode | Mapping[str, Any]) -> TaskNode:
    return value if isinstance(value, TaskNode) else TaskNode.model_validate(value)


def _chunks(values: Sequence[RetrievedContextChunk | Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        (value.model_dump(mode="json") if isinstance(value, RetrievedContextChunk) else
         RetrievedContextChunk.model_validate(value).model_dump(mode="json"))
        for value in values
    ]


class _StructuredBuildAgent:
    agent_name: str
    prompt_version: str
    response_model: Any

    def __init__(self, *, client: ChatModelClient | None = None,
                 config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_env()
        self.client = client or LiteLLMChatClient(self.config)
        self.model_call = LangGraphModelCall(self.client)

    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        timeout_seconds: int | None,
    ) -> tuple[Any, ModelCompletion]:
        completion = self.model_call.invoke(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name=self.agent_name,
            timeout_seconds=timeout_seconds,
        )
        try:
            response = self.response_model.model_validate(json.loads(completion.content))
            self._validate_response(response)
        except (json.JSONDecodeError, ValueError) as exc:
            raise AgentOutputValidationError(
                f"{self.agent_name.title()} agent returned invalid structured output: {exc}",
                completion=completion,
            ) from exc
        return response, completion

    def _system_prompt(self) -> str:
        schema = _json(self.response_model.model_json_schema())
        return f"{_PROMPTS[self.agent_name]}\n\nExpected JSON Schema:\n{schema}"

    def _validate_response(self, response: Any) -> None:
        """Subclasses may add deterministic checks after Pydantic parsing."""


class CoderAgent(_StructuredBuildAgent):
    """Produce task-scoped file changes; workspace code enforces permissions and commits."""

    agent_name = "coder"
    prompt_version = "coder_v1"
    response_model = CoderAgentResponse

    def build_prompts(
        self,
        task_node: TaskNode | Mapping[str, Any],
        *,
        relevant_spec_context: Sequence[RetrievedContextChunk | Mapping[str, Any]],
        repository_files: Mapping[str, str],
        debugger_feedback: Mapping[str, Any] | None = None,
    ) -> tuple[str, str]:
        task = _task(task_node)
        user_context = {
            "task": task.model_dump(mode="json"),
            "relevant_project_spec_context": _chunks(relevant_spec_context),
            "repository_files": dict(sorted(repository_files.items())),
            "debugger_feedback": dict(debugger_feedback or {}),
        }
        return self._system_prompt(), _json(user_context)

    def run(
        self,
        task_node: TaskNode | Mapping[str, Any],
        *,
        relevant_spec_context: Sequence[RetrievedContextChunk | Mapping[str, Any]],
        repository_files: Mapping[str, str],
        debugger_feedback: Mapping[str, Any] | None = None,
        timeout_seconds: int | None = None,
    ) -> tuple[CoderAgentResponse, ModelCompletion]:
        system_prompt, user_prompt = self.build_prompts(
            task_node,
            relevant_spec_context=relevant_spec_context,
            repository_files=repository_files,
            debugger_feedback=debugger_feedback,
        )
        response, completion = self._invoke(
            system_prompt, user_prompt, timeout_seconds=timeout_seconds,
        )
        task = _task(task_node)
        if response.task_id != task.id:
            raise AgentOutputValidationError(
                f"Coder response task_id {response.task_id!r} does not match task {task.id!r}",
                completion=completion,
            )
        return response, completion


class TesterAgent(_StructuredBuildAgent):
    """Generate test files from acceptance criteria; sandbox execution is a separate step."""

    agent_name = "tester"
    prompt_version = "tester_v1"
    response_model = TesterAgentResponse

    def build_prompts(
        self,
        task_node: TaskNode | Mapping[str, Any],
        *,
        runtime: str,
        repository_files: Mapping[str, str],
    ) -> tuple[str, str]:
        task = _task(task_node)
        user_context = {
            "task_id": task.id,
            "task_description": task.description,
            "runtime": runtime,
            "acceptance_criteria": task.acceptance_criteria,
            "repository_files": dict(sorted(repository_files.items())),
        }
        return self._system_prompt(), _json(user_context)

    def run(
        self,
        task_node: TaskNode | Mapping[str, Any],
        *,
        runtime: str,
        repository_files: Mapping[str, str],
        timeout_seconds: int | None = None,
    ) -> tuple[TesterAgentResponse, ModelCompletion]:
        system_prompt, user_prompt = self.build_prompts(
            task_node, runtime=runtime, repository_files=repository_files,
        )
        return self._invoke(system_prompt, user_prompt, timeout_seconds=timeout_seconds)


class DebuggerAgent(_StructuredBuildAgent):
    """Diagnose failing tests and return repair guidance without modifying files."""

    agent_name = "debugger"
    prompt_version = "debugger_v1"
    response_model = DebuggerAgentResponse

    def build_prompts(
        self,
        task_node: TaskNode | Mapping[str, Any],
        *,
        test_failure: Mapping[str, Any],
        repository_files: Mapping[str, str],
        retry_number: int,
    ) -> tuple[str, str]:
        task = _task(task_node)
        user_context = {
            "task": task.model_dump(mode="json"),
            "retry_number": retry_number,
            "test_failure": dict(test_failure),
            "repository_files": dict(sorted(repository_files.items())),
        }
        return self._system_prompt(), _json(user_context)

    def run(
        self,
        task_node: TaskNode | Mapping[str, Any],
        *,
        test_failure: Mapping[str, Any],
        repository_files: Mapping[str, str],
        retry_number: int,
        timeout_seconds: int | None = None,
    ) -> tuple[DebuggerAgentResponse, ModelCompletion]:
        system_prompt, user_prompt = self.build_prompts(
            task_node,
            test_failure=test_failure,
            repository_files=repository_files,
            retry_number=retry_number,
        )
        return self._invoke(system_prompt, user_prompt, timeout_seconds=timeout_seconds)
