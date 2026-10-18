"""Model-backed academic-brief normalization with explicit clarification outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from projectforge_schemas.models import (
    ClarifyingQuestion,
    ProjectSpec,
    RequirementAgentResponse,
)

from .config import ModelConfig
from .errors import AgentOutputValidationError
from .langgraph_runner import LangGraphModelCall
from .litellm_client import ChatModelClient, LiteLLMChatClient, ModelCompletion


_PROMPT = Path(__file__).with_name("prompts").joinpath("requirement_v1.txt").read_text(
    encoding="utf-8",
)


class RequirementAgent:
    """Convert a raw brief and optional rubric into a validated Phase 1 ProjectSpec."""

    def __init__(self, *, client: ChatModelClient | None = None,
                 config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_env()
        self.client = client or LiteLLMChatClient(self.config)
        self.model_call = LangGraphModelCall(self.client)

    def build_prompts(
        self,
        raw_brief_text: str,
        *,
        rubric_text: str | None = None,
        clarification_answers: Mapping[str, str] | None = None,
        source_project_spec: ProjectSpec | Mapping[str, Any] | None = None,
    ) -> tuple[str, str]:
        brief = raw_brief_text.strip()
        if len(brief) < 20:
            raise ValueError("RequirementAgent requires at least 20 characters of brief text")
        source_spec: dict[str, Any] | None
        if isinstance(source_project_spec, ProjectSpec):
            source_spec = source_project_spec.model_dump(mode="json")
        elif source_project_spec is None:
            source_spec = None
        else:
            source_spec = ProjectSpec.model_validate(source_project_spec).model_dump(mode="json")
        context = {
            "raw_brief_text": brief,
            "rubric_text": rubric_text.strip() if isinstance(rubric_text, str) and rubric_text.strip() else None,
            "source_project_spec": source_spec,
            "clarification_answers": dict(clarification_answers or {}),
            "confidence_threshold": self.config.confidence_threshold,
        }
        schema = json.dumps(RequirementAgentResponse.model_json_schema(), ensure_ascii=False)
        return (
            f"{_PROMPT}\n\nExpected JSON Schema:\n{schema}",
            json.dumps(context, ensure_ascii=False, sort_keys=True),
        )

    def run(
        self,
        raw_brief_text: str,
        *,
        rubric_text: str | None = None,
        clarification_answers: Mapping[str, str] | None = None,
        source_project_spec: ProjectSpec | Mapping[str, Any] | None = None,
        timeout_seconds: int | None = None,
    ) -> tuple[RequirementAgentResponse, ModelCompletion]:
        system_prompt, user_prompt = self.build_prompts(
            raw_brief_text,
            rubric_text=rubric_text,
            clarification_answers=clarification_answers,
            source_project_spec=source_project_spec,
        )
        completion = self.model_call.invoke(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name="requirement",
            timeout_seconds=timeout_seconds,
        )
        try:
            decoded = json.loads(completion.content)
            response = RequirementAgentResponse.model_validate(decoded)
        except (json.JSONDecodeError, ValueError) as exc:
            raise AgentOutputValidationError(
                f"Requirement agent returned invalid structured output: {exc}",
                completion=completion,
            ) from exc

        if response.status == "completed" and response.confidence < self.config.confidence_threshold:
            questions = response.questions or self._questions_from_uncertainties(response.uncertainties)
            response = RequirementAgentResponse(
                status="needs_clarification",
                confidence=response.confidence,
                project_spec=None,
                questions=questions,
                uncertainties=response.uncertainties,
            )
        # A fully validated ProjectSpec enforces that every rubric entry has exactly
        # one of the contract's two types: automatable or human-judgment.
        if response.status == "completed":
            ProjectSpec.model_validate(response.project_spec)
        return response, completion

    @staticmethod
    def _questions_from_uncertainties(uncertainties: list[str]) -> list[ClarifyingQuestion]:
        items = uncertainties[:20] or [
            "The brief does not provide enough detail to produce a reliable project specification."
        ]
        questions: list[ClarifyingQuestion] = []
        for index, uncertainty in enumerate(items, start=1):
            detail = uncertainty.strip()
            question = detail if "?" in detail else f"Please clarify: {detail}"
            if len(question) < 8:
                question = f"Please clarify {question}."
            questions.append(ClarifyingQuestion(
                question_id=f"confidence_{index}",
                question=question[:2000],
                why_it_matters=(
                    f"This uncertainty affects the requirements or evaluation: {detail}"
                )[:2000],
            ))
        return questions
