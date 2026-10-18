from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from projectforge_agents.config import ModelConfig
from projectforge_agents.errors import AgentOutputValidationError
from projectforge_agents.litellm_client import (
    LiteLLMChatClient,
    ModelCompletion,
    PromptTokenLimitExceeded,
)
from projectforge_agents.planner import PlannerAgent
from projectforge_agents.requirement import RequirementAgent
from projectforge_schemas.models import (
    PlannerAgentResponse,
    ProjectSpec,
    RequirementAgentResponse,
    TaskGraph,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "projectforge"
FIXTURE_NAMES = ("simple_script", "web_app", "data_analysis", "erdos_gyarfas")


class FixtureModelClient:
    """Offline model double returning versioned evaluation fixture responses."""

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str,
                 agent_name: str, timeout_seconds: int | None = None) -> ModelCompletion:
        self.calls.append((agent_name, system_prompt, user_prompt))
        response = self.responses[agent_name]
        return ModelCompletion(
            content=json.dumps(response, ensure_ascii=False),
            tokens_used=128,
            cost=Decimal("0.00100000"),
            provider="mock",
            model=f"fixture-strong-{agent_name}",
        )


def _config() -> ModelConfig:
    return ModelConfig.from_env({
        "PROJECTFORGE_LLM_PROVIDER": "dahl",
        "PROJECTFORGE_LLM_TIER": "strong",
        "PROJECTFORGE_DAHL_STRONG_MODEL": "academic-eval-model",
        "DAHL_API_KEY": "not-used-by-offline-fixture-client",
        "PROJECTFORGE_LLM_RESERVATION_TOKENS": "1000",
        "PROJECTFORGE_LLM_MAX_OUTPUT_TOKENS": "512",
        "PROJECTFORGE_LLM_RESERVATION_SPEND": "0.05000000",
        "PROJECTFORGE_REQUIREMENT_CONFIDENCE": "0.70",
    })


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_offline_academic_brief_evaluations_exercise_requirement_and_planner(fixture_name):
    fixture = json.loads((FIXTURE_DIR / f"{fixture_name}.json").read_text(encoding="utf-8"))
    client = FixtureModelClient({
        "requirement": fixture["requirement_response"],
        "planner": fixture["planner_response"],
    })
    config = _config()

    requirement, requirement_completion = RequirementAgent(
        client=client, config=config,
    ).run(fixture["raw_brief_text"], rubric_text=fixture["rubric_text"])
    assert isinstance(requirement, RequirementAgentResponse)
    assert requirement.status == "completed"
    assert requirement.confidence >= config.confidence_threshold
    assert requirement.project_spec is not None
    spec = ProjectSpec.model_validate(requirement.project_spec.model_dump(mode="json"))
    assert len(spec.rubric_items) == len(fixture["requirement_response"]["project_spec"]["rubric_items"])
    assert {item.type for item in spec.rubric_items} <= {"automatable", "human-judgment"}
    assert requirement_completion.provider == "mock"

    plan, planner_completion = PlannerAgent(client=client, config=config).run(spec)
    assert isinstance(plan, PlannerAgentResponse)
    graph = TaskGraph.model_validate(plan.task_graph.model_dump(mode="json"))
    expected = fixture["expected_task_count"]
    assert expected["min"] <= len(graph.nodes) <= expected["max"]
    assert all(node.assigned_agent == "coder" for node in graph.nodes)
    assert all(node.acceptance_criteria for node in graph.nodes)
    assert planner_completion.provider == "mock"
    assert [call[0] for call in client.calls] == ["requirement", "planner"]
    assert all("Expected JSON Schema" in system_prompt for _, system_prompt, _ in client.calls)


def test_low_confidence_requirement_output_becomes_a_question_not_a_guessed_spec():
    fixture = json.loads((FIXTURE_DIR / "simple_script.json").read_text(encoding="utf-8"))
    low_confidence = dict(fixture["requirement_response"])
    low_confidence["confidence"] = 0.42
    low_confidence["uncertainties"] = ["The required input file format is not specified clearly"]
    client = FixtureModelClient({"requirement": low_confidence})

    response, _ = RequirementAgent(client=client, config=_config()).run(
        fixture["raw_brief_text"], rubric_text=fixture["rubric_text"],
    )
    assert response.status == "needs_clarification"
    assert response.project_spec is None
    assert response.questions
    assert response.questions[0].question_id == "confidence_1"


def test_planner_rejects_cycles_and_vague_acceptance_criteria():
    fixture = json.loads((FIXTURE_DIR / "simple_script.json").read_text(encoding="utf-8"))
    cyclic = json.loads(json.dumps(fixture["planner_response"]))
    cyclic["task_graph"]["nodes"][0]["dependencies"] = ["tests-and-readme"]
    client = FixtureModelClient({"planner": cyclic})
    spec = ProjectSpec.model_validate(fixture["requirement_response"]["project_spec"])
    with pytest.raises(AgentOutputValidationError, match="acyclic"):
        PlannerAgent(client=client, config=_config()).run(spec)

    vague = json.loads(json.dumps(fixture["planner_response"]))
    vague["task_graph"]["nodes"][0]["acceptance_criteria"] = ["Works correctly"]
    client = FixtureModelClient({"planner": vague})
    with pytest.raises(AgentOutputValidationError, match="vague"):
        PlannerAgent(client=client, config=_config()).run(spec)


def test_dahl_model_tier_is_configurable_and_shared_by_both_agents():
    config = _config()
    assert config.provider == "dahl"
    assert config.tier == "strong"
    assert config.model_for("requirement") == "openai/academic-eval-model"
    assert config.model_for("planner") == "openai/academic-eval-model"
    per_agent = ModelConfig.from_env({
        "PROJECTFORGE_LLM_PROVIDER": "dahl",
        "PROJECTFORGE_LLM_TIER": "strong",
        "PROJECTFORGE_DAHL_STRONG_MODEL": "shared",
        "PROJECTFORGE_PLANNER_MODEL": "planner-specialist",
        "PROJECTFORGE_CODER_MODEL": "coder-specialist",
        "PROJECTFORGE_TESTER_MODEL": "tester-specialist",
        "PROJECTFORGE_DEBUGGER_MODEL": "debugger-specialist",
        "PROJECTFORGE_REVIEWER_MODEL": "reviewer-specialist",
        "PROJECTFORGE_PACKAGER_MODEL": "packager-specialist",
    })
    assert per_agent.model_for("requirement") == "openai/shared"
    assert per_agent.model_for("planner") == "openai/planner-specialist"
    assert per_agent.model_for("coder") == "openai/coder-specialist"
    assert per_agent.model_for("tester") == "openai/tester-specialist"
    assert per_agent.model_for("debugger") == "openai/debugger-specialist"
    assert per_agent.model_for("reviewer") == "openai/reviewer-specialist"
    assert per_agent.model_for("packager") == "openai/packager-specialist"


def test_litellm_adapter_uses_dahl_route_and_extracts_usage_without_network(monkeypatch):
    class MockLiteLLM:
        call: dict[str, Any] = {}

        @classmethod
        def completion(cls, **kwargs):
            cls.call = kwargs
            return {
                "choices": [{"message": {"content": json.dumps({"status": "completed"})}}],
                "usage": {"total_tokens": 45},
                "_hidden_params": {"response_cost": "0.002"},
            }

    monkeypatch.setitem(sys.modules, "litellm", MockLiteLLM)
    config = _config()
    completion = LiteLLMChatClient(config).complete(
        system_prompt="system",
        user_prompt="user",
        agent_name="requirement",
        timeout_seconds=9,
    )
    assert completion.content == json.dumps({"status": "completed"})
    assert completion.tokens_used == 45
    assert completion.cost == Decimal("0.002")
    assert completion.provider == "dahl"
    assert completion.model == "openai/academic-eval-model"
    assert MockLiteLLM.call["api_base"] == "https://inference.dahl.global/v1"
    assert MockLiteLLM.call["api_key"] == "not-used-by-offline-fixture-client"
    assert MockLiteLLM.call["max_tokens"] == 512
    assert MockLiteLLM.call["timeout"] == 9
    assert MockLiteLLM.call["num_retries"] == 0


def test_litellm_client_blocks_prompt_that_exceeds_reserved_token_ceiling(monkeypatch):
    class NoNetworkLiteLLM:
        @staticmethod
        def completion(**kwargs):
            raise AssertionError("provider must not be called for an oversized prompt")

    monkeypatch.setitem(sys.modules, "litellm", NoNetworkLiteLLM)
    config = ModelConfig.from_env({
        "PROJECTFORGE_LLM_PROVIDER": "dahl",
        "PROJECTFORGE_LLM_TIER": "strong",
        "PROJECTFORGE_DAHL_STRONG_MODEL": "token-budget-model",
        "DAHL_API_KEY": "offline-test-key",
        "PROJECTFORGE_LLM_MAX_OUTPUT_TOKENS": "32",
        "PROJECTFORGE_LLM_RESERVATION_TOKENS": "64",
    })
    with pytest.raises(PromptTokenLimitExceeded, match="per-call token reservation"):
        LiteLLMChatClient(config).complete(
            system_prompt="brief instructions " * 200,
            user_prompt="a short brief",
            agent_name="requirement",
        )
