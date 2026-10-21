"""End-to-end research integrity: pipeline, orchestrator wiring and the CLI flags.

These tests never touch a provider or the network. The planner/reviewer/reviser are a
fixture stub, the executor is a real subprocess running a tiny deterministic program,
and the source retriever is a static fixture.

They assert the property the whole layer exists for: a well-supported manuscript can
reach "ready for final review", while an overclaiming one cannot - no matter how good
its prose score is.
"""

from __future__ import annotations

import json
import os

import pytest

from articlegen.cli import build_parser
from articlegen.events import EventBus
from articlegen.llm import LLMResponse
from articlegen.memory import MemoryStore
from articlegen.models import Draft, Section
from articlegen.orchestrator import Orchestrator, OrchestratorConfig
from articlegen.research.pipeline import ResearchPipeline
from articlegen.research.report import apply_bibliography
from articlegen.research.sources import NullSourceRetriever, StaticSourceRetriever
from tests.conftest import TestLLMClient
from tests import research_fixtures as fixtures


def _payload(plan, code: str = fixtures.GOOD_CODE,
             tests: str = fixtures.GOOD_TESTS) -> dict:
    payload = plan.to_dict()
    payload["experiments"] = [fixtures.spec_payload("exp-1", code, tests)]
    payload["source_queries"] = ["monochromatic cycle edge coloured complete graph"]
    return payload


def _pipeline(tmp_path, brief, client, **kwargs) -> ResearchPipeline:
    store = MemoryStore(str(tmp_path / "runs"), brief.task_id)
    return ResearchPipeline(client=client, store=store, brief=brief, enabled=True,
                            **kwargs)


def _events(events: EventBus) -> list:
    return [event.payload.get("stage") for event in events.events
            if event.type.value == "research_stage"]


def test_supported_study_reaches_ready_for_final_review(tmp_path):
    brief = fixtures.study_brief()
    plan = fixtures.good_plan(brief)
    client = fixtures.FakeResearchClient(plan_payload=_payload(plan))
    events = EventBus()
    pipeline = _pipeline(
        tmp_path, brief, client, review_iterations=1,
        retriever=StaticSourceRetriever(fixtures.source_records(), mark_verified=True),
        events=events)

    pipeline.begin()
    draft = pipeline.complete(fixtures.good_draft(brief))
    apply_bibliography(draft, pipeline.outcome.library)

    assert pipeline.outcome.plan.validated is True
    assert pipeline.outcome.plan.decision == "proceed"
    assert pipeline.outcome.plan.missing_elements() == []
    assert [run.status.value for run in pipeline.outcome.runs] == ["ok"]
    assert pipeline.outcome.runs[0].tests_passed == 2
    assert pipeline.outcome.library.report().verified == 3
    assert pipeline.outcome.validation.summary()["fail"] == 0
    assert pipeline.outcome.gate.status.value == "ready_for_final_review"
    assert pipeline.outcome.gate.states.to_dict()["sources_verified"] is True
    assert pipeline.outcome.gate.states.to_dict()["experiments_executed"] is True

    # Every stage is emitted so a dashboard can show the run while it happens.
    assert {"sources", "experiment", "plan", "peer_review", "validation",
            "quality_gate"} <= set(_events(events))

    research_dir = os.path.join(pipeline.store.root, "research")
    for name in ("plan.json", "sources.json", "experiments.json", "validation.json",
                 "review.json", "quality_gate.json", "report.md"):
        assert os.path.exists(os.path.join(research_dir, name)), name
    report = open(os.path.join(research_dir, "report.md"), encoding="utf-8").read()
    assert "Ready for final human review" in report
    assert "not a guarantee of scientific correctness" in report.lower()


def test_overclaiming_study_is_blocked_even_with_a_recorded_run(tmp_path):
    brief = fixtures.study_brief(vector="erdos")
    plan = fixtures.erdos_plan(brief, fixtures.verified_library())
    client = fixtures.FakeResearchClient(
        plan_payload=_payload(plan, fixtures.ERDOS_CODE, fixtures.ERDOS_TESTS))
    pipeline = _pipeline(tmp_path, brief, client, review_iterations=0,
                         retriever=fixtures_static())

    pipeline.begin()
    draft = pipeline.complete(fixtures.erdos_draft(brief))
    apply_bibliography(draft, pipeline.outcome.library)

    assert pipeline.outcome.plan.validated is False
    assert pipeline.outcome.plan.decision == "narrow_claims"
    assert pipeline.outcome.gate.status.value == "incomplete_or_unverified"
    checks = {check.check_id for check in pipeline.outcome.validation.problems()}
    for expected in ("plan.gap_established", "plan.novelty_supported",
                     "code.generality", "performance.measured", "proof.claims"):
        assert expected in checks


def fixtures_static() -> StaticSourceRetriever:
    return StaticSourceRetriever(fixtures.source_records(), mark_verified=True)


def test_planner_failure_never_invents_a_plan_or_a_pass(tmp_path):
    brief = fixtures.study_brief()
    client = fixtures.FakeResearchClient(raise_on_planner=True)
    pipeline = _pipeline(tmp_path, brief, client, review_iterations=0,
                         retriever=fixtures_static())

    pipeline.begin()
    draft = pipeline.complete(fixtures.good_draft(brief))
    apply_bibliography(draft, pipeline.outcome.library)

    assert pipeline.outcome.plan.origin == "deterministic"
    assert pipeline.outcome.plan.validated is False
    assert any("model planning unavailable" in note for note in pipeline.outcome.notes)
    assert pipeline.outcome.gate.status.value == "incomplete_or_unverified"
    assert "plan" in " ".join(pipeline.outcome.gate.reasons).lower()


def test_missing_retrieval_is_reported_and_never_fabricated(tmp_path):
    brief = fixtures.study_brief()
    plan = fixtures.good_plan(brief)
    client = fixtures.FakeResearchClient(plan_payload=_payload(plan))
    pipeline = _pipeline(tmp_path, brief, client, review_iterations=0,
                         retriever=NullSourceRetriever())

    pipeline.begin()
    draft = pipeline.complete(fixtures.good_draft(brief))

    assert pipeline.outcome.library.records == []
    assert pipeline.outcome.library.bibliography().strip() == ""
    assert pipeline.outcome.retrieval["available"] is False
    assert pipeline.outcome.gate.status.value == "incomplete_or_unverified"
    reasons = " ".join(pipeline.outcome.gate.reasons)
    assert "source" in reasons.lower()
    # No reference entry may be written without a record behind it.
    references = next((s for s in draft.sections if s.id == "references"), None)
    assert references is None or "[1]" not in references.body


def test_experiments_disabled_leaves_the_study_incomplete(tmp_path):
    brief = fixtures.study_brief()
    plan = fixtures.good_plan(brief)
    client = fixtures.FakeResearchClient(plan_payload=_payload(plan))
    pipeline = _pipeline(tmp_path, brief, client, review_iterations=0,
                         execute_experiments=False, retriever=fixtures_static())

    pipeline.begin()

    assert [run.status.value for run in pipeline.outcome.runs] == ["not_run"]
    assert any("execution is disabled" in note for note in pipeline.outcome.notes)


# --- orchestrator wiring ---------------------------------------------------


class _OrchestratorClient(TestLLMClient):
    """Writer/Tester/... from the shared fixture, research prompts from ours."""

    def __init__(self, plan_payload: dict, reviewer_payload: dict | None = None) -> None:
        self.plan_payload = plan_payload
        self.reviewer_payload = reviewer_payload or {"issues": []}

    def complete(self, system: str, user: str, *, max_output_tokens: int = 1000):
        if "# Research Planner" in system:
            return LLMResponse(text=json.dumps(self.plan_payload), tokens_in=10,
                               tokens_out=10, engine="test-llm")
        if "# Scientific Reviewer" in system:
            return LLMResponse(text=json.dumps(self.reviewer_payload), tokens_in=10,
                               tokens_out=5, engine="test-llm")
        if "# Targeted Reviser" in system:
            return LLMResponse(text=json.dumps({"sections": [], "addressed": [],
                                                "notes": []}),
                               tokens_in=10, tokens_out=5, engine="test-llm")
        return super().complete(system, user, max_output_tokens=max_output_tokens)


def _run_orchestrator(tmp_path, brief, client, **overrides) -> object:
    data = {
        "max_iterations": 1,
        "workspace": str(tmp_path / "runs"),
        "changelog_path": str(tmp_path / "change_log.md"),
        "source_retrieval": "none",
        "review_iterations": 1,
    }
    data.update(overrides)
    return Orchestrator(OrchestratorConfig.from_dict(data), client=client).run(brief)


def test_orchestrator_reports_needs_human_when_research_is_incomplete(tmp_path):
    brief = fixtures.study_brief()
    plan = fixtures.derive_plan(brief)
    result = _run_orchestrator(tmp_path, brief,
                               _OrchestratorClient(_payload(plan)))

    assert result.status == "NEEDS_HUMAN"
    assert result.quality_status == "incomplete_or_unverified"
    assert "research validation status" in result.stop_reason
    assert "quality_gate" in result.research
    assert result.research["quality_gate"]["status"] == "incomplete_or_unverified"
    assert os.path.exists(os.path.join(result.workspace, "research", "report.md"))
    log = open(tmp_path / "change_log.md", encoding="utf-8").read()
    assert "RESEARCH_BEGIN" in log and "RESEARCH_GATE" in log


def test_writing_score_cannot_override_a_research_failure():
    from articlegen.models import GateResult

    gate = GateResult(ok=True, stage="draft_verification")
    assert Orchestrator._final_status(0.99, gate, [], score_target=0.5,
                                      quality_status="needs_revision") == "NEEDS_HUMAN"
    assert Orchestrator._final_status(0.99, gate, [], score_target=0.5,
                                      quality_status="incomplete_or_unverified") \
        == "NEEDS_HUMAN"
    assert Orchestrator._final_status(0.99, gate, [], score_target=0.5,
                                      quality_status="ready_for_final_review") == "PASS"


def test_research_can_be_disabled_for_a_run(tmp_path):
    brief = fixtures.study_brief()
    result = _run_orchestrator(tmp_path, brief, _OrchestratorClient({}), research=False)

    assert result.quality_status == ""
    assert result.research == {}
    assert not os.path.exists(os.path.join(result.workspace, "research"))


# --- CLI -------------------------------------------------------------------


def test_cli_exposes_the_research_controls():
    parser = build_parser()
    args = parser.parse_args([
        "run", "brief.json",
        "--article-type", "computational",
        "--citation-style", "ieee",
        "--review-iterations", "2",
        "--source-retrieval", "crossref",
        "--max-sources", "12",
        "--no-experiments",
        "--experiment-timeout", "45",
    ])
    assert args.article_type == "computational"
    assert args.citation_style == "ieee"
    assert args.review_iterations == 2
    assert args.source_retrieval == "crossref"
    assert args.max_sources == 12
    assert args.no_experiments is True
    assert args.experiment_timeout == 45


def test_cli_can_turn_research_off():
    parser = build_parser()
    args = parser.parse_args(["run", "brief.json", "--no-research"])
    assert args.no_research is True


def test_cli_rejects_an_unknown_retrieval_policy():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "brief.json",
                           "--source-retrieval", "invented-index"])
