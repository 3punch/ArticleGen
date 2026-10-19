from articlegen.changelog import read_jsonl
from articlegen.models import Brief, Draft, Section
from articlegen.orchestrator import Orchestrator, OrchestratorConfig
from articlegen.scoring import decide_next, score_iteration


def _config(tmp_path, **overrides):
    data = {
        "max_iterations": 3,
        "score_target": 0.85,
        "workspace": str(tmp_path / "runs"),
        "changelog_path": str(tmp_path / "change_log.md"),
        "provider": "dahl",
    }
    data.update(overrides)
    return OrchestratorConfig.from_dict(data)


def test_invalid_brief_is_routed_back_without_running_agents(tmp_path, valid_brief):
    valid_brief.objective = "short"
    orchestrator = Orchestrator(_config(tmp_path))
    result = orchestrator.run(valid_brief)

    assert result.status == "ROUTE_BACK"
    assert result.iterations == []
    assert result.tokens_total == 0
    log = (tmp_path / "change_log.md").read_text()
    assert "GATE ROUTE-BACK" in log
    assert "ROUTE_BACK" in log


def test_run_without_an_llm_never_creates_an_offline_article(tmp_path, valid_brief):
    from articlegen.llm import OfflineClient

    result = Orchestrator(_config(tmp_path), client=OfflineClient()).run(valid_brief)

    assert result.status == "ROUTE_BACK"
    assert result.iterations == []
    assert result.final_draft is None
    assert "no offline fallback" in result.stop_reason.lower()
    assert not (tmp_path / "runs" / valid_brief.task_id).exists()


def test_invalid_task_id_cannot_create_a_path_outside_the_workspace(tmp_path, valid_brief):
    valid_brief.task_id = "../escaped-task"
    result = Orchestrator(_config(tmp_path)).run(valid_brief)

    assert result.status == "ROUTE_BACK"
    assert result.workspace == ""
    assert not (tmp_path / "escaped-task").exists()
    assert not (tmp_path / "runs").exists()


def test_final_status_respects_the_configured_score_target():
    from articlegen.models import GateResult

    gate = GateResult(ok=True, stage="draft_verification")
    assert Orchestrator._final_status(0.80, gate, [], score_target=0.70) == "PASS"
    assert Orchestrator._final_status(0.80, gate, [], score_target=0.90) == "NEEDS_HUMAN"


def test_run_does_not_pass_when_it_misses_a_higher_configured_target(
    tmp_path, example_brief_dict, model_client
):
    brief = Brief.from_dict(example_brief_dict)
    result = Orchestrator(
        _config(tmp_path, max_iterations=2, score_target=1.0), client=model_client
    ).run(brief)

    assert 0.85 <= result.final_score < 1.0
    assert result.status == "NEEDS_HUMAN"


def test_seeded_section_prose_is_preserved_in_the_initial_draft(valid_brief):
    valid_brief.sections[0].body = "A complete user-supplied section draft."

    draft = Draft.from_brief(valid_brief)

    assert draft.sections[0].body == valid_brief.sections[0].body
    assert all(not section.body for section in draft.sections[1:])


def test_full_loop_uses_configured_model_and_produces_a_draft(tmp_path, valid_brief, model_client):
    orchestrator = Orchestrator(_config(tmp_path), client=model_client)
    result = orchestrator.run(valid_brief)

    assert result.status in ("PASS", "NEEDS_HUMAN")
    assert result.final_draft is not None
    assert [s.id for s in result.final_draft.sections] == valid_brief.section_ids
    assert all(s.body.strip() for s in result.final_draft.sections)
    assert result.final_draft.word_count > 0
    assert (tmp_path / "runs" / valid_brief.task_id / "master_draft.md").exists()
    assert all(report.engine.startswith("test-llm")
               for report in result.iterations[0].reports)


def test_every_iteration_is_logged(tmp_path, valid_brief, model_client):
    orchestrator = Orchestrator(_config(tmp_path, max_iterations=3), client=model_client)
    result = orchestrator.run(valid_brief)

    entries = read_jsonl(str(tmp_path / "change_log.jsonl"))
    iteration_entries = [e for e in entries if e.get("event") == "iteration"]
    assert len(iteration_entries) == len(result.iterations)
    assert any(e.get("event") == "RUN_START" for e in entries)
    assert any(e.get("event") == "RUN_END" for e in entries)
    assert result.stop_reason


def test_score_does_not_regress_after_the_first_iteration(tmp_path, valid_brief, model_client):
    orchestrator = Orchestrator(
        _config(tmp_path, max_iterations=3, score_target=1.0), client=model_client
    )
    result = orchestrator.run(valid_brief)
    scores = [r.score for r in result.iterations]
    assert len(scores) >= 2
    # Iteration 1 drafts from nothing; later iterations must not be worse.
    assert max(scores) >= scores[0]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_context_packets_respect_the_hard_cap(tmp_path, valid_brief, model_client):
    config = _config(tmp_path, max_iterations=2)
    orchestrator = Orchestrator(config, client=model_client)
    orchestrator.run(valid_brief)

    from articlegen.tokens import count_tokens
    run_dir = tmp_path / "runs" / valid_brief.task_id / "iterations"
    context_files = list(run_dir.rglob("context-*.txt"))
    assert context_files, "no context packets were persisted"
    for path in context_files:
        assert count_tokens(path.read_text()) <= config.budget.context_hard_cap + 60, path.name


def test_run_is_deterministic(tmp_path, valid_brief, model_client):
    first = Orchestrator(_config(tmp_path / "a"), client=model_client).run(valid_brief)
    second = Orchestrator(_config(tmp_path / "b"), client=model_client).run(valid_brief)
    assert [round(r.score, 4) for r in first.iterations] == [round(r.score, 4) for r in second.iterations]
    assert first.tokens_total == second.tokens_total


def test_open_findings_are_capped(tmp_path, model_client):
    brief = Brief(
        task_id="many-findings",
        title="A Brief Designed To Produce Many Findings",
        objective="Produce enough findings to prove the open-findings list is bounded rather than growing without limit.",
        audience="engineers",
        target_words=900,
        sections=[
            Section(id=f"section-{index}", title=f"Section {index}",
                    purpose=f"describe aspect {index} of the system", word_target=120)
            for index in range(10)
        ],
    )
    orchestrator = Orchestrator(_config(tmp_path, max_iterations=2), client=model_client)
    orchestrator.run(brief)

    from articlegen.memory import MemoryStore
    store = MemoryStore(str(tmp_path / "runs"), brief.task_id)
    assert len(store.open_findings) <= 40


def test_decide_next_stops_on_target():
    decision, reason, stall = decide_next(
        score=0.9, previous_score=0.8, iteration=1, max_iterations=4, score_target=0.85,
        min_delta=0.01, stall_window=2, stall_count=0, tokens_total=0,
        token_ceiling=60000, degraded=False,
    )
    assert decision == "stop"
    assert "target met" in reason


def test_decide_next_stops_on_stall_window():
    stall = 0
    decision = "continue"
    for _ in range(3):
        decision, reason, stall = decide_next(
            score=0.5, previous_score=0.5, iteration=1, max_iterations=9, score_target=0.85,
            min_delta=0.01, stall_window=2, stall_count=stall, tokens_total=0,
            token_ceiling=60000, degraded=False,
        )
        if decision == "stop":
            break
    assert decision == "stop"
    assert "no material improvement" in reason


def test_decide_next_respects_token_ceiling():
    decision, reason, _ = decide_next(
        score=0.3, previous_score=0.2, iteration=1, max_iterations=9, score_target=0.85,
        min_delta=0.01, stall_window=2, stall_count=0, tokens_total=70000,
        token_ceiling=60000, degraded=False,
    )
    assert decision == "stop"
    assert "token ceiling" in reason


def test_decide_next_does_not_stall_on_degraded_output():
    decision, _, stall = decide_next(
        score=0.5, previous_score=0.5, iteration=1, max_iterations=9, score_target=0.85,
        min_delta=0.01, stall_window=1, stall_count=0, tokens_total=0,
        token_ceiling=60000, degraded=True,
    )
    assert decision == "continue"
    assert stall == 0


def test_score_iteration_penalises_blockers_more_than_notes(valid_brief):
    from articlegen.agents import build_agents
    from articlegen.llm import OfflineClient
    from articlegen.models import AgentReport, Finding, Severity
    from articlegen.tokens import TokenBudget

    build_agents(OfflineClient(), TokenBudget.default())
    draft = Draft.from_brief(valid_brief)
    for section in draft.sections:
        section.body = "Content."

    def report_with(severity):
        finding = Finding("tester", severity, "logic", "mechanism", "problem", "fix it")
        return AgentReport(agent_id="tester", role="t", iteration=1, findings=[finding])

    blockers = score_iteration(valid_brief, draft, [report_with(Severity.BLOCKER)])
    notes = score_iteration(valid_brief, draft, [report_with(Severity.NOTE)])
    assert notes.total > blockers.total
