import os

from articlegen.changelog import ChangeLog, read_jsonl
from articlegen.gate import verify_brief
from articlegen.models import AgentReport, AgentStatus, Finding, IterationRecord, Severity


def test_header_is_written_once(tmp_path):
    path = str(tmp_path / "change_log.md")
    ChangeLog(path)
    ChangeLog(path)
    text = open(path, encoding="utf-8").read()
    assert text.count("# ArticleGen Change Log") == 1


def test_event_appends_markdown_and_jsonl(tmp_path):
    log = ChangeLog(str(tmp_path / "change_log.md"))
    log.log_event("RUN_START", "demo", {"status": "started"})

    assert "RUN_START" in open(str(tmp_path / "change_log.md"), encoding="utf-8").read()
    entries = read_jsonl(str(tmp_path / "change_log.jsonl"))
    assert entries[-1]["event"] == "RUN_START"
    assert entries[-1]["task_id"] == "demo"


def test_appends_are_append_only(tmp_path):
    path = str(tmp_path / "change_log.md")
    log = ChangeLog(path)
    log.log_event("A", "demo", {"status": "first"})
    first = open(path, encoding="utf-8").read()
    log.log_event("B", "demo", {"status": "second"})
    second = open(path, encoding="utf-8").read()
    assert second.startswith(first.rstrip("\n")[:len(first.rstrip("\n"))])
    assert "A" in second and "B" in second


def test_route_back_block_lists_corrections(tmp_path, valid_brief):
    valid_brief.objective = "too short"
    gate = verify_brief(valid_brief)
    log = ChangeLog(str(tmp_path / "change_log.md"))
    log.log_gate_routed_back(valid_brief.task_id, gate)

    text = open(str(tmp_path / "change_log.md"), encoding="utf-8").read()
    assert "GATE ROUTE-BACK" in text
    assert "Corrections required" in text
    assert "objective" in text


def test_iteration_block_records_scores_and_evictions(tmp_path):
    record = IterationRecord(
        iteration=2,
        score=0.7123,
        previous_score=0.6000,
        delta=0.1123,
        reports=[
            AgentReport(agent_id="writer", role="w", iteration=2, status=AgentStatus.OK,
                        engine="offline-heuristic"),
            AgentReport(agent_id="tester", role="t", iteration=2,
                        findings=[
                            Finding("tester", Severity.BLOCKER, "code", "mechanism",
                                    "block does not compile", "fix the syntax", auto_fixable=False),
                            Finding("tester", Severity.MINOR, "links", "intro",
                                    "anchor unresolved", "drop the anchor", auto_fixable=True),
                        ]),
        ],
        evicted=["rolling_history"],
        tokens_total=1234,
        decision="continue",
        applied=1,
        deferred=1,
    )
    log = ChangeLog(str(tmp_path / "change_log.md"))
    log.log_iteration("demo", record, context_describe="cap=6000 evicted=rolling_history",
                      draft_words=900)

    text = open(str(tmp_path / "change_log.md"), encoding="utf-8").read()
    assert "iteration 2" in text
    assert "0.6000" in text and "0.7123" in text
    assert "evicted from context: rolling_history" in text
    assert "Applied by Synthesizer" in text
    assert "Deferred" in text

    entry = read_jsonl(str(tmp_path / "change_log.jsonl"))[-1]
    assert entry["event"] == "iteration"
    assert entry["iteration"] == 2
    assert entry["applied"] == 1 and entry["deferred"] == 1
    assert entry["evicted"] == ["rolling_history"]


def test_pipe_characters_in_findings_do_not_break_tables(tmp_path):
    record = IterationRecord(
        iteration=1, score=0.5, previous_score=None, delta=0.5,
        reports=[AgentReport(agent_id="tester", role="t", iteration=1,
                             findings=[Finding("tester", Severity.MINOR, "logic", "intro",
                                               "a | b pipe", "fix | it", auto_fixable=True)])],
        applied=1, deferred=0,
    )
    log = ChangeLog(str(tmp_path / "change_log.md"))
    log.log_iteration("demo", record)
    text = open(str(tmp_path / "change_log.md"), encoding="utf-8").read()
    assert r"a \| b pipe" in text
