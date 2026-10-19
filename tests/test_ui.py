"""Tests for the web dashboard.

Two layers, deliberately split:

* ``articlegen.ui.state`` / ``articlegen.ui.builder`` are Streamlit-free, so they are
  tested as ordinary units - including by replaying a real orchestrator run's events.
* ``app.py`` is driven through Streamlit's own ``AppTest`` harness, which executes the
  actual script and asserts on the widgets it produced. No browser, no network.
"""

import importlib.util
import json
import os

import pytest

from articlegen.events import EventBus, EventType, PipelineEvent
from articlegen.ui.builder import (
    SECTION_TEMPLATE,
    SectionParseError,
    build_brief_dict,
    format_issues,
    parse_sections,
    sections_to_text,
    validate,
)
from articlegen.ui.state import AGENT_ORDER, TrackerState

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Drive the packaged script, which is what `articlegen ui` actually launches.
from articlegen.ui import DASHBOARD_PATH as APP_PATH  # noqa: E402


# -- builder --------------------------------------------------------------

def test_parse_sections_reads_the_template():
    sections = parse_sections(SECTION_TEMPLATE)
    assert len(sections) == 4
    assert sections[0]["id"] == "introduction"
    assert sections[0]["word_target"] == 200
    assert sections[2]["title"] == "Worked Example"


def test_parse_sections_skips_blanks_and_comments():
    text = "# a comment\n\nintroduction | Intro | Frame the problem for the reader | 150\n\n"
    assert len(parse_sections(text)) == 1


@pytest.mark.parametrize("bad,expected", [
    ("only | three | fields", "expected 4 fields"),
    ("intro | Intro | purpose here | notanumber", "is not an integer"),
    (" | Intro | purpose here | 100", "section id is empty"),
    ("intro |  | purpose here | 100", "section title is empty"),
])
def test_parse_sections_reports_the_offending_line(bad, expected):
    with pytest.raises(SectionParseError) as excinfo:
        parse_sections("good | Good | A valid purpose for the section | 100\n" + bad)
    assert expected in str(excinfo.value)
    assert "line 2" in str(excinfo.value)


def test_sections_round_trip_through_text():
    sections = parse_sections(SECTION_TEMPLATE)
    assert parse_sections(sections_to_text(sections)) == sections


def test_build_brief_dict_coerces_constraints_and_ignores_bad_sections():
    data = build_brief_dict(
        task_id="demo", title="T", objective="O", audience="A",
        constraints_text="- include a worked example\n# comment\navoid invented numbers\n",
        sections_text="broken line with the wrong shape",
    )
    assert data["constraints"] == ["include a worked example", "avoid invented numbers"]
    assert data["sections"] == []
    assert "line 1" in data["_section_parse_error"]


def test_validate_preserves_the_bad_section_line_for_the_ui():
    data = build_brief_dict(
        task_id="good-brief", title="A Complete And Valid Article Title",
        objective="Explain how the pipeline converges and what the reader should decide next.",
        audience="engineers", target_words=600,
        sections_text=("introduction | Intro | Frame the problem for the reader | 150\n"
                       "broken line"),
    )
    brief, gate, errors = validate(data)

    assert brief is not None
    assert gate.routed_back is True
    assert errors == ["sections: line 2: expected 4 fields separated by '|', found 1"]


def test_validate_returns_gate_blockers_for_a_thin_brief():
    data = build_brief_dict(task_id="x", title="T", objective="short",
                            audience="", target_words=10, sections_text="")
    brief, gate, errors = validate(data)
    assert brief is not None
    assert gate.routed_back is True
    assert any("objective" in i.section for i in gate.blockers)


def test_validate_passes_a_complete_brief():
    data = build_brief_dict(
        task_id="good-brief",
        title="A Complete And Valid Article Title",
        objective="Explain how the pipeline converges and what the reader should decide next.",
        audience="engineers",
        target_words=600,
        sections_text=SECTION_TEMPLATE,
    )
    brief, gate, errors = validate(data)
    assert errors == []
    assert gate.ok is True, [i.claim for i in gate.issues]


def test_format_issues_escapes_pipes():
    data = build_brief_dict(task_id="x", title="T", objective="short",
                            audience="", target_words=10, sections_text="")
    _, gate, _ = validate(data)
    markdown = format_issues(gate)
    assert markdown.startswith("| severity |")
    assert "\n| " in markdown


# -- tracker state --------------------------------------------------------

def _event(event_type, **kwargs):
    """Build an event the way ``EventBus.emit`` does: known fields, rest in payload."""
    known = {"task_id", "iteration", "agent", "message"}
    fields = {key: value for key, value in kwargs.items() if key in known}
    payload = {key: value for key, value in kwargs.items() if key not in known}
    return PipelineEvent(type=event_type, payload=payload, **fields)


def test_tracker_starts_idle_at_zero_progress():
    tracker = TrackerState()
    assert tracker.status == "idle"
    assert tracker.progress == 0.0
    assert tracker.headline() == "Idle"
    assert tracker.is_finished is False


def test_tracker_marks_all_five_agents_pending_on_run_start():
    tracker = TrackerState()
    tracker.consume(_event(EventType.RUN_START, task_id="demo", max_iterations=3))
    assert tracker.status == "starting"
    assert [a.agent_id for a in tracker.agents()] == AGENT_ORDER
    assert all(a.state == "pending" for a in tracker.agents())


def test_tracker_lights_up_the_running_agent():
    tracker = TrackerState()
    tracker.consume(_event(EventType.RUN_START, max_iterations=2))
    tracker.consume(_event(EventType.GATE_PASS, message="ok"))
    tracker.consume(_event(EventType.ITERATION_START, iteration=1))
    tracker.consume(_event(EventType.AGENT_START, agent="tester", message="working"))

    states = {a.agent_id: a.state for a in tracker.agents()}
    assert states["tester"] == "running"
    assert states["writer"] == "pending"
    assert "2 · Tester" in tracker.headline()


def test_tracker_records_findings_and_tokens_from_agent_done():
    tracker = TrackerState()
    tracker.consume(_event(EventType.AGENT_DONE, agent="tester", iteration=1,
                           message="3 issues", role="Tester/Validator", status="ok",
                           findings=3, blockers=1, tokens_in=500, tokens_out=120,
                           context_tokens=1800, evicted=0, engine="offline-heuristic",
                           summary="3 issues"))
    view = next(a for a in tracker.agents() if a.agent_id == "tester")
    assert view.state == "done"
    assert view.findings == 3 and view.blockers == 1
    assert view.tokens_in == 500 and view.tokens_out == 120
    assert view.context_tokens == 1800
    assert view.icon == "🟢"


def test_tracker_flags_a_degraded_agent_visually():
    tracker = TrackerState()
    tracker.consume(_event(EventType.AGENT_DONE, agent="writer", status="degraded"))
    view = next(a for a in tracker.agents() if a.agent_id == "writer")
    assert view.state == "degraded"
    assert view.icon == "🟠"


def test_progress_advances_by_agent_within_an_iteration():
    tracker = TrackerState(max_iterations=2)
    tracker.consume(_event(EventType.RUN_START, max_iterations=2))
    tracker.consume(_event(EventType.GATE_PASS))
    tracker.consume(_event(EventType.ITERATION_START, iteration=1))
    assert tracker.progress == 0.0

    for index, agent_id in enumerate(AGENT_ORDER, start=1):
        tracker.consume(_event(EventType.AGENT_DONE, agent=agent_id, iteration=1))
        expected = index / (2 * len(AGENT_ORDER))
        assert tracker.progress == pytest.approx(expected)


def test_progress_resets_per_iteration_and_caps_below_one_until_finished():
    tracker = TrackerState(max_iterations=2)
    tracker.consume(_event(EventType.RUN_START, max_iterations=2))
    tracker.consume(_event(EventType.GATE_PASS))
    tracker.consume(_event(EventType.ITERATION_START, iteration=1))
    for agent_id in AGENT_ORDER:
        tracker.consume(_event(EventType.AGENT_DONE, agent=agent_id, iteration=1))
    tracker.consume(_event(EventType.ITERATION_START, iteration=2))
    tracker.consume(_event(EventType.AGENT_START, agent="writer", iteration=2))

    # Iteration 1's five agents are done; iteration 2 has started but completed
    # nothing yet, so exactly half the run is finished.
    assert tracker.progress < 1.0
    assert tracker.progress == pytest.approx(0.5)
    tracker.consume(_event(EventType.RUN_END, status="PASS", final_score=0.9,
                           tokens_total=100, stop_reason="target met"))
    assert tracker.progress == 1.0
    assert tracker.is_finished is True


def test_score_series_pins_iteration_zero_as_the_baseline():
    tracker = TrackerState()
    tracker.consume(_event(EventType.ITERATION_DONE, iteration=1, score=0.42,
                           components={"correctness": 0.5, "structure": 0.4}))
    tracker.consume(_event(EventType.ITERATION_DONE, iteration=2, score=0.88,
                           components={"correctness": 0.9, "structure": 0.85}))

    series = tracker.score_series()
    assert series[0] == {"Iteration": 0, "Score": 0.0}
    assert [point["Score"] for point in series] == [0.0, 0.42, 0.88]
    assert tracker.final_score == 0.88


def test_component_series_is_long_form_for_the_multi_line_chart():
    tracker = TrackerState()
    tracker.consume(_event(EventType.ITERATION_DONE, iteration=1, score=0.5,
                           components={"correctness": 0.6, "readability": 0.4}))
    rows = tracker.component_series()
    assert {row["Component"] for row in rows} == {"Correctness", "Readability"}
    assert all(row["Iteration"] == 1 for row in rows)


def test_route_back_is_terminal_and_carries_the_issues():
    tracker = TrackerState()
    tracker.consume(_event(EventType.RUN_START))
    tracker.consume(_event(EventType.GATE_ROUTE_BACK, message="3 blockers",
                           issues=[{"severity": "blocker", "claim": "objective too short"}]))
    tracker.consume(_event(EventType.RUN_END, status="ROUTE_BACK", final_score=0.0,
                           tokens_total=0, stop_reason="3 blockers"))

    assert tracker.status == "route_back"
    assert tracker.is_finished is True
    assert tracker.progress == 1.0
    assert tracker.route_back_issues[0]["claim"] == "objective too short"
    assert "Routed back" in tracker.headline()


def test_tracker_distinguishes_missing_provider_from_a_bad_brief():
    tracker = TrackerState()
    tracker.consume(_event(EventType.GATE_ROUTE_BACK, message="Dahl key required",
                           issues=[{"severity": "blocker", "category": "provider",
                                    "claim": "Dahl key required"}]))

    assert "No draft generated" in tracker.headline()
    assert "input gate" not in tracker.headline()


def test_tracker_resets_cleanly_between_runs():
    tracker = TrackerState()
    tracker.consume(_event(EventType.RUN_START))
    tracker.consume(_event(EventType.GATE_PASS))
    tracker.consume(_event(EventType.ITERATION_DONE, iteration=1, score=0.7))
    tracker.reset()

    assert tracker.status == "idle"
    assert tracker.iterations == []
    assert tracker.events == []
    assert tracker.final_score == 0.0


def test_tracker_is_driven_by_a_real_orchestrator_run(tmp_path, model_client):
    """Integration: replay the events a genuine run emits and check the tracker tracks."""
    from articlegen.models import Brief
    from articlegen.orchestrator import Orchestrator, OrchestratorConfig

    with open(os.path.join(REPO_ROOT, "examples", "brief_example.json"),
              encoding="utf-8") as handle:
        brief = Brief.from_dict(json.load(handle))

    tracker = TrackerState()
    events = EventBus()
    events.subscribe(tracker.consume)

    config = OrchestratorConfig.from_dict({
        "max_iterations": 2,
        "workspace": str(tmp_path / "runs"),
        "changelog_path": str(tmp_path / "change_log.md"),
        "provider": "dahl",
    })
    result = Orchestrator(config, client=model_client, events=events).run(brief)

    assert tracker.status == "done"
    assert tracker.result_status == result.status
    assert tracker.final_score == pytest.approx(result.final_score)
    assert len(tracker.iterations) == len(result.iterations)
    assert tracker.tokens_total == result.tokens_total
    assert tracker.gate_passed is True

    # Every agent must have reported, in every iteration.
    done = [e for e in tracker.events if e.type == EventType.AGENT_DONE]
    assert len(done) == len(result.iterations) * len(AGENT_ORDER)
    assert tracker.agent_rows()
    assert tracker.score_series()[0]["Score"] == 0.0


def test_event_bus_isolates_broken_callbacks(tmp_path):
    """A raising subscriber must not abort the run."""
    bus = EventBus()

    def explode(_event):
        raise RuntimeError("dashboard exploded")

    collected = []
    bus.subscribe(explode)
    bus.subscribe(collected.append)

    bus.emit(EventType.GATE_PASS, task_id="demo")
    assert len(collected) == 1
    assert "dashboard exploded" in bus.errors[0]


# -- the actual Streamlit app --------------------------------------------

def _fill_and_validate(test, example):
    """Fill the brief form and submit it in ONE batch.

    Streamlit form widgets only commit their values when the form is submitted, so
    setting values and running before the click silently discards them.
    """
    for widget in test.sidebar.text_input:
        if widget.label == "Workspace" and test.session_state.get("_ws"):
            widget.set_value(test.session_state["_ws"])
        elif widget.label == "Change log" and test.session_state.get("_cl"):
            widget.set_value(test.session_state["_cl"])
        elif widget.label == "Dahl API Key":
            widget.set_value("test-dahl-key")

    for widget in test.text_input:
        if widget.label == "Task ID":
            widget.set_value(example["task_id"])
        elif widget.label == "Title":
            widget.set_value(example["title"])
        elif widget.label == "Audience":
            widget.set_value(example["audience"])
    for widget in test.text_area:
        if "Objective" in widget.label:
            widget.set_value(example["objective"])
        elif "Sections" in widget.label:
            widget.set_value(sections_to_text(example["sections"]))

    [b for b in test.button if "Validate brief" in b.label][0].click()
    test.run()
    return test




# Scoped per test rather than module-wide: the CLI test below deliberately simulates
# streamlit being ABSENT, so a blanket importorskip would skip exactly the test that
# covers the no-dashboard environment.
_HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None
requires_streamlit = pytest.mark.skipif(
    not _HAS_STREAMLIT, reason="streamlit not installed (pip install -e '.[ui]')"
)

if _HAS_STREAMLIT:
    from streamlit.testing.v1 import AppTest
else:  # pragma: no cover - only reached when the extra is not installed
    AppTest = None


@pytest.fixture(scope="module")
def app_test():
    return AppTest.from_file(APP_PATH, default_timeout=120)


@requires_streamlit
def test_app_boots_with_all_five_tabs(app_test):
    app_test.run()
    assert not app_test.exception, app_test.exception
    labels = [tab.label for tab in app_test.tabs]
    # The Draft and Change Log tabs contain nested sub-tabs, so AppTest reports
    # more than five; assert the five top-level ones appear in order at the front.
    top_level = labels[:5]
    assert [label for label in top_level] == [
        "📋 Brief", "⚡ Execution", "📈 Scores", "📄 Draft", "📜 Change Log",
    ], labels


@requires_streamlit
def test_app_exposes_the_run_settings(app_test):
    app_test.run()
    assert not app_test.exception
    assert app_test.sidebar.slider[0].label == "Max iterations"
    assert app_test.sidebar.selectbox[0].label == "Provider"
    assert "Workspace" in app_test.sidebar.text_input[0].label


@requires_streamlit
def test_sidebar_exposes_dahl_key_and_model_fields():
    test = AppTest.from_file(APP_PATH, default_timeout=120)
    test.run()

    labels = [widget.label for widget in test.sidebar.text_input]
    assert "Dahl API Key" in labels
    assert "Dahl Model" in labels
    assert test.sidebar.selectbox[0].value == "dahl"
    assert any("Dahl API key is required" in error.value for error in test.error)


@requires_streamlit
def test_run_button_starts_disabled_until_a_brief_is_validated(app_test):
    app_test.run()
    run_buttons = [b for b in app_test.button if "Run orchestrated" in b.label]
    assert run_buttons, "run button not found"
    assert run_buttons[0].disabled is True


@requires_streamlit
def test_validating_the_bundled_example_passes_the_gate(app_test):
    app_test.run()
    assert not app_test.exception

    with open(os.path.join(REPO_ROOT, "examples", "brief_example.json"),
              encoding="utf-8") as handle:
        example = json.load(handle)

    _fill_and_validate(app_test, example)

    assert not app_test.exception, app_test.exception
    assert app_test.session_state["gate_result"].ok is True
    assert any("Gate passed" in success.value for success in app_test.success)


@requires_streamlit
def test_full_run_through_the_ui_produces_a_downloadable_draft(tmp_path, monkeypatch, model_client):
    """Drives app.py end to end with a fake model provider, never offline prose."""
    client_options = {}

    def fake_resolve_client(**kwargs):
        client_options.update(kwargs)
        return model_client

    monkeypatch.setattr("articlegen.llm.resolve_client", fake_resolve_client)
    test = AppTest.from_file(APP_PATH, default_timeout=300)
    test.run()
    assert not test.exception

    with open(os.path.join(REPO_ROOT, "examples", "brief_example.json"),
              encoding="utf-8") as handle:
        example = json.load(handle)

    # Point the run at a throwaway workspace/log before the first form interaction.
    for widget in test.sidebar.text_input:
        if widget.label == "Workspace":
            widget.set_value(str(tmp_path / "runs"))
        elif widget.label == "Change log":
            widget.set_value(str(tmp_path / "change_log.md"))
    test.run()
    assert test.session_state["workspace"] == str(tmp_path / "runs")

    _fill_and_validate(test, example)
    assert test.session_state["gate_result"].ok is True

    run_buttons = [b for b in test.button if "Run orchestrated" in b.label]
    assert run_buttons[0].disabled is False
    run_buttons[0].click()
    test.run()

    assert not test.exception, test.exception
    assert client_options["provider"] == "dahl"
    assert client_options["api_key"] == "test-dahl-key"
    assert client_options["model"]
    assert "test-dahl-key" not in (tmp_path / "change_log.md").read_text()
    tracker = test.session_state["tracker"]
    assert tracker.status == "done"
    assert tracker.iterations
    assert test.session_state["draft_markdown"].strip()

    # EventBus deliberately swallows subscriber exceptions so a broken redraw cannot
    # abort a pipeline run. That means a failing live-render callback would leave the
    # run and every other assertion looking fine while the tracker never updated, so
    # the swallowed-error list is asserted explicitly.
    bus = test.session_state["event_bus"]
    assert bus.errors == [], f"live render callback raised: {bus.errors[:3]}"
    # And it must have been invoked for every event the orchestrator emitted.
    assert len(bus.events) == len(tracker.events) > 0
    assert {e.type.value for e in bus.events} >= {
        "run_start", "gate_pass", "agent_start", "agent_done",
        "iteration_done", "run_end",
    }

    # AppTest exposes a download button's label/mime/enabled state but not its
    # payload (that is served from a URL), so the payload is asserted on the
    # session state it is read from, and the button wiring is asserted separately.
    downloads = [d for d in test.download_button if "Download Article" in d.label]
    assert downloads, "no Download Article button"
    assert downloads[0].disabled is False

    markdown = test.session_state["draft_markdown"]
    assert markdown.startswith("# " + example["title"])
    for section in example["sections"]:
        assert "## " + section["title"] in markdown

    bundles = [d for d in test.download_button if "run bundle" in d.label]
    assert bundles and bundles[0].disabled is False

    # The Change Log tab must render the parsed JSONL mirror, one row per entry.
    # This is the table whose mixed-dtype columns originally broke Arrow, so assert
    # the row count matches the file rather than merely that no exception was raised.
    from articlegen.changelog import read_jsonl

    entries = read_jsonl(str(tmp_path / "change_log.jsonl"))
    assert entries, "the run produced no change_log.jsonl"

    log_tables = [d for d in test.dataframe
                  if "timestamp" in getattr(d.value, "columns", [])]
    assert log_tables, "Change Log tab rendered no parsed-events table"
    assert len(log_tables[0].value) == len(entries)
    # Iteration events must carry their score; lifecycle events must be None, not "".
    iteration_rows = log_tables[0].value[log_tables[0].value["event"] == "iteration"]
    assert len(iteration_rows) == len(tracker.iterations)
    assert iteration_rows["score"].notna().all()


def test_cli_rejects_ui_command_without_streamlit(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "streamlit":
            raise ImportError("no streamlit")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from articlegen.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["ui"])
    assert args.func(args) == 2


@requires_streamlit
def test_export_payload_bundles_draft_scores_and_stop_reason():
    from articlegen.ui.render import export_payload

    tracker = TrackerState()
    tracker.consume(_event(EventType.RUN_START, task_id="demo", max_iterations=2))
    tracker.consume(_event(EventType.GATE_PASS))
    tracker.consume(_event(EventType.ITERATION_DONE, iteration=1, score=0.91,
                           components={"correctness": 0.9, "structure": 0.88},
                           findings=4, applied=3, deferred=1, tokens_total=1200))
    tracker.consume(_event(EventType.RUN_END, status="PASS", final_score=0.91,
                           tokens_total=1200, stop_reason="score target met"))

    bundle = export_payload("# Title\n\nBody", tracker, {"workspace": "runs/demo"})

    assert bundle["task_id"] == "demo"
    assert bundle["status"] == "PASS"
    assert bundle["stop_reason"] == "score target met"
    assert bundle["final_score"] == 0.91
    assert bundle["tokens_total"] == 1200
    assert bundle["markdown"] == "# Title\n\nBody"
    assert bundle["iterations"][0]["Score"] == 0.91
    assert bundle["meta"]["workspace"] == "runs/demo"
    # Must be JSON-serialisable: the UI passes it straight to json.dumps.
    assert json.loads(json.dumps(bundle))["final_score"] == 0.91


@requires_streamlit
def test_cmd_ui_execs_rather_than_forking_an_orphanable_child(monkeypatch):
    """`articlegen ui` must replace its own process.

    A launcher that forks leaves the real server reparented to PID 1 when the
    wrapper is killed, so it survives the job and keeps holding its port. exec
    also lets SIGTERM reach Streamlit instead of an idle wrapper.
    """
    import os

    from articlegen.cli import build_parser

    captured = {}

    def fake_execvp(program, argv):
        captured["program"] = program
        captured["argv"] = list(argv)
        raise AssertionError("execvp must not return in this test")

    monkeypatch.setattr(os, "execvp", fake_execvp)

    parser = build_parser()
    args = parser.parse_args(["ui", "--port", "8601", "--address", "127.0.0.1", "--no-cors"])

    with pytest.raises(AssertionError, match="must not return"):
        args.func(args)

    argv = captured["argv"]
    # The command is either [streamlit, run, app, ...] or
    # [python, -m, streamlit, run, app, ...], so anchor on the "run" token.
    assert "run" in argv, argv
    run_at = argv.index("run")
    assert run_at > 0, "nothing to exec before 'run'"

    assert "--server.port" in argv
    assert argv[argv.index("--server.port") + 1] == "8601"
    assert "--server.address" in argv
    assert argv[argv.index("--server.address") + 1] == "127.0.0.1"
    # --no-cors must disable both checks, or a proxied preview is blocked.
    assert "--server.enableCORS" in argv
    assert argv[argv.index("--server.enableCORS") + 1] == "false"
    assert "--server.enableXsrfProtection" in argv
    assert argv[argv.index("--server.enableXsrfProtection") + 1] == "false"


@requires_streamlit
def test_cmd_ui_launches_the_packaged_dashboard(monkeypatch):
    """The resolved script must be the packaged dashboard, not a repo-root app.py.

    Resolving from the package is what lets `articlegen ui` work after a plain
    `pip install`, where no repo checkout exists.
    """
    import os

    from articlegen.cli import build_parser
    from articlegen.ui import DASHBOARD_PATH

    captured = {}

    def fake_execvp(program, argv):
        captured["argv"] = list(argv)

    monkeypatch.setattr(os, "execvp", fake_execvp)

    args = build_parser().parse_args(["ui"])
    assert args.func(args) in (None, 0, 127)

    argv = captured["argv"]
    app_arg = argv[argv.index("run") + 1]
    assert app_arg == DASHBOARD_PATH
    assert os.path.exists(app_arg), f"resolved app path does not exist: {app_arg}"
    # It must live inside the package, or it will not ship with an install.
    assert os.sep + os.path.join("articlegen", "ui") + os.sep in app_arg
    # Defaults must bind every interface so reverse-proxied previews can reach it.
    assert argv[argv.index("--server.address") + 1] == "0.0.0.0"


def test_dashboard_script_ships_inside_the_package():
    """Guard against the script being moved back out of package data."""
    import articlegen.ui as ui

    assert os.path.exists(ui.DASHBOARD_PATH)
    assert os.path.dirname(ui.DASHBOARD_PATH) == os.path.dirname(ui.__file__)


def test_changelog_rows_keep_every_column_a_single_dtype():
    """The exact shape that broke Arrow: events with and without numeric fields.

    RUN_START/RUN_END carry no score or iteration, so a naive build produces columns
    mixing floats with "" and pyarrow refuses the table.
    """
    from articlegen.ui.builder import changelog_rows

    entries = [
        {"timestamp": "2026-10-11T16:43:51Z", "event": "RUN_START", "task_id": "demo",
         "title": "A Title", "mode": "article"},
        {"timestamp": "2026-10-11T16:43:52Z", "event": "iteration", "task_id": "demo",
         "iteration": 1, "score": 0.7241, "applied": 4, "deferred": 5,
         "tokens_total": 11709},
        {"timestamp": "2026-10-11T16:43:53Z", "event": "iteration", "task_id": "demo",
         "iteration": 2, "score": 0.9169, "applied": 1, "deferred": 5,
         "tokens_total": 15341},
        {"timestamp": "2026-10-11T16:43:54Z", "event": "RUN_END", "task_id": "demo",
         "status": "PASS", "stop_reason": "score target met"},
    ]
    rows = changelog_rows(entries)

    assert len(rows) == 4
    # Non-iteration events must be None, not "" - None keeps the column nullable
    # numeric instead of degrading it to object.
    assert rows[0]["score"] is None
    assert rows[0]["iteration"] is None
    assert rows[3]["tokens"] is None
    assert rows[1]["score"] == 0.7241
    assert rows[1]["iteration"] == 1
    assert rows[2]["tokens"] == 15341

    # Every column must hold at most one non-None type.
    for column in rows[0]:
        types = {type(r[column]) for r in rows if r[column] is not None}
        assert len(types) <= 1, f"column {column!r} mixes types: {types}"


def test_changelog_rows_survive_malformed_entries():
    from articlegen.ui.builder import changelog_rows

    rows = changelog_rows([
        {"event": "iteration", "score": "not-a-number", "iteration": "x"},
        "a bare string, not a dict",
        {},
    ])
    assert len(rows) == 2          # the non-dict entry is skipped
    assert rows[0]["score"] is None
    assert rows[0]["iteration"] is None
    assert rows[0]["event"] == "iteration"


def test_changelog_rows_are_arrow_serialisable():
    """Round-trip through pyarrow, which is what st.dataframe actually does."""
    pandas = pytest.importorskip("pandas")
    pyarrow = pytest.importorskip("pyarrow")

    from articlegen.changelog import read_jsonl
    from articlegen.ui.builder import changelog_rows

    entries = read_jsonl(os.path.join(REPO_ROOT, "change_log.jsonl"))
    assert entries, "repo change_log.jsonl is empty; nothing to verify"

    frame = pandas.DataFrame(changelog_rows(entries))
    table = pyarrow.Table.from_pandas(frame, preserve_index=False)
    assert table.num_rows == len(entries)
    # No object-dtype columns: that is the failure mode the coercion prevents.
    for field in table.schema:
        assert "object" not in str(field.type).lower(), field


@requires_streamlit
def test_run_stays_disabled_when_the_gate_routes_the_brief_back():
    """Stage 1 must actually block Stage 2.

    Regression: `brief_data` was stored on every validate, and the Run button only
    checked that key existed, so a brief with blockers still enabled the button and
    let a user spend a pipeline run on input the gate had already rejected.
    """
    test = AppTest.from_file(APP_PATH, default_timeout=180)
    test.run()

    for widget in test.text_input:
        if widget.label == "Task ID":
            widget.set_value("bad-brief")
        elif widget.label == "Title":
            widget.set_value("T")
        elif widget.label == "Audience":
            widget.set_value("")
    for widget in test.text_area:
        if "Objective" in widget.label:
            widget.set_value("short")
        elif "Sections" in widget.label:
            widget.set_value("this line is not valid section syntax")

    run_before = [b for b in test.button if "Run orchestrated" in b.label][0]
    assert run_before.disabled is True, "Run must start disabled before any validation"

    [b for b in test.button if "Validate brief" in b.label][0].click()
    test.run()

    gate = test.session_state["gate_result"]
    assert gate.routed_back is True
    assert gate.blockers, "expected the gate to raise blockers"

    run_after = [b for b in test.button if "Run orchestrated" in b.label][0]
    assert run_after.disabled is True, "Run must stay disabled for a routed-back brief"
    assert any("routed this brief back" in e.value for e in test.error)


@requires_streamlit
def test_run_becomes_enabled_once_the_gate_passes():
    """Guard against over-correcting the routed-back fix into blocking valid briefs."""
    test = AppTest.from_file(APP_PATH, default_timeout=180)
    test.run()

    with open(os.path.join(REPO_ROOT, "examples", "brief_example.json"),
              encoding="utf-8") as handle:
        example = json.load(handle)

    _fill_and_validate(test, example)

    assert test.session_state["gate_result"].ok is True
    run = [b for b in test.button if "Run orchestrated" in b.label][0]
    assert run.disabled is False


@requires_streamlit
def _count_chart_specs(test):
    """Count Vega-Lite chart specs in the element tree.

    AppTest exposes no accessor for st.line_chart (charts land in UnknownElement),
    so the proto has to be inspected directly. A naive `test.get("line_chart")`
    returns empty even when the charts render fine.
    """
    found = []

    def walk(nodes):
        for node in nodes:
            if type(node).__name__ == "UnknownElement":
                proto = getattr(node, "proto", None)
                if proto is not None:
                    for field, _value in proto.ListFields():
                        if field.name == "spec":
                            found.append(proto)
                        break
            children = getattr(node, "children", None)
            if children:
                walk(list(children.values()) if isinstance(children, dict)
                     else list(children))

    walk(list(test.main.children.values()))
    return found


@requires_streamlit
def test_score_and_component_charts_actually_render(tmp_path, monkeypatch, model_client):
    """The Scores tab must emit both charts after a fake-model run."""
    monkeypatch.setattr("articlegen.llm.resolve_client", lambda **_kwargs: model_client)
    test = AppTest.from_file(APP_PATH, default_timeout=300)
    test.run()
    for widget in test.sidebar.slider:
        if widget.label == "Max iterations":
            widget.set_value(2)
        elif widget.label == "Target score":
            widget.set_value(0.99)
    test.run()

    with open(os.path.join(REPO_ROOT, "examples", "brief_example.json"),
              encoding="utf-8") as handle:
        example = json.load(handle)

    for widget in test.sidebar.text_input:
        if widget.label == "Workspace":
            widget.set_value(str(tmp_path / "runs"))
        elif widget.label == "Change log":
            widget.set_value(str(tmp_path / "change_log.md"))
    test.run()

    _fill_and_validate(test, example)
    [b for b in test.button if "Run orchestrated" in b.label][0].click()
    test.run()

    assert not test.exception, test.exception
    assert len(test.session_state["tracker"].iterations) >= 2, \
        "need 2+ iterations for the progression chart to draw a line"

    specs = _count_chart_specs(test)
    # One score-progression chart plus one component-breakdown chart.
    assert len(specs) == 2, f"expected 2 charts, found {len(specs)}"
    for spec in specs:
        assert spec.spec, "chart spec is empty"


@requires_streamlit
def test_editing_the_form_after_validating_blocks_the_run(tmp_path):
    """The pipeline runs the validated snapshot, so silent drift must be surfaced.

    Run consumes st.session_state['brief_data'], never the live widgets - which is
    correct, because the gate verdict must describe what actually executes. But
    without a drift check the user edits the form, clicks Run, and gets a draft of
    the *previous* brief with nothing indicating that happened.
    """
    test = AppTest.from_file(APP_PATH, default_timeout=300)
    test.run()

    with open(os.path.join(REPO_ROOT, "examples", "brief_example.json"),
              encoding="utf-8") as handle:
        example = json.load(handle)

    for widget in test.sidebar.text_input:
        if widget.label == "Workspace":
            widget.set_value(str(tmp_path / "runs"))
        elif widget.label == "Change log":
            widget.set_value(str(tmp_path / "change_log.md"))
    test.run()

    _fill_and_validate(test, example)
    assert [b for b in test.button if "Run orchestrated" in b.label][0].disabled is False

    # Edit the form, then click Run in the same batch.
    for widget in test.text_input:
        if widget.label == "Title":
            widget.set_value("A Completely Different Title")
    [b for b in test.button if "Run orchestrated" in b.label][0].click()
    test.run()

    run = [b for b in test.button if "Run orchestrated" in b.label][0]
    assert run.disabled is True, "drifted form must not be runnable"
    assert any("changed since you validated" in w.value for w in test.warning)
    assert test.session_state["tracker"].status == "idle", "the run must not have executed"
    assert not test.session_state["draft_markdown"]


@requires_streamlit
def test_unchanged_form_stays_runnable_after_validation():
    """Guard against drift detection firing when nothing actually changed."""
    test = AppTest.from_file(APP_PATH, default_timeout=180)
    test.run()

    with open(os.path.join(REPO_ROOT, "examples", "brief_example.json"),
              encoding="utf-8") as handle:
        example = json.load(handle)

    _fill_and_validate(test, example)

    assert [b for b in test.button if "Run orchestrated" in b.label][0].disabled is False
    assert not any("changed since you validated" in w.value for w in test.warning)


@requires_streamlit
def test_empty_generation_shows_failure_instead_of_article_download(tmp_path, monkeypatch, model_client,
                                                                  example_brief_dict):
    from articlegen.llm import LLMResponse

    original_complete = model_client.complete

    def empty_synthesis(system, user, *, max_output_tokens=1000):
        response = original_complete(system, user, max_output_tokens=max_output_tokens)
        if "# Agent 5 — Synthesizer" not in system:
            return response
        payload = json.loads(response.text)
        for section in payload["sections"]:
            section["body"] = ""
        return LLMResponse(text=json.dumps(payload), tokens_in=10, tokens_out=10,
                           engine="test-llm:empty")

    monkeypatch.setattr(model_client, "complete", empty_synthesis)
    monkeypatch.setattr("articlegen.llm.resolve_client", lambda **kwargs: model_client)
    test = AppTest.from_file(APP_PATH, default_timeout=120)
    test.run()
    for widget in test.sidebar.text_input:
        if widget.label == "Workspace":
            widget.set_value(str(tmp_path / "runs"))
        elif widget.label == "Change log":
            widget.set_value(str(tmp_path / "change_log.md"))
    test.sidebar.slider[0].set_value(1)
    test.run()
    _fill_and_validate(test, example_brief_dict)
    [b for b in test.button if "Run orchestrated" in b.label][0].click()
    test.run()

    assert not test.exception, test.exception
    assert test.session_state["run_meta"]["status"] == "NEEDS_HUMAN"
    assert test.session_state["draft_markdown"] == ""
    assert any("No article was produced" in error.value for error in test.error)
    assert not [d for d in test.download_button if "Download Article" in d.label]
