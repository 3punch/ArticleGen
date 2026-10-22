"""ArticleGen Web Dashboard.

Launch with either:

    streamlit run app.py
    python -m articlegen ui

The app script stays thin: brief parsing lives in ``articlegen.ui.builder``, tracker
state in ``articlegen.ui.state``, and rendering in ``articlegen.ui.render``. Those are
covered by ``tests/test_ui.py`` without needing a browser.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import streamlit as st

from articlegen.changelog import read_jsonl
from articlegen.events import EventBus
from articlegen.llm import DahlClient, resolve_client, resolve_resilient_client
from articlegen.orchestrator import Orchestrator, OrchestratorConfig
from articlegen.cli import TEMPLATE_BRIEF
from articlegen.ui import builder, render
from articlegen.ui.state import TrackerState

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Run artefacts default to the caller's working directory, not the install
# location: a pip-installed package may be read-only (site-packages), and users
# expect `runs/` and `change_log.md` to land where they launched the app.
DEFAULT_WORKSPACE = os.path.join(os.getcwd(), "runs")
DEFAULT_CHANGELOG = os.path.join(os.getcwd(), "change_log.md")

# examples/ only exists in a source checkout, so the bundled-example button falls
# back to the in-package template. cli.TEMPLATE_BRIEF is the single source of truth
# for that fallback, which keeps an installed dashboard working.
EXAMPLE_BRIEF = os.path.join(REPO_ROOT, "examples", "brief_example.json")

st.set_page_config(page_title="ArticleGen", page_icon="📝", layout="wide")


# -- session state --------------------------------------------------------

def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "tracker": TrackerState(),
        "draft_markdown": "",
        "run_meta": {},
        "score_target": 0.85,
        "changelog_path": DEFAULT_CHANGELOG,
        "workspace": DEFAULT_WORKSPACE,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()


def _load_example() -> Dict[str, Any]:
    """Return the bundled example brief.

    ``examples/`` is only present in a source checkout, so an installed dashboard
    falls back to ``cli.TEMPLATE_BRIEF`` rather than raising FileNotFoundError on a
    button click.
    """
    if os.path.exists(EXAMPLE_BRIEF):
        with open(EXAMPLE_BRIEF, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(json.dumps(TEMPLATE_BRIEF))


# -- sidebar --------------------------------------------------------------

with st.sidebar:
    st.title("📝 ArticleGen")
    st.caption("Iterative multi-agent article pipeline")

    st.subheader("Run settings")
    iterations = st.slider("Max iterations", 1, 8, 4)
    score_target = st.slider("Target score", 0.50, 0.99, 0.85, 0.01)
    provider = st.selectbox(
        "Provider", ["dahl", "auto", "anthropic", "openai"],
        help="Article generation requires a Dahl, Anthropic or OpenAI API key. Offline templating is disabled.",
    )
    st.session_state.score_target = score_target

    st.subheader("Research mode")
    st.caption("Research mode adds a validated research plan, verified sources, code "
               "and experiment execution, a peer-review revision loop and a "
               "three-outcome quality gate. 'General' keeps the classic workflow.")
    article_type = st.selectbox(
        "Article type",
        ["general", "educational", "conceptual", "empirical", "computational",
         "theoretical", "methods", "systematic_review", "narrative_review"],
        help="Sets the required sections and which checks apply. Research types "
             "activate the evidence pipeline.",
    )
    citation_style = st.selectbox("Citation style", ["apa", "ieee", "vancouver", "plain"])
    review_iterations = st.slider("Review/revision iterations", 0, 3, 1)
    execute_experiments = st.checkbox(
        "Execute planned experiments", value=True,
        help="Runs the plan's code in an isolated subprocess and records results. "
             "Without execution the run cannot be reported as a completed study.",
    )
    source_retrieval = st.selectbox(
        "Source retrieval", ["auto", "crossref", "user", "none"],
        help="'auto' queries Crossref. 'user' uses only sources supplied in the brief. "
             "'none' records that no bibliography can be verified.",
    )
    st.session_state["research_settings"] = {
        "article_type": article_type,
        "citation_style": citation_style,
        "review_iterations": int(review_iterations),
        "execute_experiments": bool(execute_experiments),
        "source_retrieval": source_retrieval,
    }

    st.subheader("Paths")
    workspace = st.text_input("Workspace", st.session_state.workspace)
    changelog_path = st.text_input("Change log", st.session_state.changelog_path)
    st.session_state.workspace = workspace
    st.session_state.changelog_path = changelog_path

    st.subheader("LLM credentials — Dahl")
    st.caption("Enter your Dahl key and model here (or set DAHL_API_KEY env var). Stored only in session memory.")
    dahl_api_key = st.text_input(
        "Dahl API Key",
        value=st.session_state.get("dahl_api_key", os.environ.get("DAHL_API_KEY", "")),
        type="password",
        help="Bearer token for https://inference.dahl.global/v1 — leave empty to use env var DAHL_API_KEY",
    )
    dahl_model = st.text_input(
        "Dahl Model",
        value=st.session_state.get("dahl_model", os.environ.get("DAHL_MODEL", DahlClient.DEFAULT_MODEL)),
        help="Exact model ID from Dahl /v1/models, e.g. zai-org/GLM-5.3-Flash",
    )
    dahl_base_url = st.text_input(
        "Dahl Base URL",
        value=st.session_state.get("dahl_base_url", os.environ.get("DAHL_BASE_URL", DahlClient.DEFAULT_BASE_URL)),
        help="Override only if using a custom Dahl endpoint",
    )
    # Persist in session so the run block can read them without re-entering
    st.session_state["dahl_api_key"] = dahl_api_key
    st.session_state["dahl_model"] = dahl_model
    st.session_state["dahl_base_url"] = dahl_base_url

    # Validation feedback before run
    _effective_dahl_key = (dahl_api_key or os.environ.get("DAHL_API_KEY", "")).strip()
    _effective_anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    _effective_openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    _has_any_key = bool(_effective_dahl_key or _effective_anthropic_key or _effective_openai_key)
    if provider == "dahl" and not _effective_dahl_key:
        st.error("Dahl API key is required. Enter it above or set DAHL_API_KEY in the environment.")
    elif provider == "anthropic" and not _effective_anthropic_key:
        st.error("Anthropic API key is required for this provider (set ANTHROPIC_API_KEY).")
    elif provider == "openai" and not _effective_openai_key:
        st.error("OpenAI API key is required for this provider (set OPENAI_API_KEY).")
    elif provider == "auto" and not _has_any_key:
        st.error("No LLM key found — enter a Dahl key above or set DAHL_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY, then re-run.")

    st.divider()
    st.caption(
        "The orchestrator runs in-process. Progress updates stream into the "
        "Execution view while it works."
    )

# -- tabs -----------------------------------------------------------------

# -- One-prompt autonomous mode -----------------------------------------
# The whole point of the rewrite: one sentence in, a full article out. This builds the
# brief, makes the client self-healing (retry + model fallback on rate limits) and runs
# the pipeline to a ready quality gate, storing results in the same session keys the
# tabs below read from.
with st.expander("⚡ One-prompt mode — describe the article in a single sentence",
                 expanded=False):
    st.caption(
        "Type one prompt. The pipeline builds the brief, plans, retrieves and verifies "
        "sources, runs experiments, writes, reviews and revises until the article is "
        "complete. No manual brief editing required."
    )
    one_prompt = st.text_area(
        "Your prompt", key="one_prompt", height=110,
        placeholder=("e.g. Write a computational article explaining how to verify the "
                     "Erdős–Gyárfás conjecture on a sample of cubic graphs, with a "
                     "runnable Python experiment."),
    )
    one_rounds = st.number_input("Max autonomous rounds", 1, 8, 4, key="one_rounds")
    _one_can_run = bool(
        (st.session_state.get("dahl_api_key", "") or os.environ.get("DAHL_API_KEY", "")).strip()
        or os.environ.get("ANTHROPIC_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    one_run = st.button("🚀 Generate full article", type="primary", key="one_run",
                        disabled=not _one_can_run)
    one_container = st.container()
    if one_run and one_prompt.strip():
        tracker = TrackerState(max_iterations=iterations)
        st.session_state["tracker"] = tracker
        events = EventBus()
        events.subscribe(tracker.consume)
        events.subscribe(
            lambda e: render.render_tracker(one_container, tracker, score_target)
        )
        llm_client = resolve_resilient_client(
            provider=provider,
            model=st.session_state.get("dahl_model", "") if provider == "dahl" else None,
            api_key=st.session_state.get("dahl_api_key", "") if provider == "dahl" else None,
            base_url=st.session_state.get("dahl_base_url", "") if provider == "dahl" else None,
            dahl_api_key=st.session_state.get("dahl_api_key", ""),
            dahl_model=st.session_state.get("dahl_model", ""),
            dahl_base_url=st.session_state.get("dahl_base_url", ""),
        )
        config = OrchestratorConfig.from_dict({
            "max_iterations": iterations,
            "score_target": score_target,
            "provider": provider,
            "workspace": workspace,
            "changelog_path": changelog_path,
            "research": True,
            "auto": True,
            "max_autonomous_rounds": int(one_rounds),
            "source_retrieval": st.session_state["research_settings"]["source_retrieval"],
            "review_iterations": st.session_state["research_settings"]["review_iterations"],
            "execute_experiments":
                st.session_state["research_settings"]["execute_experiments"],
        })
        with st.spinner("Autonomously generating the full article…"):
            orchestrator = Orchestrator(config, client=llm_client, events=events)
            result = orchestrator.run_autonomous(one_prompt)
        st.session_state["draft_markdown"] = (
            result.final_draft.to_markdown()
            if result.final_draft and any(s.body.strip() for s in result.final_draft.sections)
            else ""
        )
        research = result.research or {}
        gate = research.get("quality_gate") or {}
        st.session_state["run_meta"] = {
            "status": result.status,
            "stop_reason": result.stop_reason,
            "final_score": round(result.final_score, 4),
            "iterations": len(result.iterations),
            "tokens_total": result.tokens_total,
            "workspace": result.workspace,
            "quality_status": result.quality_status,
            "quality_label": gate.get("label", ""),
            "research_report": os.path.join(result.workspace, "research", "report.md")
            if result.workspace else "",
        }
        st.session_state["research_report_markdown"] = research.get("markdown", "")
        render.render_tracker(one_container, tracker, score_target)
        if result.quality_status == "ready_for_final_review":
            st.success("Autonomous run finished — article reached 'Ready for final review'. "
                       "See the Execution, Research, Scores and Draft tabs.")
        else:
            st.warning(
                f"Autonomous run finished — {result.status}: {result.stop_reason}. "
                "The article is produced as a proposal/draft; see the tabs for detail."
            )

tab_brief, tab_exec, tab_research, tab_scores, tab_draft, tab_log = st.tabs(
    ["📋 Brief", "⚡ Execution", "🔬 Research", "📈 Scores", "📄 Draft", "📜 Change Log"]
)

# -- Tab 1: brief + run ---------------------------------------------------

with tab_brief:
    st.header("Task brief")

    col_left, col_right = st.columns([1, 1])
    with col_right:
        st.markdown("**Load a brief**")
        uploaded = st.file_uploader("Upload brief JSON", type=["json"], key="upload")
        if uploaded is not None:
            try:
                loaded = json.loads(uploaded.getvalue().decode("utf-8"))
                if not isinstance(loaded, dict):
                    st.error("Invalid brief: the top-level JSON value must be an object.")
                else:
                    from articlegen.models import Brief

                    Brief.from_dict(loaded)  # validate types before using them as widget defaults
                    st.session_state["loaded_brief"] = loaded
                    st.success(f"Loaded brief '{loaded.get('task_id', '?')}'.")
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                st.error(f"Invalid JSON: {exc}")
            except (TypeError, ValueError) as exc:
                st.error(f"Invalid brief structure: {exc}")
        if st.button("Use bundled example", width="stretch"):
            st.session_state["loaded_brief"] = _load_example()
            st.success("Loaded examples/brief_example.json")

    loaded = st.session_state.get("loaded_brief") or {}

    with st.form("brief_form", clear_on_submit=False):
        with col_left:
            task_id = st.text_input("Task ID", loaded.get("task_id", "my-article"),
                                    help="lowercase, digits, '-' or '_'")
            title = st.text_input("Title", loaded.get("title", ""))
            audience = st.text_input("Audience", loaded.get("audience", ""))
            mode = st.selectbox("Mode", ["article", "code"],
                                index=0 if loaded.get("mode", "article") == "article" else 1)
            tone = st.text_input("Tone", loaded.get("tone", "neutral, precise, professional"))
        objective = st.text_area(
            "Objective / requirements",
            loaded.get("objective", ""),
            height=100,
            help="Minimum 40 characters. State the outcome the reader gets.",
        )
        mcol1, mcol2, mcol3 = st.columns(3)
        target_words = mcol1.number_input("Target words", 200, 12000,
                                          int(loaded.get("target_words", 1000) or 1000), 50)
        requires_citations = mcol2.checkbox("Require citations",
                                            bool(loaded.get("requires_citations", False)))
        mcol3.write("")
        constraints_text = st.text_area(
            "Constraints (one per line)",
            "\n".join(loaded.get("constraints") or []),
            height=80,
        )
        research_cols = st.columns(2)
        research_question = research_cols[0].text_input(
            "Research question (optional)",
            loaded.get("research_question", ""),
            help="A precise, answerable question. Required in practice for research "
                 "article types; the planner will otherwise report the gap.",
        )
        venue = research_cols[1].text_input("Target venue (optional)",
                                           loaded.get("venue", ""))
        contribution_type = st.selectbox(
            "Contribution type",
            ["", "novel_theory", "novel_empirical", "novel_method", "novel_algorithm",
             "synthesis", "replication", "routine_implementation", "educational",
             "conceptual"],
            index=0,
            help="Novelty is never assumed: a routine implementation will be reframed "
                 "as a replication unless the literature search supports the claim.",
        )
        requirements_cols = st.columns(2)
        requires_experiments = requirements_cols[0].checkbox(
            "This article requires experiments",
            bool(loaded.get("requires_experiments", False)),
        )
        novelty_claim = requirements_cols[1].checkbox(
            "Explicit novelty claim requested",
            bool(loaded.get("novelty_claim", False)),
        )
        sections_text = st.text_area(
            "Sections  —  `id | Title | purpose | word_target`",
            loaded.get("_sections_text")
            or builder.sections_to_text(loaded.get("sections") or [])
            or builder.SECTION_TEMPLATE,
            height=180,
        )
        submitted = st.form_submit_button("✅ Validate brief", type="primary",
                                          width="stretch")

    if submitted:
        data = builder.build_brief_dict(
            task_id=task_id, title=title, objective=objective, audience=audience,
            mode=mode, tone=tone, target_words=int(target_words),
            sections_text=sections_text, constraints_text=constraints_text,
            requires_citations=requires_citations,
            article_type=st.session_state["research_settings"]["article_type"],
            citation_style=st.session_state["research_settings"]["citation_style"],
            research_question=research_question,
            contribution_type=contribution_type,
            venue=venue,
            requires_experiments=requires_experiments,
            execute_experiments=st.session_state["research_settings"]["execute_experiments"],
            review_iterations=st.session_state["research_settings"]["review_iterations"],
            source_retrieval=st.session_state["research_settings"]["source_retrieval"],
            novelty_claim=novelty_claim,
        )
        st.session_state["brief_data"] = data
        brief, gate, parse_errors = builder.validate(data)
        st.session_state["gate_result"] = gate
        for error in parse_errors:
            st.error(error)
        if gate.routed_back:
            st.error("⛔ The decision gate routed this brief back. Fix the blockers below.")
        elif gate.issues:
            st.warning("The gate passes, but these non-blocking issues will be handed "
                       "to the agents as work items.")
        else:
            st.success("✅ Gate passed — no issues.")
        st.markdown(builder.format_issues(gate))

    st.divider()

    # The gate is the whole point of Stage 1, so a brief it routed back must not be
    # runnable. Checking only for brief_data would let a user spend a pipeline run on
    # input the gate already rejected.
    validated_gate = st.session_state.get("gate_result")
    validated_brief = st.session_state.get("brief_data")

    # Running uses the validated snapshot, never the live widgets, so that the gate
    # verdict always describes what actually executes. The consequence is that edits
    # made after validating are silently ignored - so surface the drift instead of
    # letting the user believe they are running what is on screen.
    current_brief = builder.build_brief_dict(
        task_id=task_id, title=title, objective=objective, audience=audience,
        mode=mode, tone=tone, target_words=int(target_words),
        sections_text=sections_text, constraints_text=constraints_text,
        requires_citations=requires_citations,
        article_type=st.session_state["research_settings"]["article_type"],
        citation_style=st.session_state["research_settings"]["citation_style"],
        research_question=research_question,
        contribution_type=contribution_type,
        venue=venue,
        requires_experiments=requires_experiments,
        execute_experiments=st.session_state["research_settings"]["execute_experiments"],
        review_iterations=st.session_state["research_settings"]["review_iterations"],
        source_retrieval=st.session_state["research_settings"]["source_retrieval"],
        novelty_claim=novelty_claim,
    )
    form_drifted = validated_brief is not None and current_brief != validated_brief

    _can_run_llm = True
    _effective_dahl_key = (st.session_state.get("dahl_api_key", "") or os.environ.get("DAHL_API_KEY", "")).strip()
    _effective_anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    _effective_openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if provider == "dahl":
        _can_run_llm = bool(_effective_dahl_key)
    elif provider == "anthropic":
        _can_run_llm = bool(_effective_anthropic_key)
    elif provider == "openai":
        _can_run_llm = bool(_effective_openai_key)
    elif provider == "auto":
        _can_run_llm = bool(_effective_dahl_key or _effective_anthropic_key or _effective_openai_key)

    can_run = (validated_brief is not None and validated_gate is not None
               and validated_gate.ok and not form_drifted and _can_run_llm)

    st.markdown("### Run the pipeline")
    if validated_brief is None:
        st.info("Validate a brief first, then the run button becomes available.")
    elif form_drifted:
        st.warning("The form has changed since you validated it. The pipeline runs the "
                   "**validated** brief, not what is on screen — re-validate to run these "
                   "edits.")
    elif not _can_run_llm:
        st.error("Enter the selected provider's API key before running. Offline article generation is disabled.")
    elif not can_run:
        st.error("The decision gate routed this brief back. Resolve the blockers above "
                 "before running.")
    run_clicked = st.button("▶️ Run orchestrated pipeline", type="primary",
                            disabled=not can_run, width="stretch")

    # Live tracker placeholders. Created *before* the run so the event callback can
    # redraw into them while the orchestrator is still working.
    live_container = st.container()
    route_back_container = st.container()

    if run_clicked and can_run:
        from articlegen.models import Brief

        data = st.session_state["brief_data"]
        try:
            sections = builder.parse_sections(
                builder.sections_to_text(data.get("sections") or [])
            )
        except builder.SectionParseError as exc:
            st.error(f"Sections could not be parsed: {exc}")
            sections = []

        data = dict(data)
        data["sections"] = sections
        brief = Brief.from_dict(data)

        tracker = TrackerState(max_iterations=iterations)
        st.session_state["tracker"] = tracker

        config = OrchestratorConfig.from_dict({
            "max_iterations": iterations,
            "score_target": score_target,
            "provider": provider,
            "workspace": workspace,
            "changelog_path": changelog_path,
            "research": True,
            "source_retrieval": st.session_state["research_settings"]["source_retrieval"],
            "review_iterations": st.session_state["research_settings"]["review_iterations"],
            "execute_experiments":
                st.session_state["research_settings"]["execute_experiments"],
        })
        # Build LLM client from sidebar inputs (takes precedence over env vars)
        llm_client = resolve_client(
            provider=provider,
            model=st.session_state.get("dahl_model", "") if provider == "dahl" else None,
            api_key=st.session_state.get("dahl_api_key", "") if provider == "dahl" else None,
            base_url=st.session_state.get("dahl_base_url", "") if provider == "dahl" else None,
            dahl_api_key=st.session_state.get("dahl_api_key", ""),
            dahl_model=st.session_state.get("dahl_model", ""),
            dahl_base_url=st.session_state.get("dahl_base_url", ""),
        )
        events = EventBus()
        events.subscribe(tracker.consume)
        events.subscribe(lambda e: render.render_tracker(live_container, tracker, score_target))
        # Kept in session state on purpose: EventBus swallows subscriber exceptions so
        # a broken redraw can never abort a run, which means this list is the only
        # evidence that the live tracker actually rendered. Surfacing it makes a
        # silent failure observable instead of invisible.
        st.session_state["event_bus"] = events

        with st.spinner("Orchestrator running…"):
            orchestrator = Orchestrator(config, client=llm_client, events=events)
            result = orchestrator.run(brief)

        st.session_state["draft_markdown"] = (
            result.final_draft.to_markdown()
            if result.final_draft and any(s.body.strip() for s in result.final_draft.sections)
            else ""
        )
        research = result.research or {}
        gate = research.get("quality_gate") or {}
        st.session_state["run_meta"] = {
            "status": result.status,
            "stop_reason": result.stop_reason,
            "final_score": round(result.final_score, 4),
            "iterations": len(result.iterations),
            "tokens_total": result.tokens_total,
            "workspace": result.workspace,
            "quality_status": result.quality_status,
            "quality_label": gate.get("label", ""),
            "research_report": os.path.join(result.workspace, "research", "report.md")
            if result.workspace else "",
        }
        st.session_state["research_report_markdown"] = research.get("markdown", "")
        render.render_tracker(live_container, tracker, score_target)
        if tracker.status == "route_back":
            render.render_route_back(route_back_container, tracker)
        if result.status == "PASS":
            st.success(f"Run finished — {result.status}. Switch to the "
                       f"**Execution**, **Research**, **Scores** and **Draft** tabs "
                       f"for detail.")
        else:
            st.error(f"Run finished — {result.status}: {result.stop_reason}")

# -- Tab 2: execution -----------------------------------------------------

with tab_exec:
    st.header("Execution tracker")
    tracker: TrackerState = st.session_state["tracker"]
    if not tracker.events:
        st.info("No run yet. Validate a brief and run it from the Brief tab.")
    else:
        exec_headline = st.container()
        exec_progress = st.container()
        exec_cards = st.container()
        exec_table = st.container()
        exec_timeline = st.container()

        render.render_headline(exec_headline, tracker)
        render.render_progress(exec_progress, tracker, st.session_state.score_target)
        render.render_agent_cards(exec_cards, tracker)

        st.subheader("Per-agent detail")
        render.render_agent_table(exec_table, tracker)

        if tracker.status == "route_back":
            render.render_route_back(st.container(), tracker)

        st.subheader("Event timeline")
        render.render_timeline(exec_timeline, tracker)

# -- Tab 3: research ------------------------------------------------------

with tab_research:
    st.header("Research integrity")
    tracker = st.session_state["tracker"]
    render.render_research(st.container(), tracker)

    report_markdown = st.session_state.get("research_report_markdown") or ""
    if report_markdown:
        st.subheader("Validated export")
        st.caption("The manuscript with its integrity appendix: status, plan, verified "
                   "sources, executed experiments, failed checks and unresolved review "
                   "issues. Generated from recorded artefacts, not model text.")
        st.download_button(
            "⬇️ Download article + integrity appendix (.md)",
            data=report_markdown,
            file_name=f"{tracker.task_id or 'article'}_with_appendix.md",
            mime="text/markdown",
            width="stretch",
        )
        with st.expander("Preview appendix", expanded=False):
            st.markdown(report_markdown[:20000])

# -- Tab 4: scores --------------------------------------------------------

with tab_scores:
    st.header("Score dashboard")
    tracker = st.session_state["tracker"]
    if not tracker.iterations:
        st.info("Scores appear once the first iteration completes.")
    else:
        metrics_row = st.container()
        render.render_metrics(metrics_row, tracker, st.session_state.score_target)

        st.subheader("Score progression")
        st.line_chart_placeholder = None
        score_container = st.container()
        render.render_score_chart(score_container, tracker, st.session_state.score_target)

        st.subheader("Component breakdown")
        render.render_component_chart(st.container(), tracker)

        st.subheader("Per-iteration metrics")
        render.render_iteration_table(st.container(), tracker)

        if tracker.iterations:
            best = max(tracker.iterations, key=lambda v: v.score)
            st.caption(
                f"Best score {best.score:.4f} at iteration {best.iteration} · "
                f"target {st.session_state.score_target} · "
                f"stop reason: {tracker.stop_reason or 'n/a'}"
            )

# -- Tab 5: draft ---------------------------------------------------------

with tab_draft:
    st.header("Master draft")
    markdown = st.session_state.get("draft_markdown", "")
    meta = st.session_state.get("run_meta") or {}
    if not markdown:
        if meta:
            st.error(f"No article was produced: {meta.get('stop_reason', 'generation failed')}")
        else:
            st.info("No draft yet. Run the pipeline from the Brief tab.")
    else:
        if meta.get("status") != "PASS":
            st.warning(f"This draft needs review: {meta.get('stop_reason', 'quality target not met')}")
        quality_status = meta.get("quality_status") or ""
        if quality_status == "ready_for_final_review":
            st.success("Research validation: Ready for final review — automated checks "
                       "passed. This does not guarantee acceptance or correctness.")
        elif quality_status == "needs_revision":
            st.warning("Research validation: Needs revision — unresolved findings remain.")
        elif quality_status:
            st.error("Research validation: Incomplete or unverified — required evidence "
                     "or execution is missing. Do not present this as a completed study.")
        toolbar = st.columns([2, 1, 1, 2])
        toolbar[0].metric("Words", len(markdown.split()))
        toolbar[1].metric("Score", f"{meta.get('final_score', 0):.4f}")
        toolbar[2].metric("Iters", meta.get("iterations", 0))
        toolbar[3].write("")

        view, source = st.tabs(["Rendered", "Markdown source"])
        with view:
            st.markdown(markdown)
        with source:
            st.code(markdown, language="markdown")

        filename = f"{meta.get('task_id') or st.session_state['tracker'].task_id or 'article'}.md"
        st.download_button(
            "⬇️ Download Article (.md)",
            data=markdown,
            file_name=filename,
            mime="text/markdown",
            type="primary",
            width="stretch",
        )
        bundle = json.dumps(render.export_payload(markdown, st.session_state["tracker"], meta),
                            indent=2)
        st.download_button("⬇️ Download run bundle (.json)", data=bundle,
                           file_name=f"{filename[:-3]}_run.json", mime="application/json",
                           width="stretch")

# -- Tab 6: change log ----------------------------------------------------

with tab_log:
    st.header("Change log")
    changelog_path = st.session_state["changelog_path"]
    jsonl_path = os.path.splitext(changelog_path)[0] + ".jsonl"

    if not os.path.exists(changelog_path):
        st.info(f"No change log at `{changelog_path}` yet. Run the pipeline first.")
    else:
        raw_view, parsed_view = st.tabs(["change_log.md", "Parsed events"])
        with raw_view:
            with open(changelog_path, "r", encoding="utf-8") as handle:
                content = handle.read()
            st.caption(f"{len(content):,} characters · append-only")
            st.code(content, language="markdown")
            st.download_button("⬇️ Download change_log.md", data=content,
                               file_name="change_log.md", mime="text/markdown")
        with parsed_view:
            entries = read_jsonl(jsonl_path)
            if not entries:
                st.info("No JSONL mirror found.")
            else:
                import pandas as pd

                # Row shaping lives in builder.changelog_rows so it is unit-testable
                # and so every column arrives as a single Arrow-compatible dtype.
                st.dataframe(pd.DataFrame(builder.changelog_rows(entries)),
                             width="stretch", hide_index=True)
