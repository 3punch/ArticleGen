"""Streamlit rendering helpers.

The only module in the package that imports ``streamlit``. Every function takes an
explicit container so the live tracker can be redrawn into the *same* placeholders
while the orchestrator is still running - that is what makes the updates feel
real-time instead of appearing all at once when the run finishes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from .state import TrackerState

STATUS_METRIC_COLORS = {
    "done": "normal",
    "route_back": "inverse",
    "running": "normal",
    "starting": "normal",
    "idle": "off",
}


def render_headline(container, tracker: TrackerState) -> None:
    """One-line status under the progress bar."""
    with container:
        st.markdown(f"**{tracker.headline()}**")


def render_progress(container, tracker: TrackerState, score_target: float = 0.85) -> None:
    with container:
        st.progress(tracker.progress, text=f"Run progress — {tracker.progress:.0%}")


def render_agent_cards(container, tracker: TrackerState) -> None:
    """Five status cards, one per agent, in mandatory execution order."""
    with container:
        columns = st.columns(len(tracker.agents()) or 1)
        for column, view in zip(columns, tracker.agents()):
            with column:
                st.markdown(
                    f"### {view.icon}\n\n**{view.label}**\n\n"
                    f"<span style='font-size:0.8rem'>{view.role or '—'}</span>",
                    unsafe_allow_html=True,
                )
                if view.state in ("done", "degraded"):
                    st.caption(
                        f"{view.findings} findings · {view.blockers} blockers · "
                        f"{view.context_tokens} ctx tok"
                    )
                else:
                    st.caption(view.state)


def render_agent_table(container, tracker: TrackerState) -> None:
    with container:
        rows = tracker.agent_rows()
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_metrics(container, tracker: TrackerState, score_target: float) -> None:
    with container:
        left, middle, right = st.columns(3)
        left.metric("Final score", f"{tracker.final_score:.4f}",
                    help=f"Target is {score_target}")
        middle.metric("Iterations", len(tracker.iterations))
        right.metric("Tokens used", f"{tracker.tokens_total:,}")


def render_score_chart(container, tracker: TrackerState, score_target: float) -> None:
    with container:
        series = tracker.score_series()
        if len(series) < 2:
            st.info("No completed iterations yet — the score line appears after iteration 1.")
            return
        frame = pd.DataFrame(series)
        target = pd.DataFrame({"Iteration": frame["Iteration"],
                               "Target": [score_target] * len(frame)})
        st.line_chart(pd.concat([frame, target], axis=1), height=280)


def render_component_chart(container, tracker: TrackerState) -> None:
    with container:
        rows = tracker.component_series()
        if not rows:
            return
        frame = pd.DataFrame(rows)
        pivot = frame.pivot_table(index="Iteration", columns="Component",
                                  values="Score", aggfunc="first").reset_index()
        st.line_chart(pivot.set_index("Iteration"), height=260)


def render_iteration_table(container, tracker: TrackerState) -> None:
    with container:
        rows = tracker.iteration_rows()
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_timeline(container, tracker: TrackerState, limit: int = 40) -> None:
    with container:
        if not tracker.timeline:
            st.caption("No events yet.")
            return
        frame = pd.DataFrame(tracker.timeline[-limit:])
        st.dataframe(frame, width="stretch", hide_index=True)


def render_tracker(container, tracker: TrackerState, score_target: float = 0.85) -> None:
    """Redraw the whole live tracker into one container.

    Called from the orchestrator's event callback, so it runs many times during a
    single run. Streamlit pushes each redraw to the browser as it happens.
    """
    with container:
        render_headline(st.empty(), tracker)
        render_progress(st.empty(), tracker, score_target)
        render_agent_cards(st.empty(), tracker)


def render_route_back(container, tracker: TrackerState) -> None:
    with container:
        st.error(f"Input gate routed this task back: {tracker.stop_reason}")
        issues = tracker.route_back_issues
        if issues:
            rows = [{
                "severity": issue.get("severity", ""),
                "category": issue.get("category", ""),
                "target": issue.get("section", "-"),
                "issue": issue.get("claim", ""),
                "action": issue.get("action", ""),
            } for issue in issues if issue.get("severity") != "note"]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def export_payload(draft_markdown: str, tracker: TrackerState,
                   result_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Bundle the draft, scores and stop reason into one exportable dict."""
    return {
        "task_id": tracker.task_id,
        "status": tracker.result_status,
        "stop_reason": tracker.stop_reason,
        "final_score": round(tracker.final_score, 4),
        "tokens_total": tracker.tokens_total,
        "iterations": tracker.iteration_rows(),
        "agents": tracker.agent_rows(),
        "meta": result_meta or {},
        "markdown": draft_markdown,
    }
