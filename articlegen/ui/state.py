"""Streamlit-independent view state for the dashboard.

Everything here is plain Python: it consumes ``PipelineEvent`` objects and exposes
render-ready structures. Keeping it free of ``streamlit`` imports means the tracker
logic is covered by ordinary unit tests, and only the thin rendering layer needs a
running Streamlit process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..events import PipelineEvent

#: The mandatory order from claude.md. The tracker renders in this order regardless
#: of the order events arrive in.
AGENT_ORDER: List[str] = ["writer", "tester", "reviewer", "architect", "synthesizer"]

AGENT_LABELS: Dict[str, str] = {
    "writer": "1 · Writer",
    "tester": "2 · Tester",
    "reviewer": "3 · Reviewer",
    "architect": "4 · Architect",
    "synthesizer": "5 · Synthesizer",
}

#: Icon per lifecycle state, for the visual indicator column.
STATE_ICONS: Dict[str, str] = {
    "pending": "⚪",
    "running": "🔵",
    "done": "🟢",
    "degraded": "🟠",
    "skipped": "⚫",
    "failed": "🔴",
}


@dataclass
class AgentView:
    """One row of the execution tracker."""

    agent_id: str
    label: str
    role: str = ""
    state: str = "pending"
    findings: int = 0
    blockers: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    context_tokens: int = 0
    evicted: int = 0
    engine: str = ""
    summary: str = ""
    notes: List[str] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def icon(self) -> str:
        return STATE_ICONS.get(self.state, "⚪")

    def as_row(self) -> Dict[str, Any]:
        return {
            "Agent": self.label,
            "Role": self.role,
            "State": f"{self.icon} {self.state}",
            "Findings": self.findings,
            "Blockers": self.blockers,
            "Context tok": self.context_tokens,
            "In/Out tok": f"{self.tokens_in}/{self.tokens_out}",
            "Engine": self.engine,
        }


@dataclass
class IterationView:
    iteration: int
    score: float
    previous_score: Optional[float] = None
    delta: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    findings: int = 0
    applied: int = 0
    deferred: int = 0
    tokens_total: int = 0
    draft_words: int = 0
    draft_version: int = 0
    evicted: List[str] = field(default_factory=list)
    gate_ok: bool = True
    gate_blockers: int = 0

    def as_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "Iteration": self.iteration,
            "Score": round(self.score, 4),
            "Δ": round(self.delta, 4),
            "Findings": self.findings,
            "Applied": self.applied,
            "Deferred": self.deferred,
            "Words": self.draft_words,
            "Tokens": self.tokens_total,
        }
        for key, value in sorted(self.components.items()):
            row[key.capitalize()] = round(value, 3)
        return row


class TrackerState:
    """Accumulates orchestrator events into dashboard state.

    ``consume`` is the only mutator, so a UI can drive the tracker by replaying an
    event list - which is exactly what the tests do.
    """

    def __init__(self, agent_ids: Optional[List[str]] = None, max_iterations: int = 4) -> None:
        self.agent_ids = list(agent_ids or AGENT_ORDER)
        self.max_iterations = max(1, max_iterations)
        self.reset()

    def reset(self) -> None:
        self.status: str = "idle"
        self.task_id: str = ""
        self.message: str = ""
        self.stop_reason: str = ""
        self.result_status: str = ""
        self.final_score: float = 0.0
        self.tokens_total: int = 0
        self.elapsed: float = 0.0
        self.route_back_issues: List[Dict[str, Any]] = []
        self.gate_passed: bool = False
        self.current_iteration: int = 0
        self._agents: Dict[str, AgentView] = {
            agent_id: AgentView(agent_id=agent_id, label=AGENT_LABELS.get(agent_id, agent_id))
            for agent_id in self.agent_ids
        }
        self._iteration_agents_done: int = 0
        self.iterations: List[IterationView] = []
        self.timeline: List[Dict[str, Any]] = []
        self.events: List[PipelineEvent] = []

    # -- ingestion --------------------------------------------------------
    def consume(self, event: PipelineEvent) -> None:
        self.events.append(event)
        self.elapsed = event.elapsed
        if event.task_id:
            self.task_id = event.task_id
        if event.message:
            self.message = event.message

        handler = getattr(self, f"_on_{event.type.value}", None)
        if handler is not None:
            handler(event)

        self.timeline.append({
            "t": round(event.elapsed, 2),
            # Stringified deliberately: a column mixing int and "-" is not
            # Arrow-serialisable, and st.dataframe needs a single dtype per column.
            "iteration": str(event.iteration) if event.iteration else "-",
            "agent": event.agent or "-",
            "event": event.type.value,
            "message": event.message,
        })

    def consume_all(self, events) -> None:
        for event in events:
            self.consume(event)

    # -- per-event handlers ----------------------------------------------
    def _on_run_start(self, event: PipelineEvent) -> None:
        self.status = "starting"
        self.max_iterations = int(event.payload.get("max_iterations") or self.max_iterations)
        for view in self._agents.values():
            view.state = "pending"

    def _on_gate_pass(self, event: PipelineEvent) -> None:
        self.status = "running"
        self.gate_passed = True

    def _on_gate_route_back(self, event: PipelineEvent) -> None:
        self.status = "route_back"
        self.gate_passed = False
        self.stop_reason = event.message
        self.route_back_issues = list(event.payload.get("issues") or [])

    def _on_iteration_start(self, event: PipelineEvent) -> None:
        self.status = "running"
        self.current_iteration = event.iteration
        self._iteration_agents_done = 0
        for view in self._agents.values():
            view.state = "pending"
            view.findings = 0
            view.blockers = 0

    def _on_agent_start(self, event: PipelineEvent) -> None:
        view = self._agents.get(event.agent)
        if view is not None:
            view.state = "running"
            view.role = event.payload.get("role", view.role)

    def _on_agent_done(self, event: PipelineEvent) -> None:
        view = self._agents.get(event.agent)
        if view is not None:
            payload = event.payload
            status = str(payload.get("status", "ok"))
            if str(payload.get("engine", "")) == "skipped":
                view.state = "skipped"
            elif status in ("degraded", "failed"):
                view.state = "degraded"
            else:
                view.state = "done"
            view.role = payload.get("role", view.role)
            view.findings = int(payload.get("findings", 0))
            view.blockers = int(payload.get("blockers", 0))
            view.tokens_in = int(payload.get("tokens_in", 0))
            view.tokens_out = int(payload.get("tokens_out", 0))
            view.context_tokens = int(payload.get("context_tokens", 0))
            view.evicted = int(payload.get("evicted", 0))
            view.engine = payload.get("engine", "")
            view.summary = payload.get("summary", "")
            view.notes = list(payload.get("notes") or [])
            view.elapsed = event.elapsed
        self._iteration_agents_done += 1

    def _on_iteration_done(self, event: PipelineEvent) -> None:
        payload = event.payload
        self.iterations.append(IterationView(
            iteration=event.iteration,
            score=float(payload.get("score", 0.0)),
            previous_score=payload.get("previous_score"),
            delta=float(payload.get("delta", 0.0)),
            components=dict(payload.get("components") or {}),
            findings=int(payload.get("findings", 0)),
            applied=int(payload.get("applied", 0)),
            deferred=int(payload.get("deferred", 0)),
            tokens_total=int(payload.get("tokens_total", 0)),
            draft_words=int(payload.get("draft_words", 0)),
            draft_version=int(payload.get("draft_version", 0)),
            evicted=list(payload.get("evicted") or []),
            gate_ok=bool(payload.get("gate_ok", True)),
            gate_blockers=int(payload.get("gate_blockers", 0)),
        ))
        self.final_score = float(payload.get("score", 0.0))
        self.tokens_total += int(payload.get("tokens_total", 0))

    def _on_run_end(self, event: PipelineEvent) -> None:
        payload = event.payload
        self.status = "route_back" if payload.get("status") == "ROUTE_BACK" else "done"
        self.result_status = str(payload.get("status", ""))
        self.stop_reason = str(payload.get("stop_reason") or event.message)
        self.final_score = float(payload.get("final_score", self.final_score))
        self.tokens_total = int(payload.get("tokens_total", self.tokens_total))

    # -- derived views ----------------------------------------------------
    def agents(self) -> List[AgentView]:
        return [self._agents[agent_id] for agent_id in self.agent_ids if agent_id in self._agents]

    def agent_rows(self) -> List[Dict[str, Any]]:
        return [view.as_row() for view in self.agents()]

    def iteration_rows(self) -> List[Dict[str, Any]]:
        return [view.as_row() for view in self.iterations]

    def score_series(self) -> List[Dict[str, Any]]:
        """Iteration 0 is pinned at 0.0 so the chart shows the starting point."""
        series = [{"Iteration": 0, "Score": 0.0}]
        for view in self.iterations:
            series.append({"Iteration": view.iteration, "Score": round(view.score, 4)})
        return series

    def component_series(self) -> List[Dict[str, Any]]:
        """Long-form rows: one per (iteration, component) for a stacked/multi-line chart."""
        rows: List[Dict[str, Any]] = []
        for view in self.iterations:
            for component, value in sorted(view.components.items()):
                rows.append({"Iteration": view.iteration,
                             "Component": component.capitalize(),
                             "Score": round(float(value), 4)})
        return rows

    @property
    def progress(self) -> float:
        """0..1 across the whole run, so a progress bar is meaningful mid-flight."""
        if self.status in ("done",):
            return 1.0
        if self.status == "route_back":
            return 1.0
        if not self.gate_passed:
            return 0.02 if self.status == "starting" else 0.0
        total_steps = self.max_iterations * max(1, len(self.agent_ids))
        completed = ((self.current_iteration - 1) * len(self.agent_ids)
                     + self._iteration_agents_done)
        return min(0.99, max(0.0, completed / total_steps))

    @property
    def is_finished(self) -> bool:
        return self.status in ("done", "route_back")

    def status_label(self) -> str:
        return {
            "idle": "Idle",
            "starting": "Starting",
            "running": "Running",
            "done": "Finished",
            "route_back": "Routed back",
        }.get(self.status, self.status)

    def headline(self) -> str:
        if self.status == "route_back":
            return f"⛔ Routed back at the input gate — {self.stop_reason}"
        if self.status == "done":
            return (f"✅ {self.result_status} — score {self.final_score:.4f} "
                    f"({self.stop_reason})")
        if self.status == "running" and self.current_iteration:
            running = [v.label for v in self.agents() if v.state == "running"]
            who = running[0] if running else "between agents"
            return f"▶️ Iteration {self.current_iteration}/{self.max_iterations} — {who}"
        return self.message or "Idle"
