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


#: Human labels for the research workflow states shown in the dashboard.
RESEARCH_STAGE_LABELS: Dict[str, str] = {
    "plan": "Research plan",
    "sources": "Sources retrieved / verified",
    "experiment": "Experiments executed",
    "code_execution": "Displayed code executed",
    "validation": "Deterministic validation",
    "peer_review": "Scientific peer review",
    "quality_gate": "Quality gate",
}

GATE_LABELS: Dict[str, str] = {
    "ready_for_final_review": "✅ Ready for final review",
    "needs_revision": "🟠 Needs revision",
    "incomplete_or_unverified": "🔴 Incomplete or unverified",
}


@dataclass
class ResearchStageView:
    """One research stage: what ran, and what it concluded."""

    stage: str
    label: str = ""
    status: str = "pending"
    summary: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def icon(self) -> str:
        return {
            "pending": "⚪", "running": "🔵", "done": "🟢", "ok": "🟢",
            "failed": "🔴", "incomplete": "🟠", "unverified": "🟠",
        }.get(self.status, "⚪")

    def as_row(self) -> Dict[str, Any]:
        return {"Stage": self.label or self.stage,
                "State": f"{self.icon} {self.status}",
                "Summary": self.summary}


@dataclass
class ResearchIssueView:
    issue_id: str
    severity: str
    dimension: str
    section: str
    claim: str
    correction: str
    status: str


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
        # -- research state (empty for a classic article run) --------------
        self.research_stages: Dict[str, ResearchStageView] = {}
        self.research_gate: str = ""
        self.research_gate_label: str = ""
        self.research_states: Dict[str, Any] = {}
        self.research_reasons: List[str] = []
        self.research_notes: List[str] = []
        self.research_issues: List[ResearchIssueView] = []
        self.research_counts: Dict[str, Any] = {}
        self.plan_summary: Dict[str, Any] = {}
        self.sources_summary: Dict[str, Any] = {}
        self.experiment_rows: List[Dict[str, Any]] = []
        self.validation_counts: Dict[str, int] = {}
        self.validation_failures: List[Dict[str, str]] = []

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

    def _on_research_stage(self, event: PipelineEvent) -> None:
        """Track plan, sources, experiment, review and gate progress."""
        payload = dict(event.payload)
        stage = str(payload.pop("stage", "") or "stage")
        status = str(payload.pop("status", "") or "")
        summary = str(payload.pop("summary", "") or event.message)
        view = self.research_stages.get(stage) or ResearchStageView(stage=stage)
        view.label = RESEARCH_STAGE_LABELS.get(stage, view.label or stage)
        view.status = status or view.status
        view.summary = summary or view.summary
        view.details.update(payload)
        self.research_stages[stage] = view
        self.research_counts[stage] = view.status

        if stage == "quality_gate":
            self.research_gate = status
            self.research_gate_label = GATE_LABELS.get(status, status)
            self.research_reasons = list(payload.get("reasons") or [])
            self.research_states = dict(payload.get("states") or {})
        elif stage == "validation":
            self.validation_counts = dict(payload.get("counts") or {})
            self.validation_failures = [
                {key: str(item.get(key, "")) for key in
                 ("check", "severity", "details", "effect")}
                for item in (payload.get("failing") or []) if isinstance(item, dict)
            ]
        elif stage == "sources":
            self.sources_summary = {key: payload.get(key) for key in
                                    ("records", "verified", "user_provided",
                                     "unverified", "provider", "queries")}
        elif stage == "plan":
            self.plan_summary = {
                "article_type": payload.get("article_type", ""),
                "contribution_type": payload.get("contribution_type", ""),
                "research_question": payload.get("research_question", ""),
                "contribution": payload.get("contribution", ""),
                "decision": payload.get("decision", ""),
                "validated": payload.get("validated", False),
                "missing_elements": list(payload.get("missing_elements") or []),
                "decision_reasons": list(payload.get("decision_reasons") or []),
            }
        elif stage == "experiment":
            row = {
                "Experiment": str(payload.get("experiment_id", "")),
                "State": str(payload.get("status", "")),
                "Exit": str(payload.get("exit_code", "-")),
                "Tests": f"{payload.get('tests_passed', 0)}/{payload.get('tests_run', 0)}",
            }
            self.experiment_rows.append(row)
        elif stage == "peer_review":
            issues = payload.get("issues") or []
            if issues:
                self.research_issues = [
                    ResearchIssueView(
                        issue_id=str(item.get("issue_id", "")),
                        severity=str(item.get("severity", "")),
                        dimension=str(item.get("dimension", "")),
                        section=str(item.get("section", "")),
                        claim=str(item.get("claim", "")),
                        correction=str(item.get("correction", "")),
                        status=str(item.get("status", "open")),
                    ) for item in issues if isinstance(item, dict)
                ]

    def _on_run_end(self, event: PipelineEvent) -> None:
        payload = event.payload
        self.status = "route_back" if payload.get("status") == "ROUTE_BACK" else "done"
        self.result_status = str(payload.get("status", ""))
        self.stop_reason = str(payload.get("stop_reason") or event.message)
        self.final_score = float(payload.get("final_score", self.final_score))
        self.tokens_total = int(payload.get("tokens_total", self.tokens_total))
        quality_status = str(payload.get("quality_status") or "")
        if quality_status:
            self.research_gate = self.research_gate or quality_status
            self.research_gate_label = (
                self.research_gate_label or GATE_LABELS.get(quality_status, quality_status)
            )
        self.research_notes = list(payload.get("research_notes") or self.research_notes)

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

    def research_rows(self) -> List[Dict[str, Any]]:
        order = ["plan", "sources", "experiment", "code_execution", "validation",
                 "peer_review", "quality_gate"]
        views = [self.research_stages[stage] for stage in order
                 if stage in self.research_stages]
        views += [view for key, view in self.research_stages.items()
                  if key not in order]
        return [view.as_row() for view in views]

    def research_state_rows(self) -> List[Dict[str, str]]:
        """Workflow states reported by the gate, with 'not required' kept distinct."""
        rows: List[Dict[str, str]] = []
        labels = {
            "draft_generated": "Draft generated",
            "plan_validated": "Research plan validated",
            "sources_retrieved": "Sources retrieved",
            "sources_verified": "Sources verified",
            "experiments_executed": "Experiments executed",
            "tests_passed": "Tests passed",
            "technical_checks_completed": "Technical checks completed",
            "peer_review_completed": "Peer review completed",
            "revision_completed": "Revision completed",
            "ready_for_human_review": "Ready for human review",
        }
        for key, label in labels.items():
            if key not in self.research_states:
                continue
            value = bool(self.research_states.get(key))
            rows.append({"State": label, "Reached": "yes" if value else "no"})
        return rows

    @property
    def has_research(self) -> bool:
        return bool(self.research_stages)

    def headline(self) -> str:
        if self.status == "route_back":
            if any(issue.get("category") == "provider" for issue in self.route_back_issues):
                return f"⛔ No draft generated — {self.stop_reason}"
            return f"⛔ Routed back at the input gate — {self.stop_reason}"
        if self.status == "done":
            gate = f" · research: {self.research_gate_label}" if self.research_gate else ""
            icon = "✅" if self.result_status == "PASS" else "⚠️"
            return (f"{icon} {self.result_status} — score {self.final_score:.4f}"
                    f"{gate} ({self.stop_reason})")
        if self.status == "running" and self.current_iteration:
            running = [v.label for v in self.agents() if v.state == "running"]
            who = running[0] if running else "between agents"
            return f"▶️ Iteration {self.current_iteration}/{self.max_iterations} — {who}"
        return self.message or "Idle"
