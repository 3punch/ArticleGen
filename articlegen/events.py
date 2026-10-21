"""Structured progress events emitted by the orchestrator.

The orchestrator stays UI-agnostic: it emits plain dataclass events through an
optional callback, and consumers (the Streamlit dashboard, a future websocket
bridge, a progress-printing CLI) decide how to render them. Nothing in here imports
Streamlit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(str, Enum):
    RUN_START = "run_start"
    GATE_PASS = "gate_pass"
    GATE_ROUTE_BACK = "gate_route_back"
    AGENT_START = "agent_start"
    AGENT_DONE = "agent_done"
    ITERATION_START = "iteration_start"
    ITERATION_DONE = "iteration_done"
    DRAFT_UPDATED = "draft_updated"
    #: A research stage (plan, sources, experiment, code check, review, gate)
    #: changed state. The payload carries the stage name and its outcome.
    RESEARCH_STAGE = "research_stage"
    RUN_END = "run_end"
    LOG = "log"


@dataclass
class PipelineEvent:
    type: EventType
    task_id: str = ""
    iteration: int = 0
    agent: str = ""
    message: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    #: Monotonic seconds since the run began - lets a UI plot elapsed time.
    elapsed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "task_id": self.task_id,
            "iteration": self.iteration,
            "agent": self.agent,
            "message": self.message,
            "payload": self.payload,
            "elapsed": round(self.elapsed, 3),
        }


EventCallback = Callable[[PipelineEvent], None]


class EventBus:
    """Collects events and fans them out to any registered callbacks.

    A callback that raises must not be able to kill a pipeline run - a broken
    dashboard is a reporting problem, not a reason to lose the work.
    """

    def __init__(self, callbacks: Optional[List[EventCallback]] = None) -> None:
        self.callbacks: List[EventCallback] = list(callbacks or [])
        self.events: List[PipelineEvent] = []
        self._start = time.monotonic()
        self.errors: List[str] = []

    def subscribe(self, callback: EventCallback) -> None:
        self.callbacks.append(callback)

    def emit(self, event_type: EventType, *, task_id: str = "", iteration: int = 0,
             agent: str = "", message: str = "", **payload: Any) -> PipelineEvent:
        event = PipelineEvent(
            type=event_type,
            task_id=task_id,
            iteration=iteration,
            agent=agent,
            message=message,
            payload=payload,
            elapsed=time.monotonic() - self._start,
        )
        self.events.append(event)
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as exc:  # noqa: BLE001 - UI must never break the run
                self.errors.append(f"{type(exc).__name__}: {exc}")
        return event

    def by_type(self, event_type: EventType) -> List[PipelineEvent]:
        return [e for e in self.events if e.type == event_type]
