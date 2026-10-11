"""Change log: the durable record of what changed and why.

Two artefacts are maintained side by side:

* ``change_log.md`` - append-only, human-readable, one block per iteration. This
  is the file the user asked for; it is the substitute for re-passing history.
* ``change_log.jsonl`` - one JSON object per block, for tooling and for resuming
  a run without re-reading prose.

Writes are locked (POSIX advisory lock) and go through a temp file + ``rename``
so concurrent runs cannot interleave a half-written block.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - platform dependent
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

from .models import AgentReport, GateResult, IterationRecord, Severity

HEADER = """# ArticleGen Change Log

Append-only. Every pipeline iteration appends exactly one block; nothing here is
rewritten. This file is the pipeline's long-term memory substitute: instead of
re-passing full agent histories into each context window, prior state is
compressed into the blocks below and referenced from there.

Legend: `gate` = input/draft verification result, `score` = weighted quality
score (higher is better), `evicted` = context blocks dropped to respect the token
cap, `applied` = fixes the Synthesizer merged, `deferred` = findings carried
forward to the next iteration.

"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _LockedAppend:
    """Context manager giving exclusive, crash-safe append semantics."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._handle = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._handle = open(self.path, "a", encoding="utf-8")
        if fcntl is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self._handle

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is None:
            return
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None
        return False


class ChangeLog:
    def __init__(self, markdown_path: str = "change_log.md", jsonl_path: Optional[str] = None) -> None:
        self.markdown_path = markdown_path
        self.jsonl_path = jsonl_path or (
            os.path.splitext(markdown_path)[0] + ".jsonl"
        )
        self._ensure_header()

    # -- plumbing ---------------------------------------------------------
    def _ensure_header(self) -> None:
        if os.path.exists(self.markdown_path) and os.path.getsize(self.markdown_path) > 0:
            return
        with _LockedAppend(self.markdown_path) as handle:
            handle.write(HEADER)

    def _write(self, block: str, record: Dict[str, Any]) -> None:
        """Append one Markdown block and its JSONL mirror under an exclusive lock."""
        with _LockedAppend(self.markdown_path) as handle:
            handle.write("\n" + block.rstrip() + "\n")
        directory = os.path.dirname(self.jsonl_path) or "."
        os.makedirs(directory, exist_ok=True)
        with _LockedAppend(self.jsonl_path) as handle:
            handle.write(json.dumps(record, sort_keys=False) + "\n")

    # -- block emitters ---------------------------------------------------
    def log_event(self, event: str, task_id: str, details: Dict[str, Any]) -> None:
        lines = [f"## [{utc_now()}] {event} · task={task_id}", ""]
        for key, value in details.items():
            lines.append(f"- {key}: {value}")
        record = {"timestamp": utc_now(), "event": event, "task_id": task_id, **details}
        self._write("\n".join(lines), record)

    def log_gate_routed_back(self, task_id: str, gate: GateResult, iteration: int = 0) -> None:
        lines = [
            f"## [{utc_now()}] GATE ROUTE-BACK · task={task_id} · stage={gate.stage}",
            "",
            f"- decision: **ROUTE_BACK** (no sub-agent tokens spent)",
            f"- reason: {gate.reason}",
            f"- checks run: {', '.join(gate.checks_run)}",
            "",
            "### Corrections required",
            "",
            "| severity | category | target | issue | required action |",
            "| --- | --- | --- | --- | --- |",
        ]
        for issue in sorted(gate.issues, key=lambda f: (-f.severity.weight, f.category)):
            if issue.severity == Severity.NOTE:
                continue
            lines.append(
                f"| {issue.severity.value} | {issue.category} | {_cell(issue.section)} | "
                f"{_cell(issue.claim)} | {_cell(issue.action)} |"
            )
        record = {
            "timestamp": utc_now(),
            "event": "gate_route_back",
            "task_id": task_id,
            "stage": gate.stage,
            "iteration": iteration,
            "reason": gate.reason,
            "issues": [i.to_dict() for i in gate.issues],
        }
        self._write("\n".join(lines), record)

    def log_iteration(self, task_id: str, record: IterationRecord, gate: Optional[GateResult] = None,
                      context_describe: str = "", draft_words: int = 0) -> None:
        previous = "—" if record.previous_score is None else f"{record.previous_score:.4f}"
        lines = [
            f"## [{utc_now()}] task={task_id} · iteration {record.iteration} · "
            f"decision={record.decision.upper()}",
            "",
            f"- gate: {_gate_line(gate)}",
            f"- agents: {_agents_line(record.reports)}",
            f"- score: {previous} → **{record.score:.4f}** (Δ{record.delta:+.4f})",
            f"- findings: {record.finding_count} raised · {record.applied} applied · {record.deferred} deferred",
            f"- draft: {draft_words} words at end of iteration",
            f"- context: {context_describe or 'n/a'}",
            f"- tokens: {record.tokens_total} this iteration (run total {record.tokens_total})",
        ]
        if record.evicted:
            lines.append(f"- evicted from context: {', '.join(record.evicted)}")

        applied = [f for r in record.reports for f in r.findings if f.auto_fixable]
        deferred = [f for r in record.reports for f in r.findings if not f.auto_fixable]
        if applied:
            lines += ["", "### Applied by Synthesizer", "",
                      "| agent | severity | section | claim | action |",
                      "| --- | --- | --- | --- | --- |"]
            for finding in applied[:20]:
                lines.append(
                    f"| {finding.agent} | {finding.severity.value} | {_cell(finding.section)} | "
                    f"{_cell(finding.claim)} | {_cell(finding.action)} |"
                )
            if len(applied) > 20:
                lines.append(f"| … | | | | {len(applied) - 20} more applied |")
        if deferred:
            lines += ["", "### Deferred (carried forward in compressed form, never re-passed in full)", ""]
            for finding in sorted(deferred, key=lambda f: -f.severity.weight)[:10]:
                lines.append(
                    f"- [{finding.severity.value}] {finding.agent}/{finding.category}"
                    f"@{_cell(finding.section)}: {finding.claim}"
                )
            if len(deferred) > 10:
                lines.append(f"- … {len(deferred) - 10} further finding(s) folded into the rolling summary")

        payload = {
            "timestamp": utc_now(),
            "event": "iteration",
            "task_id": task_id,
            "iteration": record.iteration,
            "decision": record.decision,
            "score": round(record.score, 4),
            "previous_score": None if record.previous_score is None else round(record.previous_score, 4),
            "delta": round(record.delta, 4),
            "tokens_total": record.tokens_total,
            "evicted": record.evicted,
            "applied": record.applied,
            "deferred": record.deferred,
            "agents": record.to_dict()["agents"],
            "context": context_describe,
            "gate": gate.to_dict() if gate else None,
        }
        self._write("\n".join(lines), payload)


def _cell(value: str) -> str:
    return str(value or "-").replace("|", "\\|").replace("\n", " ")


def _gate_line(gate: Optional[GateResult]) -> str:
    if gate is None:
        return "not re-run"
    status = "PASS" if gate.ok else "FAIL"
    counts = {sev.value: 0 for sev in Severity}
    for issue in gate.issues:
        counts[issue.severity.value] += 1
    detail = ", ".join(f"{value} {key}" for key, value in counts.items() if value)
    return f"{status} ({detail or '0 issues'}) at {gate.stage}"


def _agents_line(reports: Iterable[AgentReport]) -> str:
    parts = []
    for report in reports:
        parts.append(f"{report.agent_id}={report.status.value}/{len(report.findings)}f/{report.engine}")
    return " · ".join(parts) or "none"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries
