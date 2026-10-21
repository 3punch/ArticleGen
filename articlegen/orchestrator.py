"""Stage 2 - Master Orchestrator.

Owns the loop: gate -> mandatory agent sequence -> score -> decide -> log. Sub-agents
never call each other and never see each other's raw transcripts; the orchestrator is
the only thing that routes state between them, and it does so through bounded context
packets.

Termination is never silent: every exit path produces a ``stop_reason`` that is written
to ``change_log.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agents import build_agents
from .agents.synthesizer import publish_draft
from .changelog import ChangeLog
from .events import EventBus, EventType
from .gate import verify_brief, verify_draft
from .llm import LLMClient, resolve_client
from .memory import MemoryStore, findings_block
from .models import (
    AgentReport,
    AgentStatus,
    Brief,
    Draft,
    Finding,
    GateResult,
    IterationRecord,
    RunResult,
    Severity,
)
from .prompts import rules_digest
from .scoring import ScoreCard, decide_next, draft_is_publishable, score_iteration
from .tokens import TokenBudget


@dataclass
class OrchestratorConfig:
    max_iterations: int = 4
    score_target: float = 0.85
    min_delta: float = 0.01
    stall_window: int = 2
    provider: str = "auto"
    model: Optional[str] = None
    workspace: str = "runs"
    changelog_path: str = "change_log.md"
    prompt_dir: Optional[str] = None
    rules_path: Optional[str] = None
    budget: TokenBudget = field(default_factory=TokenBudget.default)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestratorConfig":
        base = cls()
        return cls(
            max_iterations=int(data.get("max_iterations", base.max_iterations)),
            score_target=float(data.get("score_target", base.score_target)),
            min_delta=float(data.get("min_delta", base.min_delta)),
            stall_window=int(data.get("stall_window", base.stall_window)),
            provider=str(data.get("provider", base.provider)),
            model=data.get("model"),
            workspace=str(data.get("workspace", base.workspace)),
            changelog_path=str(data.get("changelog_path", base.changelog_path)),
            prompt_dir=data.get("prompt_dir"),
            rules_path=data.get("rules_path"),
            budget=TokenBudget.from_dict(data.get("token_budget") or {}, base.budget),
        )


class Orchestrator:
    def __init__(self, config: Optional[OrchestratorConfig] = None,
                 client: Optional[LLMClient] = None,
                 changelog: Optional[ChangeLog] = None,
                 events: Optional[EventBus] = None) -> None:
        self.config = config or OrchestratorConfig()
        self.client = client or resolve_client(self.config.provider, self.config.model)
        self.changelog = changelog or ChangeLog(self.config.changelog_path)
        self.agents = build_agents(self.client, self.config.budget,
                                   prompt_dir=self.config.prompt_dir,
                                   rules_path=self.config.rules_path)
        self._digest = rules_digest(self.config.rules_path)
        #: Optional progress bus. The dashboard subscribes to this; the CLI does not.
        self.events = events or EventBus()

    # -- entry point ------------------------------------------------------
    def run(self, brief: Brief) -> RunResult:
        self.changelog.log_event("RUN_START", brief.task_id, {
            "title": brief.title,
            "mode": brief.mode.value,
            "target_words": brief.target_words,
            "sections": len(brief.sections),
            "engine": self.client.name() if self.client.available else "not-configured",
            "max_iterations": self.config.max_iterations,
            "score_target": self.config.score_target,
            "context_hard_cap": self.config.budget.context_hard_cap,
        })

        # -- Stage 1: decision gate --------------------------------------
        self.events.emit(EventType.RUN_START, task_id=brief.task_id,
                         message=f"verifying input for {brief.task_id}",
                         title=brief.title, sections=len(brief.sections),
                         max_iterations=self.config.max_iterations,
                         score_target=self.config.score_target,
                         engine=self.client.name() if self.client.available
                         else "not-configured")

        gate = verify_brief(brief)
        if gate.routed_back:
            self.changelog.log_gate_routed_back(brief.task_id, gate)
            self.changelog.log_event("RUN_ABORTED", brief.task_id, {
                "reason": "input verification failed; routed back for correction",
                "blockers": len(gate.blockers),
                "tokens_spent": 0,
            })
            self.events.emit(EventType.GATE_ROUTE_BACK, task_id=brief.task_id,
                             message=gate.reason, blockers=len(gate.blockers),
                             issues=[i.to_dict() for i in gate.issues])
            self.events.emit(EventType.RUN_END, task_id=brief.task_id, status="ROUTE_BACK",
                             message=gate.reason, final_score=0.0, iterations=0,
                             tokens_total=0)
            return RunResult(
                task_id=brief.task_id, status="ROUTE_BACK", gate=gate,
                workspace="", stop_reason=gate.reason,
            )

        self.events.emit(EventType.GATE_PASS, task_id=brief.task_id,
                         message="input verification passed; entering agent loop",
                         issues=len(gate.issues))

        # Generation is LLM-only. A missing key (including explicit offline mode)
        # must stop before the Writer can produce any draft text.
        if not self.client.available:
            reason = (
                f"LLM provider '{self.config.provider}' is not configured. "
                "Article generation has no offline fallback; enter a Dahl API key "
                "and model in the UI sidebar or set DAHL_API_KEY / DAHL_MODEL, "
                "then run again."
            )
            self.changelog.log_event("RUN_ABORTED", brief.task_id, {
                "reason": reason,
                "provider": self.config.provider,
                "blockers": 0,
                "tokens_spent": 0,
            })
            self.events.emit(EventType.GATE_ROUTE_BACK, task_id=brief.task_id,
                             message=reason, blockers=0,
                             issues=[{"severity": "blocker", "category": "provider",
                                      "section": "run settings", "claim": reason,
                                      "action": "Enter a valid LLM API key and model, then retry."}])
            self.events.emit(EventType.RUN_END, task_id=brief.task_id, status="ROUTE_BACK",
                             message=reason, final_score=0.0, iterations=0,
                             tokens_total=0)
            return RunResult(
                task_id=brief.task_id, status="ROUTE_BACK", gate=gate,
                workspace="", stop_reason=reason,
            )

        # Only create a per-task directory after the gate has validated task_id.
        # Otherwise an invalid id such as "../outside" could escape the workspace
        # before the gate gets a chance to route the brief back.
        store = MemoryStore(self.config.workspace, brief.task_id)
        draft = Draft.from_brief(brief)
        store.write_master_draft(draft)
        store.absorb_findings(gate.issues)   # gate's non-blocking issues become work items
        store.save_state()

        previous_score: Optional[float] = None
        stall_count = 0
        tokens_total = 0
        records: List[IterationRecord] = []
        stop_reason = ""
        status = "NEEDS_HUMAN"

        for iteration in range(1, self.config.max_iterations + 1):
            iteration_record, draft = self._run_iteration(iteration, brief, draft, store,
                                                          previous_score)
            tokens_total += iteration_record.tokens_total
            records.append(iteration_record)
            previous_score = iteration_record.score

            decision, stop_reason, stall_count = decide_next(
                score=iteration_record.score,
                previous_score=None if iteration == 1 else records[-2].score,
                iteration=iteration,
                max_iterations=self.config.max_iterations,
                score_target=self.config.score_target,
                min_delta=self.config.min_delta,
                stall_window=self.config.stall_window,
                stall_count=stall_count,
                tokens_total=tokens_total,
                token_ceiling=self.config.budget.run_token_ceiling,
                degraded=any(r.status == AgentStatus.DEGRADED for r in iteration_record.reports),
            )
            # An empty outline used to score exactly 0.85 (every weight except
            # completeness) and stop the loop as if the article were done.
            if (decision == "stop" and stop_reason.startswith("score target met")
                    and not draft_is_publishable(draft, brief)):
                if iteration >= self.config.max_iterations:
                    decision = "stop"
                    stop_reason = (
                        f"max iterations reached ({self.config.max_iterations}); "
                        "score looked complete but the draft has no publishable article text"
                    )
                else:
                    decision = "continue"
                    stop_reason = ""
            if (decision == "continue" and iteration >= 2
                    and not any(section.body.strip() for section in draft.sections)):
                decision = "stop"
                stop_reason = "no article text after 2 iterations"
            iteration_record.decision = decision
            store.record_iteration(iteration_record.to_dict())

            if decision == "stop":
                break

        final_score = records[-1].score if records else 0.0
        final_gate = verify_draft(draft, brief)
        status = self._final_status(final_score, final_gate, records,
                                    score_target=self.config.score_target)

        failures = [
            f"{report.agent_id}: {note}"
            for record in records for report in record.reports
            if report.status == AgentStatus.DEGRADED
            for note in report.notes
        ]
        draft_empty = not any(section.body.strip() for section in draft.sections)
        if draft_empty:
            # Prefer the provider/parse note over the generic "no usable draft"
            # trailer, which is appended last and used to hide the real cause.
            cause = next((item for item in failures if "no usable draft" not in item), None)
            if cause is None and failures:
                cause = failures[-1]
            detail = f" {cause}." if cause else ""
            stop_reason = (
                f"No article content was generated.{detail} {stop_reason or 'loop exited'}"
            ).strip()
        elif failures:
            agents = ", ".join(sorted({item.split(":", 1)[0] for item in failures}))
            stop_reason = f"{stop_reason or 'loop exited'}; degraded agent(s): {agents}"

        store.write_master_draft(draft)
        store.drop_notes()
        store.save_state()

        self.changelog.log_event("RUN_END", brief.task_id, {
            "status": status,
            "stop_reason": stop_reason or "loop exited",
            "iterations": len(records),
            "final_score": round(final_score, 4),
            "score_target": self.config.score_target,
            "tokens_total": tokens_total,
            "draft_words": draft.word_count,
            "final_draft_gate": "PASS" if final_gate.ok else f"FAIL ({len(final_gate.blockers)} blockers)",
            "master_draft": os.path.join(store.root, "master_draft.md"),
            "open_findings": len(store.open_findings),
        })

        self.events.emit(EventType.RUN_END, task_id=brief.task_id, status=status,
                         message=stop_reason or "loop exited",
                         final_score=final_score, iterations=len(records),
                         tokens_total=tokens_total, draft_words=draft.word_count,
                         draft_version=draft.version,
                         stop_reason=stop_reason or "loop exited",
                         open_findings=len(store.open_findings),
                         gate_ok=final_gate.ok)

        return RunResult(
            task_id=brief.task_id, status=status, iterations=records, gate=final_gate,
            final_score=final_score, final_draft=draft, workspace=store.root,
            tokens_total=tokens_total, stop_reason=stop_reason or "loop exited",
        )

    # -- one iteration ----------------------------------------------------
    def _run_iteration(self, iteration: int, brief: Brief, draft: Draft, store: MemoryStore,
                       previous_score: Optional[float]) -> tuple:
        reports: List[AgentReport] = []
        evicted: List[str] = []
        tokens_used = 0
        context_describe = ""

        self.events.emit(EventType.ITERATION_START, task_id=brief.task_id, iteration=iteration,
                         message=f"iteration {iteration}: {len(self.agents)} agents queued")

        for agent in self.agents:
            self.events.emit(EventType.AGENT_START, task_id=brief.task_id, iteration=iteration,
                             agent=agent.agent_id, message=f"{agent.role} is working")
            packet = store.build_packet(
                agent_id=agent.agent_id,
                brief=brief,
                draft=draft,
                rules_digest=self._digest,
                budget=self.config.budget,
                iteration=iteration,
                draft_required=agent.draft_required,
                peer_reports=reports,
            )
            evicted.extend(packet.evicted)
            store.write_text(iteration, f"context-{agent.agent_id}.txt", packet.text)

            if agent.agent_id != "synthesizer":
                report = agent.run(packet, brief, draft, iteration, peer_reports=reports,
                                   open_findings=store.open_findings)
                reports.append(report)
                store.write_json(iteration, f"report-{agent.agent_id}.json", report.to_dict())
                store.absorb_findings(report.findings)
                tokens_used += report.tokens_in + report.tokens_out
                self._emit_agent_done(agent, report, iteration, brief.task_id,
                                      len(packet.evicted), packet.tokens)
                continue

            report = agent.run(packet, brief, draft, iteration, peer_reports=reports,
                               open_findings=store.open_findings)
            reports.append(report)
            tokens_used += report.tokens_in + report.tokens_out

            # A degraded model call still has the Writer report in hand. Publish
            # that prose instead of discarding the iteration — it is not invented
            # text, and it is the only way an unusable synthesizer JSON cannot
            # wipe a finished article.
            fallback = agent.deterministic(
                brief, draft, iteration, reports, store.open_findings,
            )
            model_payload = {} if report.status == AgentStatus.DEGRADED else report.payload
            new_draft, recovered = publish_draft(
                model_payload, fallback, brief, draft.version, report.engine,
            )
            if new_draft is None:
                report.status = AgentStatus.DEGRADED
                report.notes.append(
                    "synthesizer returned no usable draft (expected outline sections "
                    "with article text); master draft unchanged"
                )
            else:
                if recovered:
                    report.payload = {
                        "sections": [section.to_dict() for section in new_draft.sections],
                        "title": new_draft.title,
                        "applied": list(fallback.get("applied") or [])
                        + ["synthesizer:published writer prose"],
                        "deferred": fallback.get("deferred") or [],
                        "rejected": fallback.get("rejected") or [],
                        "summary": fallback.get("summary") or report.summary,
                    }
                    report.summary = str(report.payload["summary"])
                    report.notes.append(
                        "model draft was unusable; published the Writer's section text"
                    )
                    if report.status == AgentStatus.DEGRADED:
                        report.status = AgentStatus.OK
                draft = new_draft
                self.events.emit(EventType.DRAFT_UPDATED, task_id=brief.task_id,
                                 iteration=iteration, agent=agent.agent_id,
                                 message=f"master draft now at version {draft.version}",
                                 version=draft.version, words=draft.word_count,
                                 sections=len(draft.sections),
                                 applied=len(report.payload.get("applied") or []),
                                 deferred=len(report.payload.get("deferred") or []))
            # Persist after validation so disk reports agree with events/status.
            store.write_json(iteration, "report-synthesizer.json", report.to_dict())
            self._emit_agent_done(agent, report, iteration, brief.task_id,
                                  len(packet.evicted), packet.tokens)

        # -- validate what the synthesizer produced ------------------------
        draft_gate = verify_draft(draft, brief)
        store.write_json(iteration, "draft_gate.json", draft_gate.to_dict())

        # Resolve findings the draft no longer exhibits.
        self._reconcile_open_findings(store, brief, draft, reports)

        score_card: ScoreCard = score_iteration(brief, draft, reports, draft_gate)
        synthesizer = next((r for r in reports if r.agent_id == "synthesizer"), None)
        applied = len(synthesizer.payload.get("applied") or []) if synthesizer else 0
        deferred = len(synthesizer.payload.get("deferred") or []) if synthesizer else 0

        context_describe = (
            f"cap={self.config.budget.context_hard_cap} "
            f"evicted={','.join(sorted(set(evicted))) or 'none'}"
        )

        record = IterationRecord(
            iteration=iteration,
            score=score_card.total,
            previous_score=previous_score,
            delta=0.0 if previous_score is None else score_card.total - previous_score,
            reports=reports,
            evicted=sorted(set(evicted)),
            tokens_total=tokens_used,
            decision="continue",
            applied=applied,
            deferred=deferred,
        )
        store.write_text(iteration, "draft.md", draft.to_markdown())
        self.changelog.log_iteration(brief.task_id, record, gate=draft_gate,
                                     context_describe=context_describe,
                                     draft_words=draft.word_count)

        self.events.emit(
            EventType.ITERATION_DONE, task_id=brief.task_id, iteration=iteration,
            message=f"iteration {iteration} scored {record.score:.4f}",
            score=record.score,
            previous_score=record.previous_score,
            delta=record.delta,
            components=score_card.components,
            findings=record.finding_count,
            applied=applied,
            deferred=deferred,
            tokens_total=tokens_used,
            draft_words=draft.word_count,
            draft_version=draft.version,
            evicted=record.evicted,
            gate_ok=draft_gate.ok,
            gate_blockers=len(draft_gate.blockers),
        )

        # Compress this iteration into the rolling digest - it is not retained in full.
        store.rolling.absorb(
            f"Iteration {iteration}: score {record.score:.4f} "
            f"({', '.join(f'{k}={v:.2f}' for k, v in score_card.components.items())}). "
            f"{record.finding_count} findings raised, {applied} applied, {deferred} deferred. "
            f"{findings_block(store.open_findings, limit=4).replace(chr(10), ' ')}"
        )
        store.save_state()

        return record, draft

    # -- helpers ----------------------------------------------------------
    def _emit_agent_done(self, agent, report: AgentReport, iteration: int, task_id: str,
                         evicted_count: int, context_tokens: int) -> None:
        """Report one agent's completion with the numbers a dashboard cares about."""
        self.events.emit(
            EventType.AGENT_DONE,
            task_id=task_id,
            iteration=iteration,
            agent=agent.agent_id,
            message=report.summary or f"{agent.role} finished",
            role=agent.role,
            status=report.status.value,
            engine=report.engine,
            findings=len(report.findings),
            blockers=len([f for f in report.findings if f.severity == Severity.BLOCKER]),
            tokens_in=report.tokens_in,
            tokens_out=report.tokens_out,
            context_tokens=context_tokens,
            evicted=evicted_count,
            summary=report.summary,
            notes=list(report.notes),
        )

    def _reconcile_open_findings(self, store: MemoryStore, brief: Brief, draft: Draft,
                                 reports: List[AgentReport]) -> None:
        """Drop open findings the current draft no longer reproduces.

        Re-running the deterministic checkers against the new draft is the only honest
        way to know a finding is resolved; assuming resolution because the synthesizer
        claimed a fix would let defects disappear from the record.
        """
        if not store.open_findings:
            return
        current: List[Finding] = []
        for agent in self.agents:
            if agent.agent_id in ("writer", "synthesizer"):
                continue
            try:
                payload = agent.deterministic(brief, draft, 0, [])
            except Exception:  # pragma: no cover - defensive
                continue
            for item in payload.get("findings") or []:
                try:
                    current.append(Finding.from_dict(item))
                except (KeyError, TypeError, ValueError):
                    continue

        current_keys = {f.key for f in current}
        # Keep only findings that are still reproducible, plus advisory notes.
        store.open_findings = [
            f for f in store.open_findings
            if f.key in current_keys or f.severity == Severity.NOTE or f.agent not in
            ("tester", "reviewer", "architect")
        ]

    @staticmethod
    def _final_status(score: float, gate: GateResult, records: List[IterationRecord],
                      score_target: float = 0.85) -> str:
        if gate.routed_back:
            return "NEEDS_HUMAN"
        if any(r.status == AgentStatus.DEGRADED for record in records for r in record.reports):
            return "NEEDS_HUMAN"
        return "PASS" if score >= score_target else "NEEDS_HUMAN"
