"""Shared machinery for the five specialised sub-agents.

Every agent runs the same skeleton: assemble prompt -> call the configured LLM ->
normalise the result into an ``AgentReport``. Rule-based deterministic hooks remain
for isolated checker tests; they are not a substitute for model-generated writing.
Production orchestration fails before the agent loop if no LLM is configured.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..llm import LLMClient, ProviderError, ResponseParseError, parse_json, salvage_truncated_json
from ..memory import AssembledContext
from ..models import AgentReport, AgentStatus, Brief, Draft, Finding
from ..prompts import load_template, rules_digest
from ..tokens import TokenBudget, count_tokens
from .payload import sections_from_markdown


class BaseAgent:
    agent_id = "base"
    role = "base"
    #: Whether the master draft must survive in this agent's context even under pressure.
    draft_required = True

    def __init__(self, client: LLMClient, budget: TokenBudget, *,
                 prompt_dir: Optional[str] = None, rules_path: Optional[str] = None) -> None:
        self.client = client
        self.budget = budget
        self._template: Optional[str] = None
        self._digest: Optional[str] = None
        self._prompt_dir = prompt_dir
        self._rules_path = rules_path

    # -- prompt assembly --------------------------------------------------
    @property
    def template(self) -> str:
        if self._template is None:
            self._template = load_template(self.agent_id, self._prompt_dir)
        return self._template

    @property
    def digest(self) -> str:
        if self._digest is None:
            self._digest = rules_digest(self._rules_path)
        return self._digest

    def system_prompt(self) -> str:
        return (
            f"{self.digest}\n\n---\n\n{self.template}\n\n"
            "Output contract: reply with one JSON object only. Reasoning effort is low — "
            "do not spend the output budget on analysis, and do not emit think tags."
        )

    def output_token_limit(self, brief: Brief) -> int:
        """Ceiling for one completion. Not a spend target — the model stops early.

        The default Dahl model (GLM-5.3-Flash) thinks at maximum effort unless
        asked not to, and a 2k ``max_tokens`` is then consumed before any JSON
        is emitted. Leave room for that trace plus the article itself.
        """
        configured = self.budget.agent(self.agent_id).max_output_tokens
        if self.agent_id in ("writer", "synthesizer"):
            article = int(max(brief.target_words, 200) * 1.8) + 1200
            return min(12288, max(configured, article + 6000, 8192))
        return max(configured, 4096)

    # -- execution --------------------------------------------------------
    def should_skip(self, brief: Brief, draft: Draft, peer_reports: List[AgentReport],
                    open_findings: Optional[List[Finding]] = None) -> Optional[str]:
        """Return a human-readable skip reason when this agent has no work left.

        A skipped agent never reaches the provider - the cheapest token is the
        one never spent - and still produces a report so the change log, the
        event stream and the dashboard keep recording the mandatory order.
        Returns ``None`` when the agent must run.
        """
        return None

    def run(self, context: AssembledContext, brief: Brief, draft: Draft, iteration: int,
            peer_reports: Optional[List[AgentReport]] = None,
            open_findings: Optional[List[Finding]] = None) -> AgentReport:
        """Run one agent turn against an already-assembled context packet."""
        system = self.system_prompt()
        reports = peer_reports or []
        skip_reason = self.should_skip(brief, draft, reports, open_findings)
        if skip_reason:
            return self.base_report(
                iteration=iteration, status=AgentStatus.OK,
                notes=[f"skipped: {skip_reason}; zero tokens spent"],
                engine="skipped", tokens_in=0, tokens_out=0, findings=[],
                summary=f"skipped: {skip_reason}", payload={"skipped": skip_reason},
            )
        tokens_in = context.tokens + count_tokens(system)
        status = AgentStatus.OK
        notes: List[str] = []
        payload: Dict[str, Any] = {}
        engine = "heuristic"
        response_tokens_out = 0

        if context.evicted:
            notes.append(f"context evicted to meet cap: {', '.join(context.evicted)}")
        if context.truncated:
            notes.append(f"context truncated to meet cap: {', '.join(context.truncated)}")

        if context.breached_cap:
            status = AgentStatus.DEGRADED
            notes.append("Required context does not fit the token budget; no model call was made.")
            engine = "context:error"
            payload = {"_error": "required context does not fit", "sections": [], "findings": []}
        elif not self.client.available:
            status = AgentStatus.DEGRADED
            notes.append(
                f"LLM provider '{self.client.engine_name}' is not configured; "
                "no content was generated."
            )
            engine = f"{self.client.engine_name}:not-configured"
            payload = {"_error": "provider not configured", "sections": [], "findings": []}
        else:
            response = None
            try:
                response = self.client.complete(
                    system, context.text,
                    max_output_tokens=self.output_token_limit(brief),
                )
                engine = response.engine
                tokens_in = max(tokens_in, response.tokens_in)
                response_tokens_out = response.tokens_out
                try:
                    payload = parse_json(response.text)
                except ResponseParseError as exc:
                    repaired = salvage_truncated_json(response.text)
                    salvaged = []
                    if repaired is None and self.agent_id in ("writer", "synthesizer"):
                        salvaged = sections_from_markdown(response.text, brief)
                    if repaired is not None:
                        payload = repaired
                        notes.append("model output was truncated; recovered a partial JSON object")
                    elif salvaged:
                        payload = {
                            "sections": salvaged,
                            "summary": "extracted section bodies from a non-JSON model reply",
                        }
                        notes.append(
                            "model reply was not JSON; section bodies were extracted from headings"
                        )
                    else:
                        raise exc
                if not isinstance(payload, dict):
                    payload = {"payload": payload}
            except (ProviderError, ResponseParseError) as exc:
                # Never substitute deterministic prose or findings when a real
                # provider call fails; mark this agent degraded and preserve the
                # failure for the run report.
                status = AgentStatus.DEGRADED
                detail = str(exc)
                if response is not None and response.finish_reason == "length":
                    detail = f"{detail} (output truncated at {response.tokens_out} tokens)"
                notes.append(f"LLM call failed: {detail}")
                engine = f"{self.client.engine_name}:error"
                payload = {"_error": str(exc), "sections": [], "findings": []}

        report = self.compose(
            payload=payload,
            brief=brief,
            draft=draft,
            iteration=iteration,
            status=status,
            notes=notes,
            engine=engine,
            tokens_in=tokens_in,
            tokens_out=response_tokens_out or self._estimate_output_tokens(payload),
            peer_reports=peer_reports or [],
        )
        return report

    # -- subclass hooks ---------------------------------------------------
    def deterministic(self, brief: Brief, draft: Draft, iteration: int,
                      peer_reports: List[AgentReport],
                      open_findings: Optional[List[Finding]] = None) -> Dict[str, Any]:
        """Expose rule-based test logic; this is never a production LLM fallback."""
        raise NotImplementedError

    def compose(self, *, payload: Dict[str, Any], brief: Brief, draft: Draft, iteration: int,
                status: AgentStatus, notes: List[str], engine: str, tokens_in: int,
                tokens_out: int, peer_reports: List[AgentReport]) -> AgentReport:
        raise NotImplementedError

    # -- helpers ----------------------------------------------------------
    def _estimate_output_tokens(self, payload: Dict[str, Any]) -> int:
        import json

        return count_tokens(json.dumps(payload, ensure_ascii=False, default=str))

    def base_report(self, *, iteration: int, status: AgentStatus, notes: List[str], engine: str,
                    tokens_in: int, tokens_out: int, findings: List[Finding], summary: str,
                    payload: Dict[str, Any]) -> AgentReport:
        return AgentReport(
            agent_id=self.agent_id,
            role=self.role,
            iteration=iteration,
            status=status,
            findings=findings,
            summary=summary,
            payload=payload,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            engine=engine,
            notes=notes,
        )
