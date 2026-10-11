"""Shared machinery for the five specialised sub-agents.

Every agent runs the same skeleton: assemble prompt -> call the engine (LLM if a
provider is configured, deterministic engine otherwise) -> normalise the result into
an ``AgentReport``. Subclasses only supply the deterministic engine and the mapping
from payload to findings.

The split matters for verification: when no API key exists the loop still runs end to
end, and its behaviour is checkable by tests rather than dependent on a model call.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..llm import LLMClient, ProviderError, ResponseParseError, parse_json
from ..memory import AssembledContext
from ..models import AgentReport, AgentStatus, Brief, Draft, Finding
from ..prompts import load_template, rules_digest
from ..tokens import TokenBudget, count_tokens


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
        return f"{self.digest}\n\n---\n\n{self.template}"

    # -- execution --------------------------------------------------------
    def run(self, context: AssembledContext, brief: Brief, draft: Draft, iteration: int,
            peer_reports: Optional[List[AgentReport]] = None) -> AgentReport:
        """Run one agent turn against an already-assembled context packet."""
        system = self.system_prompt()
        tokens_in = context.tokens + count_tokens(system)
        agent_budget = self.budget.agent(self.agent_id)
        status = AgentStatus.OK
        notes: List[str] = []
        payload: Dict[str, Any] = {}
        engine = "heuristic"
        response_tokens_out = 0

        if context.evicted:
            notes.append(f"context evicted to meet cap: {', '.join(context.evicted)}")
        if context.truncated:
            notes.append(f"context truncated to meet cap: {', '.join(context.truncated)}")

        if self.client.available:
            try:
                response = self.client.complete(system, context.text,
                                                max_output_tokens=agent_budget.max_output_tokens)
                payload = parse_json(response.text)
                if not isinstance(payload, dict):
                    payload = {"payload": payload}
                engine = response.engine
                tokens_in = max(tokens_in, response.tokens_in)
                response_tokens_out = response.tokens_out
            except (ProviderError, ResponseParseError) as exc:
                # Degrade loudly rather than inventing a result.
                status = AgentStatus.DEGRADED
                notes.append(f"provider unavailable/unparseable, fell back to deterministic engine: {exc}")
                engine = "heuristic"
                payload = self.deterministic(brief, draft, iteration, peer_reports or [])
        else:
            engine = "offline-heuristic"
            payload = self.deterministic(brief, draft, iteration, peer_reports or [])

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
                      peer_reports: List[AgentReport]) -> Dict[str, Any]:
        """Produce the same payload an LLM would, using rule-based analysis."""
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
