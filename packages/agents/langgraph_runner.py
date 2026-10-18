"""One-node LangGraph adapter for an individual deterministic model call.

Workflow control flow remains in the Postgres FSM. This graph has exactly one model
call node and one END edge; it contains no conditional/LLM-driven routing.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, TypedDict

from .config import ModelConfigurationError
from .litellm_client import ChatModelClient, ModelCompletion


class ModelCallState(TypedDict, total=False):
    system_prompt: str
    user_prompt: str
    agent_name: str
    timeout_seconds: int | None
    completion: ModelCompletion


class LangGraphModelCall:
    """Run a single provider call in a compiled, linear LangGraph subgraph."""

    def __init__(self, client: ChatModelClient) -> None:
        self.client = client
        self._graph: Any | None = None
        self._compile_lock = Lock()

    def invoke(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        agent_name: str,
        timeout_seconds: int | None = None,
    ) -> ModelCompletion:
        graph = self._compiled_graph()
        state: ModelCallState = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "agent_name": agent_name,
            "timeout_seconds": timeout_seconds,
        }
        result = graph.invoke(state)
        completion = result.get("completion")
        if not isinstance(completion, ModelCompletion):
            raise RuntimeError("LangGraph model-call node returned no ModelCompletion")
        return completion

    def _compiled_graph(self) -> Any:
        if self._graph is not None:
            return self._graph
        with self._compile_lock:
            if self._graph is not None:
                return self._graph
            try:
                from langgraph.graph import END, StateGraph
            except ImportError as exc:
                raise ModelConfigurationError(
                    "LangGraph is not installed; install the 'projectforge-agents' extra"
                ) from exc

            def invoke_model(state: ModelCallState) -> dict[str, ModelCompletion]:
                completion = self.client.complete(
                    system_prompt=state["system_prompt"],
                    user_prompt=state["user_prompt"],
                    agent_name=state["agent_name"],
                    timeout_seconds=state.get("timeout_seconds"),
                )
                return {"completion": completion}

            builder = StateGraph(ModelCallState)
            builder.add_node("model_call", invoke_model)
            builder.set_entry_point("model_call")
            builder.add_edge("model_call", END)
            self._graph = builder.compile()
            return self._graph
