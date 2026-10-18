"""Hard-coded deterministic agent responses for Phase 2.

These stubs never call a model, write files, or execute code. ``test_failures`` is a
fault-injection seam for retry/escalation tests; it is not read from project input.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Mapping

from .results import AgentInvocationResult


STUB_AGENT_SCOPES: dict[str, tuple[str, ...]] = {
    "requirement": ("read_project_spec", "normalize_requirements"),
    "planner": ("read_task_graph", "sequence_tasks"),
    "coder": ("read_task", "retrieve_spec_context", "write_declared_paths", "commit_changes"),
    "tester": ("read_acceptance_criteria", "write_tests_only_if_missing", "sandbox_execute", "commit_changes"),
    "debugger": ("read_failed_test_output", "read_relevant_code", "propose_fix_only"),
    "reviewer": ("read_rubric_and_evidence", "emit_review_result"),
    "packager": ("read_audit_trail", "write_package_documentation", "commit_changes",
                 "emit_package_metadata_and_disclosure"),
}


@dataclass(frozen=True)
class StubResult(AgentInvocationResult):
    is_stub: bool = True


class StubAgentSuite:
    """Pure deterministic stand-ins for requirement/planning/build/review roles."""

    def __init__(self, *, test_failures: Mapping[str, int] | None = None) -> None:
        self._remaining_test_failures = dict(test_failures or {})
        if any(not isinstance(count, int) or count < 0
               for count in self._remaining_test_failures.values()):
            raise ValueError("test failure counts must be nonnegative integers")
        self._lock = Lock()

    def invoke(self, agent_name: str, payload: Mapping[str, Any]) -> StubResult:
        """Return a stable, auditable response without external side effects."""
        scope = list(payload.get("agent_scope", STUB_AGENT_SCOPES.get(agent_name, ())))
        if agent_name == "requirement":
            from projectforge_schemas.models import ProjectSpec

            source = payload.get("source_project_spec") or payload.get("spec")
            if source is not None:
                project_spec = ProjectSpec.model_validate(source).model_dump(mode="json")
            else:
                brief = str(payload.get("raw_brief_text", "Academic project brief"))
                project_spec = ProjectSpec.model_validate({
                    "title": brief.splitlines()[0][:300] or "Academic project",
                    "description": brief,
                    "deliverables": ["Implementation described by the submitted brief"],
                    "constraints": [],
                    "tech_stack_hints": [],
                    "rubric_items": [{
                        "criterion": "Required deliverable is present",
                        "type": "automatable",
                        "check_method": "Verify the named deliverable file and required output fields exist.",
                    }],
                }).model_dump(mode="json")
            return StubResult({
                "status": "completed",
                "confidence": 1.0,
                "project_spec": project_spec,
                "questions": [],
                "uncertainties": [],
                "evidence": ["Deterministic test stub; no external model was called."],
                "scope": scope,
                "stub": True,
            })
        if agent_name == "planner":
            from projectforge_schemas.models import ProjectSpec, TaskGraph

            seed_graph = payload.get("seed_graph")
            if seed_graph is not None:
                task_graph = TaskGraph.model_validate(seed_graph).model_dump(mode="json")
                for node in task_graph["nodes"]:
                    node["status"] = "pending"
                    node["retries"] = 0
            else:
                spec = ProjectSpec.model_validate(payload.get("spec", {}))
                deliverables = spec.deliverables
                nodes = []
                previous_id = None
                for index, deliverable in enumerate(deliverables, start=1):
                    node_id = f"deliverable-{index}"
                    nodes.append({
                        "id": node_id,
                        "description": f"Implement the required deliverable: {deliverable}",
                        "dependencies": [previous_id] if previous_id else [],
                        "acceptance_criteria": [
                            f"The {deliverable[:150]} artifact exists and contains the required, brief-defined output."
                        ],
                        "assigned_agent": "coder",
                        "read_set": [],
                        "write_set": [f"deliverables/{index}"],
                        "timeout_seconds": 300,
                    })
                    previous_id = node_id
                nodes.append({
                    "id": "verify-deliverables",
                    "description": "Verify all deliverables and rubric-linked requirements.",
                    "dependencies": [previous_id] if previous_id else [],
                    "acceptance_criteria": [
                        "A verification record lists every deliverable and its observable acceptance result."
                    ],
                    "assigned_agent": "coder",
                    "read_set": ["deliverables"],
                    "write_set": ["verification"],
                    "timeout_seconds": 300,
                })
                task_graph = TaskGraph.model_validate({"nodes": nodes}).model_dump(mode="json")
            return StubResult({
                "plan_summary": "Deterministic test plan over the supplied requirements.",
                "task_graph": task_graph,
                "evidence": ["Planner test stub produced a schema-valid DAG without an LLM."],
                "scope": scope,
                "stub": True,
            })
        if agent_name == "coder":
            return StubResult({
                "task_id": str(payload.get("task_id", "")),
                "phase": "CODE",
                "status": "stub_complete",
                "files_written": [],
                "code_executed": False,
                "evidence": ["Phase 2 coder stub; no repository files changed."],
                "scope": scope,
                "stub": True,
            })
        if agent_name == "tester":
            phase = str(payload.get("phase", "TEST"))
            if phase == "INTEGRATION":
                return StubResult({
                    "passed": True,
                    "phase": phase,
                    "code_executed": False,
                    "evidence": ["Integration stub accepted persisted task outputs; nothing executed."],
                    "scope": scope,
                    "stub": True,
                })
            task_id = str(payload.get("task_id", ""))
            with self._lock:
                failures_left = self._remaining_test_failures.get(task_id, 0)
                if failures_left > 0:
                    self._remaining_test_failures[task_id] = failures_left - 1
                    passed = False
                else:
                    passed = True
            return StubResult({
                "task_id": task_id,
                "phase": "TEST",
                "passed": passed,
                "code_executed": False,
                "evidence": [
                    "Deterministic stub verdict; no test process or project code was run."
                ],
                "scope": scope,
                "stub": True,
            })
        if agent_name == "debugger":
            return StubResult({
                "task_id": str(payload.get("task_id", "")),
                "status": "stub_repair_advice",
                "changes_made": False,
                "evidence": ["Phase 2 debugger stub; no files modified."],
                "scope": scope,
                "stub": True,
            })
        if agent_name == "reviewer":
            return StubResult({
                "passed": True,
                "reviewed_criteria": list(payload.get("criteria", [])),
                "evidence": ["Deterministic stub review; no model or code execution."],
                "scope": scope,
                "stub": True,
            })
        if agent_name == "packager":
            return StubResult({
                "status": "package_ready",
                "files_created": [],
                "disclosure": "Disclosure is assembled from persisted project agent-run records.",
                "evidence": ["Package metadata only; no project files were created."],
                "scope": scope,
                "stub": True,
            })
        raise ValueError(f"unsupported stub agent: {agent_name}")
