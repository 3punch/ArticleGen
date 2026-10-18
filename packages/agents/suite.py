"""Production suite with real Requirement, Planner, Coder, Tester, and Debugger workers."""

from __future__ import annotations

from decimal import Decimal
import os
from typing import Any, Mapping, Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from projectforge_api.orm import ProjectSpecRow
from projectforge_sandbox import SandboxManager
from projectforge_schemas.models import (
    AgentFileChange,
    ProjectSpec,
    ReviewerAgentResponse,
    TaskGraph,
    TaskNode,
)

from .build_agents import CoderAgent, DebuggerAgent, TesterAgent
from .config import ModelConfig
from .package_export import ProjectPackageManager
from .packager import PackagerAgent
from .reviewer import ReviewerAgent
from .errors import AgentOutputValidationError
from .git_workspace import CommitRecord, GitWorkspaceManager
from .litellm_client import ChatModelClient, LiteLLMChatClient
from .planner import PlannerAgent
from .rag import PgVectorContextRetriever
from .requirement import RequirementAgent
from .results import AgentInvocationResult
from .stubs import STUB_AGENT_SCOPES, StubAgentSuite


class ProjectForgeAgentSuite:
    """Real model-backed workers with filesystem writes and execution behind controls.

    The LLMs return structured proposals only. The deterministic suite validates path
    permissions, writes/commits through ``GitWorkspaceManager``, and runs tests only
    through the isolated ``SandboxManager``. Integration/review/packaging remain the
    existing stubs until their phases are explicitly implemented.
    """

    def __init__(
        self,
        *,
        config: ModelConfig | None = None,
        client: ChatModelClient | None = None,
        stubs: StubAgentSuite | None = None,
        session_factory: sessionmaker[Session] | None = None,
        workspace_manager: GitWorkspaceManager | None = None,
        sandbox_manager: SandboxManager | None = None,
        context_retriever: PgVectorContextRetriever | None = None,
    ) -> None:
        self.config = config or ModelConfig.from_env()
        model_client = client or LiteLLMChatClient(self.config)
        self.requirement_agent = RequirementAgent(client=model_client, config=self.config)
        self.planner_agent = PlannerAgent(client=model_client, config=self.config)
        self.coder_agent = CoderAgent(client=model_client, config=self.config)
        self.tester_agent = TesterAgent(client=model_client, config=self.config)
        self.debugger_agent = DebuggerAgent(client=model_client, config=self.config)
        self.reviewer_agent = ReviewerAgent(client=model_client, config=self.config)
        self.packager_agent = PackagerAgent(client=model_client, config=self.config)
        self.stubs = stubs or StubAgentSuite()
        self.session_factory = session_factory
        self.context_retriever = context_retriever or (
            PgVectorContextRetriever(session_factory) if session_factory is not None else None
        )
        self.workspace_manager = workspace_manager or GitWorkspaceManager(
            os.environ.get("PROJECTFORGE_REPO_ROOT", "/data/repos"),
        )
        self.sandbox_manager = sandbox_manager or SandboxManager(
            default_timeout_seconds=min(60, self.config.timeout_seconds),
            default_cpu_limit=1.0,
            default_memory_limit="512m",
            default_pids_limit=128,
            max_output_bytes=256 * 1024,
            network_allowlist=(),
        )
        self.package_manager = ProjectPackageManager(self.workspace_manager)
        self.reservation_tokens = self.config.reservation_tokens
        self.reservation_spend = self.config.reservation_spend
        self.reservation_ttl_seconds = self.config.reservation_ttl_seconds

    def prepare_payload(self, agent_name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve bounded task-local context once, before the durable budget reservation."""
        prepared = dict(payload)
        if agent_name in {"reviewer", "packager"}:
            project_id = self._project_id(prepared)
            self.workspace_manager.ensure_repository(project_id)
            prepared["repository_files"] = self.workspace_manager.read_repository_context(
                project_id, max_files=200, max_total_bytes=1_000_000,
            )
            return prepared
        if agent_name not in {"coder", "tester", "debugger"}:
            return prepared
        if agent_name == "tester" and prepared.get("phase") == "INTEGRATION":
            return prepared
        if self.session_factory is None or self.context_retriever is None:
            raise RuntimeError("build agents require PostgreSQL persistence and pgvector context storage")

        project_id = self._project_id(prepared)
        task = self._task_node(prepared)
        self.workspace_manager.ensure_repository(project_id)
        tech_stack_hints = self._load_tech_stack_hints(project_id)
        path_hints = [*task.read_set, *task.write_set]
        runtime = self.workspace_manager.detect_runtime(
            project_id,
            explicit_runtime=task.runtime,
            path_hints=path_hints,
            tech_stack_hints=tech_stack_hints,
        )
        task_data = task.model_dump(mode="json")
        task_data["runtime"] = runtime
        prepared.update({
            "project_id": str(project_id),
            "task_node": task_data,
            "runtime": runtime,
        })

        existing_test_files = self._tests_for_runtime(
            self.workspace_manager.discover_test_files(project_id), runtime,
        )
        if agent_name == "tester":
            prepared["existing_test_files"] = existing_test_files
            prepared["test_generation_required"] = not existing_test_files
            context = self.workspace_manager.read_context(
                project_id,
                read_scope=task.read_set,
                write_scope=task.write_set,
                extra_paths=existing_test_files,
                max_files=30,
                max_total_bytes=120_000,
            )
            prepared["repository_files"] = context
            return prepared

        context = self.workspace_manager.read_context(
            project_id,
            read_scope=task.read_set,
            write_scope=task.write_set,
            extra_paths=existing_test_files if agent_name == "debugger" else (),
            max_files=30,
            max_total_bytes=120_000,
        )
        prepared["repository_files"] = context
        if agent_name == "coder":
            query = " ".join([task.id, task.description, *task.acceptance_criteria])
            # A valid ProjectSpec always produces at least title, description,
            # deliverable, and rubric chunks. Three results therefore guarantee
            # that Coder never receives the whole specification.
            chunks = self.context_retriever.retrieve(project_id, query, top_k=3)
            if not chunks:
                self._ensure_spec_is_indexed(project_id)
                chunks = self.context_retriever.retrieve(project_id, query, top_k=3)
            prepared["relevant_spec_context"] = [
                chunk.model_dump(mode="json") for chunk in chunks
            ]
        return prepared

    def is_billable_call(self, agent_name: str, payload: Mapping[str, Any]) -> bool:
        """Reserve provider budget only when this invocation will actually call a model."""
        if agent_name in {"requirement", "planner", "coder", "debugger", "packager"}:
            return True
        if agent_name == "reviewer":
            spec = payload.get("project_spec") or payload.get("spec") or {}
            items = spec.get("rubric_items", []) if isinstance(spec, Mapping) else []
            return any(
                isinstance(item, Mapping) and item.get("type") == "human-judgment"
                for item in items
            )
        if agent_name == "tester":
            return payload.get("phase") != "INTEGRATION" and payload.get("test_generation_required") is True
        return False

    def estimate_call_tokens(self, agent_name: str, payload: Mapping[str, Any]) -> int:
        """Bound prompt bytes plus maximum output before making a billable call."""
        if agent_name == "requirement":
            system_prompt, user_prompt = self.requirement_agent.build_prompts(
                str(payload.get("raw_brief_text", "")),
                rubric_text=payload.get("rubric_text"),
                clarification_answers=payload.get("clarification_answers", {}),
                source_project_spec=payload.get("source_project_spec"),
            )
        elif agent_name == "planner":
            system_prompt, user_prompt = self.planner_agent.build_prompts(
                payload.get("spec", {}), plan_feedback=payload.get("plan_feedback", []),
            )
        elif agent_name == "coder":
            system_prompt, user_prompt = self.coder_agent.build_prompts(
                self._task_node(payload),
                relevant_spec_context=payload.get("relevant_spec_context", []),
                repository_files=payload.get("repository_files", {}),
                debugger_feedback=payload.get("debugger_feedback"),
            )
        elif agent_name == "tester":
            system_prompt, user_prompt = self.tester_agent.build_prompts(
                self._task_node(payload),
                runtime=str(payload.get("runtime", "python")),
                repository_files=payload.get("repository_files", {}),
            )
        elif agent_name == "debugger":
            system_prompt, user_prompt = self.debugger_agent.build_prompts(
                self._task_node(payload),
                test_failure=payload.get("test_failure", {}),
                repository_files=payload.get("repository_files", {}),
                retry_number=int(payload.get("retry_number", 1)),
            )
        elif agent_name == "reviewer":
            spec = ProjectSpec.model_validate(payload.get("project_spec", payload.get("spec", {})))
            human_items = [item for item in spec.rubric_items if item.type == "human-judgment"]
            if not human_items:
                return 1
            system_prompt, user_prompt = self.reviewer_agent.build_human_review_prompts(
                spec, human_items,
                repository_files=payload.get("repository_files", {}),
                test_runs=payload.get("test_runs", []),
            )
        elif agent_name == "packager":
            system_prompt, user_prompt = self.packager_agent.build_prompts(
                payload.get("project_spec", payload.get("spec", {})),
                payload.get("task_graph", {}),
                payload.get("review", {}),
                repository_files=payload.get("repository_files", {}),
                test_runs=payload.get("test_runs", []),
                final_gate_decision=payload.get("final_gate_decision", {}),
            )
        else:
            return 1
        input_bound = LiteLLMChatClient._count_prompt_tokens("", [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        return input_bound + self.config.max_output_tokens

    def invoke(self, agent_name: str, payload: Mapping[str, Any]) -> AgentInvocationResult:
        if agent_name == "requirement":
            return self._invoke_requirement(payload)
        if agent_name == "planner":
            return self._invoke_planner(payload)
        if agent_name == "coder":
            return self._invoke_coder(payload)
        if agent_name == "tester":
            if payload.get("phase") == "INTEGRATION":
                return self.stubs.invoke(agent_name, payload)
            return self._invoke_tester(payload)
        if agent_name == "debugger":
            return self._invoke_debugger(payload)
        if agent_name == "reviewer":
            return self._invoke_reviewer(payload)
        if agent_name == "packager":
            return self._invoke_packager(payload)
        return self.stubs.invoke(agent_name, payload)

    def _invoke_requirement(self, payload: Mapping[str, Any]) -> AgentInvocationResult:
        source_spec = payload.get("source_project_spec")
        spec = ProjectSpec.model_validate(source_spec) if source_spec is not None else None
        response, completion = self.requirement_agent.run(
            str(payload.get("raw_brief_text", "")),
            rubric_text=payload.get("rubric_text"),
            clarification_answers=payload.get("clarification_answers", {}),
            source_project_spec=spec,
            timeout_seconds=payload.get("agent_timeout_seconds"),
        )
        output = response.model_dump(mode="json")
        output.update({
            "provider": completion.provider,
            "model": completion.model,
            "prompt_version": "requirement_v1",
            "llm_calls": 1,
            "stub": False,
            "evidence": [
                "Requirement response was parsed and validated against the shared Pydantic contract.",
                "Rubric items were schema-validated as automatable or human-judgment.",
            ],
            "scope": list(STUB_AGENT_SCOPES["requirement"]),
        })
        return self._model_result(output, completion)

    def _invoke_planner(self, payload: Mapping[str, Any]) -> AgentInvocationResult:
        spec_data = payload.get("spec")
        if spec_data is None:
            raise ValueError("Planner agent requires a ProjectSpec")
        plan, completion = self.planner_agent.run(
            spec_data,
            plan_feedback=payload.get("plan_feedback", []),
            timeout_seconds=payload.get("agent_timeout_seconds"),
        )
        output = plan.model_dump(mode="json")
        output.update({
            "provider": completion.provider,
            "model": completion.model,
            "prompt_version": "planner_v1",
            "llm_calls": 1,
            "stub": False,
            "scope": list(STUB_AGENT_SCOPES["planner"]),
            "evidence": list(plan.evidence) + [
                "TaskGraph passed Pydantic node/edge, dependency, and acyclic-graph validation.",
                f"Planner produced {len(plan.task_graph.nodes)} tasks with concrete acceptance-criteria fields.",
            ],
        })
        return self._model_result(output, completion)

    def _invoke_coder(self, payload: Mapping[str, Any]) -> AgentInvocationResult:
        task = self._task_node(payload)
        response, completion = self.coder_agent.run(
            task,
            relevant_spec_context=payload.get("relevant_spec_context", []),
            repository_files=payload.get("repository_files", {}),
            debugger_feedback=payload.get("debugger_feedback"),
            timeout_seconds=payload.get("agent_timeout_seconds"),
        )
        commit = self.workspace_manager.apply_changes_and_commit(
            self._project_id(payload),
            task_id=task.id,
            agent_name="coder",
            changes=response.changes,
            write_scope=task.write_set,
            summary=response.summary,
        )
        output = {
            "task_id": task.id,
            "phase": "CODE",
            "status": "changes_committed" if commit.commit_hash else "no_content_change",
            "summary": response.summary,
            "files_written": [change.path for change in response.changes if change.action == "write"],
            "files_deleted": [change.path for change in response.changes if change.action == "delete"],
            "commit_hash": commit.commit_hash,
            "commit_message": commit.message,
            "changed_paths": list(commit.changed_paths),
            "runtime": payload.get("runtime", task.runtime or "python"),
            "prompt_version": self.coder_agent.prompt_version,
            "llm_calls": 1,
            "stub": False,
            "code_executed": False,
            "scope": list(payload.get("agent_scope", STUB_AGENT_SCOPES["coder"])),
            "evidence": list(response.evidence) + [
                f"Validated {len(response.changes)} file change(s) against the task write_set.",
                (f"Changes committed as {commit.commit_hash}." if commit.commit_hash else
                 "The proposed content matched the current repository; no empty Git commit was created."),
            ],
        }
        return self._model_result(output, completion)

    def _invoke_tester(self, payload: Mapping[str, Any]) -> AgentInvocationResult:
        task = self._task_node(payload)
        project_id = self._project_id(payload)
        runtime = str(payload.get("runtime", task.runtime or "python"))
        generated_commit: CommitRecord | None = None
        completion = None
        generation_evidence: list[str] = []
        if payload.get("test_generation_required") is True:
            response, completion = self.tester_agent.run(
                task,
                runtime=runtime,
                repository_files=payload.get("repository_files", {}),
                timeout_seconds=payload.get("agent_timeout_seconds"),
            )
            self._validate_test_filenames(response, runtime, completion)
            generated_commit = self.workspace_manager.apply_changes_and_commit(
                project_id,
                task_id=task.id,
                agent_name="tester",
                changes=response.test_files,
                write_scope=("tests",),
                summary=response.summary,
            )
            generation_evidence = list(response.evidence)
        test_files = self._tests_for_runtime(
            self.workspace_manager.discover_test_files(project_id), runtime,
        )
        if not test_files:
            logs = f"No executable {runtime} test files exist after test discovery."
            result = None
            passed = False
            execution_output = {
                "stdout": "",
                "stderr": logs,
                "exit_code": 1,
                "timed_out": False,
                "duration_seconds": 0.0,
                "output_truncated": False,
                "container_id": None,
            }
            command = []
        else:
            command = self._test_command(runtime)
            result = self.sandbox_manager.run(
                runtime,
                command,
                project_dir=self.workspace_manager.project_path(project_id),
                timeout_seconds=min(
                    int(payload.get("agent_timeout_seconds", payload.get("timeout_seconds", task.timeout_seconds))),
                    task.timeout_seconds,
                ),
                network_allowlist=(),
            )
            passed = result.exit_code == 0 and not result.timed_out
            execution_output = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "duration_seconds": result.duration_seconds,
                "output_truncated": result.output_truncated,
                "container_id": result.container_id,
            }
        evidence = generation_evidence + [
            f"Isolated {runtime} sandbox executed {' '.join(command) if command else 'test discovery'}.",
            (f"{len(test_files)} discovered test file(s): {', '.join(test_files[:30])}." if test_files else
             "No executable test files were discovered."),
            ("Sandbox returned exit code 0 without timing out." if passed else
             f"Sandbox reported exit code {execution_output['exit_code']} (timed_out={execution_output['timed_out']})."),
        ]
        output = {
            "task_id": task.id,
            "phase": "TEST",
            "passed": passed,
            "runtime": runtime,
            "command": command,
            "tested_files": test_files,
            "generated_test_files": list(generated_commit.changed_paths) if generated_commit else [],
            "test_commit_hash": generated_commit.commit_hash if generated_commit else None,
            "stdout": execution_output["stdout"],
            "stderr": execution_output["stderr"],
            "exit_code": execution_output["exit_code"],
            "timed_out": execution_output["timed_out"],
            "duration_seconds": execution_output["duration_seconds"],
            "output_truncated": execution_output["output_truncated"],
            "container_id": execution_output["container_id"],
            "code_executed": result is not None,
            "code_execution_count": int(result is not None),
            "test_generation_performed": completion is not None,
            "prompt_version": self.tester_agent.prompt_version if completion else None,
            "llm_calls": 1 if completion else 0,
            "stub": False,
            "scope": list(payload.get("agent_scope", STUB_AGENT_SCOPES["tester"])),
            "evidence": evidence,
        }
        if completion is None:
            return AgentInvocationResult(
                output=output,
                tokens_used=0,
                cost=Decimal("0"),
                llm_calls=0,
                is_stub=False,
                usage_known=True,
            )
        output.update({"provider": completion.provider, "model": completion.model})
        return self._model_result(output, completion)

    def _invoke_debugger(self, payload: Mapping[str, Any]) -> AgentInvocationResult:
        task = self._task_node(payload)
        response, completion = self.debugger_agent.run(
            task,
            test_failure=payload.get("test_failure", {}),
            repository_files=payload.get("repository_files", {}),
            retry_number=int(payload.get("retry_number", 1)),
            timeout_seconds=payload.get("agent_timeout_seconds"),
        )
        output = {
            **response.model_dump(mode="json"),
            "task_id": task.id,
            "phase": "DEBUG",
            "retry_number": int(payload.get("retry_number", 1)),
            "changes_made": False,
            "code_executed": False,
            "prompt_version": self.debugger_agent.prompt_version,
            "llm_calls": 1,
            "stub": False,
            "scope": list(payload.get("agent_scope", STUB_AGENT_SCOPES["debugger"])),
            "evidence": list(response.evidence) + [
                "Debugger returned repair guidance only; the next Coder attempt owns all file writes."
            ],
        }
        return self._model_result(output, completion)

    def _invoke_reviewer(self, payload: Mapping[str, Any]) -> AgentInvocationResult:
        spec = ProjectSpec.model_validate(payload.get("project_spec", payload.get("spec", {})))
        response, completion = self.reviewer_agent.run(
            spec,
            repository_files=payload.get("repository_files", {}),
            test_runs=payload.get("test_runs", []),
            timeout_seconds=payload.get("agent_timeout_seconds"),
        )
        output = {
            **response.model_dump(mode="json"),
            "passed": response.automatable_checks_passed,
            "phase": "FINAL_REVIEW",
            "prompt_version": self.reviewer_agent.prompt_version,
            "llm_calls": 1 if completion is not None else 0,
            "stub": False,
            "scope": list(payload.get("agent_scope", STUB_AGENT_SCOPES["reviewer"])),
            "evidence": response.evidence,
        }
        if completion is None:
            return AgentInvocationResult(
                output=output, tokens_used=0, cost=Decimal("0"),
                llm_calls=0, is_stub=False, usage_known=True,
            )
        output.update({"provider": completion.provider, "model": completion.model})
        return self._model_result(output, completion)

    def _invoke_packager(self, payload: Mapping[str, Any]) -> AgentInvocationResult:
        project_id = self._project_id(payload)
        project_spec = ProjectSpec.model_validate(
            payload.get("project_spec", payload.get("spec", {}))
        )
        graph = TaskGraph.model_validate(payload.get("task_graph", {}))
        review_payload = payload.get("review", {})
        if not isinstance(review_payload, Mapping):
            raise ValueError("Packager requires a structured Reviewer result")
        review = ReviewerAgentResponse.model_validate({
            key: review_payload[key]
            for key in (
                "summary", "criteria", "automatable_checks_passed",
                "human_review_required", "evidence",
            )
            if key in review_payload
        })
        test_runs = payload.get("test_runs", [])
        final_gate = payload.get("final_gate_decision", {})
        response, completion = self.packager_agent.run(
            project_spec,
            graph,
            review,
            repository_files=payload.get("repository_files", {}),
            test_runs=test_runs,
            final_gate_decision=final_gate,
            timeout_seconds=payload.get("agent_timeout_seconds"),
        )
        automated = [item for item in review.criteria if item.rubric_item.type == "automatable"]
        human = [item for item in review.criteria if item.rubric_item.type == "human-judgment"]
        gate_decision = str(final_gate.get("decision", "approve"))
        report_markdown = response.report_markdown.rstrip() + "\n\n## Deterministic review record\n\n"
        report_markdown += (
            f"- Automatable checks: {sum(item.status == 'passed' for item in automated)} of "
            f"{len(automated)} recorded as passed by the Reviewer Agent.\n"
        )
        report_markdown += (
            f"- Human final gate: `{gate_decision}`; comment: "
            f"{str(final_gate.get('comment', '') or 'none supplied').replace(chr(10), ' ')[:1000]}\n"
        )
        if human:
            report_markdown += (
                "- Human-judgment rubric items remain recorded as `human_review_required` with "
                "their evidence and Reviewer recommendations; the final packaging gate does "
                "not rewrite them as automated passes.\n"
            )
            for item in human:
                report_markdown += (
                    f"  - {item.rubric_item.criterion}: human review required; "
                    f"Reviewer recommendation: `{item.recommendation}`.\n"
                )
        report_markdown += (
            "\n## Sandbox test evidence\n\n"
            f"{sum(run.get('passed') is True for run in test_runs)} of {len(test_runs)} "
            "persisted TEST-phase sandbox runs passed. See `reports/TEST_RESULTS.json` in the bundle.\n"
        )
        changes = [
            AgentFileChange(path="README.md", content=response.readme_markdown.rstrip() + "\n"),
            AgentFileChange(path="reports/PROJECT_REPORT.md", content=report_markdown),
        ]
        workflow_id = str(payload.get("workflow_id", ""))
        safe_suffix = "".join(character for character in workflow_id if character.isalnum())[:16]
        commit = self.workspace_manager.apply_changes_and_commit(
            project_id,
            task_id=f"package-{safe_suffix or 'final'}",
            agent_name="packager",
            changes=changes,
            write_scope=("README.md", "reports"),
            summary=response.summary,
        )
        output = {
            "phase": "PACKAGE_DOCUMENTS",
            "summary": response.summary,
            "readme_path": "README.md",
            "report_path": "reports/PROJECT_REPORT.md",
            "documentation_commit_hash": commit.commit_hash,
            "changed_paths": list(commit.changed_paths),
            "prompt_version": self.packager_agent.prompt_version,
            "llm_calls": 1,
            "code_executed": False,
            "code_execution_count": 0,
            "stub": False,
            "scope": list(payload.get("agent_scope", STUB_AGENT_SCOPES["packager"])),
            "evidence": [
                *response.evidence,
                "README.md and reports/PROJECT_REPORT.md committed to the project Git workspace.",
                (f"Documentation commit: {commit.commit_hash}." if commit.commit_hash else
                 "Generated documentation matched the current files; no empty Git commit was created."),
                "Human-judgment criteria remain labeled for human review; no model verdict was recorded as a pass.",
            ],
        }
        output.update({"provider": completion.provider, "model": completion.model})
        return self._model_result(output, completion)

    def _load_tech_stack_hints(self, project_id: UUID) -> list[str]:
        assert self.session_factory is not None
        with self.session_factory() as session:
            spec = session.get(ProjectSpecRow, project_id)
            return list(spec.tech_stack_hints or []) if spec is not None else []

    def _ensure_spec_is_indexed(self, project_id: UUID) -> None:
        assert self.session_factory is not None and self.context_retriever is not None
        with self.session_factory() as session, session.begin():
            spec_row = session.scalar(
                select(ProjectSpecRow).where(ProjectSpecRow.project_id == project_id)
            )
            if spec_row is None:
                raise RuntimeError("Coder cannot retrieve context because the project has no persisted ProjectSpec")
            spec = ProjectSpec.model_validate({
                "title": spec_row.title,
                "description": spec_row.description,
                "deliverables": spec_row.deliverables,
                "constraints": spec_row.constraints,
                "tech_stack_hints": spec_row.tech_stack_hints,
                "rubric_items": spec_row.rubric_items,
            })
            self.context_retriever.replace_project_spec(session, project_id, spec)

    @staticmethod
    def _task_node(payload: Mapping[str, Any]) -> TaskNode:
        task_data = payload.get("task_node")
        if isinstance(task_data, TaskNode):
            return task_data
        if isinstance(task_data, Mapping):
            return TaskNode.model_validate(task_data)
        required = {
            "id": payload.get("task_id"),
            "description": payload.get("description"),
            "acceptance_criteria": payload.get("acceptance_criteria"),
            "assigned_agent": "coder",
            "read_set": payload.get("read_set", []),
            "write_set": payload.get("write_set", []),
            "runtime": payload.get("runtime"),
            "timeout_seconds": payload.get("timeout_seconds", 300),
        }
        return TaskNode.model_validate(required)

    @staticmethod
    def _project_id(payload: Mapping[str, Any]) -> UUID:
        try:
            return UUID(str(payload["project_id"]))
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("build-agent payload must include a valid project_id") from exc

    @staticmethod
    def _tests_for_runtime(paths: Sequence[str], runtime: str) -> list[str]:
        selected: list[str] = []
        for path in paths:
            name = os.path.basename(path).casefold()
            suffix = os.path.splitext(name)[1]
            if runtime == "python" and suffix == ".py" and (
                name.startswith("test_") or name.endswith("_test.py")
            ):
                selected.append(path)
            elif runtime == "node" and suffix in {".js", ".jsx", ".cjs", ".mjs", ".ts", ".tsx"} and (
                ".test." in name or ".spec." in name
            ):
                selected.append(path)
        return sorted(selected)

    @staticmethod
    def _validate_test_filenames(response: Any, runtime: str, completion: Any) -> None:
        invalid = []
        for change in response.test_files:
            name = os.path.basename(change.path).casefold()
            suffix = os.path.splitext(name)[1]
            valid = (
                runtime == "python" and suffix == ".py"
                and (name.startswith("test_") or name.endswith("_test.py"))
            ) or (
                runtime == "node" and suffix in {".js", ".jsx", ".cjs", ".mjs", ".ts", ".tsx"}
                and (".test." in name or ".spec." in name)
            )
            if not valid:
                invalid.append(change.path)
        if invalid:
            raise AgentOutputValidationError(
                f"Tester generated filenames not discovered by the {runtime} runner: {invalid}",
                completion=completion,
            )

    @staticmethod
    def _test_command(runtime: str) -> list[str]:
        if runtime == "python":
            return ["pytest", "-q"]
        if runtime == "node":
            return ["jest", "--runInBand", "--color=false"]
        raise ValueError(f"unsupported sandbox runtime: {runtime}")

    @staticmethod
    def _model_result(output: dict[str, Any], completion: Any) -> AgentInvocationResult:
        return AgentInvocationResult(
            output=output,
            tokens_used=completion.tokens_used,
            cost=completion.cost,
            provider=completion.provider,
            model=completion.model,
            llm_calls=1,
            is_stub=False,
            usage_known=completion.tokens_used is not None and completion.cost is not None,
        )
