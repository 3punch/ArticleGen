"""FastAPI persistence and workflow API for ProjectForge's Phase 3 FSM agents."""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timezone
from typing import Generator, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.responses import FileResponse
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from projectforge_api.db import build_engine, configured_database_url, make_session_factory
from projectforge_api.orm import (
    AgentRunRow,
    Base,
    BudgetLedgerRow,
    Project,
    ProjectSpecRow,
    TaskEdgeRow,
    TaskGraphRow,
    TaskNodeRow,
    WorkflowRunRow,
)
from projectforge_schemas.models import (
    BudgetLedger,
    ProjectCreate,
    ProjectRecord,
    ProjectSpec,
    ProjectUpdate,
    TaskGraph,
    TaskGraphRecord,
    WorkflowApprovalRequest,
    WorkflowCancelRequest,
    WorkflowClarificationRequest,
    WorkflowEventRecord,
    WorkflowPackageExport,
    WorkflowReviewBundle,
    WorkflowRunRecord,
    WorkflowTaskLogRecord,
    WorkflowStartRequest,
)
from projectforge_orchestrator.engine import (
    InvalidWorkflowTransition,
    PostgresFSMEngine,
    WorkflowCancelled,
    WorkflowEngine,
)
from projectforge_orchestrator.queue import RQDispatcher
from projectforge_agents.git_workspace import GitWorkspaceManager
from projectforge_agents.package_export import PackageExportError
from projectforge_agents.stubs import StubAgentSuite
from projectforge_agents.suite import ProjectForgeAgentSuite


def _project_record(project: Project) -> ProjectRecord:
    spec = project.spec
    budget = project.budget
    spec_record = None if spec is None else ProjectSpec(
        title=spec.title,
        description=spec.description,
        deliverables=spec.deliverables,
        constraints=spec.constraints,
        tech_stack_hints=spec.tech_stack_hints,
        rubric_items=spec.rubric_items,
    )
    return ProjectRecord(
        id=project.id,
        name=project.name,
        spec=spec_record,
        raw_brief_text=project.raw_brief_text,
        rubric_text=project.rubric_text,
        budget=BudgetLedger(
            project_id=budget.project_id,
            token_cap=budget.token_cap,
            spend_cap=budget.spend_cap,
            used_tokens=budget.used_tokens,
            used_spend=budget.used_spend,
            reserved_tokens=budget.reserved_tokens,
            reserved_spend=budget.reserved_spend,
        ),
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


def _graph_record(session: Session, row: TaskGraphRow) -> TaskGraphRecord:
    node_rows = session.scalars(
        select(TaskNodeRow).where(TaskNodeRow.graph_id == row.id).order_by(TaskNodeRow.id)
    ).all()
    edge_rows = session.scalars(
        select(TaskEdgeRow).where(TaskEdgeRow.graph_id == row.id)
        .order_by(TaskEdgeRow.source_id, TaskEdgeRow.target_id)
    ).all()
    graph = TaskGraph(
        nodes=[{
            "id": node.id,
            "description": node.description,
            "dependencies": node.dependencies,
            "status": node.status,
            "acceptance_criteria": node.acceptance_criteria,
            "retries": node.retries,
            "assigned_agent": node.assigned_agent,
            "read_set": node.read_set,
            "write_set": node.write_set,
            "runtime": node.runtime,
            "timeout_seconds": node.timeout_seconds,
            "review_feedback": node.review_feedback or [],
        } for node in node_rows],
        edges=[{"source": edge.source_id, "target": edge.target_id} for edge in edge_rows],
    )
    return TaskGraphRecord(id=row.id, project_id=row.project_id, version=row.version,
                           graph=graph, created_at=row.created_at)


def _insert_graph(session: Session, row: TaskGraphRow, graph: TaskGraph) -> None:
    for node in graph.nodes:
        session.add(TaskNodeRow(
            graph_id=row.id,
            id=node.id,
            description=node.description,
            dependencies=node.dependencies,
            status=node.status,
            acceptance_criteria=node.acceptance_criteria,
            retries=node.retries,
            assigned_agent=node.assigned_agent,
            read_set=node.read_set,
            write_set=node.write_set,
            runtime=node.runtime,
            timeout_seconds=node.timeout_seconds,
            review_feedback=node.review_feedback,
        ))
    session.flush()
    for edge in graph.edges:
        session.add(TaskEdgeRow(graph_id=row.id, source_id=edge.source, target_id=edge.target))


def create_app(*, database_url: Optional[str] = None,
               create_schema_for_tests: bool = False,
               workflow_engine: Optional[WorkflowEngine] = None,
               queue_dispatcher: Optional[object] = None,
               agent_suite: Optional[object] = None,
               workspace_manager: Optional[GitWorkspaceManager] = None) -> FastAPI:
    """Build the app with injectable DB configuration for API tests.

    Normal deployments must apply Alembic migrations. ``create_schema_for_tests`` is
    deliberately opt-in and exists only to make the SQLite CRUD tests self-contained.
    """
    resolved_url = configured_database_url(database_url)
    engine = build_engine(resolved_url) if resolved_url else None
    session_factory = make_session_factory(engine) if engine is not None else None
    default_agent_suite = agent_suite or (
        StubAgentSuite() if create_schema_for_tests else
        ProjectForgeAgentSuite(session_factory=session_factory)
    )
    resolved_workspace_manager = workspace_manager
    if resolved_workspace_manager is None and not create_schema_for_tests:
        resolved_workspace_manager = getattr(default_agent_suite, "workspace_manager", None)
        if resolved_workspace_manager is None:
            resolved_workspace_manager = GitWorkspaceManager(
                os.environ.get("PROJECTFORGE_REPO_ROOT", "/data/repos"),
            )
    resolved_workflow_engine = workflow_engine or (
        PostgresFSMEngine(session_factory, agents=default_agent_suite)
        if session_factory is not None else None
    )
    redis_url = os.environ.get("PROJECTFORGE_REDIS_URL")
    resolved_queue_dispatcher = queue_dispatcher or (
        RQDispatcher(
            redis_url,
            queue_name=os.environ.get("PROJECTFORGE_RQ_QUEUE", "projectforge"),
        ) if redis_url else None
    )

    if create_schema_for_tests and engine is not None:
        Base.metadata.create_all(engine)

    app = FastAPI(
        title="ProjectForge API",
        version="0.1.0",
        description=("Postgres-backed deterministic FSM with human plan/final gates, "
                     "evidence-bearing Reviewer and Packager agents, sandboxed build tests, "
                     "Git-audited changes, AI-usage disclosures, and downloadable ZIP exports."),
    )
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.workflow_engine = resolved_workflow_engine
    app.state.queue_dispatcher = resolved_queue_dispatcher
    app.state.workspace_manager = resolved_workspace_manager
    app.state.package_manager = getattr(
        resolved_workflow_engine, "package_manager",
        getattr(default_agent_suite, "package_manager", None),
    )

    def get_db() -> Generator[Session, None, None]:
        factory = app.state.session_factory
        if factory is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PROJECTFORGE_DATABASE_URL is not configured",
            )
        with factory() as session:
            yield session

    @app.get("/health")
    def health() -> dict[str, str]:
        if app.state.engine is None:
            return {"status": "not-ready", "database": "not-configured"}
        try:
            with app.state.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError:
            raise HTTPException(status_code=503, detail="database is unreachable") from None
        return {"status": "ok", "database": "reachable"}

    @app.post("/api/projects", response_model=ProjectRecord,
              status_code=status.HTTP_201_CREATED)
    def create_project(body: ProjectCreate, session: Session = Depends(get_db)):
        project_id = uuid.uuid4()
        project = Project(
            id=project_id,
            name=body.name,
            raw_brief_text=body.raw_brief_text,
            rubric_text=body.rubric_text,
        )
        if body.spec is not None:
            spec_data = body.spec.model_dump(mode="json")
            project.spec = ProjectSpecRow(
                project_id=project_id,
                title=spec_data["title"],
                description=spec_data["description"],
                deliverables=spec_data["deliverables"],
                constraints=spec_data["constraints"],
                tech_stack_hints=spec_data["tech_stack_hints"],
                rubric_items=spec_data["rubric_items"],
            )
        project.budget = BudgetLedgerRow(
            project_id=project_id,
            token_cap=body.token_cap,
            spend_cap=body.spend_cap,
            used_tokens=0,
            used_spend=0,
        )
        session.add(project)
        try:
            session.flush()
            if app.state.workspace_manager is not None:
                app.state.workspace_manager.ensure_repository(project_id)
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise HTTPException(status_code=409, detail="project could not be created") from exc
        except Exception as exc:
            session.rollback()
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "project workspace could not be initialized",
                    "error": type(exc).__name__,
                },
            ) from None
        session.refresh(project)
        if project.spec is not None:
            session.refresh(project.spec)
        session.refresh(project.budget)
        return _project_record(project)

    @app.get("/api/projects", response_model=list[ProjectRecord])
    def list_projects(limit: int = Query(default=50, ge=1, le=100),
                      offset: int = Query(default=0, ge=0),
                      session: Session = Depends(get_db)):
        projects = session.scalars(
            select(Project).order_by(Project.created_at.desc()).limit(limit).offset(offset)
        ).all()
        return [_project_record(project) for project in projects]

    @app.get("/api/projects/{project_id}", response_model=ProjectRecord)
    def get_project(project_id: uuid.UUID, session: Session = Depends(get_db)):
        project = session.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return _project_record(project)

    @app.patch("/api/projects/{project_id}", response_model=ProjectRecord)
    def update_project(project_id: uuid.UUID, body: ProjectUpdate,
                       session: Session = Depends(get_db)):
        project = session.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        supplied = body.model_fields_set
        if "name" in supplied:
            if body.name is None:
                raise HTTPException(status_code=422, detail="name may not be null")
            project.name = body.name
        if "spec" in supplied:
            if body.spec is None:
                raise HTTPException(status_code=422, detail="spec may not be null")
            spec = body.spec.model_dump(mode="json")
            if project.spec is None:
                project.spec = ProjectSpecRow(project_id=project_id, **spec)
            else:
                project.spec.title = spec["title"]
                project.spec.description = spec["description"]
                project.spec.deliverables = spec["deliverables"]
                project.spec.constraints = spec["constraints"]
                project.spec.tech_stack_hints = spec["tech_stack_hints"]
                project.spec.rubric_items = spec["rubric_items"]
        if "raw_brief_text" in supplied:
            if body.raw_brief_text is None:
                raise HTTPException(status_code=422, detail="raw_brief_text may not be null")
            project.raw_brief_text = body.raw_brief_text
        if "rubric_text" in supplied:
            if body.rubric_text is None:
                raise HTTPException(status_code=422, detail="rubric_text may not be null")
            project.rubric_text = body.rubric_text
        if "token_cap" in supplied:
            if body.token_cap is None:
                raise HTTPException(status_code=422, detail="token_cap may not be null")
            project.budget.token_cap = body.token_cap
        if "spend_cap" in supplied:
            if body.spend_cap is None:
                raise HTTPException(status_code=422, detail="spend_cap may not be null")
            project.budget.spend_cap = body.spend_cap
        project.updated_at = datetime.now(timezone.utc)
        session.commit()
        session.refresh(project)
        if project.spec is not None:
            session.refresh(project.spec)
        session.refresh(project.budget)
        return _project_record(project)

    @app.delete("/api/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_project(project_id: uuid.UUID, session: Session = Depends(get_db)):
        project = session.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        has_workflows = session.scalar(
            select(WorkflowRunRow.id).where(WorkflowRunRow.project_id == project_id).limit(1)
        )
        if has_workflows is not None:
            raise HTTPException(status_code=409,
                                detail="project is retained because it has workflow audit records")
        session.delete(project)
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise HTTPException(status_code=409,
                                detail="project is retained by persisted workflow history") from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/api/projects/{project_id}/task-graphs", response_model=TaskGraphRecord,
              status_code=status.HTTP_201_CREATED)
    def create_task_graph(project_id: uuid.UUID, graph: TaskGraph,
                          session: Session = Depends(get_db)):
        if session.get(Project, project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        version = session.scalar(
            select(func.coalesce(func.max(TaskGraphRow.version), 0))
            .where(TaskGraphRow.project_id == project_id)
        ) + 1
        row = TaskGraphRow(project_id=project_id, version=version)
        session.add(row)
        session.flush()
        _insert_graph(session, row, graph)
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise HTTPException(status_code=409, detail="task graph conflicts with persisted state") from exc
        session.refresh(row)
        return _graph_record(session, row)

    @app.get("/api/projects/{project_id}/task-graphs", response_model=list[TaskGraphRecord])
    def list_task_graphs(project_id: uuid.UUID,
                         session: Session = Depends(get_db)):
        if session.get(Project, project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        rows = session.scalars(
            select(TaskGraphRow).where(TaskGraphRow.project_id == project_id)
            .order_by(TaskGraphRow.version)
        ).all()
        return [_graph_record(session, row) for row in rows]

    @app.get("/api/projects/{project_id}/task-graphs/{graph_id}",
             response_model=TaskGraphRecord)
    def get_task_graph(project_id: uuid.UUID, graph_id: uuid.UUID,
                       session: Session = Depends(get_db)):
        row = session.scalar(select(TaskGraphRow).where(
            TaskGraphRow.project_id == project_id, TaskGraphRow.id == graph_id,
        ))
        if row is None:
            raise HTTPException(status_code=404, detail="task graph not found")
        return _graph_record(session, row)

    @app.put("/api/projects/{project_id}/task-graphs/{graph_id}",
             response_model=TaskGraphRecord)
    def replace_task_graph(project_id: uuid.UUID, graph_id: uuid.UUID, graph: TaskGraph,
                           session: Session = Depends(get_db)):
        row = session.scalar(select(TaskGraphRow).where(
            TaskGraphRow.project_id == project_id, TaskGraphRow.id == graph_id,
        ).with_for_update())
        if row is None:
            raise HTTPException(status_code=404, detail="task graph not found")
        if session.scalar(select(WorkflowRunRow.id).where(
                WorkflowRunRow.task_graph_id == graph_id).limit(1)) is not None:
            raise HTTPException(status_code=409,
                                detail="task graph is immutable after a workflow references it")
        next_version = session.scalar(
            select(func.coalesce(func.max(TaskGraphRow.version), 0))
            .where(TaskGraphRow.project_id == project_id)
        ) + 1
        session.execute(delete(TaskEdgeRow).where(TaskEdgeRow.graph_id == graph_id))
        session.execute(delete(TaskNodeRow).where(TaskNodeRow.graph_id == graph_id))
        row.version = next_version
        session.flush()
        _insert_graph(session, row, graph)
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise HTTPException(status_code=409, detail="task graph conflicts with persisted state") from exc
        session.refresh(row)
        return _graph_record(session, row)

    @app.delete("/api/projects/{project_id}/task-graphs/{graph_id}",
                status_code=status.HTTP_204_NO_CONTENT)
    def delete_task_graph(project_id: uuid.UUID, graph_id: uuid.UUID,
                          session: Session = Depends(get_db)):
        row = session.scalar(select(TaskGraphRow).where(
            TaskGraphRow.project_id == project_id, TaskGraphRow.id == graph_id,
        ).with_for_update())
        if row is None:
            raise HTTPException(status_code=404, detail="task graph not found")
        if session.scalar(select(WorkflowRunRow.id).where(
                WorkflowRunRow.task_graph_id == graph_id).limit(1)) is not None:
            raise HTTPException(status_code=409,
                                detail="task graph is retained by a workflow audit record")
        session.delete(row)
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise HTTPException(status_code=409,
                                detail="task graph is retained by a workflow audit record") from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    def require_workflow_engine() -> WorkflowEngine:
        configured = app.state.workflow_engine
        if configured is None:
            raise HTTPException(status_code=503, detail="workflow database is not configured")
        return configured

    def require_queue_dispatcher():
        configured = app.state.queue_dispatcher
        if configured is None:
            raise HTTPException(status_code=503, detail="PROJECTFORGE_REDIS_URL is not configured")
        return configured

    def dispatch_workflow(run: WorkflowRunRecord) -> None:
        dispatcher = require_queue_dispatcher()
        try:
            dispatcher.enqueue_workflow(run.id, timeout_seconds=run.timeout_seconds)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={"message": "workflow is persisted but queue dispatch failed; retry dispatch",
                        "workflow_id": str(run.id), "error": type(exc).__name__},
            ) from None

    @app.post("/api/projects/{project_id}/workflows", response_model=WorkflowRunRecord,
              status_code=status.HTTP_202_ACCEPTED)
    def start_workflow(project_id: uuid.UUID, body: WorkflowStartRequest):
        workflow = require_workflow_engine()
        require_queue_dispatcher()
        try:
            run = workflow.start(
                project_id,
                body.task_graph_id,
                raw_brief_text=body.raw_brief_text,
                rubric_text=body.rubric_text,
                max_retries=body.max_retries,
                timeout_seconds=body.timeout_seconds,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidWorkflowTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except IntegrityError as exc:
            raise HTTPException(status_code=409,
                                detail="task graph already has a workflow") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        dispatch_workflow(run)
        return run

    @app.post("/api/workflows/{workflow_id}/dispatch", response_model=WorkflowRunRecord,
              status_code=status.HTTP_202_ACCEPTED)
    def dispatch_existing_workflow(workflow_id: uuid.UUID):
        workflow = require_workflow_engine()
        try:
            run = workflow.get(workflow_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if run.state.value in {"AWAITING_PLAN_APPROVAL", "AWAITING_FINAL_APPROVAL",
                               "AWAITING_CLARIFICATION", "DONE", "FAILED", "ESCALATED"}:
            raise HTTPException(status_code=409, detail="workflow is waiting for a human gate or terminal")
        dispatch_workflow(run)
        return run

    @app.get("/api/workflows/{workflow_id}", response_model=WorkflowRunRecord)
    def get_workflow(workflow_id: uuid.UUID):
        workflow = require_workflow_engine()
        try:
            return workflow.get(workflow_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/workflows/{workflow_id}/events",
             response_model=list[WorkflowEventRecord])
    def get_workflow_events(workflow_id: uuid.UUID):
        workflow = require_workflow_engine()
        try:
            return workflow.list_events(workflow_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get(
        "/api/workflows/{workflow_id}/tasks/{task_id}/logs",
        response_model=list[WorkflowTaskLogRecord],
    )
    def get_workflow_task_logs(
        workflow_id: uuid.UUID,
        task_id: str,
        limit: int = Query(default=100, ge=1, le=200),
        session: Session = Depends(get_db),
    ):
        workflow = require_workflow_engine()
        try:
            run = workflow.get(workflow_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if run.task_graph_id is None or session.get(TaskNodeRow, (run.task_graph_id, task_id)) is None:
            raise HTTPException(status_code=404, detail="task not found for workflow")
        rows = session.scalars(
            select(AgentRunRow)
            .where(
                AgentRunRow.workflow_id == workflow_id,
                AgentRunRow.agent_name.in_(["coder", "tester", "debugger"]),
                AgentRunRow.input_payload["task_id"].as_string() == task_id,
            )
            .order_by(AgentRunRow.timestamp.desc(), AgentRunRow.id.desc())
            .limit(limit)
        ).all()
        records: list[WorkflowTaskLogRecord] = []
        actions = {
            "coder": "Code change",
            "tester": "Sandbox test",
            "debugger": "Debug guidance",
        }
        for row in reversed(rows):
            input_payload = row.input_payload or {}
            output = row.output_payload or {}
            task_node = input_payload.get("task_node")
            task_node_id = task_node.get("id") if isinstance(task_node, dict) else None
            recorded_task_id = output.get("task_id") or input_payload.get("task_id") or task_node_id
            if recorded_task_id != task_id:
                continue
            phase = str(output.get("phase") or input_payload.get("phase") or row.agent_name.upper())
            status_value = output.get("status")
            if row.agent_name == "tester" and isinstance(output.get("passed"), bool):
                status_value = "passed" if output["passed"] else "failed"
            summary_value = (
                output.get("summary") or output.get("diagnosis")
                or output.get("fix_instructions")
            )
            evidence_value = output.get("evidence", [])
            if not isinstance(evidence_value, list):
                evidence_value = [str(evidence_value)]
            timestamp = row.timestamp
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            records.append(WorkflowTaskLogRecord(
                audit_run_id=row.id,
                workflow_id=workflow_id,
                task_id=task_id,
                agent_name=row.agent_name,
                phase=phase[:32],
                action=actions[row.agent_name],
                status=str(status_value)[:80] if status_value is not None else None,
                summary=str(summary_value)[:4000] if summary_value is not None else None,
                timestamp=timestamp,
                provider=row.provider,
                model=row.model,
                tokens_used=row.tokens_used,
                spend=row.cost,
                evidence=[str(item)[:2000] for item in evidence_value[:100]],
                stdout=str(output.get("stdout", ""))[-12_000:],
                stderr=str(output.get("stderr", ""))[-12_000:],
            ))
        return records

    @app.get("/api/workflows/{workflow_id}/review", response_model=WorkflowReviewBundle)
    def get_workflow_review_bundle(workflow_id: uuid.UUID):
        workflow = require_workflow_engine()
        reviewer = getattr(workflow, "review_bundle", None)
        if not callable(reviewer):
            raise HTTPException(status_code=503, detail="workflow engine does not expose review checkpoints")
        try:
            return reviewer(workflow_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidWorkflowTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/workflows/{workflow_id}/package/metadata", response_model=WorkflowPackageExport)
    def get_workflow_package_metadata(workflow_id: uuid.UUID):
        workflow = require_workflow_engine()
        try:
            run = workflow.get(workflow_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if run.state.value != "DONE":
            raise HTTPException(status_code=409, detail="workflow package is not ready")
        raw_metadata = (run.result_payload or {}).get("package_export")
        if not isinstance(raw_metadata, dict):
            raise HTTPException(status_code=404, detail="workflow has no downloadable package")
        try:
            metadata = WorkflowPackageExport.model_validate(raw_metadata)
        except ValueError:
            raise HTTPException(status_code=500, detail="persisted package metadata is invalid") from None
        if metadata.workflow_id != workflow_id or metadata.project_id != run.project_id:
            raise HTTPException(status_code=500, detail="persisted package metadata does not match workflow")
        return metadata

    @app.get("/api/workflows/{workflow_id}/package")
    def download_workflow_package(workflow_id: uuid.UUID):
        metadata = get_workflow_package_metadata(workflow_id)
        package_manager = app.state.package_manager
        if package_manager is None:
            raise HTTPException(status_code=503, detail="package export storage is not configured")
        try:
            package_path = package_manager.package_path(workflow_id)
        except PackageExportError:
            raise HTTPException(status_code=404, detail="package artifact is unavailable") from None
        if not package_path.is_file() or package_path.is_symlink():
            raise HTTPException(status_code=404, detail="package artifact is unavailable")
        if package_path.stat().st_size != metadata.size_bytes:
            raise HTTPException(status_code=409, detail="package artifact failed its size integrity check")
        digest = hashlib.sha256()
        with package_path.open("rb") as archive:
            for chunk in iter(lambda: archive.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != metadata.sha256:
            raise HTTPException(status_code=409, detail="package artifact failed its checksum integrity check")
        return FileResponse(
            package_path,
            media_type="application/zip",
            filename=metadata.filename,
            headers={"Cache-Control": "private, no-store"},
        )

    @app.post("/api/workflows/{workflow_id}/clarifications", response_model=WorkflowRunRecord,
              status_code=status.HTTP_202_ACCEPTED)
    def submit_workflow_clarification(workflow_id: uuid.UUID,
                                      body: WorkflowClarificationRequest):
        workflow = require_workflow_engine()
        require_queue_dispatcher()
        try:
            run = workflow.submit_clarification(workflow_id, answers=body.answers)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidWorkflowTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        dispatch_workflow(run)
        return run

    @app.post("/api/workflows/{workflow_id}/approvals", response_model=WorkflowRunRecord)
    def approve_workflow(workflow_id: uuid.UUID, body: WorkflowApprovalRequest):
        workflow = require_workflow_engine()
        require_queue_dispatcher()
        try:
            run = workflow.approve(
                workflow_id,
                stage=body.stage,
                approved=body.approved,
                decision=body.decision,
                comment=body.comment,
                task_feedback=body.task_feedback,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (InvalidWorkflowTransition, WorkflowCancelled) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if run.state.value not in {"DONE", "FAILED", "ESCALATED", "AWAITING_PLAN_APPROVAL",
                                   "AWAITING_FINAL_APPROVAL", "AWAITING_CLARIFICATION"}:
            dispatch_workflow(run)
        return run

    @app.post("/api/workflows/{workflow_id}/cancellation", response_model=WorkflowRunRecord,
              status_code=status.HTTP_202_ACCEPTED)
    def cancel_workflow(workflow_id: uuid.UUID, body: WorkflowCancelRequest):
        workflow = require_workflow_engine()
        require_queue_dispatcher()
        try:
            run = workflow.cancel(workflow_id, reason=body.reason)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidWorkflowTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        dispatch_workflow(run)
        return run

    return app


app = create_app()
