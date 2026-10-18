"""Sanitized, bounded ZIP export built from a project Git workspace and audit trail."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
from uuid import UUID
import zipfile

from projectforge_schemas.models import (
    AIUsageDisclosure,
    ReviewerAgentResponse,
    WorkflowPackageExport,
)

from .git_workspace import GitWorkspaceError, GitWorkspaceManager


class PackageExportError(RuntimeError):
    """The final project package could not be safely constructed or retrieved."""


class ProjectPackageManager:
    """Create portable project ZIPs without including VCS or excluded workspace data."""

    def __init__(
        self,
        workspace_manager: GitWorkspaceManager,
        *,
        export_root: str | os.PathLike[str] | None = None,
        max_archive_bytes: int = 128_000_000,
    ) -> None:
        if not 1 <= max_archive_bytes <= 256_000_000:
            raise ValueError("max_archive_bytes must be between 1 and 256000000")
        self.workspace_manager = workspace_manager
        configured_root = export_root or (workspace_manager.root / ".exports")
        self.export_root = Path(configured_root).expanduser().absolute()
        self.max_archive_bytes = max_archive_bytes

    def package_path(self, workflow_id: UUID | str) -> Path:
        normalized = self._workflow_uuid(workflow_id)
        root = self._prepare_root()
        target = root / f"projectforge-{normalized}.zip"
        if target.is_symlink():
            raise PackageExportError("package artifact must not be a symlink")
        if not target.resolve().is_relative_to(root.resolve()):
            raise PackageExportError("package artifact escaped the configured export directory")
        return target

    def export(
        self,
        *,
        project_id: UUID | str,
        workflow_id: UUID | str,
        disclosure: AIUsageDisclosure | Mapping[str, Any],
        review: ReviewerAgentResponse | Mapping[str, Any],
        test_runs: Sequence[Mapping[str, Any]],
    ) -> tuple[WorkflowPackageExport, Path]:
        normalized_project = UUID(str(project_id))
        normalized_workflow = self._workflow_uuid(workflow_id)
        usage = disclosure if isinstance(disclosure, AIUsageDisclosure) else AIUsageDisclosure.model_validate(disclosure)
        if isinstance(review, ReviewerAgentResponse):
            review_result = review
        else:
            reviewer_fields = (
                "summary", "criteria", "automatable_checks_passed",
                "human_review_required", "evidence",
            )
            review_result = ReviewerAgentResponse.model_validate({
                key: review[key] for key in reviewer_fields if key in review
            })
        if usage.workflow_id != normalized_workflow or usage.project_id != normalized_project:
            raise PackageExportError("disclosure does not match the requested project workflow")
        if not test_runs or any(run.get("passed") is not True for run in test_runs):
            raise PackageExportError("package export requires at least one passing sandbox test run")

        try:
            files = self.workspace_manager.read_package_files(normalized_project)
        except (GitWorkspaceError, OSError) as exc:
            raise PackageExportError(f"project file snapshot failed: {exc}") from exc
        if "README.md" not in files:
            raise PackageExportError("project package is missing its generated README.md")
        if "reports/PROJECT_REPORT.md" not in files:
            raise PackageExportError("project package is missing reports/PROJECT_REPORT.md")

        generated: dict[str, bytes] = {
            "PROJECT_REPORT.md": files["reports/PROJECT_REPORT.md"],
            "AI_USAGE_DISCLOSURE.md": self.render_disclosure_markdown(usage).encode("utf-8"),
            "audit/AI_USAGE_DISCLOSURE.json": self._json_bytes(usage.model_dump(mode="json")),
            "reports/TEST_RESULTS.json": self._json_bytes(self._test_results_payload(test_runs)),
            "reports/REVIEW_RESULTS.json": self._json_bytes(review_result.model_dump(mode="json")),
        }
        for path, content in generated.items():
            if path in files and files[path] != content:
                raise PackageExportError(f"generated package artifact conflicts with project file '{path}'")
            files[path] = content
        total = sum(len(content) for content in files.values())
        if total > self.max_archive_bytes:
            raise PackageExportError("project package exceeds the configured uncompressed size limit")

        file_checksums = {
            name: hashlib.sha256(content).hexdigest()
            for name, content in sorted(files.items())
        }
        manifest = {
            "format_version": 1,
            "project_id": str(normalized_project),
            "workflow_id": str(normalized_workflow),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": [
                {"path": name, "size_bytes": len(files[name]), "sha256": digest}
                for name, digest in file_checksums.items()
            ],
        }
        files["PACKAGE_MANIFEST.json"] = self._json_bytes(manifest)
        if len(files) > 2000:
            raise PackageExportError("project package exceeds the configured file count")

        path = self.package_path(normalized_workflow)
        fd, temp_name = tempfile.mkstemp(prefix=f".{normalized_workflow}.", suffix=".zip.tmp",
                                         dir=path.parent)
        os.close(fd)
        try:
            with zipfile.ZipFile(temp_name, mode="w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6, allowZip64=False) as archive:
                for name, content in sorted(files.items()):
                    archive.writestr(self._zip_info(name), content)
            archive_size = os.path.getsize(temp_name)
            if archive_size > self.max_archive_bytes:
                raise PackageExportError("compressed package exceeds the configured archive size limit")
            with open(temp_name, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

        record = WorkflowPackageExport(
            workflow_id=normalized_workflow,
            project_id=normalized_project,
            filename=path.name,
            download_url=f"/api/workflows/{normalized_workflow}/package",
            size_bytes=archive_size,
            sha256=digest,
            files=sorted(files),
            created_at=datetime.now(timezone.utc),
        )
        return record, path

    def render_disclosure_markdown(self, disclosure: AIUsageDisclosure) -> str:
        lines = [
            "# AI Usage Disclosure",
            "",
            disclosure.statement,
            "",
            f"- Project: `{disclosure.project_id}`",
            f"- Workflow: `{disclosure.workflow_id}`",
            f"- Generated at (UTC): `{disclosure.generated_at.isoformat()}`",
            f"- Agent actions: {disclosure.agent_calls}",
            f"- LLM calls: {disclosure.llm_calls}",
            f"- Stub calls: {disclosure.stub_agent_calls}",
            f"- Code executions: {disclosure.code_execution_count}",
            f"- Tokens recorded: {disclosure.tokens_used}",
            f"- Spend recorded: {disclosure.spend}",
            "",
            "## Agent action log",
            "",
            "| Timestamp (UTC) | Agent | Action | Provider / model | Tokens | Spend | Audit run |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
        for activity in disclosure.activities:
            provider_model = "/".join(
                value for value in (activity.provider, activity.model) if value
            ) or "not reported"
            lines.append("| " + " | ".join(self._markdown_cell(value) for value in (
                activity.timestamp.isoformat(), activity.agent_name, activity.action,
                provider_model, str(activity.tokens_used), str(activity.spend),
                str(activity.audit_run_id),
            )) + " |")
        if not disclosure.activities:
            lines.append("| _No persisted agent actions were recorded._ | — | — | — | 0 | 0 | — |")
        lines.extend([
            "",
            "This disclosure is derived from persisted `AgentRun` records for this workflow. "
            "A missing model/provider value means the audit record did not report one; it is not inferred.",
            "",
        ])
        return "\n".join(lines)

    @staticmethod
    def _test_results_payload(test_runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for run in test_runs:
            results.append({
                "task_id": str(run.get("task_id", "unknown")),
                "phase": str(run.get("phase", "TEST")),
                "passed": run.get("passed") is True,
                "runtime": run.get("runtime"),
                "command": run.get("command", []),
                "tested_files": run.get("tested_files", []),
                "exit_code": run.get("exit_code"),
                "timed_out": bool(run.get("timed_out", False)),
                "duration_seconds": run.get("duration_seconds"),
                "evidence": run.get("evidence", []),
                "stdout_excerpt": str(run.get("stdout", ""))[:4000],
                "stderr_excerpt": str(run.get("stderr", ""))[:4000],
            })
        return {"all_passed": bool(results) and all(item["passed"] for item in results),
                "run_count": len(results), "runs": results}

    @staticmethod
    def _json_bytes(value: Any) -> bytes:
        return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")

    @staticmethod
    def _markdown_cell(value: Any) -> str:
        return " ".join(str(value).replace("|", "\\|").split())

    @staticmethod
    def _workflow_uuid(value: UUID | str) -> UUID:
        try:
            return UUID(str(value))
        except (ValueError, TypeError, AttributeError) as exc:
            raise PackageExportError("workflow id must be a UUID") from exc

    def _prepare_root(self) -> Path:
        if self.export_root.is_symlink():
            raise PackageExportError("package export root must not be a symlink")
        self.export_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        resolved = self.export_root.resolve()
        if not resolved.is_dir():
            raise PackageExportError("package export root is not a directory")
        return resolved

    @staticmethod
    def _zip_info(name: str) -> zipfile.ZipInfo:
        path = Path(name)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise PackageExportError("package entry path must be relative and traversal-free")
        info = zipfile.ZipInfo(path.as_posix())
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        info.create_system = 3
        return info
