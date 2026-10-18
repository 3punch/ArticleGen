"""Per-project Git workspaces with scoped, atomic agent commits and safe context reads."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
from typing import Iterator, Sequence
from uuid import UUID

from projectforge_schemas.models import AgentFileChange


_FORBIDDEN_NAMES = {
    ".git", ".env", ".netrc", ".npmrc", ".pypirc", ".ssh", ".aws", ".gnupg",
    ".git-credentials", "credentials", "id_rsa", "id_ed25519", "id_ecdsa",
}
_FORBIDDEN_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
_EXCLUDED_DIRECTORIES = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".next", "dist", "build",
}


class GitWorkspaceError(RuntimeError):
    """A Git workspace could not be safely initialized, read, or committed."""


class AgentWriteScopeError(GitWorkspaceError):
    """An agent attempted to write outside its declared path capability."""


@dataclass(frozen=True)
class CommitRecord:
    project_id: UUID
    task_id: str
    agent_name: str
    commit_hash: str | None
    message: str
    changed_paths: tuple[str, ...]


class GitWorkspaceManager:
    """Own ``{root}/{project_id}`` repositories and audit every agent write.

    Git runs only fixed local subcommands with a workspace-local identity. No agent
    text is interpolated into a shell command, and all file paths are validated and
    checked against each agent's explicit capability before touching disk.
    """

    def __init__(self, root: str | os.PathLike[str] = "/data/repos", *,
                 command_timeout_seconds: int = 30) -> None:
        self.root = Path(root).expanduser().absolute()
        if not 1 <= command_timeout_seconds <= 120:
            raise ValueError("command_timeout_seconds must be between 1 and 120")
        self.command_timeout_seconds = command_timeout_seconds

    def project_path(self, project_id: UUID | str) -> Path:
        try:
            normalized = UUID(str(project_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise GitWorkspaceError("project id must be a UUID") from exc
        return self.root / str(normalized)

    def ensure_repository(self, project_id: UUID | str) -> Path:
        normalized_project_id = UUID(str(project_id))
        with self._project_lock(normalized_project_id):
            return self._ensure_repository_locked(normalized_project_id)

    def _ensure_repository_locked(self, project_id: UUID | str) -> Path:
        project_path = self.project_path(project_id)
        self.root.mkdir(parents=True, exist_ok=True)
        if project_path.is_symlink():
            raise GitWorkspaceError("project repository directory must not be a symlink")
        project_path.mkdir(parents=True, exist_ok=True)
        git_dir = project_path / ".git"
        if git_dir.is_symlink():
            raise GitWorkspaceError("project .git directory must not be a symlink")
        if not git_dir.exists():
            self._run_git(project_path, ["init", "--quiet", "--initial-branch=main"])
            self._run_git(project_path, ["config", "user.name", "ProjectForge Agent"])
            self._run_git(project_path, ["config", "user.email", "projectforge@localhost"])
            self._run_git(project_path, ["config", "commit.gpgsign", "false"])
        elif not git_dir.is_dir():
            raise GitWorkspaceError("project .git entry is not a directory")
        top_level = self._run_git(project_path, ["rev-parse", "--show-toplevel"]).stdout.strip()
        if Path(top_level).resolve() != project_path.resolve():
            raise GitWorkspaceError("project Git root does not match its assigned workspace")
        return project_path

    def apply_changes_and_commit(
        self,
        project_id: UUID | str,
        *,
        task_id: str,
        agent_name: str,
        changes: Sequence[AgentFileChange | dict[str, object]],
        write_scope: Sequence[str],
        summary: str,
    ) -> CommitRecord:
        """Apply a bounded batch and commit it, rolling files back if commit fails."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", task_id):
            raise GitWorkspaceError("task id is not safe for a commit message")
        if agent_name not in {"coder", "tester", "debugger", "packager"}:
            raise GitWorkspaceError("only code-touching worker agents may modify a workspace")
        if not changes:
            raise GitWorkspaceError("an agent write must include at least one file change")
        normalized_changes = [
            change if isinstance(change, AgentFileChange) else AgentFileChange.model_validate(change)
            for change in changes
        ]
        paths = [self._validate_relative_path(change.path) for change in normalized_changes]
        if len(paths) != len(set(paths)):
            raise GitWorkspaceError("agent file-change paths must be unique")
        scopes = [self._validate_scope(scope) for scope in write_scope]
        if not scopes:
            raise AgentWriteScopeError(f"{agent_name} has no declared write scope")
        for relative in paths:
            if not any(self._within_scope(relative, scope) for scope in scopes):
                raise AgentWriteScopeError(
                    f"{agent_name} may not write '{relative}'; declared paths: {scopes}"
                )

        normalized_project_id = UUID(str(project_id))
        with self._project_lock(normalized_project_id):
            repo = self._ensure_repository_locked(normalized_project_id)
            status = self._run_git(
                repo, ["status", "--porcelain", "--untracked-files=all"],
            ).stdout.strip()
            if status:
                raise GitWorkspaceError(
                    "project workspace is dirty before agent write; reconcile the prior operation first"
                )

            snapshots: dict[str, tuple[bytes, int] | None] = {}
            touched: list[Path] = []
            committed = False
            message = self._commit_message(agent_name, task_id, summary)
            try:
                for change, relative in zip(normalized_changes, paths):
                    target = self._safe_target(repo, relative, create_parents=True)
                    if target.exists():
                        if not target.is_file():
                            raise GitWorkspaceError(f"agent target '{relative}' is not a regular file")
                        snapshots[relative] = (target.read_bytes(), stat.S_IMODE(target.stat().st_mode))
                    else:
                        snapshots[relative] = None
                    if change.action == "delete":
                        if target.exists():
                            target.unlink()
                        touched.append(target)
                        continue
                    assert change.content is not None
                    target.parent.mkdir(parents=True, exist_ok=True)
                    mode = snapshots[relative][1] if snapshots[relative] is not None else 0o644
                    fd, temporary_name = tempfile.mkstemp(prefix=".projectforge-", dir=target.parent)
                    try:
                        with os.fdopen(fd, "wb") as handle:
                            handle.write(change.content.encode("utf-8"))
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.chmod(temporary_name, mode)
                        os.replace(temporary_name, target)
                    finally:
                        if os.path.exists(temporary_name):
                            os.unlink(temporary_name)
                    touched.append(target)

                self._run_git(repo, ["add", "--all", "--", *paths])
                diff = self._run_git(
                    repo, ["diff", "--cached", "--quiet"], check=False,
                )
                if diff.returncode == 0:
                    self._run_git(repo, ["reset", "--quiet"], check=False)
                    return CommitRecord(
                        project_id=normalized_project_id, task_id=task_id,
                        agent_name=agent_name, commit_hash=None, message=message,
                        changed_paths=(),
                    )
                if diff.returncode != 1:
                    raise GitWorkspaceError("could not inspect staged project changes")
                self._run_git(repo, ["diff", "--cached", "--check"])
                self._run_git(repo, ["commit", "--quiet", "--no-gpg-sign", "-m", message])
                committed = True
                commit_hash = self._run_git(repo, ["rev-parse", "HEAD"]).stdout.strip()
                changed_output = self._run_git(
                    repo, ["show", "--format=", "--name-only", "HEAD"],
                ).stdout
                changed_paths = tuple(sorted(line for line in changed_output.splitlines() if line))
                return CommitRecord(
                    project_id=normalized_project_id, task_id=task_id,
                    agent_name=agent_name, commit_hash=commit_hash, message=message,
                    changed_paths=changed_paths,
                )
            except Exception:
                if not committed:
                    self._run_git(repo, ["reset", "--quiet"], check=False)
                    for relative, snapshot in snapshots.items():
                        target = self._safe_target(repo, relative, create_parents=True)
                        if snapshot is None:
                            if target.exists() and target.is_file():
                                target.unlink()
                        else:
                            content, mode = snapshot
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(content)
                            os.chmod(target, mode)
                raise

    def read_context(
        self,
        project_id: UUID | str,
        *,
        read_scope: Sequence[str] = (),
        write_scope: Sequence[str] = (),
        extra_paths: Sequence[str] = (),
        max_files: int = 30,
        max_total_bytes: int = 120_000,
        max_file_bytes: int = 24_000,
    ) -> dict[str, str]:
        """Read only scoped, non-secret regular files for a bounded agent prompt."""
        for name, value, lower, upper in (
            ("max_files", max_files, 1, 500),
            ("max_total_bytes", max_total_bytes, 1, 1_000_000),
            ("max_file_bytes", max_file_bytes, 1, 200_000),
        ):
            if not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
        repo = self.ensure_repository(project_id)
        targets: set[str] = set()
        scopes = [*read_scope, *write_scope, *extra_paths]
        for raw in scopes:
            relative = self._validate_relative_path(raw)
            target = self._safe_target(repo, relative, create_parents=False)
            if target.is_file():
                targets.add(relative)
            elif target.is_dir():
                for candidate in target.rglob("*"):
                    try:
                        candidate_relative = candidate.relative_to(repo).as_posix()
                    except ValueError:
                        continue
                    if candidate.is_file() and self._is_safe_context_path(candidate_relative):
                        targets.add(candidate_relative)
        result: dict[str, str] = {}
        total = 0
        for relative in sorted(targets):
            if len(result) >= max_files:
                break
            target = self._safe_target(repo, relative, create_parents=False)
            data = target.read_bytes()
            data = data[:max_file_bytes]
            if total + len(data) > max_total_bytes:
                data = data[:max(0, max_total_bytes - total)]
            if not data:
                continue
            total += len(data)
            result[relative] = data.decode("utf-8", errors="replace")
            if total >= max_total_bytes:
                break
        return result

    def read_repository_context(
        self,
        project_id: UUID | str,
        *,
        max_files: int = 200,
        max_total_bytes: int = 1_000_000,
        max_file_bytes: int = 200_000,
    ) -> dict[str, str]:
        """Read bounded source, test, and documentation files for review agents."""
        for name, value, lower, upper in (
            ("max_files", max_files, 1, 1000),
            ("max_total_bytes", max_total_bytes, 1, 2_000_000),
            ("max_file_bytes", max_file_bytes, 1, 500_000),
        ):
            if not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
        reviewable_suffixes = {
            ".py", ".pyi", ".js", ".jsx", ".cjs", ".mjs", ".ts", ".tsx",
            ".json", ".md", ".txt", ".csv", ".html", ".css", ".toml", ".yaml",
            ".yml", ".xml", ".svg", ".sh", ".sql", ".ini", ".cfg",
        }
        repo = self.ensure_repository(project_id)
        result: dict[str, str] = {}
        total = 0
        for candidate in sorted(repo.rglob("*")):
            relative = candidate.relative_to(repo).as_posix()
            if (not candidate.is_file() or candidate.is_symlink()
                    or not self._is_safe_context_path(relative)
                    or candidate.suffix.casefold() not in reviewable_suffixes):
                continue
            if len(result) >= max_files:
                break
            size = candidate.stat().st_size
            if size > max_file_bytes:
                continue
            if total + size > max_total_bytes:
                break
            data = candidate.read_bytes()
            result[relative] = data.decode("utf-8", errors="replace")
            total += len(data)
        return result

    def read_package_files(
        self,
        project_id: UUID | str,
        *,
        max_files: int = 2000,
        max_total_bytes: int = 64_000_000,
        max_file_bytes: int = 16_000_000,
    ) -> dict[str, bytes]:
        """Take a bounded, secret-free byte snapshot for a project ZIP export."""
        for name, value, lower, upper in (
            ("max_files", max_files, 1, 5000),
            ("max_total_bytes", max_total_bytes, 1, 128_000_000),
            ("max_file_bytes", max_file_bytes, 1, 32_000_000),
        ):
            if not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
        repo = self.ensure_repository(project_id)
        result: dict[str, bytes] = {}
        total = 0
        for candidate in sorted(repo.rglob("*")):
            relative = candidate.relative_to(repo).as_posix()
            if (not candidate.is_file() or candidate.is_symlink()
                    or not self._is_safe_context_path(relative)
                    or candidate.suffix.casefold() in {".pyc", ".pyo", ".swp", ".tmp"}):
                continue
            if len(result) >= max_files:
                raise GitWorkspaceError("project package exceeds the maximum file count")
            size = candidate.stat().st_size
            if size > max_file_bytes or total + size > max_total_bytes:
                raise GitWorkspaceError("project package exceeds its file or size limit")
            data = candidate.read_bytes()
            result[relative] = data
            total += len(data)
        if not result:
            raise GitWorkspaceError("project workspace is empty")
        return result

    def review_diff(
        self,
        project_id: UUID | str,
        *,
        max_bytes: int = 500_000,
    ) -> tuple[str, list[str], bool]:
        """Return the auditable project diff from its initial Git commit to HEAD."""
        if not isinstance(max_bytes, int) or not 1 <= max_bytes <= 2_000_000:
            raise ValueError("max_bytes must be between 1 and 2000000")
        repo = self.ensure_repository(project_id)
        head = self._run_git(repo, ["rev-parse", "--verify", "HEAD"], check=False)
        if head.returncode != 0:
            return "", [], False
        commits = self._run_git(repo, ["rev-list", "--max-parents=0", "HEAD"]).stdout.splitlines()
        if not commits:
            return "", [], False
        initial_commit = commits[-1].strip()
        names = self._run_git(
            repo, ["diff", "--name-only", "--no-ext-diff", "--no-renames", initial_commit, "HEAD", "--"],
        ).stdout
        changed_paths = sorted({
            name for name in names.splitlines()
            if name and self._is_safe_context_path(name)
        })
        diff = self._run_git(
            repo, ["diff", "--no-ext-diff", "--no-renames", "--no-color", "--src-prefix=a/",
                  "--dst-prefix=b/", "-U3", initial_commit, "HEAD", "--"],
        ).stdout
        encoded = diff.encode("utf-8")
        truncated = len(encoded) > max_bytes
        if truncated:
            diff = encoded[:max_bytes].decode("utf-8", errors="ignore")
            diff += "\\n[diff truncated at configured review limit]\\n"
        return diff, changed_paths, truncated

    def discover_test_files(self, project_id: UUID | str) -> list[str]:
        repo = self.ensure_repository(project_id)
        tests: list[str] = []
        for candidate in repo.rglob("*"):
            relative = candidate.relative_to(repo).as_posix()
            if not candidate.is_file() or not self._is_safe_context_path(relative):
                continue
            suffix = candidate.suffix.casefold()
            name = candidate.name.casefold()
            if suffix == ".py" and (name.startswith("test_") or name.endswith("_test.py")):
                tests.append(relative)
            elif suffix in {".js", ".jsx", ".cjs", ".mjs", ".ts", ".tsx"} and (
                ".test." in name or ".spec." in name
            ):
                tests.append(relative)
        return sorted(tests)

    def detect_runtime(
        self,
        project_id: UUID | str,
        *,
        explicit_runtime: str | None = None,
        path_hints: Sequence[str] = (),
        tech_stack_hints: Sequence[str] = (),
    ) -> str:
        if explicit_runtime is not None:
            if explicit_runtime not in {"python", "node"}:
                raise GitWorkspaceError("task runtime must be python or node")
            return explicit_runtime
        hints = " ".join(tech_stack_hints).casefold()
        # Select the project's declared test runtime before inspecting task-local
        # frontend paths; mixed Python API/browser projects still use pytest for tests.
        if any(term in hints for term in ("python", "pytest", "fastapi", "django", "flask")):
            return "python"
        suffixes = {Path(value).suffix.casefold() for value in path_hints}
        if suffixes & {".js", ".jsx", ".cjs", ".mjs", ".ts", ".tsx"}:
            return "node"
        if ".py" in suffixes:
            return "python"
        files = self.read_context(project_id, read_scope=path_hints, write_scope=path_hints,
                                  max_files=100, max_total_bytes=500_000)
        file_suffixes = {Path(path).suffix.casefold() for path in files}
        # Prefer Python when the scoped workspace contains both backend modules and
        # browser JavaScript: Python is the project test runner for mixed-stack apps.
        if ".py" in file_suffixes:
            return "python"
        if file_suffixes & {".js", ".jsx", ".cjs", ".mjs", ".ts", ".tsx"}:
            return "node"
        if any(term in hints for term in ("node", "javascript", "typescript", "jest")):
            return "node"
        return "python"

    @contextmanager
    def _project_lock(self, project_id: UUID) -> Iterator[None]:
        lock_dir = self.root / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{project_id}.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _validate_relative_path(value: str) -> str:
        path = PurePosixPath(value)
        if (
            not value
            or value != value.strip()
            or path.is_absolute()
            or "\\" in value
            or ":" in value
            or "\x00" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or any(part.casefold() in _FORBIDDEN_NAMES for part in path.parts)
            or path.suffix.casefold() in _FORBIDDEN_SUFFIXES
        ):
            raise GitWorkspaceError("agent path must be a safe relative path without secrets or VCS data")
        return path.as_posix()

    @classmethod
    def _validate_scope(cls, value: str) -> str:
        return cls._validate_relative_path(value.rstrip("/"))

    @staticmethod
    def _within_scope(path: str, scope: str) -> bool:
        return path == scope or path.startswith(f"{scope}/")

    @classmethod
    def _is_safe_context_path(cls, value: str) -> bool:
        path = PurePosixPath(value)
        return (
            not any(part.casefold() in _EXCLUDED_DIRECTORIES for part in path.parts)
            and not any(part.casefold() in _FORBIDDEN_NAMES for part in path.parts)
            and path.suffix.casefold() not in _FORBIDDEN_SUFFIXES
        )

    @classmethod
    def _safe_target(cls, repo: Path, relative: str, *, create_parents: bool) -> Path:
        relative = cls._validate_relative_path(relative)
        current = repo
        parts = PurePosixPath(relative).parts
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise GitWorkspaceError(f"agent path '{relative}' traverses a symlink")
            if current.exists() and current != repo and not current.is_file() and not current.is_dir():
                raise GitWorkspaceError(f"agent path '{relative}' contains a non-regular entry")
        if create_parents:
            current.parent.mkdir(parents=True, exist_ok=True)
            # Re-check after creation in case another process raced directory setup.
            resolved_parent = current.parent.resolve()
            if not resolved_parent.is_relative_to(repo.resolve()):
                raise GitWorkspaceError("agent path parent escapes the assigned repository")
        return current

    @staticmethod
    def _commit_message(agent_name: str, task_id: str, summary: str) -> str:
        cleaned = " ".join(str(summary).split())
        cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", cleaned).strip()
        if not cleaned:
            cleaned = "update project files"
        return f"{agent_name.title()} task {task_id}: {cleaned[:180]}"

    def _run_git(
        self,
        cwd: Path,
        arguments: Sequence[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update({
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_AUTHOR_NAME": "ProjectForge Agent",
            "GIT_AUTHOR_EMAIL": "projectforge@localhost",
            "GIT_COMMITTER_NAME": "ProjectForge Agent",
            "GIT_COMMITTER_EMAIL": "projectforge@localhost",
        })
        try:
            result = subprocess.run(
                ["git", *arguments], cwd=cwd, env=environment,
                capture_output=True, text=True, timeout=self.command_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GitWorkspaceError(f"git {' '.join(arguments[:2])} failed: {exc}") from exc
        if check and result.returncode != 0:
            detail = result.stderr.strip()[-2000:]
            raise GitWorkspaceError(
                f"git {' '.join(arguments[:2])} failed with exit {result.returncode}: {detail}"
            )
        return result
