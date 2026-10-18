"""Docker isolation boundary for untrusted project code.

Project files are sent through the Docker archive API (never a host bind mount),
then the short-lived container is removed in a ``finally`` block. Network isolation
is the default. Non-empty allowlists use a filtered HTTP(S) egress proxy on a
separate network; the sandbox itself is only attached to an internal Docker network.
"""

from __future__ import annotations

import io
import json
import os
import re
import stat
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .egress_proxy import normalize_allowlist


DEFAULT_IMAGES = {
    "python": "projectforge/python-test:3.11",
    "node": "projectforge/node-test:20",
}
DEFAULT_PROXY_IMAGE = "projectforge/sandbox-egress-proxy:3.11"
WORKSPACE = "/workspace"
SANDBOX_UID = 65_532
SANDBOX_GID = 65_532
MAX_MEMORY_BYTES = 4 * 1024**3
MAX_PROJECT_BYTES = 100 * 1024**2
MAX_PROJECT_FILES = 100_000
MAX_OUTPUT_BYTES = 1 * 1024**2
MAX_TIMEOUT_SECONDS = 3600
MAX_CPU_CORES = 4.0
MAX_PIDS = 512

_EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".ssh",
    ".aws",
    ".gnupg",
}
_EXCLUDED_FILE_NAMES = {
    ".env",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    "credentials",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
}
_SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


class SandboxConfigurationError(ValueError):
    """Raised for an unsafe or invalid runtime/limit configuration."""


class SandboxExecutionError(RuntimeError):
    """Raised when Docker cannot create, start, or observe a sandbox."""


class UnsafeProjectFileError(ValueError):
    """Raised when a project file could escape or expose the selected workspace."""


@dataclass(frozen=True)
class SandboxResult:
    """Captured result of one short-lived execution."""

    runtime: str
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_seconds: float
    container_id: str
    output_truncated: bool = False


class SandboxManager:
    """Create tightly constrained, disposable Docker execution containers.

    Args:
        client: Optional Docker SDK client. Inject a client in tests; otherwise the
            SDK is imported lazily and ``docker.from_env`` is used.
        images: Optional image overrides for the supported ``python`` and ``node``
            runtimes. Images should be prebuilt from ``infra/sandbox`` Dockerfiles.
        egress_proxy_image: Image used only when a non-empty host allowlist is
            requested. Its Dockerfile is ``infra/sandbox/egress-proxy/Dockerfile``.
        default_timeout_seconds: Wall-clock cap for one command (default: 60s).
        default_cpu_limit: CPU quota in cores (default: 1 core).
        default_memory_limit: Memory cap as bytes or a Docker unit string (default:
            ``512m``). Swap is set equal to memory, disabling additional swap.
        max_output_bytes: Captured output ceiling per stdout/stderr stream.
        max_project_bytes: Maximum uncompressed project input size.
        network_allowlist: Optional default allowlist. An empty list means no
            network access. A per-run ``network_allowlist`` overrides the default.
        poll_interval_seconds: Container status polling cadence.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        images: Mapping[str, str] | None = None,
        egress_proxy_image: str = DEFAULT_PROXY_IMAGE,
        default_timeout_seconds: int = 60,
        default_cpu_limit: float = 1.0,
        default_memory_limit: int | str = "512m",
        default_pids_limit: int = 128,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
        max_project_bytes: int = MAX_PROJECT_BYTES,
        network_allowlist: Sequence[str] = (),
        poll_interval_seconds: float = 0.1,
    ) -> None:
        self._client = client
        self.images = dict(DEFAULT_IMAGES)
        if images:
            self.images.update(images)
        for runtime, image in self.images.items():
            if runtime not in DEFAULT_IMAGES or not isinstance(image, str) or not image.strip():
                raise SandboxConfigurationError("images may only override python/node with non-empty tags")
        if not isinstance(egress_proxy_image, str) or not egress_proxy_image.strip():
            raise SandboxConfigurationError("egress_proxy_image must be a non-empty Docker image tag")
        self.egress_proxy_image = egress_proxy_image
        self.default_timeout_seconds = self._bounded_int(
            "default_timeout_seconds", default_timeout_seconds, 1, MAX_TIMEOUT_SECONDS,
        )
        self.default_cpu_limit = self._bounded_float(
            "default_cpu_limit", default_cpu_limit, 0.01, MAX_CPU_CORES,
        )
        self.default_memory_limit_bytes = self._memory_bytes(default_memory_limit)
        if self.default_memory_limit_bytes > MAX_MEMORY_BYTES:
            raise SandboxConfigurationError("default_memory_limit exceeds the 4 GiB hard cap")
        self.default_pids_limit = self._bounded_int("default_pids_limit", default_pids_limit, 1, MAX_PIDS)
        self.max_output_bytes = self._bounded_int("max_output_bytes", max_output_bytes, 1, 64 * 1024**2)
        self.max_project_bytes = self._bounded_int(
            "max_project_bytes", max_project_bytes, 0, MAX_PROJECT_BYTES,
        )
        try:
            self.network_allowlist = normalize_allowlist(network_allowlist)
        except ValueError as exc:
            raise SandboxConfigurationError(str(exc)) from exc
        if not (0 < poll_interval_seconds <= 1):
            raise SandboxConfigurationError("poll_interval_seconds must be in (0, 1]")
        self.poll_interval_seconds = float(poll_interval_seconds)

    def run(
        self,
        runtime: str,
        command: str | Sequence[str],
        project_files: Mapping[str, str | bytes] | None = None,
        *,
        project_dir: str | os.PathLike[str] | None = None,
        timeout_seconds: int | None = None,
        cpu_limit: float | None = None,
        memory_limit: int | str | None = None,
        pids_limit: int | None = None,
        network_allowlist: Sequence[str] | None = None,
    ) -> SandboxResult:
        """Copy a workspace into a container, run ``command``, and always remove it.

        ``project_files`` maps safe relative paths to text or bytes. Alternatively,
        ``project_dir`` can point at a local worktree; secret/config files, VCS data,
        virtual environments, caches, and symlinks are excluded or rejected. The two
        inputs are mutually exclusive. String commands run via ``/bin/sh -c``;
        sequence commands bypass the shell.

        Host allowlists support exact names and ``*.example.org`` subdomain rules.
        Only HTTP and HTTPS (ports 80/443 unless a rule names a specific port) can
        pass through the sidecar proxy. All other sandbox egress remains blocked.
        """
        runtime_name = self._normalize_runtime(runtime)
        image = self.images[runtime_name]
        command_value = self._normalize_command(command)
        timeout = self.default_timeout_seconds if timeout_seconds is None else self._bounded_int(
            "timeout_seconds", timeout_seconds, 1, MAX_TIMEOUT_SECONDS,
        )
        cpu = self.default_cpu_limit if cpu_limit is None else self._bounded_float(
            "cpu_limit", cpu_limit, 0.01, MAX_CPU_CORES,
        )
        memory = self.default_memory_limit_bytes if memory_limit is None else self._memory_bytes(memory_limit)
        if memory > MAX_MEMORY_BYTES:
            raise SandboxConfigurationError("memory_limit exceeds the 4 GiB hard cap")
        pids = self.default_pids_limit if pids_limit is None else self._bounded_int(
            "pids_limit", pids_limit, 1, MAX_PIDS,
        )
        try:
            allowlist = self.network_allowlist if network_allowlist is None else normalize_allowlist(network_allowlist)
        except ValueError as exc:
            raise SandboxConfigurationError(str(exc)) from exc
        archive = self._create_archive(project_files, project_dir)
        client = self._docker_client()
        execution_id = uuid4().hex
        labels = {
            "projectforge.sandbox": "true",
            "projectforge.execution": execution_id,
            "projectforge.created_at_epoch": str(int(time.time())),
        }
        container = None
        proxy = None
        internal_network = None
        external_network = None
        marker_path = f"{WORKSPACE}/.projectforge-ready-{execution_id}"
        start_time: float | None = None

        try:
            if allowlist:
                internal_network, external_network, proxy = self._start_egress_proxy(
                    client, execution_id, allowlist, labels,
                )

            create_kwargs: dict[str, Any] = {
                "image": image,
                "command": self._start_gate_command(command_value, marker_path),
                "name": f"projectforge-sandbox-{execution_id[:16]}",
                "working_dir": WORKSPACE,
                "environment": self._execution_environment(bool(allowlist)),
                "stdin_open": False,
                "tty": False,
                "user": f"{SANDBOX_UID}:{SANDBOX_GID}",
                "read_only": True,
                "network_mode": "none" if not allowlist else None,
                "mem_limit": memory,
                "memswap_limit": memory,
                "nano_cpus": int(cpu * 1_000_000_000),
                "pids_limit": pids,
                "cap_drop": ["ALL"],
                "security_opt": ["no-new-privileges:true"],
                "init": True,
                "tmpfs": {
                    WORKSPACE: "rw,noexec,nosuid,nodev,size=256m,mode=1777",
                    "/tmp": "rw,noexec,nosuid,nodev,size=64m,mode=1777",
                },
                "labels": labels,
                "log_config": self._log_config({"max-size": "2m", "max-file": "1"}),
            }
            if allowlist:
                create_kwargs.pop("network_mode")
                create_kwargs["network"] = internal_network.name
            container = client.containers.create(**create_kwargs)
            container.start()
            copied = container.put_archive(WORKSPACE, archive)
            if copied is False:
                raise SandboxExecutionError("Docker refused the project file archive")
            if self._touch_ready_marker(container, marker_path) != 0:
                raise SandboxExecutionError("Docker could not release the sandbox command gate")

            start_time = time.monotonic()
            timed_out = self._wait_for_exit(container, timeout)
            exit_code = self._exit_code(container, timed_out=timed_out)
            stdout, stdout_truncated = self._read_logs(container, stdout=True, stderr=False)
            stderr, stderr_truncated = self._read_logs(container, stdout=False, stderr=True)
            return SandboxResult(
                runtime=runtime_name,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                timed_out=timed_out,
                duration_seconds=max(0.0, time.monotonic() - start_time),
                container_id=str(getattr(container, "id", "")),
                output_truncated=stdout_truncated or stderr_truncated,
            )
        except SandboxExecutionError:
            raise
        except Exception as exc:
            raise SandboxExecutionError(f"Docker sandbox execution failed: {exc}") from exc
        finally:
            self._cleanup_container(container)
            self._cleanup_container(proxy)
            self._cleanup_network(internal_network)
            self._cleanup_network(external_network)

    @staticmethod
    def _start_gate_command(command: list[str], marker_path: str) -> list[str]:
        gate = 'while [ ! -f "$1" ]; do sleep 0.05; done; shift; exec "$@"'
        return ["/bin/sh", "-c", gate, "projectforge-launcher", marker_path, *command]

    @staticmethod
    def _touch_ready_marker(container: Any, marker_path: str) -> int:
        result = container.exec_run(
            ["/bin/sh", "-c", 'touch -- "$1"', "projectforge-marker", marker_path],
            stdout=False,
            stderr=False,
        )
        if isinstance(result, tuple):
            return int(result[0])
        exit_code = getattr(result, "exit_code", None)
        return int(exit_code) if exit_code is not None else 1

    def _start_egress_proxy(self, client: Any, execution_id: str, allowlist: tuple[str, ...],
                            labels: dict[str, str]) -> tuple[Any, Any, Any]:
        internal = external = proxy = None
        try:
            internal = client.networks.create(
                name=f"projectforge-internal-{execution_id[:16]}",
                driver="bridge",
                internal=True,
                labels=labels,
            )
            external = client.networks.create(
                name=f"projectforge-egress-{execution_id[:16]}",
                driver="bridge",
                internal=False,
                labels=labels,
            )
            proxy = client.containers.create(
                image=self.egress_proxy_image,
                command=["python", "/opt/projectforge/egress_proxy.py"],
                name=f"projectforge-proxy-{execution_id[:16]}",
                user=f"{SANDBOX_UID}:{SANDBOX_GID}",
                environment={
                    "PROJECTFORGE_EGRESS_ALLOWLIST": json.dumps(allowlist),
                    "PROJECTFORGE_EGRESS_PORT": "3128",
                },
                network=external.name,
                read_only=True,
                mem_limit=128 * 1024**2,
                memswap_limit=128 * 1024**2,
                nano_cpus=250_000_000,
                pids_limit=32,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                tmpfs={"/tmp": "rw,noexec,nosuid,nodev,size=16m,mode=1777"},
                labels=labels,
                log_config=self._log_config({"max-size": "256k", "max-file": "1"}),
            )
            proxy.start()
            internal.connect(proxy, aliases=["projectforge-egress-proxy"])
            self._wait_for_proxy_ready(proxy)
            return internal, external, proxy
        except Exception:
            self._cleanup_container(proxy)
            self._cleanup_network(internal)
            self._cleanup_network(external)
            raise

    @staticmethod
    def _wait_for_proxy_ready(proxy: Any) -> None:
        deadline = time.monotonic() + 10
        probe = [
            "python", "-c",
            "import socket; s=socket.create_connection(('127.0.0.1', 3128), timeout=0.2); s.close()",
        ]
        while time.monotonic() < deadline:
            proxy.reload()
            if getattr(proxy, "status", None) in {"exited", "dead"}:
                raise SandboxExecutionError("egress proxy exited during startup")
            try:
                result = proxy.exec_run(probe, stdout=False, stderr=False)
                exit_code = getattr(result, "exit_code", result[0] if isinstance(result, tuple) else None)
            except Exception:
                exit_code = None
            if exit_code == 0:
                return
            time.sleep(0.05)
        raise SandboxExecutionError("egress proxy did not become ready within 10 seconds")

    @staticmethod
    def _log_config(config: dict[str, str]) -> Any:
        try:
            from docker.types import LogConfig
        except ImportError:
            # Injected fake clients in dependency-free unit tests do not need the
            # SDK type. The real Docker client is installed via the optional extra.
            return {"type": "json-file", "config": config}
        return LogConfig(type="json-file", config=config)

    @staticmethod
    def _execution_environment(allowlist_enabled: bool) -> dict[str, str]:
        environment = {
            "HOME": "/tmp",
            "TMPDIR": "/tmp",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "CI": "1",
        }
        if allowlist_enabled:
            proxy_url = "http://projectforge-egress-proxy:3128"
            environment.update({
                "HTTP_PROXY": proxy_url,
                "HTTPS_PROXY": proxy_url,
                "ALL_PROXY": proxy_url,
                "http_proxy": proxy_url,
                "https_proxy": proxy_url,
                "all_proxy": proxy_url,
                "NO_PROXY": "",
                "no_proxy": "",
            })
        return environment

    def _wait_for_exit(self, container: Any, timeout_seconds: int) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    container.reload()
                    if getattr(container, "status", None) in {"exited", "dead"}:
                        return False
                except Exception:
                    pass
                try:
                    container.kill()
                except Exception:
                    # A process can exit between the last status check and kill.
                    pass
                try:
                    container.wait(timeout=5)
                except Exception:
                    pass
                return True
            container.reload()
            if getattr(container, "status", None) in {"exited", "dead"}:
                return False
            time.sleep(min(self.poll_interval_seconds, remaining))

    @staticmethod
    def _exit_code(container: Any, *, timed_out: bool) -> int:
        try:
            result = container.wait(timeout=5)
            if isinstance(result, dict) and result.get("StatusCode") is not None:
                return int(result["StatusCode"])
        except Exception:
            pass
        attrs = getattr(container, "attrs", None)
        if isinstance(attrs, dict):
            state = attrs.get("State", {})
            if isinstance(state, dict) and state.get("ExitCode") is not None:
                return int(state["ExitCode"])
        return 137 if timed_out else 1

    def _read_logs(self, container: Any, *, stdout: bool, stderr: bool) -> tuple[str, bool]:
        raw = container.logs(stdout=stdout, stderr=stderr)
        if isinstance(raw, str):
            data = raw.encode("utf-8")
        elif isinstance(raw, bytes):
            data = raw
        else:
            data = b"".join(chunk if isinstance(chunk, bytes) else str(chunk).encode("utf-8")
                             for chunk in raw)
        truncated = len(data) > self.max_output_bytes
        if truncated:
            marker = b"\n...[sandbox output truncated]...\n"
            if self.max_output_bytes <= len(marker):
                data = data[:self.max_output_bytes]
            else:
                data = data[:self.max_output_bytes - len(marker)] + marker
        return data.decode("utf-8", errors="replace"), truncated

    @staticmethod
    def _cleanup_container(container: Any | None) -> None:
        if container is None:
            return
        try:
            container.remove(force=True, v=True)
        except Exception:
            # Cleanup must not mask the original execution error. Docker will also
            # reap labeled containers after a worker restart (see reap_orphans).
            pass

    @staticmethod
    def _cleanup_network(network: Any | None) -> None:
        if network is None:
            return
        try:
            network.remove()
        except Exception:
            pass

    def reap_orphans(self, *, stale_after_seconds: int = MAX_TIMEOUT_SECONDS + 300) -> int:
        """Remove labeled resources older than the maximum sandbox lease plus grace.

        Call this explicitly during worker startup. The age threshold is deliberately
        longer than any allowed task timeout so another live worker's container is
        never killed during a normal run. The filter excludes unrelated Docker
        resources. Returns the number of containers removed.
        """
        stale_after_seconds = self._bounded_int(
            "stale_after_seconds", stale_after_seconds, MAX_TIMEOUT_SECONDS + 60, 86_400 * 30,
        )
        client = self._docker_client()
        cutoff = int(time.time()) - stale_after_seconds
        containers = client.containers.list(
            all=True,
            filters={"label": "projectforge.sandbox=true"},
        )
        removed = 0
        for container in containers:
            labels = self._resource_labels(container, network=False)
            created = self._label_epoch(labels)
            if created and created <= cutoff:
                self._cleanup_container(container)
                removed += 1
        for network in client.networks.list(filters={"label": "projectforge.sandbox=true"}):
            labels = self._resource_labels(network, network=True)
            created = self._label_epoch(labels)
            if created and created <= cutoff:
                self._cleanup_network(network)
        return removed

    @staticmethod
    def _resource_labels(resource: Any, *, network: bool) -> dict[str, str]:
        attrs = getattr(resource, "attrs", {})
        if not isinstance(attrs, dict):
            return {}
        if network:
            labels = attrs.get("Labels")
        else:
            config = attrs.get("Config") or {}
            labels = config.get("Labels") if isinstance(config, dict) else None
            labels = labels or attrs.get("Labels")
        return labels if isinstance(labels, dict) else {}

    @staticmethod
    def _label_epoch(labels: Mapping[str, str]) -> int:
        try:
            return int(labels.get("projectforge.created_at_epoch", "0"))
        except (TypeError, ValueError):
            return 0

    def _docker_client(self) -> Any:
        if self._client is None:
            try:
                import docker
            except ImportError as exc:
                raise SandboxConfigurationError(
                    "Docker SDK is optional; install ArticleGen with the projectforge-sandbox extra"
                ) from exc
            try:
                self._client = docker.from_env(timeout=15)
            except Exception as exc:
                raise SandboxExecutionError(f"cannot connect to Docker: {exc}") from exc
        return self._client

    def _create_archive(self, project_files: Mapping[str, str | bytes] | None,
                        project_dir: str | os.PathLike[str] | None) -> bytes:
        if project_files is not None and project_dir is not None:
            raise SandboxConfigurationError("provide project_files or project_dir, not both")
        files: dict[str, bytes] = {}
        total_bytes = 0
        if project_dir is not None:
            root = Path(project_dir).resolve(strict=True)
            if not root.is_dir():
                raise UnsafeProjectFileError("project_dir must be a directory")
            for current, dirnames, filenames in os.walk(root, followlinks=False):
                current_path = Path(current)
                safe_dirs: list[str] = []
                for dirname in sorted(dirnames):
                    child = current_path / dirname
                    if dirname.casefold() in _EXCLUDED_DIRECTORY_NAMES or dirname.casefold() == ".env":
                        continue
                    if child.is_symlink():
                        raise UnsafeProjectFileError(f"symlink directory is not permitted: {child.relative_to(root)}")
                    safe_dirs.append(dirname)
                dirnames[:] = safe_dirs
                for filename in sorted(filenames):
                    host_path = current_path / filename
                    relative = host_path.relative_to(root).as_posix()
                    if self._excluded_file(relative):
                        continue
                    if host_path.is_symlink() or not host_path.is_file():
                        raise UnsafeProjectFileError(f"only regular files may be copied: {relative}")
                    safe_path = self._safe_relative_path(relative)
                    if len(files) >= MAX_PROJECT_FILES:
                        raise UnsafeProjectFileError("project workspace exceeds max file count")
                    data = self._read_regular_file(host_path, max_bytes=self.max_project_bytes - total_bytes)
                    total_bytes += len(data)
                    if total_bytes > self.max_project_bytes:
                        raise UnsafeProjectFileError("project workspace exceeds max_project_bytes")
                    files[safe_path] = data
        elif project_files is not None:
            for raw_path, content in project_files.items():
                if len(files) >= MAX_PROJECT_FILES:
                    raise UnsafeProjectFileError("project files exceed max file count")
                safe_path = self._safe_relative_path(raw_path)
                if self._excluded_file(safe_path):
                    raise UnsafeProjectFileError(f"sensitive project file is not copied: {safe_path}")
                if not isinstance(content, (str, bytes)):
                    raise UnsafeProjectFileError("project_files values must be text or bytes")
                data = content.encode("utf-8") if isinstance(content, str) else content
                total_bytes += len(data)
                if total_bytes > self.max_project_bytes:
                    raise UnsafeProjectFileError("project files exceed max_project_bytes")
                files[safe_path] = data

        archive = io.BytesIO()
        directories: set[str] = set()
        for name in files:
            parts = PurePosixPath(name).parts
            directories.update("/".join(parts[:index]) for index in range(1, len(parts)))
        with tarfile.open(fileobj=archive, mode="w") as tar:
            for name in sorted(directories, key=lambda entry: (entry.count("/"), entry)):
                info = tarfile.TarInfo(name=name.rstrip("/") + "/")
                info.type = tarfile.DIRTYPE
                info.mode = 0o777
                info.uid = SANDBOX_UID
                info.gid = SANDBOX_GID
                info.uname = "sandbox"
                info.gname = "sandbox"
                info.mtime = 0
                tar.addfile(info)
            for name, data in sorted(files.items()):
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.mode = 0o666
                info.uid = SANDBOX_UID
                info.gid = SANDBOX_GID
                info.uname = "sandbox"
                info.gname = "sandbox"
                info.mtime = 0
                tar.addfile(info, io.BytesIO(data))
        return archive.getvalue()

    @staticmethod
    def _read_regular_file(path: Path, *, max_bytes: int) -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise UnsafeProjectFileError(f"cannot safely open project file {path.name!r}: {exc}") from exc
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise UnsafeProjectFileError(f"only regular files may be copied: {path.name}")
            if file_stat.st_size > max_bytes:
                raise UnsafeProjectFileError("project workspace exceeds max_project_bytes")
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, max_bytes - size + 1))
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise UnsafeProjectFileError("project workspace exceeds max_project_bytes")
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    @staticmethod
    def _excluded_file(path: str) -> bool:
        name = PurePosixPath(path).name.casefold()
        return (
            name in _EXCLUDED_FILE_NAMES
            or name.startswith(".env.")
            or Path(name).suffix.casefold() in _SECRET_SUFFIXES
        )

    @staticmethod
    def _safe_relative_path(value: str) -> str:
        if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
            raise UnsafeProjectFileError("project file paths must be safe relative POSIX paths")
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
            raise UnsafeProjectFileError(f"unsafe project file path: {value!r}")
        if re.match(r"^[A-Za-z]:", value) or ":" in value:
            raise UnsafeProjectFileError(f"unsafe project file path: {value!r}")
        folded_parts = [part.casefold() for part in path.parts]
        if (
            any(part in _EXCLUDED_DIRECTORY_NAMES for part in folded_parts)
            or ".env" in folded_parts[:-1]
        ):
            raise UnsafeProjectFileError(f"VCS/cache/secret paths are not copied: {value!r}")
        return path.as_posix()

    @staticmethod
    def _normalize_runtime(runtime: str) -> str:
        aliases = {
            "python": "python",
            "python3": "python",
            "python:3.11": "python",
            "node": "node",
            "nodejs": "node",
            "node:20": "node",
        }
        try:
            return aliases[runtime.strip().casefold()]
        except (AttributeError, KeyError) as exc:
            raise SandboxConfigurationError("runtime must be python or node") from exc

    @staticmethod
    def _normalize_command(command: str | Sequence[str]) -> list[str]:
        if isinstance(command, str):
            if not command.strip() or "\x00" in command:
                raise SandboxConfigurationError("command must be a non-empty string")
            return ["/bin/sh", "-c", command]
        if not isinstance(command, Sequence) or isinstance(command, (bytes, bytearray)):
            raise SandboxConfigurationError("command must be a shell string or argument sequence")
        parts = list(command)
        if not parts or any(not isinstance(part, str) or not part or "\x00" in part for part in parts):
            raise SandboxConfigurationError("command arguments must be non-empty strings")
        return parts

    @staticmethod
    def _bounded_int(name: str, value: int, minimum: int, maximum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise SandboxConfigurationError(f"{name} must be an integer between {minimum} and {maximum}")
        return value

    @staticmethod
    def _bounded_float(name: str, value: float, minimum: float, maximum: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise SandboxConfigurationError(f"{name} must be a number between {minimum} and {maximum}")
        value = float(value)
        if not minimum <= value <= maximum:
            raise SandboxConfigurationError(f"{name} must be a number between {minimum} and {maximum}")
        return value

    @staticmethod
    def _memory_bytes(value: int | str) -> int:
        if isinstance(value, bool):
            raise SandboxConfigurationError("memory_limit must be bytes or a Docker unit string")
        if isinstance(value, int):
            if value <= 0:
                raise SandboxConfigurationError("memory_limit must be positive")
            return value
        if not isinstance(value, str):
            raise SandboxConfigurationError("memory_limit must be bytes or a Docker unit string")
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b?)\s*", value, re.IGNORECASE)
        if not match:
            raise SandboxConfigurationError("memory_limit must look like '512m' or be a byte count")
        number = float(match.group(1))
        suffix = match.group(2).casefold()
        unit = suffix.rstrip("b")
        if unit.endswith("i"):
            unit = unit[:-1]
        power = {"": 0, "k": 1, "m": 2, "g": 3, "t": 4}[unit]
        multiplier = 1024**power
        result = int(number * multiplier)
        if result <= 0:
            raise SandboxConfigurationError("memory_limit must be positive")
        return result
