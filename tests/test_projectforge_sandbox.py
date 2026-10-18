from __future__ import annotations

import asyncio
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any

import pytest

from projectforge_sandbox import (
    SandboxConfigurationError,
    SandboxExecutionError,
    SandboxManager,
    UnsafeProjectFileError,
)
from projectforge_sandbox import manager as sandbox_module
from projectforge_sandbox.egress_proxy import (
    _allowed,
    _handle_client,
    _read_client_hello,
    normalize_allowlist,
)


class FakeNetwork:
    def __init__(self, name: str):
        self.name = name
        self.removed = False
        self.connected: list[tuple[Any, list[str] | None]] = []

    def connect(self, container, aliases=None):
        self.connected.append((container, aliases))

    def remove(self):
        self.removed = True


class FakeContainer:
    def __init__(self, kwargs: dict[str, Any], *, should_fail_start: bool = False,
                 run_forever: bool = False):
        self.kwargs = kwargs
        self.id = f"container-{len(kwargs)}"
        self.status = "created"
        self.attrs = {"State": {"ExitCode": 0}}
        self.removed = False
        self.killed = False
        self.archive: bytes | None = None
        self.calls: list[str] = []
        self.should_fail_start = should_fail_start
        self.run_forever = run_forever
        self.is_proxy = "egress-proxy" in kwargs["image"]

    def put_archive(self, path: str, archive: bytes) -> bool:
        self.calls.append("archive")
        assert path == "/workspace"
        self.archive = archive
        return True

    def start(self):
        self.calls.append("start")
        if self.should_fail_start:
            raise RuntimeError("start exploded")
        self.status = "running"

    def reload(self):
        return None

    def wait(self, timeout=None):
        return {"StatusCode": 0 if not self.killed else 137}

    def exec_run(self, command, *, stdout, stderr):
        self.calls.append("exec")
        if any("touch" in part for part in command):
            if not self.run_forever:
                self.status = "exited"
            return 0, b""
        assert "127.0.0.1" in command[-1]
        return 0, b""

    def logs(self, *, stdout: bool, stderr: bool):
        if self.is_proxy:
            return b""
        return b"unit stdout\n" if stdout else b"unit stderr\n"

    def kill(self):
        self.killed = True
        self.status = "exited"

    def remove(self, *, force: bool, v: bool):
        assert force and v
        self.removed = True


class FakeContainers:
    def __init__(self):
        self.created: list[FakeContainer] = []
        self.fail_next_start = False
        self.run_forever = False

    def create(self, **kwargs):
        container = FakeContainer(
            kwargs,
            should_fail_start=self.fail_next_start,
            run_forever=self.run_forever and "egress-proxy" not in kwargs["image"],
        )
        self.fail_next_start = False
        self.created.append(container)
        return container

    def list(self, **kwargs):
        return getattr(self, "listed", [])


class FakeNetworks:
    def __init__(self):
        self.created: list[FakeNetwork] = []

    def create(self, *, name: str, **kwargs):
        network = FakeNetwork(name)
        network.kwargs = kwargs
        self.created.append(network)
        return network

    def list(self, **kwargs):
        return getattr(self, "listed", [])


class FakeDockerClient:
    def __init__(self):
        self.containers = FakeContainers()
        self.networks = FakeNetworks()


def test_sandbox_copies_files_applies_hardening_and_cleans_up():
    client = FakeDockerClient()
    manager = SandboxManager(client=client, default_memory_limit="128m", default_cpu_limit=0.5)
    result = manager.run(
        "python3",
        ["python", "main.py"],
        {"main.py": "print('hello')", "data/input.txt": b"sample"},
    )

    assert result.runtime == "python"
    assert result.stdout == "unit stdout\n"
    assert result.stderr == "unit stderr\n"
    assert result.exit_code == 0
    assert not result.timed_out
    sandbox = client.containers.created[0]
    kwargs = sandbox.kwargs
    assert kwargs["network_mode"] == "none"
    assert kwargs["mem_limit"] == 128 * 1024**2
    assert kwargs["memswap_limit"] == kwargs["mem_limit"]
    assert kwargs["nano_cpus"] == 500_000_000
    assert kwargs["read_only"] is True
    assert kwargs["pids_limit"] == 128
    assert kwargs["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in kwargs["security_opt"]
    assert kwargs["user"] == "65532:65532"
    assert kwargs["tty"] is False
    assert not any(key.lower().endswith("_proxy") for key in kwargs["environment"])
    assert sandbox.removed
    assert sandbox.archive is not None
    assert sandbox.calls[:3] == ["start", "archive", "exec"]
    with tarfile.open(fileobj=io.BytesIO(sandbox.archive), mode="r:") as archive:
        assert archive.getnames() == ["data", "data/input.txt", "main.py"]
        assert archive.extractfile("main.py").read() == b"print('hello')"


def test_sandbox_bounds_captured_stdout_and_stderr():
    result = SandboxManager(client=FakeDockerClient(), max_output_bytes=8).run(
        "python", ["python", "main.py"], {"main.py": "print('hello')"},
    )
    assert result.output_truncated
    assert len(result.stdout.encode("utf-8")) <= 8
    assert len(result.stderr.encode("utf-8")) <= 8


def test_sandbox_rejects_unsafe_paths_and_secret_files_before_docker():
    client = FakeDockerClient()
    manager = SandboxManager(client=client)
    with pytest.raises(UnsafeProjectFileError, match="unsafe project file path"):
        manager.run("python", ["python", "main.py"], {"../outside": "no"})
    with pytest.raises(UnsafeProjectFileError, match="sensitive project file"):
        manager.run("python", ["python", "main.py"], {".env": "secret=1"})
    assert not client.containers.created


def test_sandbox_removes_container_when_container_start_fails():
    client = FakeDockerClient()
    client.containers.fail_next_start = True
    manager = SandboxManager(client=client)
    with pytest.raises(SandboxExecutionError, match="start exploded"):
        manager.run("python", ["python", "main.py"], {"main.py": "print('hi')"})
    assert client.containers.created[0].removed


def test_orphan_reaper_removes_only_resources_older_than_max_lease():
    import time

    client = FakeDockerClient()
    now = int(time.time())
    stale = FakeContainer({"image": "python"})
    stale.attrs = {"Labels": {
        "projectforge.sandbox": "true",
        "projectforge.created_at_epoch": str(now - 5000),
    }}
    active = FakeContainer({"image": "python"})
    active.attrs = {"Labels": {
        "projectforge.sandbox": "true",
        "projectforge.created_at_epoch": str(now),
    }}
    network = FakeNetwork("stale-network")
    network.attrs = {"Labels": {
        "projectforge.sandbox": "true",
        "projectforge.created_at_epoch": str(now - 5000),
    }}
    client.containers.listed = [stale, active]
    client.networks.listed = [network]

    removed = SandboxManager(client=client).reap_orphans()
    assert removed == 1
    assert stale.removed
    assert not active.removed
    assert network.removed


def test_wall_clock_timeout_kills_and_removes_container(monkeypatch):
    client = FakeDockerClient()
    client.containers.run_forever = True
    ticks = iter([0.0, 0.0, 0.0, 2.0, 2.0])
    monkeypatch.setattr(sandbox_module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(sandbox_module.time, "sleep", lambda _seconds: None)
    manager = SandboxManager(client=client, poll_interval_seconds=0.01)
    result = manager.run("python", ["python", "loop.py"], {"loop.py": "while True: pass"}, timeout_seconds=1)
    container = client.containers.created[0]
    assert result.timed_out
    assert result.exit_code == 137
    assert container.killed
    assert container.removed


def test_allowlist_uses_internal_sandbox_network_and_filtered_proxy():
    client = FakeDockerClient()
    manager = SandboxManager(client=client)
    result = manager.run(
        "node",
        ["node", "index.js"],
        {"index.js": "console.log('ok')"},
        network_allowlist=["example.org", "*.packages.example"],
    )
    assert result.exit_code == 0
    assert len(client.networks.created) == 2
    internal, external = client.networks.created
    assert internal.kwargs["internal"] is True
    assert external.kwargs["internal"] is False
    proxy, sandbox = client.containers.created
    assert proxy.kwargs["network"] == external.name
    assert json.loads(proxy.kwargs["environment"]["PROJECTFORGE_EGRESS_ALLOWLIST"]) == [
        "*.packages.example", "example.org",
    ]
    assert (proxy, ["projectforge-egress-proxy"]) in internal.connected
    assert sandbox.kwargs["network"] == internal.name
    assert "network_mode" not in sandbox.kwargs
    assert sandbox.kwargs["environment"]["HTTPS_PROXY"] == "http://projectforge-egress-proxy:3128"
    assert proxy.removed and sandbox.removed
    assert internal.removed and external.removed


def test_allowlist_only_matches_exact_hosts_and_explicit_subdomains():
    rules = normalize_allowlist(["Example.org", "*.packages.example:8443"])
    assert _allowed("example.org", 443, rules)
    assert not _allowed("sub.example.org", 443, rules)
    assert _allowed("a.packages.example", 8443, rules)
    assert not _allowed("packages.example", 8443, rules)
    assert not _allowed("a.packages.example", 443, rules)
    with pytest.raises(ValueError, match="wildcards"):
        normalize_allowlist(["example.*.org"])


def test_tls_client_hello_sni_is_parsed_for_allowlist_checks():
    hostname = b"example.org"
    server_name = b"\x00" + len(hostname).to_bytes(2, "big") + hostname
    sni_list = len(server_name).to_bytes(2, "big") + server_name
    sni_extension = b"\x00\x00" + len(sni_list).to_bytes(2, "big") + sni_list
    extensions = len(sni_extension).to_bytes(2, "big") + sni_extension
    body = (
        b"\x03\x03" + (b"R" * 32) + b"\x00"
        + b"\x00\x02\x13\x01" + b"\x01\x00" + extensions
    )
    handshake = b"\x01" + len(body).to_bytes(3, "big") + body
    record = b"\x16\x03\x03" + len(handshake).to_bytes(2, "big") + handshake

    async def parse() -> tuple[str | None, bytes]:
        reader = asyncio.StreamReader()
        reader.feed_data(record)
        reader.feed_eof()
        return await _read_client_hello(reader)

    sni, forwarded = asyncio.run(parse())
    assert sni == "example.org"
    assert forwarded == record


def test_egress_proxy_denies_non_allowlisted_and_private_destinations():
    async def exercise_request(rules: tuple[str, ...], target: str) -> bytes:
        server = await asyncio.start_server(
            lambda reader, writer: _handle_client(reader, writer, rules),
            "127.0.0.1", 0,
        )
        async with server:
            port = server.sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
            await writer.drain()
            response = await reader.read()
            writer.close()
            await writer.wait_closed()
            return response

    denied_host = asyncio.run(exercise_request(normalize_allowlist(["example.org"]), "blocked.example:443"))
    assert denied_host.startswith(b"HTTP/1.1 403")
    private_target = asyncio.run(exercise_request(normalize_allowlist(["127.0.0.1"]), "127.0.0.1:443"))
    assert private_target.startswith(b"HTTP/1.1 403")


def test_project_dir_excludes_secrets_and_rejects_symlinks(tmp_path: Path):
    manager = SandboxManager(client=FakeDockerClient())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "main.py").write_text("print('safe')", encoding="utf-8")
    (workspace / ".env").write_text("TOKEN=not-for-container", encoding="utf-8")
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("remote=https://user:secret@example.test", encoding="utf-8")
    archive = manager._create_archive(None, workspace)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        assert tar.getnames() == ["main.py"]

    (workspace / "outside-link").symlink_to(workspace / "main.py")
    with pytest.raises(UnsafeProjectFileError, match="regular files"):
        manager._create_archive(None, workspace)


def test_resource_limits_and_runtime_are_validated_before_docker():
    client = FakeDockerClient()
    manager = SandboxManager(client=client)
    with pytest.raises(SandboxConfigurationError, match="runtime must be python or node"):
        manager.run("ruby", ["ruby", "main.rb"])
    with pytest.raises(SandboxConfigurationError, match="timeout_seconds"):
        manager.run("python", ["python", "main.py"], timeout_seconds=3601)
    with pytest.raises(SandboxConfigurationError, match="memory_limit"):
        manager.run("python", ["python", "main.py"], memory_limit="0m")
    assert not client.containers.created


def _docker_manager() -> SandboxManager:
    require_docker = os.environ.get("PROJECTFORGE_REQUIRE_DOCKER") == "1"
    try:
        import docker
    except ImportError as exc:
        if require_docker:
            pytest.fail(f"Docker SDK is required in this test run: {exc}")
        pytest.skip("Docker SDK optional extra is not installed")
    try:
        client = docker.from_env(timeout=5)
        client.ping()
    except Exception as exc:
        if require_docker:
            pytest.fail(f"Docker daemon is required in this test run: {exc}")
        pytest.skip(f"Docker daemon is unavailable: {exc}")
    manager = SandboxManager(client=client)
    for image in manager.images.values():
        try:
            client.images.get(image)
        except Exception as exc:
            if require_docker:
                pytest.fail(f"required prebuilt image {image!r} is unavailable: {exc}")
            pytest.skip(f"required prebuilt image {image!r} is unavailable")
    return manager


@pytest.mark.docker
def test_docker_python_pytest_runner_passes_and_captures_output():
    manager = _docker_manager()
    result = manager.run(
        "python",
        ["pytest", "-q", "test_math.py"],
        {"test_math.py": "def test_addition():\n    assert 2 + 2 == 4\n"},
        timeout_seconds=20,
    )
    assert result.exit_code == 0
    assert "1 passed" in result.stdout
    assert not result.timed_out


@pytest.mark.docker
def test_docker_python_pytest_runner_reports_a_failing_test():
    manager = _docker_manager()
    result = manager.run(
        "python",
        ["pytest", "-q", "test_math.py"],
        {"test_math.py": "def test_failure():\n    assert 2 + 2 == 5\n"},
        timeout_seconds=20,
    )
    assert result.exit_code != 0
    assert "1 failed" in result.stdout
    assert not result.timed_out


@pytest.mark.docker
def test_docker_kills_infinite_loop_at_wall_clock_timeout():
    manager = _docker_manager()
    result = manager.run(
        "python",
        ["python", "loop.py"],
        {"loop.py": "while True:\n    pass\n"},
        timeout_seconds=2,
    )
    assert result.timed_out
    assert result.exit_code != 0
    assert result.duration_seconds < 15


@pytest.mark.docker
def test_docker_default_network_blocks_outbound_connection():
    manager = _docker_manager()
    script = """import socket
try:
    socket.create_connection(("1.1.1.1", 443), timeout=1)
except OSError as exc:
    print("blocked", type(exc).__name__)
else:
    raise SystemExit("network unexpectedly available")
"""
    result = manager.run(
        "python",
        ["python", "network_check.py"],
        {"network_check.py": script},
        timeout_seconds=10,
    )
    assert result.exit_code == 0
    assert "blocked" in result.stdout
    assert not result.timed_out


@pytest.mark.docker
def test_docker_node_jest_runner_executes_tests():
    manager = _docker_manager()
    result = manager.run(
        "node",
        ["jest", "--runInBand", "--color=false"],
        {"sum.test.js": "test('adds', () => { expect(2 + 2).toBe(4); });\n"},
        timeout_seconds=30,
    )
    assert result.exit_code == 0
    assert "PASS" in result.stdout
