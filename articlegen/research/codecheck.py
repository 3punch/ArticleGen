"""Code extraction, isolated execution, test capture and technical verification.

The checks in this module are about what actually ran, not what the manuscript says
ran:

* ``SubprocessExecutor`` writes the experiment and its tests into a temporary
  directory, runs them through the current interpreter in a fresh process, and keeps
  stdout, stderr, the exit code, timing, hashes, discovered test outcomes and any
  metrics the experiment wrote. Timeouts and crashes are recorded as outcomes.
* ``verify_code`` compares the displayed code with the executed code, detects
  untested/illustrative snippets, compares claimed test counts with executed tests,
  and looks for two algorithmic red flags:
    - prose claiming generality while the implementation enumerates a hard-coded
      finite domain (the Erdős–Gyárfás hard-coded cycle-length failure);
    - strong normative claims about behaviour with no enforcing guard in the code.

Nothing here "proves" an algorithm correct. Execution and passing tests are evidence
about the executed inputs, and the findings say so in those words.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..models import Draft
from .models import (
    ExperimentRun,
    ExperimentSpec,
    ExperimentStatus,
    ReviewSeverity,
    TestOutcome,
    classify_section_role,
)

CODE_BLOCK_RE = re.compile(r"```([^\n]*)\n(.*?)```", re.DOTALL)
TEST_DEF_RE = re.compile(r"(?m)^\s*def\s+(test_\w+)\s*\(")
METRIC_LINE_RE = re.compile(r"(?im)^\s*METRIC[: ]\s*([A-Za-z][\w .\-/]*?)\s*[=:]\s*"
                            r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([A-Za-z%/µ]+)?\s*$")
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
CLAIMED_TESTS_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,3})"
    r"(?:\s*[- ]case)?\s*(?:unit\s+|integration\s+|regression\s+|worked\s+)?"
    r"(?:tests?|test\s+cases?)(?:\s+in\s+the\s+(?:test\s+)?suite|\s+suite)?\b",
    re.IGNORECASE,
)
CLAIMED_TESTS_SUITE_RE = re.compile(
    r"\b(?:suite|set)\s+of\s+(one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|\d{1,3})\s+(?:unit\s+|integration\s+|regression\s+)?tests?\b",
    re.IGNORECASE,
)
GENERALITY_CLAIM_RE = re.compile(
    r"\b(arbitrary|any\s+(?:n|size|number|order|degree|graph|input|length)|"
    r"for\s+all\s+(?:n|sizes?|orders?|degrees?|graphs?|inputs?|lengths?)|"
    r"all\s+(?:supported\s+)?(?:sizes?|orders?|degrees?|graphs?|inputs?|lengths?)|"
    r"general(?:ly)?|in\s+general|powers?\s+of\s+two|2\s*\^\s*[kn])\b",
    re.IGNORECASE,
)
STRONG_NORMATIVE_RE = re.compile(
    r"\b(guarantee[sd]?|ensure[sd]?|eliminat\w+|enforc\w+|always|never|"
    r"in\s+both\s+(?:rotational\s+)?directions|both\s+directions|"
    r"preserv\w+\s+(?:the\s+)?invariant|invariant\s+is\s+maintained|"
    r"cannot\s+(?:occur|happen|traverse)|prove[sd]?\s+that)\b",
    re.IGNORECASE,
)
_ILLUSTRATIVE_MARKERS = ("...", "…#", "TODO", "omitted for brevity",
                         "pseudo-code", "pseudocode")
_DOMAIN_NAME_RE = re.compile(
    r"(length|size|order|degree|cycle|case|dimension|bound|limit|n\b|domain|"
    r"values?|inputs?|samples?|levels?|steps?)", re.IGNORECASE)


def hash_code(text: str) -> str:
    """Fingerprint a program's text.

    Trailing blank lines are formatting, not code: a fenced listing in Markdown always
    ends with the newline before the closing fence, while the executed string need not.
    Stripping them keeps "displayed == executed" a statement about the program rather
    than about its file ending.
    """
    return hashlib.sha256((text or "").rstrip("\n").encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@dataclass
class CodeBlock:
    section_id: str
    section_role: str
    index: int
    language: str
    body: str
    sha256: str = ""

    def __post_init__(self) -> None:
        if not self.sha256:
            self.sha256 = hash_code(self.body)

    @property
    def executable(self) -> bool:
        return self.language.strip().lower() in ("python", "py", "python3")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "section_role": self.section_role,
            "index": self.index,
            "language": self.language,
            "sha256": self.sha256,
            "executable": self.executable,
            "lines": len(self.body.splitlines()),
        }


def extract_code_blocks(draft: Draft) -> List[CodeBlock]:
    blocks: List[CodeBlock] = []
    for section in draft.sections:
        role = classify_section_role(section.id, section.title)
        for index, (lang, body) in enumerate(CODE_BLOCK_RE.findall(section.body), start=1):
            blocks.append(CodeBlock(section.id, role, index, lang.strip(), body))
    return blocks


def code_from_payload(payload: Dict[str, Any]) -> str:
    blocks = payload.get("code_blocks") or []
    if isinstance(blocks, list):
        return "\n\n".join(str(b.get("body") or b.get("code") or "")
                           for b in blocks if isinstance(b, dict))
    return ""


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


_RUNNER_SOURCE = r'''
import json, os, runpy, sys, time, traceback

MARKER = "__ARTICLEGEN_RESULT__"
payload = {"ok": True, "metrics": {}}
started = time.time()
try:
    runpy.run_path("artifact.py", run_name="__main__")
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    if code:
        payload.update({"ok": False, "error_type": "SystemExit",
                        "error": f"artifact exited with status {code}"})
    sys.exit(code or 0)
except BaseException as exc:
    traceback.print_exc()
    payload.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
    print(MARKER + json.dumps(payload), flush=True)
    sys.exit(1)
if os.path.exists("metrics.json"):
    try:
        loaded = json.load(open("metrics.json", encoding="utf-8"))
        if isinstance(loaded, dict):
            payload["metrics"] = loaded
    except Exception as exc:
        payload["metric_error"] = f"{type(exc).__name__}: {exc}"
payload["duration_seconds"] = time.time() - started
print(MARKER + json.dumps(payload, default=str), flush=True)
'''


_TESTS_SOURCE = r'''
import importlib.util, json, os, sys, traceback, unittest

MARKER = "__ARTICLEGEN_TESTS__"
outcomes = []
error = ""


class Collect(unittest.TestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = []

    def _detail(self, err):
        try:
            return "".join(traceback.format_exception(*err))[-1200:]
        except Exception:
            return "unavailable traceback"

    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcomes.append({"name": str(test), "status": "passed", "detail": ""})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.outcomes.append({"name": str(test), "status": "failed",
                              "detail": self._detail(err)})

    def addError(self, test, err):
        super().addError(test, err)
        self.outcomes.append({"name": str(test), "status": "error",
                              "detail": self._detail(err)})

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.outcomes.append({"name": str(test), "status": "skipped", "detail": reason})

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self.outcomes.append({"name": str(test), "status": "passed",
                              "detail": "expected failure"})

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.outcomes.append({"name": str(test), "status": "failed",
                              "detail": "unexpected success"})


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


try:
    load("artifact", "artifact.py")
    tests = load("generated_tests", "tests.py")
    suite = unittest.TestLoader().loadTestsFromModule(tests)
    runner = unittest.TextTestRunner(resultclass=Collect, verbosity=0,
                                     stream=open(os.devnull, "w"))
    result = runner.run(suite)
    outcomes = result.outcomes
    tests_run = result.testsRun
except BaseException as exc:
    traceback.print_exc()
    error = f"{type(exc).__name__}: {exc}"
    tests_run = 0

print(MARKER + json.dumps({"tests": outcomes, "tests_run": tests_run, "error": error}),
      flush=True)
sys.exit(1 if error else 0)
'''


def execution_environment() -> Dict[str, Any]:
    """Capture the metadata a reader needs to interpret a measurement."""
    dependencies: Dict[str, str] = {}
    try:
        from importlib import metadata as importlib_metadata
        for name in ("numpy", "scipy", "networkx", "pandas", "matplotlib",
                     "sympy", "pytest"):
            try:
                dependencies[name] = importlib_metadata.version(name)
            except importlib_metadata.PackageNotFoundError:
                continue
    except Exception:  # pragma: no cover - platform defensive
        pass
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "dependencies": dependencies,
        "executable": os.path.basename(sys.executable),
    }


class SubprocessExecutor:
    """Runs Python experiments in a throw-away directory and a separate process.

    This is *process* isolation, not a security sandbox: there is no container,
    no network namespace and no seccomp policy. That limitation is reported with the
    results so a reader does not mistake it for hermetic execution.
    """

    name = "subprocess"

    def __init__(self, *, timeout: int = 30, python: Optional[str] = None) -> None:
        self.timeout = max(1, int(timeout))
        self.python = python or sys.executable

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _minimal_env(workdir: str) -> Dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": workdir,
            "TMPDIR": workdir,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), 9)
            else:  # pragma: no cover - Windows
                process.kill()
        except Exception:  # pragma: no cover - defensive
            pass

    def _write(self, path: str, text: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def _spawn(self, command: List[str], workdir: str, timeout: int) -> Dict[str, Any]:
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                command, cwd=workdir, env=self._minimal_env(workdir),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, text=True,
                start_new_session=True,
            )
        except OSError as exc:
            return {"timed_out": False, "error": f"{type(exc).__name__}: {exc}",
                    "stdout": "", "stderr": "", "exit_code": None,
                    "duration": time.monotonic() - started}
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            self._kill(process)
            stdout, stderr = process.communicate()
            timed_out = True
        return {
            "timed_out": timed_out,
            "error": "",
            "stdout": stdout or "",
            "stderr": stderr or "",
            "exit_code": process.returncode,
            "duration": time.monotonic() - started,
        }

    @staticmethod
    def _extract_marker(stdout: str, marker: str) -> Dict[str, Any]:
        for line in reversed((stdout or "").splitlines()):
            if line.startswith(marker):
                try:
                    value = json.loads(line[len(marker):])
                    return value if isinstance(value, dict) else {}
                except json.JSONDecodeError:
                    return {}
        return {}

    @staticmethod
    def _parse_metric_lines(stdout: str) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        for match in METRIC_LINE_RE.finditer(stdout or ""):
            name = re.sub(r"\s+", "_", match.group(1).strip().lower())
            try:
                value: Any = float(match.group(2))
            except ValueError:
                value = match.group(2)
            metrics[name] = value
            unit = (match.group(3) or "").strip()
            if unit:
                metrics[f"{name}_unit"] = unit
        return metrics

    @staticmethod
    def _last_error_type(stderr: str) -> str:
        match = re.findall(r"(?m)^([A-Za-z_][\w.]*(?:Error|Exception|Exit))\s*:", stderr or "")
        return match[-1].split(".")[-1] if match else ""

    # -- public API -------------------------------------------------------
    def run(self, spec: ExperimentSpec, *, timeout: Optional[int] = None) -> ExperimentRun:
        timeout = max(1, int(timeout or self.timeout))
        run = ExperimentRun(
            experiment_id=spec.experiment_id,
            run_id=f"{spec.experiment_id}-{hash_code(spec.code)[:8]}",
            status=ExperimentStatus.NOT_RUN,
            code_sha256=hash_code(spec.code),
            repetitions=max(1, spec.repetitions),
            environment=execution_environment(),
            started_at=utc_now(),
        )
        language = (spec.language or "python").strip().lower()
        if language not in ("python", "py", "python3"):
            run.status = ExperimentStatus.SKIPPED
            run.error = (f"cannot execute language '{spec.language}' in this environment; "
                         "the code is unverified, not tested")
            run.finished_at = utc_now()
            return run
        if not spec.code.strip():
            run.status = ExperimentStatus.ERROR
            run.error = "experiment has no code to execute"
            run.finished_at = utc_now()
            return run

        workdir = tempfile.mkdtemp(prefix="articlegen-exp-")
        try:
            self._write(os.path.join(workdir, "artifact.py"), spec.code)
            if spec.tests.strip():
                self._write(os.path.join(workdir, "tests.py"), spec.tests)
            self._write(os.path.join(workdir, "runner.py"), _RUNNER_SOURCE)
            self._write(os.path.join(workdir, "run_tests.py"), _TESTS_SOURCE)

            repetition_metrics: List[Dict[str, Any]] = []
            primary: Optional[Dict[str, Any]] = None
            total_duration = 0.0
            last_stdout = ""
            last_stderr = ""
            last_code: Optional[int] = None
            timed_out = False
            error = ""
            for repetition in range(max(1, spec.repetitions)):
                result = self._spawn(
                    [self.python, "-I", "-B", "runner.py"], workdir, timeout
                )
                total_duration += float(result["duration"])
                last_stdout, last_stderr = result["stdout"], result["stderr"]
                last_code = result["exit_code"]
                if result["error"]:
                    error = result["error"]
                    break
                if result["timed_out"]:
                    timed_out = True
                    break
                payload = self._extract_marker(last_stdout, "__ARTICLEGEN_RESULT__")
                metrics = dict(payload.get("metrics") or {})
                metrics.update(self._parse_metric_lines(last_stdout))
                repetition_metrics.append(metrics)
                if primary is None:
                    primary = metrics
                if payload and not payload.get("ok", True):
                    error = payload.get("error") or "artifact raised an exception"
                    run.error_type = str(payload.get("error_type") or "")
                    break

            if timed_out:
                run.status = ExperimentStatus.TIMEOUT
                run.error = f"execution timed out after {timeout}s"
            elif error and not repetition_metrics:
                run.status = ExperimentStatus.ERROR
                run.error = error
            elif last_code not in (0, None) or error:
                run.status = ExperimentStatus.ERROR
                run.error = error or f"process exited with status {last_code}"
            else:
                run.status = ExperimentStatus.OK

            if run.status == ExperimentStatus.OK and primary is not None:
                aggregate: Dict[str, Any] = dict(primary)
                if len(repetition_metrics) > 1:
                    for key in list(primary):
                        if key.endswith("_unit"):
                            continue
                        values = [m[key] for m in repetition_metrics
                                  if isinstance(m.get(key), (int, float))]
                        if len(values) == len(repetition_metrics) and values:
                            aggregate[f"{key}_mean"] = sum(values) / len(values)
                            aggregate[f"{key}_min"] = min(values)
                            aggregate[f"{key}_max"] = max(values)
                            if len(values) > 1:
                                mean = aggregate[f"{key}_mean"]
                                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                                aggregate[f"{key}_sd"] = variance ** 0.5
                run.metrics = aggregate
            run.repetition_metrics = repetition_metrics
            run.stdout = last_stdout[-20000:]
            run.stderr = last_stderr[-20000:]
            run.exit_code = last_code
            run.duration_seconds = total_duration
            if run.status != ExperimentStatus.OK and not run.error_type:
                run.error_type = self._last_error_type(last_stderr)

            if spec.tests.strip():
                test_result = self._spawn(
                    [self.python, "-I", "-B", "run_tests.py"], workdir, timeout
                )
                if test_result["timed_out"]:
                    run.tests = [TestOutcome("<test suite>", "error",
                                             f"test run timed out after {timeout}s")]
                else:
                    payload = self._extract_marker(test_result["stdout"],
                                                   "__ARTICLEGEN_TESTS__")
                    run.tests = [TestOutcome.from_dict(item)
                                 for item in (payload.get("tests") or [])
                                 if isinstance(item, dict)]
                    if payload.get("error"):
                        run.tests.append(TestOutcome("<test module>", "error",
                                                     str(payload["error"])))
                    if not run.tests and test_result["exit_code"] not in (0, None):
                        run.tests.append(TestOutcome(
                            "<test suite>", "error",
                            (test_result["stderr"] or "test run failed")[-800:]))
                run.artifacts["tests_sha256"] = hash_code(spec.tests)
            for name in ("metrics.json", "results.json", "figure.png"):
                path = os.path.join(workdir, name)
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8", errors="replace") as handle:
                            content = handle.read()
                        run.artifacts[name] = content[:20000]
                    except OSError:
                        continue
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        run.finished_at = utc_now()
        return run


# ---------------------------------------------------------------------------
# Static analysis
# ---------------------------------------------------------------------------


@dataclass
class HardcodedDomain:
    name: str
    values: List[Any]
    line: int
    code: str


@dataclass
class GeneralityGap:
    domain: HardcodedDomain
    claim: str
    severity: ReviewSeverity = ReviewSeverity.MAJOR
    detail: str = ""


@dataclass
class EnforcementGap:
    claim: str
    reason: str
    line: Optional[int] = None
    severity: ReviewSeverity = ReviewSeverity.MAJOR


@dataclass
class CodeAnalysis:
    blocks: List[CodeBlock] = field(default_factory=list)
    compile_errors: List[Dict[str, Any]] = field(default_factory=list)
    unlabelled_blocks: List[CodeBlock] = field(default_factory=list)
    illustrative_blocks: List[CodeBlock] = field(default_factory=list)
    executable_blocks: List[CodeBlock] = field(default_factory=list)
    unexecuted_blocks: List[CodeBlock] = field(default_factory=list)
    hash_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    claimed_tests: Optional[int] = None
    displayed_tests: int = 0
    executed_tests: int = 0
    executed_passed: int = 0
    test_claim_text: str = ""
    generality_gaps: List[GeneralityGap] = field(default_factory=list)
    enforcement_gaps: List[EnforcementGap] = field(default_factory=list)
    runtime_errors: List[Dict[str, str]] = field(default_factory=list)
    missing_dependencies: List[str] = field(default_factory=list)
    unexecuted_code_claims: List[str] = field(default_factory=list)

    @property
    def has_executable_code(self) -> bool:
        return bool(self.executable_blocks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "compile_errors": self.compile_errors,
            "unlabelled_blocks": [b.to_dict() for b in self.unlabelled_blocks],
            "illustrative_blocks": [b.to_dict() for b in self.illustrative_blocks],
            "unexecuted_blocks": [b.to_dict() for b in self.unexecuted_blocks],
            "hash_mismatches": self.hash_mismatches,
            "claimed_tests": self.claimed_tests,
            "displayed_tests": self.displayed_tests,
            "executed_tests": self.executed_tests,
            "executed_passed": self.executed_passed,
            "test_claim_text": self.test_claim_text,
            "generality_gaps": [
                {"values": gap.domain.values, "name": gap.domain.name,
                 "line": gap.domain.line, "claim": gap.claim,
                 "severity": gap.severity.value, "detail": gap.detail}
                for gap in self.generality_gaps
            ],
            "enforcement_gaps": [
                {"claim": gap.claim, "reason": gap.reason,
                 "severity": gap.severity.value}
                for gap in self.enforcement_gaps
            ],
            "runtime_errors": self.runtime_errors,
            "missing_dependencies": self.missing_dependencies,
            "unexecuted_code_claims": self.unexecuted_code_claims,
        }


def _numeric_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
        and not isinstance(node.value, bool)


def find_hardcoded_numeric_domains(code: str) -> List[HardcodedDomain]:
    """Find finite numeric domains written out as literals instead of generated."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    domains: List[HardcodedDomain] = []

    def describe(nodes: Iterable[ast.AST]) -> Optional[List[Any]]:
        items = list(nodes)
        if len(items) >= 2 and all(_numeric_literal(item) for item in items):
            return [getattr(item, "value") for item in items]
        return None

    for node in ast.walk(tree):
        values: Optional[List[Any]] = None
        name = ""
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                values = describe(value.elts)
            target = node.targets[0] if isinstance(node, ast.Assign) and node.targets \
                else getattr(node, "target", None)
            name = getattr(target, "id", "") if isinstance(target, ast.Name) else ""
        elif isinstance(node, (ast.For, ast.comprehension)):
            iterator = getattr(node, "iter", None)
            if isinstance(iterator, (ast.List, ast.Tuple, ast.Set)):
                values = describe(iterator.elts)
                name = "for-loop domain"
        elif isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if isinstance(comparator, (ast.List, ast.Tuple, ast.Set)):
                    values = describe(comparator.elts)
                    name = name or "membership test"
        if values:
            domains.append(HardcodedDomain(
                name=name or "literal domain",
                values=values,
                line=getattr(node, "lineno", 0),
                code=ast.get_source_segment(code, node) or "",
            ))
    return domains


def _code_defines_domain_dynamically(code: str) -> bool:
    """True when the code also builds a parameterised domain (range/pow/2**)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("range", "pow"):
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            return True
    return False


def detect_generality_gaps(code: str, prose: str) -> List[GeneralityGap]:
    """Prose claims generality; code enumerates a finite literal domain.

    This is the generalised form of the Erdős–Gyárfás regression: the article said
    the verifier generated powers of two for arbitrary supported graph sizes, while
    the implementation tested ``{4, 8, 16, 32}``.
    """
    claims = [m.group(0) for m in GENERALITY_CLAIM_RE.finditer(prose or "")]
    if not claims:
        return []
    domains = [d for d in find_hardcoded_numeric_domains(code)
               if _DOMAIN_NAME_RE.search(d.name) or _DOMAIN_NAME_RE.search(d.code)]
    if not domains:
        return []
    dynamic = _code_defines_domain_dynamically(code)
    gaps: List[GeneralityGap] = []
    for domain in domains:
        detail = (f"prose claims generality ({claims[0]!r}) while the implementation "
                  f"enumerates the fixed domain {domain.name}={domain.values} "
                  f"(line {domain.line})")
        if dynamic:
            detail += ("; the file also builds a parameterised range, so check which "
                       "path the manuscript's claim actually covers")
        gaps.append(GeneralityGap(
            domain=domain,
            claim=claims[0],
            severity=ReviewSeverity.MAJOR if dynamic else ReviewSeverity.CRITICAL,
            detail=detail,
        ))
    return gaps


def detect_enforcement_gaps(code: str, prose: str) -> List[EnforcementGap]:
    """Strong normative claims that the implementation never checks.

    A sentence saying the algorithm "guarantees" or "eliminates" a behaviour must be
    backed by an assertion/guard or by the mechanism itself. The heuristic looks for
    an ``assert``/``raise`` guard or for the claim's content words inside the code.
    It reports a gap for expert review - it cannot decide semantic equivalence.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    has_guard = any(isinstance(node, (ast.Assert, ast.Raise)) for node in ast.walk(tree))
    code_lower = code.lower()
    gaps: List[EnforcementGap] = []
    for sentence in re.split(r"(?<=[.!?])\s+", prose or ""):
        if not STRONG_NORMATIVE_RE.search(sentence):
            continue
        content_words = [w for w in re.findall(r"[a-z]{6,}", sentence.lower())
                         if w not in ("always", "guarantee", "guarantees", "ensures",
                                      "eliminates", "enforces", "cannot", "directions")]
        referenced = [w for w in content_words if w in code_lower]
        if has_guard:
            continue
        if referenced:
            continue
        reason = ("the prose makes a universal behavioural claim but the code contains "
                  "no assertion/guard and never refers to the claim's key concepts")
        gaps.append(EnforcementGap(claim=sentence.strip()[:240], reason=reason,
                                   severity=ReviewSeverity.MAJOR))
    return gaps


def claimed_test_count(text: str) -> Tuple[Optional[int], str]:
    for pattern in (CLAIMED_TESTS_RE, CLAIMED_TESTS_SUITE_RE):
        for match in pattern.finditer(text or ""):
            token = match.group(1).lower()
            count = NUMBER_WORDS.get(token)
            if count is None:
                try:
                    count = int(token)
                except ValueError:
                    continue
            return count, match.group(0)
    return None, ""


def analyse_code(draft: Draft, runs: Sequence[ExperimentRun], *,
                 article_type_research: bool = False,
                 specs: Sequence[ExperimentSpec] = ()) -> CodeAnalysis:
    """Deterministically analyse every code block and every recorded run."""
    analysis = CodeAnalysis()
    analysis.blocks = extract_code_blocks(draft)
    prose = "\n\n".join(s.body for s in draft.sections)
    prose_without_code = re.sub(r"```.*?```", " ", prose, flags=re.DOTALL)

    for block in analysis.blocks:
        if not block.language.strip():
            analysis.unlabelled_blocks.append(block)
        if block.executable:
            analysis.executable_blocks.append(block)
            try:
                compile(block.body, f"<{block.section_id}:block{block.index}>", "exec")
            except SyntaxError as exc:
                analysis.compile_errors.append({
                    "section": block.section_id,
                    "block": block.index,
                    "message": exc.msg,
                    "line": exc.lineno,
                    "code": (exc.text or "").strip()[:120],
                })
        lowered = block.body.lower()
        if any(marker.lower() in lowered for marker in _ILLUSTRATIVE_MARKERS):
            analysis.illustrative_blocks.append(block)

    executed_hashes = {run.code_sha256 for run in runs if run.status.ran and run.code_sha256}
    spec_hashes = {hash_code(spec.code) for spec in specs if spec.code.strip()}
    for block in analysis.executable_blocks:
        if block.sha256 not in executed_hashes and block.sha256 not in spec_hashes:
            analysis.unexecuted_blocks.append(block)
    for block in analysis.blocks:
        if block.sha256 in executed_hashes or block.sha256 in spec_hashes:
            continue
        if not block.executable:
            continue
        for spec in specs:
            if not spec.code.strip():
                continue
            if block.section_id == spec.section_id and block.sha256 != hash_code(spec.code):
                analysis.hash_mismatches.append({
                    "section": block.section_id,
                    "experiment_id": spec.experiment_id,
                    "displayed": block.sha256,
                    "executed": hash_code(spec.code),
                    "message": ("the displayed code differs from the code that was "
                                "executed, so the reported results do not correspond "
                                "to the shown listing"),
                })

    analysis.displayed_tests = len(TEST_DEF_RE.findall(
        "\n\n".join(b.body for b in analysis.blocks)
    ))
    analysis.executed_tests = max((run.tests_run for run in runs), default=0)
    analysis.executed_passed = max((run.tests_passed for run in runs), default=0)
    claimed, claim_text = claimed_test_count(prose_without_code)
    analysis.claimed_tests = claimed
    analysis.test_claim_text = claim_text

    for run in runs:
        if run.status in (ExperimentStatus.ERROR, ExperimentStatus.TIMEOUT,
                          ExperimentStatus.FAILED):
            analysis.runtime_errors.append({
                "experiment_id": run.experiment_id,
                "status": run.status.value,
                "error_type": run.error_type,
                "error": run.error[:400],
                "stderr": run.stderr[-400:],
            })
        if run.error_type in ("ModuleNotFoundError", "ImportError"):
            match = re.search(r"No module named '([^']+)'", run.stderr or run.error)
            if match:
                analysis.missing_dependencies.append(match.group(1))

    # Generality is a claim about the whole article, not just the methods section:
    # the Erdős–Gyárfás manuscript asserted arbitrary graph sizes in its introduction.
    method_prose = re.sub(r"```.*?```", " ", prose, flags=re.DOTALL)
    for block in analysis.executable_blocks:
        gaps = detect_generality_gaps(block.body, method_prose)
        for gap in gaps:
            gap.detail = f"{block.section_id}#{block.index}: {gap.detail}"
            analysis.generality_gaps.append(gap)
    for block in analysis.executable_blocks:
        analysis.enforcement_gaps.extend(detect_enforcement_gaps(block.body, method_prose))

    if article_type_research:
        completion_claims = re.findall(
            r"(?i)\b(we\s+(?:ran|executed|tested|measured)|our\s+tests?\s+"
            r"(?:pass|passed|show|showed))\b", prose_without_code
        )
        if completion_claims and not any(run.status.succeeded for run in runs):
            analysis.unexecuted_code_claims.append(
                "the manuscript claims executed tests or measurements but no "
                "experiment run succeeded"
            )
    return analysis


def tests_claimed_but_not_run(analysis: CodeAnalysis) -> Optional[str]:
    if analysis.claimed_tests is None:
        return None
    if analysis.executed_tests == 0 and analysis.claimed_tests > 0:
        return (f"the manuscript claims {analysis.claimed_tests} test case(s) "
                f"({analysis.test_claim_text!r}) but no test was executed")
    if analysis.claimed_tests > max(analysis.executed_tests, analysis.displayed_tests):
        return (f"the manuscript claims {analysis.claimed_tests} test case(s) "
                f"({analysis.test_claim_text!r}) but only {analysis.executed_tests} "
                f"were executed and {analysis.displayed_tests} appear in the displayed code")
    return None
