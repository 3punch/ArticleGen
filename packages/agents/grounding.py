"""Deterministic evidence selection and groundedness checks for project reports.

These checks are deliberately cheap and conservative: they do not try to prove that
an LLM's prose is true, but reject missing project context, placeholder leakage,
repeated filler, and reports that omit the actual source/test artifacts supplied to
the Packager.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from projectforge_schemas.models import ProjectSpec


_TOKEN_RE = re.compile(r"(?u)[^\W_]+(?:[-‐‑‒–—][^\W_]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_TEST_RESULT_RE = re.compile(r"\b\d+\s+(?:passed|failed|skipped|xfailed|xpassed)\b", re.I)
_SOURCE_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".cjs", ".mjs", ".ts", ".tsx",
    ".html", ".css", ".sql", ".sh",
}
_TEST_PARTS = {"test", "tests", "spec", "specs", "fixtures", "fixture"}
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "if", "in", "into", "is", "it", "its", "may", "might",
    "must", "of", "on", "or", "our", "shall", "should", "so", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "those",
    "through", "to", "under", "up", "use", "using", "was", "were", "will",
    "with", "within", "without", "would",
}
_GENERIC_TERMS = {
    "acceptance", "access", "agent", "analysis", "automatable", "automated", "automation",
    "brief", "build", "check", "checks", "clear", "clearly", "code", "criterion",
    "criteria", "data", "deliverable", "deliverables", "document", "documentation",
    "expected",
    "file", "files", "fixture", "fixtures", "human", "implementation", "include",
    "input", "inputs", "item", "items", "local", "method", "methods", "missing", "output",
    "outputs", "pass", "passed", "program", "project", "provided", "readme", "record",
    "required",
    "records", "report", "reported", "reports", "result", "results", "run", "runs",
    "section", "setup", "software", "source", "static", "support", "task", "tasks",
    "test", "tested", "testing", "tests", "use", "valid", "value",
    "values", "verify", "verified", "work", "workflow", "written",
}
_SECTION_ECHO_PHRASES = (
    "the core concept",
    "worked example",
    "trade-offs and limits",
)
_META_PHRASES = (
    "the brief",
    "reviewers will judge",
    "mathematicians will judge",
    "the argument proceeds in steps",
    "the writing process",
    "this section exists to",
    "the section name",
    "the reader will judge",
)
_IGNORED_SYMBOLS = {
    "__init__", "__main__", "main", "setup", "teardown", "run", "test",
    "create", "get", "set", "add", "remove", "update", "delete",
}


@dataclass(frozen=True)
class ReportGroundedness:
    """Auditable result of the deterministic report-quality gate."""

    required_terms: tuple[str, ...]
    matched_terms: tuple[str, ...]
    code_anchors: tuple[str, ...]
    matched_code_anchors: tuple[str, ...]
    test_anchors: tuple[str, ...]
    matched_test_anchors: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "required_terms": list(self.required_terms),
            "matched_terms": list(self.matched_terms),
            "code_anchors": list(self.code_anchors),
            "matched_code_anchors": list(self.matched_code_anchors),
            "test_anchors": list(self.test_anchors),
            "matched_test_anchors": list(self.matched_test_anchors),
            "artifact_paths": list(self.artifact_paths),
            "issues": list(self.issues),
        }


def extract_key_terms(
    project_spec: ProjectSpec | Mapping[str, Any],
    *,
    max_terms: int = 20,
) -> list[str]:
    """Extract deterministic domain anchors from a ProjectSpec.

    Every meaningful title term is retained. Additional terms are ranked by how
    often they occur across independent description/deliverable/rubric fields, which
    favors the concepts central to the assignment without requiring generic words
    such as "project", "report", or "test" to appear in the report.
    """
    if not 1 <= max_terms <= 100:
        raise ValueError("max_terms must be between 1 and 100")
    spec = project_spec if isinstance(project_spec, ProjectSpec) else ProjectSpec.model_validate(project_spec)

    title_terms: list[str] = []
    for token in _tokens(spec.title):
        if _is_key_term(token) and _canonical_term(token) not in {
            _canonical_term(value) for value in title_terms
        }:
            title_terms.append(token)

    field_texts = [spec.description, *spec.deliverables, *spec.constraints]
    field_texts.extend(item.criterion for item in spec.rubric_items)
    term_counts: Counter[str] = Counter()
    term_display: dict[str, str] = {}
    term_fields: dict[str, set[int]] = defaultdict(set)
    for field_index, text in enumerate(field_texts):
        for token in _tokens(text):
            if not _is_key_term(token):
                continue
            normalized = _canonical_term(token)
            term_counts[normalized] += 1
            term_display.setdefault(normalized, token)
            term_fields[normalized].add(field_index)

    title_keys = {_canonical_term(token) for token in title_terms}
    ranked = sorted(
        (key for key in term_counts if key not in title_keys),
        key=lambda key: (
            -len(term_fields[key]),
            -term_counts[key],
            -len(key),
            key,
        ),
    )
    result = list(title_terms)
    for key in ranked:
        if len(result) >= max_terms:
            break
        # A term that appears in one field only is usually incidental. Proper nouns,
        # hyphenated mathematical/technical terms, and numerals remain useful anchors.
        display = term_display[key]
        if (len(term_fields[key]) < 2 and not _looks_specific(display)):
            continue
        result.append(display)
    return result[:max_terms]


def has_repeated_sentences(text: str, *, threshold: float = 0.9) -> bool:
    """Return true when report sentences repeat exactly or are near-duplicates.

    Markdown headings, bullets, and short labels are normalized before comparison.
    Short sentences are ignored because lists often repeat brief labels legitimately.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be greater than zero and at most one")
    seen: list[str] = []
    exact: set[str] = set()
    for raw in _SENTENCE_RE.split(text):
        normalized = _normalized_sentence(raw)
        if len(normalized.split()) < 8:
            continue
        if normalized in exact:
            return True
        if any(SequenceMatcher(None, normalized, prior, autojunk=False).ratio() >= threshold
               for prior in seen):
            return True
        seen.append(normalized)
        exact.add(normalized)
    return False


def inspect_report_groundedness(
    report_text: str,
    project_spec: ProjectSpec | Mapping[str, Any],
    *,
    repository_files: Mapping[str, str],
    test_runs: Sequence[Mapping[str, Any]],
    agent_runs: Sequence[Mapping[str, Any]],
    review_results: Sequence[Mapping[str, Any]],
) -> ReportGroundedness:
    """Assess a generated report against its ProjectSpec and persisted artifacts."""
    spec = project_spec if isinstance(project_spec, ProjectSpec) else ProjectSpec.model_validate(project_spec)
    lowered = report_text.casefold()
    required_terms = extract_key_terms(spec)
    matched_terms = tuple(term for term in required_terms if _term_present(term, lowered))
    missing_terms = [term for term in required_terms if term not in matched_terms]
    issues: list[str] = []
    if not report_text.strip():
        issues.append("Report body is empty.")
    if missing_terms:
        issues.append("Report is missing expected ProjectSpec domain terms: " + ", ".join(missing_terms))

    for phrase in _SECTION_ECHO_PHRASES:
        if lowered.count(phrase) > 2:
            issues.append(f"Suspicious section-name echo appears more than twice: {phrase!r}.")
    if has_repeated_sentences(report_text, threshold=0.9):
        issues.append("Repeated near-identical sentences detected; likely template filler.")
    leaked_meta = [phrase for phrase in _META_PHRASES if phrase in lowered]
    if leaked_meta:
        issues.append("Report contains writing-process or reviewer meta-commentary: "
                      + ", ".join(repr(phrase) for phrase in leaked_meta) + ".")

    source_paths = tuple(sorted(
        path for path in repository_files
        if _is_source_path(path)
    ))
    if not repository_files:
        issues.append("Repository artifacts are missing; report generation is not grounded.")
    if not source_paths:
        issues.append("No implementation source files were supplied to ground the report.")

    code_anchors = tuple(_code_anchors(repository_files))
    matched_code_anchors = tuple(
        anchor for anchor in code_anchors if _literal_present(anchor, lowered)
    )
    if code_anchors and not matched_code_anchors:
        issues.append("Report does not identify any supplied implementation symbol or source path: "
                      + ", ".join(code_anchors[:8]) + ".")

    passing_runs = [
        run for run in test_runs
        if str(run.get("phase", "TEST")).upper() == "TEST"
        and run.get("passed") is True
        and run.get("exit_code") == 0
        and not bool(run.get("timed_out", False))
    ]
    if not test_runs:
        issues.append("Persisted sandbox test results are missing.")
    elif len(passing_runs) != len(test_runs):
        issues.append("One or more supplied sandbox TEST runs did not pass with exit code zero.")

    if not agent_runs:
        issues.append("Persisted AgentRun evidence is missing from the report context.")
    if not review_results:
        issues.append("Persisted ReviewResult evidence is missing from the report context.")

    test_anchors = tuple(_test_anchors(test_runs))
    matched_test_anchors = tuple(
        anchor for anchor in test_anchors if _literal_present(anchor, lowered)
    )
    test_file_anchors = tuple(dict.fromkeys(
        path
        for run in test_runs
        for path in (
            run.get("tested_files")
            if isinstance(run.get("tested_files"), (list, tuple)) else []
        )
        if isinstance(path, str) and path.strip()
    ))
    matched_test_files = tuple(
        path for path in test_file_anchors if _literal_present(path, lowered)
    )
    test_result_anchors = tuple(dict.fromkeys(
        anchor for anchor in test_anchors if _TEST_RESULT_RE.fullmatch(anchor)
    ))
    matched_test_results = tuple(
        anchor for anchor in test_result_anchors if _literal_present(anchor, lowered)
    )
    if test_file_anchors and not matched_test_files:
        issues.append("Report does not identify a supplied sandbox test file: "
                      + ", ".join(test_file_anchors[:8]) + ".")
    if test_result_anchors and not matched_test_results:
        issues.append("Report does not cite a concrete result from the supplied sandbox output: "
                      + ", ".join(test_result_anchors[:8]) + ".")
    if test_runs and test_anchors and not matched_test_anchors:
        issues.append("Report does not cite any supplied sandbox test file, task, or result: "
                      + ", ".join(test_anchors[:8]) + ".")

    # Every supplied artifact path is retained in the validation record, including
    # result files that are optional for projects without experiments.
    artifact_paths = tuple(sorted(set(repository_files)))
    return ReportGroundedness(
        required_terms=tuple(required_terms),
        matched_terms=matched_terms,
        code_anchors=code_anchors,
        matched_code_anchors=matched_code_anchors,
        test_anchors=test_anchors,
        matched_test_anchors=matched_test_anchors,
        artifact_paths=artifact_paths,
        issues=tuple(issues),
    )


def validate_report_groundedness(
    report_text: str,
    project_spec: ProjectSpec | Mapping[str, Any],
    *,
    repository_files: Mapping[str, str],
    test_runs: Sequence[Mapping[str, Any]],
    agent_runs: Sequence[Mapping[str, Any]],
    review_results: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return actionable deterministic issues; an empty list means checks passed."""
    return list(inspect_report_groundedness(
        report_text,
        project_spec,
        repository_files=repository_files,
        test_runs=test_runs,
        agent_runs=agent_runs,
        review_results=review_results,
    ).issues)


def build_code_summary(
    repository_files: Mapping[str, str],
    *,
    max_files: int = 8,
    max_total_chars: int = 8_000,
    max_excerpt_chars: int = 1_400,
) -> dict[str, Any]:
    """Create a bounded, source-backed module/function summary without executing code."""
    source_paths = sorted(path for path in repository_files if _is_source_path(path))
    ranked_paths = sorted(source_paths, key=lambda path: (
        _is_test_path(path),
        0 if any(part.casefold() in {"src", "app", "server", "analysis", "graphs", "lib"}
                 for part in Path(path).parts) else 1,
        path.casefold(),
    ))
    entries: list[dict[str, Any]] = []
    used_chars = 0
    for path in ranked_paths:
        if len(entries) >= max_files or used_chars >= max_total_chars:
            break
        content = repository_files[path]
        excerpt = _source_excerpt(content, max_excerpt_chars)
        remaining = max_total_chars - used_chars
        excerpt = excerpt[:remaining]
        if not excerpt.strip():
            continue
        entries.append({
            "path": path,
            "size_chars": len(content),
            "symbols": _symbols_in_file(path, content)[:20],
            "excerpt": excerpt,
        })
        used_chars += len(excerpt)
    return {
        "source_file_count": len(source_paths),
        "included_file_count": len(entries),
        "omitted_file_count": max(0, len(source_paths) - len(entries)),
        "files": entries,
    }


def extract_experiment_output(
    repository_files: Mapping[str, str],
    *,
    max_artifacts: int = 4,
    max_total_chars: int = 3_000,
    max_artifact_chars: int = 1_200,
) -> dict[str, Any]:
    """Select bounded, likely result artifacts (CSV/JSON/text/SVG), excluding tests."""
    result_markers = {"result", "results", "experiment", "experiments", "metric", "metrics",
                      "summary", "output", "outputs", "benchmark", "benchmarks"}
    candidates: list[str] = []
    for path in sorted(repository_files):
        parsed = Path(path)
        if parsed.suffix.casefold() not in {".json", ".csv", ".tsv", ".txt", ".svg"}:
            continue
        if _is_test_path(path):
            continue
        components = {part.casefold() for part in parsed.parts}
        stem_tokens = {token.casefold() for token in re.split(r"[^A-Za-z0-9]+", parsed.stem)}
        if components & result_markers or stem_tokens & result_markers:
            candidates.append(path)
    artifacts: list[dict[str, Any]] = []
    used_chars = 0
    for path in candidates[:max_artifacts]:
        remaining = max_total_chars - used_chars
        if remaining <= 0:
            break
        content = repository_files[path][:min(max_artifact_chars, remaining)]
        if not content.strip():
            continue
        artifacts.append({
            "path": path,
            "size_chars": len(repository_files[path]),
            "content_excerpt": content,
            "truncated": len(content) < len(repository_files[path]),
        })
        used_chars += len(content)
    return {
        "artifact_count": len(candidates),
        "included_artifact_count": len(artifacts),
        "artifacts": artifacts,
        "note": ("No result-like experiment artifact was found in the repository."
                 if not candidates else "Artifacts are repository files; no values were inferred."),
    }


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _is_key_term(token: str) -> bool:
    canonical = _canonical_term(token)
    if canonical in _STOP_WORDS or canonical in _GENERIC_TERMS:
        return False
    return len(canonical) >= 3 or canonical.isdigit()


def _looks_specific(token: str) -> bool:
    pieces = re.split(r"[-‐‑‒–—]", token)
    return (
        any(char.isupper() for char in token)
        or any(char.isdigit() for char in token)
        or len(pieces) > 1
    )


def _canonical_term(term: str) -> str:
    normalized = term.casefold().replace("‐", "-").replace("‑", "-")
    normalized = normalized.replace("‒", "-").replace("–", "-").replace("—", "-")
    return normalized


def _term_present(term: str, text: str) -> bool:
    text_tokens: set[str] = set()
    for token in _tokens(text):
        text_tokens.add(_canonical_word(token))
        # Keep compounds searchable as a whole and by their component words.
        # This lets a brief's ``Erdős–Gyárfás`` anchor match that exact spelling,
        # while still matching prose that writes ``Erdős Gyárfás``.
        text_tokens.update(
            _canonical_word(part)
            for part in re.split(r"[-‐‑‒–—]+", token)
            if part
        )
    required_tokens = [
        _canonical_word(token)
        for token in re.split(r"[-‐‑‒–—\s]+", term)
        if token and token.casefold() not in _STOP_WORDS
    ]
    return bool(required_tokens) and all(token in text_tokens for token in required_tokens)


def _canonical_word(token: str) -> str:
    value = _canonical_term(token).strip("-'")
    special = {
        "vertices": "vertex",
        "indices": "index",
        "analyses": "analysis",
        "hypotheses": "hypothesis",
        "properties": "property",
    }
    if value in special:
        return special[value]
    if len(value) > 4 and value.endswith("ies"):
        return value[:-3] + "y"
    if len(value) > 4 and value.endswith("s") and not value.endswith(("ss", "us", "is")):
        return value[:-1]
    return value


def _normalized_sentence(sentence: str) -> str:
    cleaned = re.sub(r"^\s*(?:#{1,6}\s*|[-*+]\s+|\d+[.)]\s+|>\s*)", "", sentence)
    return " ".join(_canonical_word(token) for token in _tokens(cleaned))


def _is_source_path(path: str) -> bool:
    parsed = Path(path)
    return parsed.suffix.casefold() in _SOURCE_SUFFIXES and not _is_test_path(path)


def _is_test_path(path: str) -> bool:
    parsed = Path(path)
    lowered_parts = {part.casefold() for part in parsed.parts}
    name = parsed.name.casefold()
    return bool(
        lowered_parts & _TEST_PARTS
        or name.startswith("test_")
        or name.endswith("_test.py")
        or ".test." in name
        or ".spec." in name
    )


def _code_anchors(repository_files: Mapping[str, str]) -> list[str]:
    symbols: list[str] = []
    source_paths = [path for path in sorted(repository_files) if _is_source_path(path)]
    for path in source_paths:
        for symbol in _symbols_in_file(path, repository_files[path]):
            normalized = symbol.casefold().strip("_")
            if (len(normalized) < 5 or normalized in _IGNORED_SYMBOLS
                    or normalized.startswith("test_") or normalized.startswith("_")
                    or normalized in {item + "s" for item in _IGNORED_SYMBOLS}):
                continue
            if symbol not in symbols:
                symbols.append(symbol)
            if len(symbols) >= 12:
                return symbols
    if symbols:
        return symbols
    return source_paths[:8]


def _symbols_in_file(path: str, content: str) -> list[str]:
    suffix = Path(path).suffix.casefold()
    if suffix in {".py", ".pyi"}:
        try:
            tree = ast.parse(content, filename=path)
        except (SyntaxError, ValueError):
            return []
        return [
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
    patterns = (
        r"(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)",
        r"(?:export\s+)?class\s+([A-Za-z_$][\w$]*)",
        r"(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function|\([^)]*\)\s*=>)",
    )
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, content))
    return list(dict.fromkeys(matches))


def _source_excerpt(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    # Prefer an excerpt that includes real declarations instead of only imports.
    lines = content.splitlines(keepends=True)
    declaration_indexes = [
        index for index, line in enumerate(lines)
        if re.match(r"\s*(?:async\s+)?def\s+|\s*class\s+|\s*(?:export\s+)?(?:async\s+)?function\s+", line)
    ]
    if not declaration_indexes:
        return content[:limit]
    start = max(0, declaration_indexes[0] - 2)
    excerpt = ""
    for line in lines[start:]:
        if len(excerpt) + len(line) > limit:
            break
        excerpt += line
    return excerpt or content[:limit]


def _test_anchors(test_runs: Sequence[Mapping[str, Any]]) -> list[str]:
    anchors: list[str] = []
    for run in test_runs:
        files = run.get("tested_files", [])
        if isinstance(files, (list, tuple)):
            for path in files:
                if isinstance(path, str) and path.strip():
                    anchors.extend((path, Path(path).name))
        task_id = run.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            anchors.append(task_id)
        output = "\n".join(str(run.get(key, "")) for key in ("stdout", "stderr", "evidence"))
        for match in _TEST_RESULT_RE.findall(output):
            anchors.append(match)
    return list(dict.fromkeys(anchors))[:20]


def _literal_present(value: str, text: str) -> bool:
    normalized_value = " ".join(_tokens(value)).casefold()
    normalized_text = " ".join(_tokens(text)).casefold()
    if not normalized_value:
        return False
    if normalized_value in normalized_text:
        return True
    # A report may use a path basename instead of the full test path.
    basename = Path(value).name
    if basename != value:
        return " ".join(_tokens(basename)).casefold() in normalized_text
    return False
