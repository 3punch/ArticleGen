from __future__ import annotations

import asyncio
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4
import zipfile

import httpx
import pytest

from projectforge_agents.config import ModelConfig
from projectforge_agents.git_workspace import GitWorkspaceManager
from projectforge_agents.litellm_client import ModelCompletion
from projectforge_agents.suite import ProjectForgeAgentSuite
from projectforge_api.db import build_engine, make_session_factory
from projectforge_api.main import create_app
from projectforge_api.orm import Base
from projectforge_orchestrator.engine import PostgresFSMEngine
from projectforge_sandbox import SandboxManager, SandboxResult
from projectforge_schemas.models import WorkflowState


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "projectforge"


SIMPLE_SOURCE = '''from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any


def summarize_csv(input_path: str | Path) -> dict[str, Any]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        missing = [name for name in ("date", "count") if name not in headers]
        if missing:
            raise ValueError(f"missing required CSV header(s): {', '.join(missing)}")
        counts: list[int] = []
        for line_number, row in enumerate(reader, start=2):
            date_value = (row.get("date") or "").strip()
            count_value = (row.get("count") or "").strip()
            try:
                date.fromisoformat(date_value)
                count = int(count_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid date or count on row {line_number}") from exc
            if count < 0:
                raise ValueError(f"count must not be negative on row {line_number}")
            counts.append(count)
    if not counts:
        raise ValueError("CSV contains no data rows")
    total = sum(counts)
    return {
        "row_count": len(counts),
        "total_count": total,
        "mean_count": total / len(counts),
        "maximum_daily_count": max(counts),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize daily campus bicycle counts.")
    parser.add_argument("input_csv", help="UTF-8 CSV containing date and count columns")
    parser.add_argument("--output", required=True, help="path for the deterministic JSON summary")
    args = parser.parse_args(argv)
    try:
        summary = summarize_csv(args.input_csv)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

SIMPLE_TESTS = '''import json
from pathlib import Path

import pytest

from src.bicycle_summary import main, summarize_csv


FIXTURE = "date,count\\n2026-01-01,4\\n2026-01-02,8\\n2026-01-03,12\\n"


def test_fixture_row_count_sum_mean_and_maximum_match_expected_values(tmp_path):
    source = tmp_path / "counts.csv"
    source.write_text(FIXTURE, encoding="utf-8")
    summary = summarize_csv(source)
    assert summary["row_count"] == 3
    assert summary["total_count"] == sum([4, 8, 12]) == 24
    assert summary["mean_count"] == 8
    assert summary["maximum_daily_count"] == max([4, 8, 12]) == 12
    target = tmp_path / "summary.json"
    assert main([str(source), "--output", str(target)]) == 0
    assert json.loads(target.read_text(encoding="utf-8")) == summary


def test_missing_header_and_malformed_date_or_negative_count_are_actionable(tmp_path):
    missing = tmp_path / "missing.csv"
    missing.write_text("day,count\\n2026-01-01,4\\n", encoding="utf-8")
    with pytest.raises(ValueError, match="date"):
        summarize_csv(missing)
    malformed = tmp_path / "malformed.csv"
    malformed.write_text("date,count\\nnot-a-date,4\\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 2"):
        summarize_csv(malformed)
    negative = tmp_path / "negative.csv"
    negative.write_text("date,count\\n2026-01-01,-1\\n", encoding="utf-8")
    with pytest.raises(ValueError, match="negative.*row 2"):
        summarize_csv(negative)
'''

WEB_SOURCE = '''from __future__ import annotations

from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class ExperimentValidationError(ValueError):
    def __init__(self, errors: dict[str, str]):
        super().__init__("Experiment validation failed")
        self.errors = errors


class ExperimentStore:
    """Local in-memory record store with field-specific server validation."""

    def __init__(self):
        self._records: list[dict[str, str]] = []
        self._next_id = 1

    def create(self, payload: dict[str, object]) -> dict[str, str]:
        errors: dict[str, str] = {}
        title = str(payload.get("title", "")).strip()
        if not title:
            errors["title"] = "Title is required."
        raw_date = str(payload.get("date", "")).strip()
        try:
            normalized_date = date.fromisoformat(raw_date).isoformat()
        except ValueError:
            normalized_date = ""
            errors["date"] = "Date must be a valid ISO calendar date."
        if errors:
            raise ExperimentValidationError(errors)
        record = {
            "id": str(self._next_id),
            "title": title,
            "date": normalized_date,
            "research_question": str(payload.get("research_question", "")).strip(),
            "notes": str(payload.get("notes", "")).strip(),
        }
        self._next_id += 1
        self._records.append(record)
        return dict(record)

    def list_experiments(self, title_query: str = "") -> list[dict[str, str]]:
        query = title_query.casefold().strip()
        ordered = sorted(self._records, key=lambda item: (item["date"], int(item["id"])), reverse=True)
        if query:
            ordered = [item for item in ordered if query in item["title"].casefold()]
        return [dict(item) for item in ordered]


STORE = ExperimentStore()


class ExperimentHandler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, value: object) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/experiments":
            query = parse_qs(parsed.query).get("title", [""])[0]
            self._send_json(200, {"experiments": STORE.list_experiments(query)})
            return
        if parsed.path == "/":
            content = Path("web/index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/experiments":
            self._send_json(404, {"error": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 1_000_000:
                self._send_json(400, {"error": "invalid request size"})
                return
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(payload, dict):
                self._send_json(400, {"error": "request must be a JSON object"})
                return
            self._send_json(201, STORE.create(payload))
        except ExperimentValidationError as exc:
            self._send_json(422, {"errors": exc.errors})
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "invalid JSON request"})

    def log_message(self, format: str, *args: object) -> None:
        return


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    with ThreadingHTTPServer((host, port), ExperimentHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
'''

WEB_TESTS = '''import pytest

from server.app import ExperimentStore, ExperimentValidationError


def test_creating_listing_sorting_and_case_insensitive_filtering_return_expected_records():
    store = ExperimentStore()
    older = store.create({"title": "Cell Culture", "date": "2025-02-01", "research_question": "growth", "notes": "older"})
    newer = store.create({"title": "Microscope Trial", "date": "2025-03-15", "research_question": "focus", "notes": "newer"})
    assert [item["id"] for item in store.list_experiments()] == [newer["id"], older["id"]]
    assert [item["title"] for item in store.list_experiments("cElL")] == ["Cell Culture"]
    assert store.list_experiments("absent") == []


def test_missing_title_and_invalid_date_are_rejected_with_field_specific_errors():
    store = ExperimentStore()
    with pytest.raises(ExperimentValidationError) as missing_title:
        store.create({"title": "", "date": "2025-04-01"})
    assert "title" in missing_title.value.errors
    with pytest.raises(ExperimentValidationError) as invalid_date:
        store.create({"title": "Valid title", "date": "not-a-date"})
    assert "date" in invalid_date.value.errors
    assert store.list_experiments() == []
'''

WEB_HTML = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Laboratory Experiment Log</title>
  <link rel="stylesheet" href="/web/styles.css">
</head>
<body>
  <main class="layout">
    <h1>Laboratory Experiment Log</h1>
    <form id="experiment-form" novalidate>
      <label for="title">Experiment title</label>
      <input id="title" name="title" required aria-describedby="title-error">
      <p id="title-error" class="error" aria-live="polite"></p>
      <label for="date">Experiment date</label>
      <input id="date" name="date" type="date" required aria-describedby="date-error">
      <p id="date-error" class="error" aria-live="polite"></p>
      <label for="research-question">Research question</label>
      <textarea id="research-question" name="research_question"></textarea>
      <label for="notes">Notes</label>
      <textarea id="notes" name="notes"></textarea>
      <button type="submit">Save experiment</button>
      <p id="form-status" role="status" aria-live="polite"></p>
    </form>
    <section aria-labelledby="list-heading">
      <h2 id="list-heading">Experiments</h2>
      <label for="title-filter">Filter titles</label>
      <input id="title-filter" type="search">
      <p id="empty-state">No experiments yet. Add one to begin.</p>
      <ul id="experiment-list" aria-live="polite"></ul>
    </section>
  </main>
  <script src="/web/app.js" defer></script>
</body>
</html>
'''

WEB_JS = '''"use strict";

const form = document.querySelector("#experiment-form");
const filter = document.querySelector("#title-filter");
const list = document.querySelector("#experiment-list");
const empty = document.querySelector("#empty-state");

async function loadExperiments() {
  const query = encodeURIComponent(filter.value.trim());
  const response = await fetch(`/api/experiments?title=${query}`);
  const payload = await response.json();
  list.replaceChildren();
  for (const experiment of payload.experiments) {
    const item = document.createElement("li");
    item.textContent = `${experiment.date} — ${experiment.title}`;
    list.append(item);
  }
  empty.hidden = payload.experiments.length > 0;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const values = Object.fromEntries(new FormData(form).entries());
  const response = await fetch("/api/experiments", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(values),
  });
  const payload = await response.json();
  for (const field of ["title", "date"]) {
    const error = document.querySelector(`#${field}-error`);
    error.textContent = payload.errors?.[field] ?? "";
  }
  document.querySelector("#form-status").textContent = response.ok
    ? "Experiment saved."
    : "Please correct the fields with errors.";
  if (response.ok) {
    form.reset();
    await loadExperiments();
  }
});
filter.addEventListener("input", loadExperiments);
loadExperiments();
'''

WEB_CSS = '''* { box-sizing: border-box; }
body { margin: 0; font: 1rem/1.5 system-ui, sans-serif; color: #152033; background: #f3f6fb; }
.layout { width: min(100% - 2rem, 74rem); margin: 1.5rem auto; display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 1.5rem; }
h1 { grid-column: 1 / -1; }
form, section { min-width: 0; padding: 1.25rem; border-radius: .75rem; background: white; box-shadow: 0 .25rem 1rem #16294a12; }
label { display: block; margin-top: .75rem; font-weight: 650; }
input, textarea, button { width: 100%; min-height: 2.75rem; padding: .6rem; font: inherit; }
textarea { min-height: 5rem; }
button { margin-top: 1rem; color: white; background: #174b8a; border: 0; border-radius: .35rem; cursor: pointer; }
.error { color: #a11927; min-height: 1.5rem; }
@media (max-width: 42rem) { .layout { grid-template-columns: minmax(0, 1fr); width: min(100% - 1rem, 42rem); } h1 { grid-column: auto; } }
'''

DATA_SOURCE = '''from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics
from typing import Any


REQUIRED_COLUMNS = ("station", "month", "millimeters")


def load_measurements(path: str | Path) -> list[dict[str, Any]]:
    measurements: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("missing required columns: " + ", ".join(missing))
        for row_number, row in enumerate(reader, start=2):
            station = (row.get("station") or "").strip()
            month = (row.get("month") or "").strip()
            raw_value = (row.get("millimeters") or "").strip()
            try:
                value = float(raw_value)
                if not station or not month:
                    raise ValueError("station and month are required")
            except ValueError as exc:
                raise ValueError(f"invalid row {row_number}: missing or nonnumeric millimeter measurement") from exc
            measurements.append({"station": station, "month": month, "millimeters": value})
    if not measurements:
        raise ValueError("CSV contains no valid measurement rows")
    return measurements


def analyze(path: str | Path) -> dict[str, Any]:
    rows = load_measurements(path)
    by_station: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_station[row["station"]].append(row["millimeters"])
    summaries = {}
    for station, values in sorted(by_station.items()):
        summaries[station] = {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    wettest = max(rows, key=lambda row: (row["millimeters"], row["station"], row["month"]))
    return {
        "station_summaries": summaries,
        "wettest_station_month": {
            "station": wettest["station"],
            "month": wettest["month"],
            "millimeters": wettest["millimeters"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze local monthly rainfall measurements.")
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="outputs/rainfall_summary.json")
    args = parser.parse_args(argv)
    try:
        result = analyze(args.csv_path)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

DATA_TESTS = '''import csv
import statistics

import pytest

from analysis.rainfall import analyze, load_measurements


FIXTURE_ROWS = [
    ("North", "2025-05", 10), ("North", "2025-06", 19),
    ("South", "2025-05", 8), ("South", "2025-06", 16),
    ("East", "2025-05", 5), ("East", "2025-06", 11),
    ("West", "2025-05", 7), ("West", "2025-06", 13),
]


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["station", "month", "millimeters"])
        writer.writerows(rows)


def test_per_station_summaries_and_wettest_station_month_match_fixture_values(tmp_path):
    source = tmp_path / "rainfall.csv"
    _write_csv(source, FIXTURE_ROWS)
    result = analyze(source)
    station_summaries = result["station_summaries"]
    north_values = [10.0, 19.0]
    assert station_summaries["North"]["count"] == len(north_values) == 2
    assert station_summaries["North"]["mean"] == statistics.mean(north_values)
    assert station_summaries["North"]["median"] == statistics.median(north_values)
    assert station_summaries["North"]["sample_stddev"] == pytest.approx(statistics.stdev(north_values))
    assert set(station_summaries) == {"North", "South", "East", "West"}
    wettest_station_month = result["wettest_station_month"]
    assert wettest_station_month == {"station": "North", "month": "2025-06", "millimeters": 19.0}


def test_missing_and_nonnumeric_measurements_report_invalid_source_rows(tmp_path):
    blank = tmp_path / "blank.csv"
    _write_csv(blank, [("North", "2025-05", ""), ("South", "2025-06", 4)])
    with pytest.raises(ValueError, match="row 2"):
        load_measurements(blank)
    nonnumeric = tmp_path / "nonnumeric.csv"
    _write_csv(nonnumeric, [("North", "2025-05", "rain"), ("South", "2025-06", 4)])
    with pytest.raises(ValueError, match="row 2"):
        load_measurements(nonnumeric)
'''

DATA_CHART = '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420" role="img" aria-labelledby="title description">
  <title id="title">Monthly rainfall comparison by station</title>
  <desc id="description">Comparison of rainfall observations for North, South, East, and West stations.</desc>
  <rect width="720" height="420" fill="#ffffff"/>
  <text x="360" y="36" text-anchor="middle" font-family="sans-serif" font-size="22">Monthly rainfall comparison by station</text>
  <text x="24" y="220" transform="rotate(-90 24 220)" text-anchor="middle" font-family="sans-serif" font-size="15">Rainfall (mm)</text>
  <line x1="80" y1="340" x2="680" y2="340" stroke="#344054"/>
  <rect x="145" y="122" width="70" height="218" fill="#3274a1"/><text x="180" y="365" text-anchor="middle">North</text>
  <rect x="280" y="202" width="70" height="138" fill="#e1812c"/><text x="315" y="365" text-anchor="middle">South</text>
  <rect x="415" y="250" width="70" height="90" fill="#3a923a"/><text x="450" y="365" text-anchor="middle">East</text>
  <rect x="550" y="225" width="70" height="115" fill="#c03d3e"/><text x="585" y="365" text-anchor="middle">West</text>
</svg>
'''


class Phase6FixtureClient:
    def __init__(self, fixture: dict[str, Any]) -> None:
        self.fixture = fixture
        self.calls: list[str] = []
        self.coder_calls: dict[str, int] = {}
        self.generated_files = self._files_for(fixture["name"])
        self.tester_output = self._tests_for(fixture["name"])

    def complete(self, *, system_prompt: str, user_prompt: str,
                 agent_name: str, timeout_seconds: int | None = None):
        self.calls.append(agent_name)
        if agent_name == "requirement":
            output = self.fixture["requirement_response"]
        elif agent_name == "planner":
            output = self.fixture["planner_response"]
        elif agent_name == "coder":
            prompt = json.loads(user_prompt)
            task = prompt["task"]
            task_id = task["id"]
            self.coder_calls[task_id] = self.coder_calls.get(task_id, 0) + 1
            changes = self.generated_files.get(task_id)
            if changes is None:
                repository_files = prompt["repository_files"]
                write_set = task["write_set"]
                existing = [
                    (path, content) for path, content in sorted(repository_files.items())
                    if any(path == scope or path.startswith(scope.rstrip("/") + "/")
                           for scope in write_set)
                ]
                if existing:
                    path, content = existing[0]
                elif "README.md" in write_set:
                    path, content = "README.md", "# Setup guide\n\nProjectForge will finalize this guide during packaging.\n"
                else:
                    raise AssertionError(f"no fixture file is available for task {task_id!r}")
                changes = [{"path": path, "content": content}]
            output = {
                "task_id": task_id,
                "summary": f"Implement the scoped deliverable for task {task_id} and preserve existing behavior.",
                "changes": changes,
                "evidence": [f"Changes address the fixture acceptance criteria for {task_id}."],
            }
        elif agent_name == "tester":
            output = self.tester_output
        elif agent_name == "reviewer":
            payload = json.loads(user_prompt)
            output = {
                "summary": "The supplied evidence is relevant, and the subjective rubric item remains for a human decision.",
                "recommendations": [{
                    "criterion": item["criterion"],
                    "recommendation": "approve",
                    "rationale": "The evidence is plausible and the final human checkpoint must decide whether it satisfies the qualitative criterion.",
                    "evidence": [self._human_evidence(self.fixture["name"])],
                } for item in payload["human_judgment_criteria"]],
            }
        elif agent_name == "packager":
            output = self._package_documents(self.fixture["name"])
        else:
            raise AssertionError(f"unexpected model-backed role: {agent_name}")
        return ModelCompletion(
            content=json.dumps(output, ensure_ascii=False),
            tokens_used=240,
            cost=Decimal("0.00100000"),
            provider="offline-phase6",
            model=f"fixture-{agent_name}-v1",
        )

    def _files_for(self, name: str) -> dict[str, list[dict[str, str]]]:
        if name == "simple-script":
            return {
                "validate-csv": [
                    {"path": "src/__init__.py", "content": ""},
                    {"path": "src/bicycle_summary.py", "content": SIMPLE_SOURCE},
                ],
            }
        if name == "web-app":
            return {
                "data-model": [
                    {"path": "server/__init__.py", "content": ""},
                    {"path": "server/app.py", "content": WEB_SOURCE},
                ],
                "accessible-form": [
                    {"path": "web/index.html", "content": WEB_HTML},
                    {"path": "web/app.js", "content": WEB_JS},
                    {"path": "web/styles.css", "content": WEB_CSS},
                ],
            }
        return {
            "load-and-validate": [
                {"path": "analysis/__init__.py", "content": ""},
                {"path": "analysis/rainfall.py", "content": DATA_SOURCE},
            ],
            "comparison-chart": [
                {"path": "outputs/rainfall_comparison.svg", "content": DATA_CHART},
            ],
        }

    def _tests_for(self, name: str) -> dict[str, Any]:
        if name == "simple-script":
            return {
                "summary": "Test fixture row count, sum, mean, maximum, and malformed CSV behavior.",
                "test_files": [{"path": "tests/test_bicycle_summary.py", "content": SIMPLE_TESTS}],
                "evidence": ["Fixed fixture values and missing, malformed, and negative rows are asserted."],
            }
        if name == "web-app":
            return {
                "summary": "Test experiment creation, listing, sorting, filtering, and field-specific validation.",
                "test_files": [{"path": "tests/test_experiments.py", "content": WEB_TESTS}],
                "evidence": ["Create/list/filter ordering and missing title/invalid date cases are asserted."],
            }
        return {
            "summary": "Test station summaries, wettest station-month, and invalid measurement rows.",
            "test_files": [{"path": "tests/test_rainfall.py", "content": DATA_TESTS}],
            "evidence": ["Fixed station fixture values, sample deviation, maximum pair, blank and nonnumeric rows are asserted."],
        }

    def _package_documents(self, name: str) -> dict[str, Any]:
        if name == "simple-script":
            title = "Campus Bicycle Count Summary CLI"
            command = "python -m src.bicycle_summary counts.csv --output reports/summary.json"
            notes = "The CSV uses ISO dates and nonnegative integer counts; malformed rows are reported with row numbers."
        elif name == "web-app":
            title = "Laboratory Experiment Log Web App"
            command = "python -m server.app"
            notes = "The local API and accessible browser UI require no external service."
        else:
            title = "Reproducible Multi-Station Rainfall Analysis"
            command = "python -m analysis.rainfall rainfall.csv --output outputs/rainfall_summary.json"
            notes = "The report uses sample standard deviation and records missing or nonnumeric measurements as invalid rows."
        return {
            "readme_markdown": (
                f"# {title}\n\n{notes}\n\n## Run locally\n\n`{command}`\n\n"
                "## Tests\n\nRun `python -m pytest -q` from the project root.\n"
            ),
            "report_markdown": (
                f"# Project Report: {title}\n\n"
                "## Deliverables\n\nThe project contains the implementation, focused regression tests, and local usage guidance.\n\n"
                "## Verification\n\nThe final recorded sandbox test runs completed successfully with exit code zero.\n"
            ),
            "summary": f"Generate a concise README and project report for {title}.",
            "evidence": ["The validated ProjectSpec, repository inventory, review, and sandbox test results were used."],
        }

    @staticmethod
    def _human_evidence(name: str) -> str:
        if name == "simple-script":
            return "README.md: describes the CLI, date/count inputs, and malformed-row behavior."
        if name == "web-app":
            return "web/index.html and web/app.js: contain associated labels, live error text, and the empty state."
        return "README.md: explains sample standard deviation and invalid-row exclusion policy."


class RecordingDispatcher:
    def __init__(self) -> None:
        self.jobs: list[tuple[UUID, int]] = []

    def enqueue_workflow(self, workflow_id: UUID, *, timeout_seconds: int) -> str:
        self.jobs.append((workflow_id, timeout_seconds))
        return f"phase6-job-{len(self.jobs)}"


class SyncASGIClient:
    def __init__(self, app: Any) -> None:
        self.app = app
        self.transport = httpx.ASGITransport(app=app)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.app.state.engine is not None:
            self.app.state.engine.dispose()

    def request(self, method: str, url: str, **kwargs):
        async def _request():
            async with httpx.AsyncClient(
                transport=self.transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(method, url, **kwargs)
        return asyncio.run(_request())

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)


def _config() -> ModelConfig:
    return ModelConfig.from_env({
        "PROJECTFORGE_LLM_PROVIDER": "dahl",
        "PROJECTFORGE_LLM_TIER": "strong",
        "PROJECTFORGE_DAHL_STRONG_MODEL": "offline-phase6-model",
        "PROJECTFORGE_LLM_RESERVATION_TOKENS": "40000",
        "PROJECTFORGE_LLM_MAX_OUTPUT_TOKENS": "4096",
        "PROJECTFORGE_LLM_RESERVATION_SPEND": "0.05000000",
        "PROJECTFORGE_LLM_RESERVATION_TTL_SECONDS": "600",
        "PROJECTFORGE_LLM_TIMEOUT_SECONDS": "60",
    })


def _docker_sandbox_or_skip() -> SandboxManager:
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
    sandbox = SandboxManager(client=client, default_timeout_seconds=60)
    try:
        client.images.get(sandbox.images["python"])
    except Exception as exc:
        if require_docker:
            pytest.fail(f"required Python sandbox image {sandbox.images['python']!r} is unavailable: {exc}")
        pytest.skip(f"required Python sandbox image {sandbox.images['python']!r} is unavailable")
    return sandbox


def _run_phase6_package_workflows(tmp_path: Path, sandbox: Any, filenames: tuple[str, ...] | list[str]) -> None:
    for filename in filenames:
        fixture = json.loads((FIXTURE_DIR / filename).read_text(encoding="utf-8"))
        case_dir = tmp_path / fixture["name"]
        case_dir.mkdir()
        database_path = case_dir / "projectforge.sqlite"
        database_url = f"sqlite:///{database_path}"
        db_engine = build_engine(database_url)
        Base.metadata.create_all(db_engine)
        session_factory = make_session_factory(db_engine)
        repository_manager = GitWorkspaceManager(case_dir / "repos")
        model_client = Phase6FixtureClient(fixture)
        suite = ProjectForgeAgentSuite(
            config=_config(),
            client=model_client,
            session_factory=session_factory,
            workspace_manager=repository_manager,
            sandbox_manager=sandbox,
        )
        workflow_engine = PostgresFSMEngine(
            session_factory, agents=suite, max_parallel_tasks=1,
        )
        dispatcher = RecordingDispatcher()
        app = create_app(
            database_url=database_url,
            create_schema_for_tests=True,
            workflow_engine=workflow_engine,
            queue_dispatcher=dispatcher,
            agent_suite=suite,
            workspace_manager=repository_manager,
        )
        try:
            with SyncASGIClient(app) as client:
                created = client.post("/api/projects", json={
                    "name": fixture["name"],
                    "raw_brief_text": fixture["raw_brief_text"],
                    "rubric_text": fixture["rubric_text"],
                    "token_cap": 100_000,
                    "spend_cap": "10.00",
                })
                assert created.status_code == 201, created.text
                project_id = created.json()["id"]
                started = client.post(f"/api/projects/{project_id}/workflows", json={
                    "max_retries": 3,
                    "timeout_seconds": 3600,
                })
                assert started.status_code == 202, started.text
                workflow_id = UUID(started.json()["id"])

                plan_gate = app.state.workflow_engine.drive(workflow_id)
                assert plan_gate.state is WorkflowState.AWAITING_PLAN_APPROVAL, plan_gate.failure_reason
                plan_bundle = client.get(f"/api/workflows/{workflow_id}/review")
                assert plan_bundle.status_code == 200, plan_bundle.text
                assert plan_bundle.json()["stage"] == "plan"
                assert plan_bundle.json()["task_graph"]["nodes"]
                approved_plan = client.post(f"/api/workflows/{workflow_id}/approvals", json={
                    "stage": "plan", "decision": "approve", "comment": "Plan inspected and approved.",
                })
                assert approved_plan.status_code == 200, approved_plan.text

                final_gate = app.state.workflow_engine.drive(workflow_id)
                assert final_gate.state is WorkflowState.AWAITING_FINAL_APPROVAL, final_gate.failure_reason
                final_bundle = client.get(f"/api/workflows/{workflow_id}/review")
                assert final_bundle.status_code == 200, final_bundle.text
                review = final_bundle.json()["review"]
                assert review is not None
                assert review["automatable_checks_passed"] is True
                assert any(item["status"] == "human_review_required" for item in review["criteria"])
                assert all(
                    item["status"] != "passed"
                    for item in review["criteria"]
                    if item["rubric_item"]["type"] == "human-judgment"
                )
                assert final_bundle.json()["changed_files"]
                assert final_bundle.json()["code_diff"]

                approved_final = client.post(f"/api/workflows/{workflow_id}/approvals", json={
                    "stage": "final",
                    "decision": "approve",
                    "comment": "I reviewed the evidence and approve packaging.",
                })
                assert approved_final.status_code == 200, approved_final.text
                done = app.state.workflow_engine.drive(workflow_id)
                assert done.state is WorkflowState.DONE, done.failure_reason
                assert done.result_payload["package_export"]["download_url"] == (
                    f"/api/workflows/{workflow_id}/package"
                )

                metadata = client.get(f"/api/workflows/{workflow_id}/package/metadata")
                assert metadata.status_code == 200, metadata.text
                download = client.get(f"/api/workflows/{workflow_id}/package")
                assert download.status_code == 200, download.text[:1000]
                assert download.headers["content-type"].startswith("application/zip")
                zip_path = case_dir / "downloaded.zip"
                zip_path.write_bytes(download.content)
                with zipfile.ZipFile(zip_path) as archive:
                    names = set(archive.namelist())
                    assert {
                        "README.md", "PROJECT_REPORT.md", "reports/PROJECT_REPORT.md",
                        "AI_USAGE_DISCLOSURE.md", "audit/AI_USAGE_DISCLOSURE.json",
                        "reports/TEST_RESULTS.json", "reports/REVIEW_RESULTS.json",
                        "PACKAGE_MANIFEST.json",
                    } <= names
                    assert any(name.startswith("tests/") for name in names)
                    assert not any(name.startswith(".git/") or name.startswith(".venv/") for name in names)
                    test_results = json.loads(archive.read("reports/TEST_RESULTS.json"))
                    assert test_results["all_passed"] is True
                    assert test_results["run_count"] >= 1
                    assert all(run["passed"] and run["exit_code"] == 0 for run in test_results["runs"])
                    disclosure = json.loads(archive.read("audit/AI_USAGE_DISCLOSURE.json"))
                    assert disclosure["workflow_id"] == str(workflow_id)
                    assert disclosure["agent_calls"] == len(disclosure["activities"])
                    assert any(
                        activity["agent_name"] == "reviewer"
                        and activity["model"] == "fixture-reviewer-v1"
                        for activity in disclosure["activities"]
                    )
                    assert any(
                        activity["agent_name"] == "packager"
                        and activity["model"] == "fixture-packager-v1"
                        for activity in disclosure["activities"]
                    )
                    disclosure_markdown = archive.read("AI_USAGE_DISCLOSURE.md").decode("utf-8")
                    assert "Timestamp (UTC)" in disclosure_markdown
                    assert "fixture-reviewer-v1" in disclosure_markdown
                    report = archive.read("reports/PROJECT_REPORT.md").decode("utf-8")
                    assert "human_review_required" in report
                    assert "does not rewrite them as automated passes" in report

                repo = repository_manager.project_path(project_id)
                assert (repo / ".git").is_dir()
                assert (repo / "README.md").is_file()
                assert any("Packager task" in line for line in (repo / ".git" / "logs" / "HEAD").read_text().splitlines())
        finally:
            db_engine.dispose()


class PassingSandboxStub:
    """Offline control-flow stub; it does not execute project code."""

    def __init__(self) -> None:
        self.calls = 0

    def run(self, runtime, command, *, project_dir, timeout_seconds, network_allowlist):
        self.calls += 1
        return SandboxResult(
            runtime=runtime,
            stdout="1 passed (sandbox execution is stubbed in this unit-level flow test)\n",
            stderr="",
            exit_code=0,
            timed_out=False,
            duration_seconds=0.001,
            container_id=f"offline-sandbox-stub-{self.calls}",
        )


@pytest.mark.docker
def test_all_phase3_fixtures_reach_downloadable_reviewed_packages(tmp_path: Path):
    """Docker-backed E2E: all Phase 3 fixtures execute tests and export audit-ready ZIPs."""
    _run_phase6_package_workflows(
        tmp_path,
        _docker_sandbox_or_skip(),
        ("simple_script.json", "web_app.json", "data_analysis.json"),
    )


@pytest.mark.parametrize("filename", ["simple_script.json", "web_app.json", "data_analysis.json"])
def test_all_phase3_fixture_package_api_flows_with_offline_sandbox_stub(tmp_path: Path, filename: str):
    """Exercise API, FSM, disclosure, and ZIP assembly without executing code."""
    _run_phase6_package_workflows(tmp_path, PassingSandboxStub(), [filename])
