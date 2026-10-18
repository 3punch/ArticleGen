from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4
import zipfile

import httpx
import pytest
from sqlalchemy import select

from projectforge_agents.config import ModelConfig
from projectforge_agents.git_workspace import GitWorkspaceManager
from projectforge_agents.litellm_client import ModelCompletion
from projectforge_agents.suite import ProjectForgeAgentSuite
from projectforge_api.db import build_engine, make_session_factory
from projectforge_api.main import create_app
from projectforge_api.orm import AgentRunRow, Base
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

ERDOS_SOURCE = '''from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def validate_simple_graph(adjacency: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    """Validate and normalize a finite undirected simple graph."""
    if not isinstance(adjacency, Mapping):
        raise ValueError("adjacency must be a mapping")
    vertices = set(adjacency)
    normalized: dict[str, tuple[str, ...]] = {}
    for vertex, raw_neighbors in adjacency.items():
        if not isinstance(vertex, str) or not vertex:
            raise ValueError("vertex labels must be non-empty strings")
        if isinstance(raw_neighbors, (str, bytes)):
            raise ValueError("neighbors must be a sequence of vertex labels")
        neighbors = tuple(raw_neighbors)
        if len(neighbors) != len(set(neighbors)):
            raise ValueError(f"duplicate neighbor in adjacency for {vertex}")
        if vertex in neighbors:
            raise ValueError(f"self-loop at {vertex} is not allowed in a simple graph")
        unknown = set(neighbors) - vertices
        if unknown:
            raise ValueError("unknown neighbor(s): " + ", ".join(sorted(unknown)))
        normalized[vertex] = tuple(sorted(neighbors))
    for vertex, neighbors in normalized.items():
        for neighbor in neighbors:
            if vertex not in normalized[neighbor]:
                raise ValueError(f"adjacency must be symmetric for edge {vertex}-{neighbor}")
    return dict(sorted(normalized.items()))


def vertex_count(adjacency: Mapping[str, Sequence[str]]) -> int:
    return len(validate_simple_graph(adjacency))


def edge_count(adjacency: Mapping[str, Sequence[str]]) -> int:
    graph = validate_simple_graph(adjacency)
    return sum(map(len, graph.values())) // 2


def minimum_degree(adjacency: Mapping[str, Sequence[str]]) -> int:
    graph = validate_simple_graph(adjacency)
    return min((len(neighbors) for neighbors in graph.values()), default=0)


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _canonical_cycle(cycle: Sequence[str]) -> tuple[str, ...]:
    path = tuple(cycle)
    orientations = (path, tuple(reversed(path)))
    return min(
        orientation[index:] + orientation[:index]
        for orientation in orientations
        for index in range(len(path))
    )


def enumerate_simple_cycles(adjacency: Mapping[str, Sequence[str]]) -> list[tuple[str, ...]]:
    """Enumerate each undirected simple cycle once; worst-case search is factorial."""
    graph = validate_simple_graph(adjacency)
    vertices = sorted(graph)
    rank = {vertex: index for index, vertex in enumerate(vertices)}
    found: set[tuple[str, ...]] = set()

    def visit(start: str, current: str, path: list[str]) -> None:
        for neighbor in graph[current]:
            if neighbor == start and len(path) >= 3:
                found.add(_canonical_cycle(path))
            elif neighbor not in path and rank[neighbor] >= rank[start]:
                visit(start, neighbor, [*path, neighbor])

    for start in vertices:
        visit(start, start, [start])
    return sorted(found, key=lambda cycle: (len(cycle), cycle))


def power_of_two_cycle_lengths(adjacency: Mapping[str, Sequence[str]]) -> list[int]:
    return sorted({
        len(cycle) for cycle in enumerate_simple_cycles(adjacency)
        if is_power_of_two(len(cycle))
    })


def petersen_graph() -> dict[str, list[str]]:
    """Return the Petersen graph as a 3-regular adjacency-list fixture."""
    return {
        "0": ["1", "4", "5"], "1": ["0", "2", "6"],
        "2": ["1", "3", "7"], "3": ["2", "4", "8"],
        "4": ["0", "3", "9"], "5": ["0", "7", "8"],
        "6": ["1", "8", "9"], "7": ["2", "5", "9"],
        "8": ["3", "5", "6"], "9": ["4", "6", "7"],
    }


def summarize_graph(adjacency: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    graph = validate_simple_graph(adjacency)
    cycles = enumerate_simple_cycles(graph)
    return {
        "vertex_count": len(graph),
        "edge_count": sum(map(len, graph.values())) // 2,
        "minimum_degree": min((len(neighbors) for neighbors in graph.values()), default=0),
        "cycle_count": len(cycles),
        "cycle_lengths": sorted({len(cycle) for cycle in cycles}),
        "power_of_two_cycle_lengths": sorted({
            len(cycle) for cycle in cycles if is_power_of_two(len(cycle))
        }),
    }


def write_petersen_result(output_path: str | Path) -> Path:
    """Write a reproducible JSON result from the enumerated Petersen graph."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = {"graph": "Petersen graph", **summarize_graph(petersen_graph())}
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return destination
'''

ERDOS_TESTS = '''import json

import pytest

from graphs.cycles import (
    enumerate_simple_cycles,
    is_power_of_two,
    petersen_graph,
    summarize_graph,
    validate_simple_graph,
    write_petersen_result,
)


def test_petersen_graph_is_cubic_and_has_an_eight_cycle():
    result = summarize_graph(petersen_graph())
    assert result["vertex_count"] == 10
    assert result["edge_count"] == 15
    assert result["minimum_degree"] == 3
    assert result["cycle_lengths"] == [5, 6, 8, 9]
    assert result["power_of_two_cycle_lengths"] == [8]
    cycles = enumerate_simple_cycles(petersen_graph())
    canonical_cycles = {
        min(
            tuple(orientation[index:] + orientation[:index])
            for orientation in (cycle, tuple(reversed(cycle)))
            for index in range(len(cycle))
        )
        for cycle in cycles
    }
    assert set(cycles) == canonical_cycles
    assert len(cycles) == len(canonical_cycles) == 57


def test_petersen_result_artifact_is_written_from_the_computed_graph(tmp_path):
    from graphs.cycles import write_petersen_result

    artifact = write_petersen_result(tmp_path / "results" / "petersen_cycle_summary.json")
    result = json.loads(artifact.read_text(encoding="utf-8"))
    assert result == {
        "graph": "Petersen graph",
        "vertex_count": 10,
        "edge_count": 15,
        "minimum_degree": 3,
        "cycle_count": 57,
        "cycle_lengths": [5, 6, 8, 9],
        "power_of_two_cycle_lengths": [8],
    }


def test_invalid_undirected_simple_graphs_are_rejected():
    with pytest.raises(ValueError, match="symmetric"):
        validate_simple_graph({"a": ["b"], "b": []})
    with pytest.raises(ValueError, match="self-loop"):
        validate_simple_graph({"a": ["a"]})
    with pytest.raises(ValueError, match="unknown"):
        validate_simple_graph({"a": ["missing"]})
    with pytest.raises(ValueError, match="duplicate neighbor"):
        validate_simple_graph({"a": ["b", "b"], "b": ["a"]})


def test_power_of_two_check_rejects_zero_and_non_powers():
    assert is_power_of_two(8)
    assert not is_power_of_two(0)
    assert not is_power_of_two(6)
'''

ERDOS_RESULT = '''{
  "cycle_count": 57,
  "cycle_lengths": [5, 6, 8, 9],
  "edge_count": 15,
  "graph": "Petersen graph",
  "minimum_degree": 3,
  "power_of_two_cycle_lengths": [8],
  "vertex_count": 10
}
'''


class Phase6FixtureClient:
    def __init__(
        self,
        fixture: dict[str, Any],
        *,
        fail_packager: bool = False,
        invalid_first_packager: bool = False,
    ) -> None:
        self.fixture = fixture
        self.fail_packager = fail_packager
        self.invalid_first_packager = invalid_first_packager
        self.calls: list[str] = []
        self.packager_inputs: list[dict[str, Any]] = []
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
            packager_input = json.loads(user_prompt)
            self.packager_inputs.append(packager_input)
            if self.fail_packager:
                raise RuntimeError("simulated packager provider outage")
            output = self._package_documents(self.fixture["name"], packager_input)
            if self.invalid_first_packager and len(self.packager_inputs) == 1:
                output["report_markdown"] = (
                    "# Project Report\n\n"
                    "## The core concept\n\n"
                    "The project follows a clear process and delivers useful results for its audience.\n"
                )
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
        if name == "erdos-gyarfas":
            return {
                "validate-graph": [
                    {"path": "graphs/__init__.py", "content": ""},
                    {"path": "graphs/cycles.py", "content": ERDOS_SOURCE},
                ],
                "analyze-petersen": [
                    {"path": "results/petersen_cycle_summary.json", "content": ERDOS_RESULT},
                ],
                "cycle-regressions": [
                    {"path": "tests/test_erdos_gyarfas.py", "content": ERDOS_TESTS},
                ],
                "readme-and-limitations": [
                    {"path": "README.md", "content": (
                        "# Erdős–Gyárfás Cycles in Cubic Graphs\n\n"
                        "Run the local graph tests with `python -m pytest -q`. "
                        "`graphs/cycles.py` contains the cycle enumerator. The search is "
                        "factorial in the worst case; the Petersen result is one finite "
                        "example, not a proof of the conjecture.\n"
                    )},
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
        if name == "erdos-gyarfas":
            return {
                "summary": "Test simple-cycle canonicalization, invalid graph rejection, and Petersen graph cycle lengths.",
                "test_files": [{"path": "tests/test_erdos_gyarfas.py", "content": ERDOS_TESTS}],
                "evidence": ["Petersen assertions check 10 vertices, 15 edges, minimum degree 3, cycle lengths 5/6/8/9, and power-of-two length 8."],
            }
        return {
            "summary": "Test station summaries, wettest station-month, and invalid measurement rows.",
            "test_files": [{"path": "tests/test_rainfall.py", "content": DATA_TESTS}],
            "evidence": ["Fixed station fixture values, sample deviation, maximum pair, blank and nonnumeric rows are asserted."],
        }

    def _package_documents(self, name: str, prompt_payload: dict[str, Any]) -> dict[str, Any]:
        test_result = "1 passed"
        for run in prompt_payload.get("sandbox_test_results", []):
            match = re.search(r"\b\d+\s+(?:passed|failed|skipped)\b", str(run.get("stdout", "")), re.I)
            if match:
                test_result = match.group(0)
                break
        if name == "simple-script":
            title = "Campus Bicycle Count Summary CLI"
            command = "python -m src.bicycle_summary counts.csv --output reports/summary.json"
            readme_notes = "The CSV uses ISO dates and nonnegative integer counts; malformed rows are reported with row numbers."
            report = (
                "## Input and implementation\n\n`src/bicycle_summary.py` provides the Python 3.11 CLI program and `summarize_csv`. It reads UTF-8 CSV files and writes deterministic JSON output. It validates required CSV columns, ISO dates, and nonnegative counts; malformed rows are rejected with row numbers and errors reported clearly. The README documents local input and output paths.\n\n"
                "## Behavior and verification\n\nThe summary calculates row count, total, mean, and maximum daily count as deterministic output from the supplied input, not hard-coded values. The program runs offline without network access. `tests/test_bicycle_summary.py` recorded " + test_result
                + " in the supplied sandbox output (exit code 0); no coverage or performance measurement is claimed.\n"
            )
        elif name == "web-app":
            title = "Laboratory Experiment Log Web App"
            command = "python -m server.app"
            readme_notes = "The local API and accessible browser UI require no external service."
            report = (
                "## API and browser implementation\n\n`server/app.py` implements `ExperimentStore` in the Python HTTP API for creating and listing local records; it returns records newest-first and supports case-insensitive title filtering. Server-side validation produces field-specific errors for missing titles and invalid dates. `web/index.html`, `web/app.js`, and `web/styles.css` provide the accessible browser interface, associated labels, and empty state. `README.md` is the local setup guide.\n\n"
                "## Verification\n\n`tests/test_experiments.py` recorded " + test_result
                + " in the supplied sandbox output (exit code 0). No remote service or coverage result is claimed.\n"
            )
        elif name == "erdos-gyarfas":
            title = "Erdős–Gyárfás Cycles in Cubic Graphs"
            command = "python -m pytest -q"
            readme_notes = "`graphs/cycles.py` validates undirected simple graphs and enumerates canonical simple cycles."
            report = (
                "## Implementation and graph model\n\n`graphs/cycles.py` validates finite undirected simple graphs and implements `validate_simple_graph`, `enumerate_simple_cycles`, and `power_of_two_cycle_lengths`; rotations and reverse traversals are canonicalized. The Petersen graph is cubic with 10 vertices, 15 edges, and minimum degree 3.\n\n"
                "## Petersen graph result\n\n`results/petersen_cycle_summary.json` records 57 simple cycles with distinct lengths 5, 6, 8, and 9. The recorded power-of-two cycle length is 8. `tests/test_erdos_gyarfas.py` checks these values and graph validation; the supplied sandbox output records " + test_result + " (exit code 0).\n\n"
                "## Scope and complexity\n\nThis finite computation on the Petersen graph is an example with minimum degree 3, not a proof of the Erdős–Gyárfás conjecture for all graphs. The depth-first search enumerates simple paths, a factorial worst-case search; canonicalizing rotations and reversals adds quadratic work per discovered cycle, and storing results requires output-sized space. No general theorem or coverage result is claimed.\n"
            )
        else:
            title = "Reproducible Multi-Station Rainfall Analysis"
            command = "python -m analysis.rainfall rainfall.csv --output outputs/rainfall_summary.json"
            readme_notes = "The report uses sample standard deviation and records missing or nonnumeric measurements as invalid rows."
            report = (
                "## CSV analysis and comparison\n\nThe Python module `analysis/rainfall.py` uses `load_measurements` to validate the required station, month, and millimeters columns in each CSV, and `analyze` computes per-station summaries plus the wettest station-month. `outputs/rainfall_comparison.svg` is the supplied comparison chart; its title identifies the monthly rainfall comparison by station.\n\n"
                "## Statistical choices and exclusions\n\nThe test fixture records two measurements each for North, South, East, and West; North is wettest in 2025-06 at 19.0 mm. The methods section records the statistical choices: `sample_stddev` uses sample standard deviation. Blank and nonnumeric measurements are reported as invalid source rows and rejected rather than silently excluded; the regression tests cover those choices.\n\n"
                "## Verification\n\n`tests/test_rainfall.py` recorded " + test_result
                + " in the supplied sandbox output (exit code 0). The chart and fixed test fixture are the available comparison evidence; no independent measurement run or coverage result is claimed.\n"
            )
        return {
            "readme_markdown": (
                f"# {title}\n\n{readme_notes}\n\n## Run locally\n\n`{command}`\n\n"
                "## Tests\n\nRun `python -m pytest -q` from the project root.\n"
            ),
            "report_markdown": f"# Project Report: {title}\n\n{report}",
            "summary": f"Generate artifact-grounded README and project report for {title}.",
            "evidence": ["The validated ProjectSpec, source excerpts, persisted AgentRun and ReviewResult records, sandbox output, and result artifacts were used."],
        }

    @staticmethod
    def _human_evidence(name: str) -> str:
        if name == "simple-script":
            return "README.md: describes the CLI, date/count inputs, and malformed-row behavior."
        if name == "web-app":
            return "web/index.html and web/app.js: contain associated labels, live error text, and the empty state."
        if name == "erdos-gyarfas":
            return "graphs/cycles.py and results/petersen_cycle_summary.json: show the finite Petersen graph analysis; human review must assess the conjecture and complexity caveat."
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


def _run_phase6_package_workflows(
    tmp_path: Path,
    sandbox: Any,
    filenames: tuple[str, ...] | list[str],
    *,
    fail_packager: bool = False,
    invalid_first_packager: bool = False,
) -> None:
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
        model_client = Phase6FixtureClient(
            fixture,
            fail_packager=fail_packager,
            invalid_first_packager=invalid_first_packager,
        )
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
                if fail_packager:
                    assert final_gate.state is WorkflowState.FAILED
                    assert "packager provider call failed" in final_gate.failure_reason.casefold()
                    assert "simulated packager provider outage" not in final_gate.failure_reason
                    with session_factory() as session:
                        failed_packager_run = session.scalar(select(AgentRunRow).where(
                            AgentRunRow.workflow_id == workflow_id,
                            AgentRunRow.agent_name == "packager",
                        ))
                    assert failed_packager_run is not None
                    assert failed_packager_run.output_payload["provider_error"] is True
                    assert failed_packager_run.output_payload["llm_calls"] == 1
                    written_files = repository_manager.read_repository_context(project_id)
                    assert "reports/PROJECT_REPORT.md" not in written_files
                    expected_coder_readme = model_client.generated_files["readme-and-limitations"][0]["content"]
                    assert written_files["README.md"].rstrip() == expected_coder_readme.rstrip()
                    assert "ProjectForge will finalize this guide" not in written_files["README.md"]
                    assert model_client.calls.count("packager") == 1
                    continue
                assert final_gate.state is WorkflowState.AWAITING_FINAL_APPROVAL, final_gate.failure_reason
                groundedness = final_gate.result_payload["report_groundedness"]
                assert groundedness["passed"] is True, groundedness["issues"]
                assert groundedness["matched_terms"]
                assert groundedness["matched_code_anchors"]
                assert groundedness["matched_test_anchors"]
                expected_packager_attempts = 2 if invalid_first_packager else 1
                assert len(model_client.packager_inputs) == expected_packager_attempts
                assert groundedness["regeneration_attempts"] == expected_packager_attempts - 1
                packager_input = model_client.packager_inputs[-1]
                if invalid_first_packager:
                    assert "Previous report was rejected" in packager_input["regeneration_feedback"]
                assert packager_input["code_summary"]["files"]
                assert packager_input["sandbox_test_results"]
                assert packager_input["agent_run_records"]
                assert packager_input["review_result_records"]
                if fixture["name"] == "erdos-gyarfas":
                    artifacts = packager_input["experiment_output"]["artifacts"]
                    assert any(
                        item["path"] == "results/petersen_cycle_summary.json"
                        and '"cycle_lengths": [5, 6, 8, 9]' in item["content_excerpt"]
                        for item in artifacts
                    )
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
                assert "README.md" in final_bundle.json()["changed_files"]
                assert "reports/PROJECT_REPORT.md" in final_bundle.json()["changed_files"]
                assert final_bundle.json()["report_groundedness"]["passed"] is True
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
                    report_text = archive.read("reports/PROJECT_REPORT.md").decode("utf-8")
                    assert "## Final human checkpoint" in report_text
                    assert "Decision: `approved`" in report_text
                    if fixture["name"] == "erdos-gyarfas":
                        assert "distinct lengths 5, 6, 8, and 9" in report_text
                        assert "not a proof" in report_text
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


def test_erdos_gyarfas_fixture_code_passes_its_regressions_locally(tmp_path: Path):
    """Execute the fixture's generated source and tests without requiring Docker."""
    (tmp_path / "graphs").mkdir()
    (tmp_path / "graphs" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "graphs" / "cycles.py").write_text(ERDOS_SOURCE, encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_erdos_gyarfas.py").write_text(ERDOS_TESTS, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_erdos_gyarfas.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\\n" + completed.stderr
    assert "4 passed" in completed.stdout


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
        ("simple_script.json", "web_app.json", "data_analysis.json", "erdos_gyarfas.json"),
    )


def test_packager_provider_error_fails_workflow_without_generating_boilerplate(tmp_path: Path):
    """A transport failure must not produce or export canned documentation."""
    _run_phase6_package_workflows(
        tmp_path,
        PassingSandboxStub(),
        ["erdos_gyarfas.json"],
        fail_packager=True,
    )


def test_invalid_grounding_is_regenerated_once_before_final_checkpoint(tmp_path: Path):
    """Invalid generated content is retried once and only validated output is reviewed."""
    _run_phase6_package_workflows(
        tmp_path,
        PassingSandboxStub(),
        ["erdos_gyarfas.json"],
        invalid_first_packager=True,
    )


@pytest.mark.parametrize("filename", ["simple_script.json", "web_app.json", "data_analysis.json", "erdos_gyarfas.json"])
def test_all_phase3_fixture_package_api_flows_with_offline_sandbox_stub(tmp_path: Path, filename: str):
    """Exercise API, FSM, disclosure, and ZIP assembly without executing code."""
    _run_phase6_package_workflows(tmp_path, PassingSandboxStub(), [filename])
