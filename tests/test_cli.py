import json
from pathlib import Path

import pytest

from articlegen.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_BRIEF = REPO_ROOT / "examples" / "brief_example.json"


def test_init_template_is_routed_back_until_user_replaces_placeholders(tmp_path, capsys):
    brief_path = tmp_path / "brief.json"

    assert main(["init", str(brief_path)]) == 0
    template = json.loads(brief_path.read_text(encoding="utf-8"))
    assert "TODO" in template["title"]
    assert main(["verify", str(brief_path)]) == 1

    output = capsys.readouterr().out
    assert "decision: ROUTE_BACK" in output
    assert "unfinished placeholder" in output


@pytest.mark.parametrize(
    "payload,expected_error",
    [
        (["not", "a", "brief"], "top level"),
        ({"task_id": "test-article", "sections": "not-an-array"},
         "sections must be an array"),
        ({"task_id": "test-article", "requires_citations": "false"},
         "requires_citations must be a boolean"),
    ],
)
def test_verify_reports_malformed_brief_json_without_traceback(
    tmp_path, capsys, payload, expected_error
):
    brief_path = tmp_path / "malformed.json"
    brief_path.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["verify", str(brief_path)]) == 2
    error = capsys.readouterr().err
    assert "error: invalid input:" in error
    assert expected_error in error
    assert "Traceback" not in error


def test_run_cli_refuses_to_generate_without_a_model(tmp_path, capsys, monkeypatch):
    workspace = tmp_path / "runs"
    changelog = tmp_path / "change_log.md"
    result_path = tmp_path / "result.json"
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DAHL_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    exit_code = main([
        "run", str(EXAMPLE_BRIEF),
        "--provider", "dahl",
        "--iterations", "2",
        "--workspace", str(workspace),
        "--changelog", str(changelog),
        "--json", str(result_path),
    ])

    output = capsys.readouterr().out
    assert exit_code == 1, output
    assert "status    : ROUTE_BACK" in output
    assert "no offline fallback" in output.lower()
    assert "draft     : not created" in output

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "ROUTE_BACK"
    assert result["iterations"] == []
    assert result["draft"] is None
    assert not (workspace / "demo-article").exists()
