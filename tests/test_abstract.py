from __future__ import annotations

import json

import pytest

from articlegen.abstract import (
    AbstractError,
    generate_abstract,
    unsupported_facts,
)
from articlegen.cli import main
from articlegen.llm import LLMResponse
from articlegen.manuscript import parse_manuscript

SOURCE = """# Widgets Under Load

## Abstract
We studied 42 widgets and observed a 31% improvement over baseline.

## Introduction
Prior work studied widgets [1] and gadgets (Smith, 2020). We tested 42 widgets in 2020.

## Methods
Each widget ran three trials under load.

## Results
The measured improvement was 31%.

## References
[1] Smith, J. (2020). Widget behaviour. Journal of Widgets.
"""


class FakeClient:
    available = True
    engine_name = "test-abstract"

    def __init__(self, payload):
        self.payload = payload
        self.last_user = ""

    def complete(self, system, user, *, max_output_tokens=1000):
        self.last_user = user
        return LLMResponse(json.dumps(self.payload), engine="test-abstract:fake")

    def name(self):
        return self.engine_name


def clean_payload():
    return {
        "abstract": {
            "background": "Widgets are used under load.",
            "objective": "We measured widget behaviour under load.",
            "methods": "We tested 42 widgets in 2020.",
            "results": "The measured improvement was 31%.",
            "conclusions": "Widgets improved under load at the measured scale.",
        },
        "titles": ["Widgets Under Load", "Widgets: 31% improvement under load"],
        "keywords": ["widgets", "load testing", "reliability"],
        "notes": ["One experiment was reported."],
    }


def payload_with_violations():
    data = clean_payload()
    data["abstract"]["results"] = ("The measured improvement was 31%. "
                                   "A second cohort improved by 57%.")
    data["titles"].append("Quantum Widgets Institute report")
    data["keywords"].append("Quantum Widgets Institute")
    return data


def package(payload):
    return generate_abstract(
        parse_manuscript(SOURCE, source="paper.md"), client=FakeClient(payload))


def test_clean_output_passes_the_fact_lock():
    result = package(clean_payload())
    assert result.fact_lock_ok is True
    assert result.violations == []
    assert list(result.abstract) == [
        "background", "objective", "methods", "results", "conclusions"]
    assert len(result.titles) == 2
    assert result.keywords == ["widgets", "load testing", "reliability"]
    assert result.engine == "test-abstract:fake"
    assert "PASSED" in result.to_markdown()


def test_unsupported_number_quarantines_only_its_sentence():
    result = package(payload_with_violations())
    assert result.fact_lock_ok is False
    assert "57%" not in result.abstract["results"]
    assert "31%" in result.abstract["results"]
    assert len(result.dropped_sentences) == 1
    assert result.dropped_sentences[0]["field"] == "results"
    violation = next(v for v in result.violations if v.kind == "number")
    assert (violation.field, violation.token) == ("results", "57%")


def test_unsupported_proper_name_rejects_title_and_keyword():
    result = package(payload_with_violations())
    assert result.titles == [
        "Widgets Under Load", "Widgets: 31% improvement under load"]
    assert result.rejected_titles == ["Quantum Widgets Institute report"]
    assert result.keywords == ["widgets", "load testing", "reliability"]
    assert result.rejected_keywords == ["Quantum Widgets Institute"]
    fields = {(v.field, v.kind, v.token) for v in result.violations}
    assert ("title", "name", "Quantum Widgets Institute") in fields
    assert ("keyword", "name", "Quantum Widgets Institute") in fields


def test_fact_lock_accepts_number_and_name_variants():
    source = "Erdős–Gyárfás reported 1,024 runs; Smith (2020) agreed."
    assert unsupported_facts(
        "The Erdős-Gyárfás study used 1024 runs.", source) == []


def test_fact_lock_flags_absent_number_and_name():
    found = unsupported_facts(
        "The Widget Institute improved results by 57%.", "Widgets improved by 31%.")
    assert ("number", "57%") in found
    assert ("name", "Widget Institute") in found


def test_sentence_starting_with_an_ordinary_capitalised_word_is_not_a_name():
    assert unsupported_facts(
        "The study measured widgets.", "The study measured widgets.") == []
    # A discourse word absent from the source must not drop the sentence as a "name".
    assert unsupported_facts(
        "Crucially, widgets improved.", "Widgets improved.") == []


def test_sentence_initial_acronym_is_still_name_locked():
    assert unsupported_facts("WHO reported widgets.", "Widgets improved.") == [
        ("name", "WHO")]


def test_source_too_short_is_rejected():
    with pytest.raises(AbstractError, match="too short"):
        generate_abstract(parse_manuscript("Too short."), client=FakeClient(clean_payload()))


def test_missing_provider_raises_an_actionable_error():
    with pytest.raises(AbstractError, match="API key"):
        generate_abstract(parse_manuscript(SOURCE), client=None)


def test_reply_without_abstract_object_is_rejected():
    with pytest.raises(AbstractError, match="missing the 'abstract'"):
        package({"titles": ["Widgets"]})


def test_no_surviving_section_is_an_error_not_an_empty_package():
    with pytest.raises(AbstractError, match="no abstract section survived"):
        package({"abstract": {"results": "A 99% improvement."}})


def test_abstract_command_writes_outputs_and_reports_the_lock(tmp_path, capsys,
                                                              monkeypatch):
    path = tmp_path / "paper.md"
    path.write_text(SOURCE, encoding="utf-8")
    json_path, markdown_path = tmp_path / "abstract.json", tmp_path / "abstract.md"
    client = FakeClient(clean_payload())
    monkeypatch.setattr("articlegen.cli.resolve_resilient_client",
                        lambda **kwargs: client)

    code = main(["abstract", str(path), "--json", str(json_path),
                 "--out", str(markdown_path)])

    captured = capsys.readouterr()
    assert code == 0
    assert "# Abstract package: Widgets Under Load" in captured.out
    assert markdown_path.read_text(encoding="utf-8").startswith("# Abstract package")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["fact_lock_ok"] is True
    assert data["abstract"]["results"] == "The measured improvement was 31%."
    assert data["keywords"] == ["widgets", "load testing", "reliability"]


def test_abstract_command_exits_non_zero_when_the_lock_rejects_anything(
        tmp_path, monkeypatch):
    path = tmp_path / "paper.md"
    path.write_text(SOURCE, encoding="utf-8")
    monkeypatch.setattr(
        "articlegen.cli.resolve_resilient_client",
        lambda **kwargs: FakeClient(payload_with_violations()))

    assert main(["abstract", str(path)]) == 1
