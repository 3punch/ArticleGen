from __future__ import annotations

import json
from types import SimpleNamespace

from articlegen.cli import main
from articlegen.llm import LLMResponse
from articlegen.manuscript import numeric_tokens, parse_manuscript
from articlegen.manuscript_review import ManuscriptReview, review_manuscript

MANUSCRIPT = """# Widgets Under Load

## Abstract
We studied 42 widgets and observed a 37% improvement over baseline.

## Introduction
Prior work studied widgets [1] and gadgets (Smith, 2020). We tested 42 widgets.

## Methods
Each widget ran three trials [3].

## Results
The measured improvement was 31%.

## References
[1] Smith, J. (2020). Widget behaviour. Journal of Widgets. https://doi.org/10.1000/xyz123
[2] Doe, A. (2019). Gadget survey.
"""

CLEAN = """# A Clean Manuscript

## Abstract
This article describes a method for testing widgets and reports the outcome of that
test in plain language.

## Introduction
We describe a method for testing widgets under load [1].

## Method
Widgets are tested under load in three stages, and each stage is recorded.

## References
[1] Smith, J. (2020). Widget testing. Journal of Widgets.
"""

PLAIN_TEXT = """Reliability of Widgets

Abstract
We studied 42 widgets and observed a 31% improvement.

Introduction
Prior work studied widgets [1]. We measured a 31% improvement on 42 widgets.

References
[1] Smith, J. (2020). Widget behaviour. Journal of Widgets. https://doi.org/10.1000/xyz123
"""

DOI_DOCUMENT = """# DOI Fixture

## Introduction
Three sources are cited [1] [2] [3].

## References
[1] Alpha, A. (2020). Resolvable work. https://doi.org/10.1000/good
[2] Beta, B. (2020). Absent work. https://doi.org/10.1000/missing
[3] Gamma, C. (2020). Offline work. https://doi.org/10.1000/offline
"""


class FakeJudge:
    available = True
    engine_name = "test-reviewer"

    def __init__(self, payload):
        self.payload = payload
        self.last_user = ""

    def complete(self, system, user, *, max_output_tokens=1000):
        self.last_user = user
        return LLMResponse(json.dumps(self.payload), engine="test-reviewer:fake")

    def name(self):
        return self.engine_name


class FakeDoiRetriever:
    """Duck-typed DOI retriever: a record, a 404, or a raised network error."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def lookup_doi(self, doi):
        self.calls.append(doi)
        value = self.table[doi]
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(**value)


def model_payload(issues):
    return {
        "summary": "A method paper with a thin empirical section.",
        "strengths": ["The method is stated once and used consistently."],
        "issues": issues,
        "limitations": ["No external fact-checking was possible."],
        "readiness": "accept",
    }


def model_issue(quote, severity="major", dimension="analysis_and_claims"):
    return {"dimension": dimension, "severity": severity, "section": "introduction",
            "quote": quote, "claim": "The claim is not supported.",
            "action": "Add the missing support."}


def review(text, **kwargs) -> ManuscriptReview:
    return review_manuscript(parse_manuscript(text), **kwargs)


def check_ids(report: ManuscriptReview) -> set:
    return {finding.check_id for finding in report.findings}


def test_parses_title_sections_references_and_author_year_citation():
    manuscript = parse_manuscript(MANUSCRIPT, source="widgets.md")
    assert manuscript.title == "Widgets Under Load"
    assert [section.title for section in manuscript.sections] == [
        "Abstract", "Introduction", "Methods", "Results"]
    assert [entry.key for entry in manuscript.references] == ["1", "2"]
    assert manuscript.references[0].doi == "10.1000/xyz123"
    assert ("smith", "2020") in manuscript.cited_author_years


def test_plain_text_without_headings_does_not_invent_structure():
    manuscript = parse_manuscript("Just prose without any headings or references.")
    assert [(section.id, section.title) for section in manuscript.sections] == [
        ("document", "Document")]
    assert manuscript.abstract_heading_present is False


def test_numeric_tokens_ignore_years_single_digits_and_code():
    tokens = numeric_tokens("In 2020 we ran 3 trials on 250 samples with a 12% gain.\n"
                            "```python\nx = 99\n```")
    assert tokens == {"250": 1, "12%": 1}


def test_plain_text_abstract_and_references_lines_are_headings():
    manuscript = parse_manuscript(PLAIN_TEXT, source="plain.txt")
    assert [section.title for section in manuscript.sections] == [
        "Abstract", "Introduction"]
    assert [entry.key for entry in manuscript.references] == ["1"]
    assert manuscript.reference_heading_line is not None

    report = review(PLAIN_TEXT, model_review=False)
    assert "citations.no_reference_list" not in check_ids(report)
    assert "citations.undefined_key" not in check_ids(report)
    assert "abstract.number_not_in_body" not in check_ids(report)


def test_plain_text_heading_variants_with_bold_and_trailing_colon():
    text = ("**Abstract**\nWe tested 42 widgets in total.\n\n"
            "References:\n[1] Smith, J. (2020). Widgets.\n")
    manuscript = parse_manuscript(text)
    assert [section.title for section in manuscript.sections] == ["Abstract"]
    assert [entry.key for entry in manuscript.references] == ["1"]
    # No body besides the abstract: the number comparison must not fire.
    assert "abstract.number_not_in_body" not in check_ids(review(text, model_review=False))


def test_a_prose_sentence_mentioning_references_is_not_a_heading():
    text = ("References are listed at the end of this document.\n"
            "The survey is cited as [1].\n")
    manuscript = parse_manuscript(text)
    assert [section.id for section in manuscript.sections] == ["document"]
    assert manuscript.references == []


def test_deterministic_layer_reports_citation_number_and_uncited_reference():
    report = review(MANUSCRIPT, model_review=False)
    assert check_ids(report) >= {
        "citations.undefined_key", "references.uncited", "abstract.number_not_in_body"}
    assert "citations.undefined_author_year" not in check_ids(report)
    assert (report.readiness, report.counts["major"], report.counts["blocker"]) == (
        "minor_revision", 2, 0)


def test_reference_cited_by_author_year_is_not_reported_uncited():
    text = """# T

## Introduction
The survey is summarised in (Smith, 2020).

## References
[1] Smith, J. (2020). The survey. Journal of Widgets.
"""
    assert "references.uncited" not in check_ids(review(text, model_review=False))


def test_empty_section_is_major():
    text = """# T

## Introduction
We cite a source [1].

## Discussion

## References
[1] Smith, J. (2020). Something.
"""
    assert "structure.empty_section" in check_ids(review(text, model_review=False))


def test_abstract_heading_without_text_is_not_reported_as_missing():
    text = """# T

## Abstract

## Introduction
We cite a source [1].

## References
[1] Smith, J. (2020). Something.
"""
    report = review(text, model_review=False)
    assert "abstract.empty" in check_ids(report)
    assert "abstract.missing" not in check_ids(report)
    assert "The abstract heading is present but empty" in report.to_markdown()


def test_placeholder_blocks_and_cli_exits_non_zero(tmp_path):
    path = tmp_path / "draft.md"
    path.write_text("# T\n\n## Introduction\nTODO: write the introduction.\n",
                    encoding="utf-8")
    report = review(path.read_text(encoding="utf-8"), model_review=False)
    assert report.readiness == "major_revision"
    assert [f.check_id for f in report.blockers] == ["text.placeholder"]
    assert main(["review", str(path), "--no-model"]) == 1


def test_duplicate_doi_and_entry_are_reported():
    text = """# T

## Introduction
Both sources are cited [1] [2].

## References
[1] Smith, J. (2020). A paper. https://doi.org/10.1000/abc
[2] Smith, J. (2020). A paper. https://doi.org/10.1000/abc
"""
    assert {"references.duplicate_doi", "references.duplicate_entry"} <= check_ids(
        review(text, model_review=False))


def test_doi_verification_reports_resolved_absent_and_network_error():
    retriever = FakeDoiRetriever({
        "10.1000/good": {"ok": True, "sources": ["record"], "error": ""},
        "10.1000/missing": {
            "ok": False, "sources": [],
            "error": "Crossref HTTP 404 for https://api.crossref.org/works/10.1000/missing"},
        "10.1000/offline": RuntimeError("Crossref network error: timed out"),
    })
    report = review(DOI_DOCUMENT, model_review=False, verify_dois=True,
                    retriever=retriever)

    unverified = [f for f in report.findings
                  if f.check_id == "references.doi_unverified"]
    assert len(unverified) == 1
    assert "10.1000/missing" in unverified[0].title
    assert "not proof that the reference is fabricated" in unverified[0].detail
    assert retriever.calls == ["10.1000/good", "10.1000/missing", "10.1000/offline"]
    assert any("10.1000/offline" in note and "could not be verified" in note
               for note in report.notes)
    assert any("resolved 1 of 3" in note for note in report.notes)


def test_doi_verification_without_any_doi_is_reported_as_a_note():
    report = review(CLEAN, model_review=False, verify_dois=True,
                    retriever=FakeDoiRetriever({}))
    assert any("has no DOI" in note for note in report.notes)


def test_model_issue_without_verbatim_quote_is_discarded():
    judge = FakeJudge(model_payload([
        model_issue("We describe a method for testing widgets under load"),
        model_issue("This sentence does not appear anywhere in the manuscript."),
    ]))
    report = review(CLEAN, client=judge)

    assert report.model.ran is True
    assert report.model.engine == "test-reviewer:fake"
    assert len(report.advisory) == 1
    assert report.advisory[0].quote.startswith("We describe")
    assert report.model.dropped_quotes == 1
    assert any("does not appear verbatim" in note for note in report.model.notes)


def test_model_cannot_change_readiness_or_supply_a_verdict():
    judge = FakeJudge(model_payload([
        model_issue("Widgets are tested under load in three stages",
                    severity="critical", dimension="structure_and_clarity"),
    ]))
    report = review(CLEAN, client=judge)

    assert report.readiness == "no_blocking_findings"
    assert report.counts["blocker"] == 0
    assert report.advisory[0].severity == "critical"
    assert any("verdict was ignored" in note for note in report.model.notes)


def test_missing_provider_keeps_deterministic_report():
    report = review(CLEAN, client=None)
    assert report.model.ran is False
    assert any("No model provider" in note for note in report.model.notes)
    assert report.readiness == "no_blocking_findings"


def test_review_command_writes_markdown_and_json(tmp_path, capsys):
    path = tmp_path / "clean.md"
    path.write_text(CLEAN, encoding="utf-8")
    json_path, markdown_path = tmp_path / "review.json", tmp_path / "review.md"

    code = main(["review", str(path), "--no-model",
                 "--json", str(json_path), "--out", str(markdown_path)])

    captured = capsys.readouterr()
    assert code == 0
    assert "# Manuscript review: A Clean Manuscript" in captured.out
    assert markdown_path.read_text(encoding="utf-8").startswith("# Manuscript review")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["readiness"] == "no_blocking_findings"
    assert data["stats"]["references"] == 1
    assert data["model"]["ran"] is False
    assert data["limits"] and "Advisory model findings never change this value" in \
        data["readiness_rule"]
