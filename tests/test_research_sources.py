"""Source retrieval, verification and the citation↔bibliography bijection."""

from __future__ import annotations

import pytest

from articlegen.models import CitationStyle
from articlegen.research.models import SourceRecord, normalise_doi
from articlegen.research.sources import (
    NullSourceRetriever,
    SourceLibrary,
    StaticSourceRetriever,
    UserSourceRetriever,
    resolve_retriever,
)
from tests.research_fixtures import (
    clean_review,
    erdos_draft,
    execute,
    good_draft,
    good_plan,
    good_spec,
    source_records,
    status_of,
    study_brief,
    unverified_library,
    validate,
    verified_library,
)


def test_user_supplied_records_are_not_presented_as_verified():
    library = unverified_library()
    report = library.report()
    assert report.total == 3
    assert report.verified == 0
    assert report.user_provided == 3
    # Usable (they have complete metadata) but not verified: the gate must say so.
    assert all(record.is_usable for record in library.records)


def test_retrieval_unavailable_produces_no_bibliography():
    library = SourceLibrary(CitationStyle.IEEE)
    outcome = library.retrieve(["monochromatic cycle"], NullSourceRetriever())
    library.verify(NullSourceRetriever())
    # The attempt happened and failed; it is reported rather than hidden.
    assert outcome.attempted is True
    assert outcome.available is False
    assert outcome.errors
    assert library.records == []
    assert library.bibliography().strip() == ""


def test_fabricated_doi_is_not_normalised_into_a_doi():
    assert normalise_doi("https://doi.org/10.1002/jgt.3190050307") == \
        "10.1002/jgt.3190050307"
    assert normalise_doi("not a doi at all") == ""
    record = SourceRecord(title="Invented", authors=["A. Author"], year=2020,
                          venue="Journal of Nothing", doi="fake-doi-123")
    assert record.doi == ""
    assert "missing DOI or stable URL" in record.metadata_issues()


def test_incomplete_metadata_records_are_not_citable():
    record = SourceRecord(title="Something", verification="verified-index")
    assert record.metadata_issues()
    assert record.is_usable is False
    library = SourceLibrary(CitationStyle.APA)
    library.add(record)
    assert "Nothing" not in library.bibliography()


def test_index_verification_promotes_records_and_formats_them():
    library = verified_library()
    bibliography = library.bibliography()
    assert "[1]" in bibliography and "[2]" in bibliography and "[3]" in bibliography
    assert "doi: 10.1002/jgt.3190050307" in bibliography
    assert library.report().complete is True


def test_author_year_style_uses_citation_keys():
    library = verified_library(style=CitationStyle.APA)
    record = library.ordered_records()[0]
    key = library.citation_key_for(record)
    surname = record.authors[0].split()[-1]
    assert key.startswith(surname.lower())
    assert str(record.year) in key
    # APA prose cites author-year, and the marker must resolve to this record.
    resolution = library.resolve_citations(
        f"An established result ({surname}, {record.year}).")
    assert not resolution.fabricated_entries


def test_citation_without_a_bibliography_entry_is_blocked():
    brief = study_brief()
    draft = good_draft(brief)
    draft.section_map()["introduction"].body = (
        "A claim attributed to a study that is not in the library [9].")
    bundle = validate(draft, brief, library=verified_library())
    assert status_of(bundle, "citations.resolution") == "fail"
    assert bundle["gate"].status.value != "ready_for_final_review"


def test_reference_entry_with_no_source_record_is_blocked():
    brief = study_brief()
    draft = good_draft(brief)
    draft.sections.append(
        __import__("articlegen.models", fromlist=["Section"]).Section(
            id="references", title="References", purpose="", word_target=50,
            body="[99] A. Invented. A paper that does not exist. Journal of Nothing, 2050."))
    bundle = validate(draft, brief, library=verified_library(),
                      plan=good_plan(brief), apply_bibliography_first=False)
    assert status_of(bundle, "citations.bibliography_grounded") == "fail"
    assert bundle["gate"].critical_failures, "a fabricated entry is a critical failure"


def test_uncited_bibliography_entry_is_reported_minor():
    brief = study_brief()
    draft = good_draft(brief)
    draft.section_map()["introduction"].body = (
        "Monochromatic cycle problems in edge-coloured complete graphs are a classical "
        "topic [1], and later work studies colourings [2]. Implementations rarely state "
        "the size domain they enumerate.")
    spec = good_spec()
    run = execute(spec)
    bundle = validate(draft, brief, library=verified_library(), plan=good_plan(brief),
                      runs=[run], specs=[spec], review=clean_review())
    check = bundle["validation"].get("citations.all_entries_used")
    assert check.status.value == "fail"
    assert check.severity.value == "minor"
    # A minor citation issue must not on its own block a validated manuscript.
    assert bundle["gate"].status.value != "incomplete_or_unverified"
    assert all("all_entries_used" not in reason for reason in bundle["gate"].reasons)


def test_retrieval_policy_resolution():
    assert isinstance(resolve_retriever("none"), NullSourceRetriever)
    assert isinstance(resolve_retriever("user", user_sources=source_records()),
                      UserSourceRetriever)
    with pytest.raises(ValueError):
        resolve_retriever("no-such-provider")


def test_retrieval_failure_is_reported_not_invented():
    class BrokenRetriever(StaticSourceRetriever):
        name = "broken-index"

        def available(self) -> bool:
            return False

        def search(self, query: str, limit: int = 10):
            raise RuntimeError("index down")

    library = SourceLibrary()
    outcome = library.retrieve(["query"], BrokenRetriever([]))
    assert outcome.attempted is True
    assert "index down" in " ".join(outcome.errors)
    assert library.records == []
