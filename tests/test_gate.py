import json
import os

import pytest

from articlegen.gate import verify_brief, verify_draft
from articlegen.models import Brief, Draft, Section, Severity

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def test_valid_brief_passes(valid_brief):
    result = verify_brief(valid_brief)
    assert result.ok is True
    assert result.routed_back is False
    assert not result.blockers
    assert "sections" in result.checks_run


def test_missing_task_id_routes_back(valid_brief):
    valid_brief.task_id = ""
    result = verify_brief(valid_brief)
    assert result.routed_back is True
    assert any(i.category == "structure" and i.section == "task_id" for i in result.blockers)


def test_short_objective_routes_back(valid_brief):
    valid_brief.objective = "too short"
    result = verify_brief(valid_brief)
    assert result.routed_back is True
    assert any("objective" in i.section for i in result.blockers)


def test_target_words_out_of_range_routes_back(valid_brief):
    valid_brief.target_words = 50
    assert verify_brief(valid_brief).routed_back is True
    valid_brief.target_words = 999999
    assert verify_brief(valid_brief).routed_back is True


def test_too_few_sections_routes_back(valid_brief):
    valid_brief.sections = valid_brief.sections[:2]
    result = verify_brief(valid_brief)
    assert result.routed_back is True


def test_duplicate_section_ids_route_back(valid_brief):
    valid_brief.sections[1].id = valid_brief.sections[0].id
    result = verify_brief(valid_brief)
    assert result.routed_back is True
    assert any("duplicate section id" in i.claim for i in result.blockers)


def test_placeholder_in_objective_routes_back(valid_brief):
    valid_brief.objective = "Write a thorough guide. TODO: finish this later with real detail."
    result = verify_brief(valid_brief)
    assert result.routed_back is True
    assert any(i.category == "formatting" for i in result.blockers)


def test_unbalanced_code_fence_in_seed_body_routes_back(valid_brief):
    valid_brief.sections[0].body = "Some text.\n\n```python\nprint('hi')\n"
    result = verify_brief(valid_brief)
    assert result.routed_back is True
    assert any("unbalanced" in i.claim for i in result.blockers)


def test_broken_python_block_is_a_blocker(valid_brief):
    valid_brief.sections[0].body = "```python\ndef broken(:\n    pass\n```"
    result = verify_brief(valid_brief)
    assert result.routed_back is True
    assert any(i.category == "correctness" for i in result.blockers)


def test_shipped_invalid_example_is_rejected():
    with open(os.path.join(EXAMPLES, "brief_invalid.json"), "r", encoding="utf-8") as handle:
        brief = Brief.from_dict(json.load(handle))
    result = verify_brief(brief)
    assert result.routed_back is True
    assert len(result.blockers) >= 3


def test_shipped_valid_example_passes(example_brief_dict):
    result = verify_brief(Brief.from_dict(example_brief_dict))
    assert result.ok is True, [i.claim for i in result.issues]


def test_draft_missing_a_section_is_a_blocker(valid_brief):
    draft = Draft.from_brief(valid_brief)
    for section in draft.sections:
        section.body = "Real content that satisfies the section purpose adequately."
    draft.sections = draft.sections[:2]
    result = verify_draft(draft, valid_brief)
    assert result.routed_back is True
    assert any("missing outline section" in i.claim for i in result.blockers)


def test_draft_extra_section_is_major_not_blocker(valid_brief):
    draft = Draft.from_brief(valid_brief)
    for section in draft.sections:
        section.body = "Real content that satisfies the section purpose adequately."
    draft.sections.append(Section(id="rogue", title="Rogue", body="Not in the outline at all."))
    result = verify_draft(draft, valid_brief)
    assert result.routed_back is False
    assert any(i.severity == Severity.MAJOR and "not in the outline" in i.claim
               for i in result.issues)


def test_draft_requires_citations_when_brief_says_so(valid_brief):
    valid_brief.requires_citations = True
    draft = Draft.from_brief(valid_brief)
    for section in draft.sections:
        section.body = "Content with no sources whatsoever, just assertions."
    result = verify_draft(draft, valid_brief)
    assert any(i.category == "citations" for i in result.issues)


def test_malformed_citation_url_is_flagged(valid_brief):
    valid_brief.requires_citations = True
    draft = Draft.from_brief(valid_brief)
    draft.sections[0].body = "See the source at https://notaurl for details."
    for section in draft.sections[1:]:
        section.body = "Supporting content that meets the word target for this section."
    result = verify_draft(draft, valid_brief)
    assert any("malformed citation" in i.claim for i in result.issues)
