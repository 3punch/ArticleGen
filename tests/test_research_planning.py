"""The research plan: ten elements, article-type roles, and the novelty policy."""

from __future__ import annotations

import pytest

from articlegen.models import ArticleType
from articlegen.research.models import ContributionType, profile_for
from articlegen.research.planning import (
    derive_plan,
    outline_additions,
    plan_from_payload,
    validate_plan,
)
from tests.research_fixtures import source_records, study_brief, verified_library


def test_plan_requires_all_ten_elements_for_a_research_article():
    brief = study_brief()
    # A brief may carry a question; remove it so the plan has to supply everything.
    brief.research_question = ""
    plan = derive_plan(brief)
    assert plan.missing_elements(), "an empty plan must report its missing elements"
    validate_plan(plan, brief, library=verified_library(),
                  retrieval_attempted=True, retrieval_available=True)
    failed = {check.check_id for check in plan.issues
              if check.status.value != "pass"}
    for check_id in ("plan.research_question", "plan.contribution",
                     "plan.methodology", "plan.limitations", "plan.evaluation"):
        assert check_id in failed
    assert plan.validated is False
    assert plan.decision in ("narrow_claims", "needs_user_direction")


def test_unsupported_novelty_claim_is_blocked_and_narrowed():
    brief = study_brief(vector="erdos")
    plan = derive_plan(brief)
    plan.problem = brief.objective
    plan.research_question = brief.research_question
    plan.gap = ""                       # no gap can be established
    plan.gap_evidence = []
    plan.contribution = "The first traversal that eliminates every monochromatic cycle."
    plan.novelty_claims = ["the first method of its kind"]
    plan.methodology = "Traverse in both directions and verify small cases."
    plan.methodology_justification = "It is simple."
    plan.limitations = ["Finite size range."]
    plan.failure_modes = ["Larger sizes untested."]
    plan.evaluation_criteria = ["Counts match the listed sizes."]

    validate_plan(plan, brief, library=verified_library(),
                  retrieval_attempted=True, retrieval_available=True)

    novelty = next(c for c in plan.issues if c.check_id == "plan.novelty_supported")
    assert novelty.status.value == "fail"
    assert plan.decision == "narrow_claims"
    assert plan.validated is False
    assert any("novelty" in reason for reason in plan.decision_reasons)


def test_routine_implementation_is_reframed_as_replication():
    brief = study_brief()
    plan = derive_plan(brief)
    plan.research_question = brief.research_question
    plan.gap = "No implementation states the size domain it enumerates."
    plan.gap_evidence = ["[1]", "[2]"]
    plan.contribution = "A re-implementation of the standard cycle checker."
    plan.contribution_type = ContributionType.ROUTINE_IMPLEMENTATION
    plan.novelty_claims = ["the first implementation"]
    plan.methodology = "Re-implement the checker and record its outputs precisely."
    plan.methodology_justification = "The reference description is complete."
    plan.limitations = ["Only the default exponent range is exercised."]
    plan.failure_modes = ["Larger exponents are not examined."]
    plan.evaluation_criteria = ["Counts match the generated sizes."]

    validate_plan(plan, brief, library=verified_library(),
                  retrieval_attempted=True, retrieval_available=True)

    assert plan.contribution_type == ContributionType.REPLICATION
    assert plan.decision == "narrow_claims"
    assert any("routine implementation" in reason for reason in plan.decision_reasons)


def test_gap_without_literature_evidence_is_reported():
    brief = study_brief()
    plan = derive_plan(brief)
    plan.research_question = brief.research_question
    plan.gap = "Nobody has enumerated these sizes."
    plan.gap_evidence = []
    plan.contribution = "A harness that enumerates the sizes and records them."
    plan.methodology = "Enumerate sizes and record outputs with a correctness argument."
    plan.methodology_justification = "It makes the domain explicit."
    plan.limitations = ["Finite exponent range only."]
    plan.failure_modes = ["No statement about larger sizes."]
    plan.evaluation_criteria = ["The generated sizes are recorded."]

    validate_plan(plan, brief, library=verified_library(),
                  retrieval_attempted=True, retrieval_available=True)
    gap_check = next(c for c in plan.issues if c.check_id == "plan.gap_established")
    assert gap_check.status.value == "fail"


def test_required_roles_depend_on_article_type():
    computational = profile_for(ArticleType.COMPUTATIONAL)
    educational = profile_for(ArticleType.EDUCATIONAL)
    assert "results" in computational.required_roles
    assert "results" not in educational.required_roles
    assert computational.requires_experiments is True
    assert educational.requires_experiments is False


def test_outline_additions_only_add_missing_required_roles():
    brief = study_brief()
    brief.sections = [s for s in brief.sections if s.id != "limitations"]
    plan = derive_plan(brief)
    plan.article_type = ArticleType.COMPUTATIONAL
    plan.outline = [
        {"id": "introduction", "title": "Introduction", "role": "introduction"},
        {"id": "limitations", "title": "Limitations", "role": "limitations"},
        {"id": "results", "title": "Results", "role": "results"},
    ]
    additions = outline_additions(plan, brief)
    assert [section.id for section in additions] == ["limitations"]


def test_plan_payload_parsing_rejects_unknown_shapes():
    brief = study_brief()
    plan = plan_from_payload({"plan": {"research_question": "Does it work?",
                                       "outline": "not-a-list"}}, brief)
    assert plan.research_question == "Does it work?"
    assert plan.outline, "the brief outline must be used when the payload has none"
    assert all(isinstance(item, dict) for item in plan.outline)


def test_plan_keeps_unknowns_unknown_instead_of_inventing_them():
    brief = study_brief()
    plan = derive_plan(brief)
    assert plan.gap == ""
    assert plan.gap_evidence == []
    assert plan.limitations == []
    assert "gap" in plan.missing_elements()


def test_plan_serialisation_round_trips():
    from articlegen.research.models import ResearchPlan

    brief = study_brief()
    plan = derive_plan(brief)
    plan.gap = "A real gap statement."
    plan.gap_evidence = ["[1]", "[2]"]
    restored = ResearchPlan.from_dict(plan.to_dict())
    assert restored.gap == plan.gap
    assert restored.article_type == plan.article_type
    assert restored.to_dict()["gap_evidence"] == ["[1]", "[2]"]
