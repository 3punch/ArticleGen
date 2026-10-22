"""Tests for one-prompt -> Brief construction (autonomous entry point)."""

import json

from articlegen.gate import verify_brief
from articlegen.llm import LLMClient, LLMResponse, OfflineClient
from articlegen.prompt2brief import brief_from_prompt


class _FakeModel(LLMClient):
    def __init__(self, label, response=None, side_effect=None, available=True):
        self.engine_name = label
        self.model = label
        self._available = available
        self.response = response
        self.side_effect = side_effect

    @property
    def available(self):
        return self._available

    def complete(self, system, user, *, max_output_tokens=1000):
        if self.side_effect is not None:
            return self.side_effect()
        return self.response


def test_fallback_brief_is_gate_valid():
    b = brief_from_prompt(
        "Write a computational article about verifying the Erdos-Gyarfas conjecture "
        "with a runnable Python experiment on cubic graphs",
        OfflineClient(),
    )
    assert b.article_type.value == "computational"
    assert b.requires_citations is True
    assert b.requires_experiments is True
    gate = verify_brief(b)
    assert gate.ok, [i.claim for i in gate.issues if i.severity.value == "blocker"]


def test_review_prompt_infers_review_type():
    b = brief_from_prompt("A narrative review of transformer attention mechanisms",
                          OfflineClient())
    assert b.article_type.value == "narrative_review"
    assert b.requires_citations is True


def test_model_dict_overrides_fallback():
    model = _FakeModel("m", response=LLMResponse(text=json.dumps({
        "title": "Custom Title",
        "objective": "x" * 50,
        "audience": "working engineers",
        "article_type": "empirical",
        "requires_citations": True,
        "requires_experiments": True,
        "target_words": 900,
        "sections": [
            {"id": "intro", "title": "Intro", "purpose": "Set up the problem for the reader.", "word_target": 300},
            {"id": "method", "title": "Method", "purpose": "Describe the approach clearly.", "word_target": 400},
            {"id": "res", "title": "Results", "purpose": "Report what was found.", "word_target": 200},
        ],
    }), engine="m"))
    b = brief_from_prompt("do a study", model)
    assert b.title == "Custom Title"
    assert b.target_words == 900
    assert len(b.sections) == 3
    assert b.sections[0].id == "intro"


def test_too_few_model_sections_keep_defaults():
    model = _FakeModel("m", response=LLMResponse(text=json.dumps({
        "title": "T",
        "objective": "y" * 50,
        "audience": "a",
        "sections": [{"id": "only", "title": "Only", "purpose": "x", "word_target": 100}],
    }), engine="m"))
    b = brief_from_prompt("thing", model)
    assert len(b.sections) >= 3  # fell back to the default outline


def test_malformed_model_output_uses_fallback():
    model = _FakeModel("m", response=LLMResponse(text="not json at all", engine="m"))
    b = brief_from_prompt("anything", model)
    gate = verify_brief(b)
    assert gate.ok
