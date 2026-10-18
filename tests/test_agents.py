import pytest

from articlegen.agents import build_agents
from articlegen.agents.synthesizer import _apply_text_fixes, build_draft
from articlegen.agents.writer import scaffold_body
from articlegen.llm import OfflineClient, coerce_findings, parse_json, ResponseParseError
from articlegen.models import AgentStatus, Draft, Section, Severity
from articlegen.tokens import TokenBudget


@pytest.fixture
def agents():
    return {a.agent_id: a for a in build_agents(OfflineClient(), TokenBudget.default())}


def test_scaffold_body_reaches_the_word_target(valid_brief):
    section = valid_brief.sections[1]
    body = scaffold_body(section, valid_brief)
    assert len(body.split()) >= section.word_target
    assert len(body.split()) <= section.word_target * 1.6


def test_scaffold_body_is_deterministic(valid_brief):
    section = valid_brief.sections[0]
    assert scaffold_body(section, valid_brief) == scaffold_body(section, valid_brief)


def test_scaffold_body_contains_no_placeholder_markers(valid_brief):
    body = scaffold_body(valid_brief.sections[0], valid_brief).lower()
    for marker in ("todo", "tbd", "fixme", "lorem ipsum", "placeholder"):
        assert marker not in body


def test_writer_fills_empty_sections(agents, valid_brief):
    draft = Draft.from_brief(valid_brief)
    payload = agents["writer"].deterministic(valid_brief, draft, 1, [])
    assert len(payload["sections"]) == len(valid_brief.sections)
    assert all(item["body"].strip() for item in payload["sections"])


def test_writer_skips_sections_that_are_already_long_enough(agents, valid_brief):
    draft = Draft.from_brief(valid_brief)
    draft.sections[0].body = " ".join(["already complete content"] * 200)
    payload = agents["writer"].deterministic(valid_brief, draft, 1, [])
    assert all(item["id"] != "introduction" for item in payload["sections"])


def test_tester_flags_broken_python_as_blocker(agents, valid_brief):
    draft = Draft.from_brief(valid_brief)
    for section in draft.sections:
        section.body = "Some explanatory prose for this section."
    draft.sections[0].body += "\n\n```python\ndef f(:\n    pass\n```"
    payload = agents["tester"].deterministic(valid_brief, draft, 1, [])
    findings = coerce_findings(payload["findings"], "tester", 1)
    assert any(f.severity == Severity.BLOCKER and f.category == "code" for f in findings)


def test_tester_accepts_valid_python(agents, valid_brief):
    draft = Draft.from_brief(valid_brief)
    for section in draft.sections:
        section.body = "Prose here.\n\n```python\nvalue = 1 + 1\n```"
    payload = agents["tester"].deterministic(valid_brief, draft, 1, [])
    findings = coerce_findings(payload["findings"], "tester", 1)
    assert not any(f.category == "code" and "does not compile" in f.claim for f in findings)


def test_tester_detects_contradictory_metric_claims(agents, valid_brief):
    draft = Draft.from_brief(valid_brief)
    draft.sections[0].body = "The cache latency is 20ms under normal load."
    draft.sections[1].body = "Measured in production, cache latency is 85ms under normal load."
    draft.sections[2].body = "Further discussion of the trade-offs involved here."
    payload = agents["tester"].deterministic(valid_brief, draft, 1, [])
    findings = coerce_findings(payload["findings"], "tester", 1)
    assert any("conflicting values" in f.claim for f in findings)


def test_tester_flags_unsupported_numeric_claim(agents, valid_brief):
    draft = Draft.from_brief(valid_brief)
    draft.sections[0].body = "This improves throughput by 40% in every deployment."
    for section in draft.sections[1:]:
        section.body = "Neutral prose with no numbers in it at all."
    payload = agents["tester"].deterministic(valid_brief, draft, 1, [])
    findings = coerce_findings(payload["findings"], "tester", 1)
    assert any(f.category == "fabrication" for f in findings)


def test_tester_resolves_real_anchors(agents, valid_brief):
    draft = Draft.from_brief(valid_brief)
    for section in draft.sections:
        section.body = "Prose."
    draft.sections[0].body = "See [limits](#limits) and [ghost](#does-not-exist)."
    payload = agents["tester"].deterministic(valid_brief, draft, 1, [])
    findings = coerce_findings(payload["findings"], "tester", 1)
    claims = " ".join(f.claim for f in findings)
    assert "does-not-exist" in claims
    assert "#limits" not in claims


def test_reviewer_computes_metrics(agents, valid_brief):
    draft = Draft.from_brief(valid_brief)
    draft.sections[0].body = "Short sentence. Another short one."
    draft.sections[1].body = ("This sentence is deliberately extremely long and keeps going and "
                              "going with clause after clause until it is very hard to follow at "
                              "all because it never once stops for breath. " * 3)
    draft.sections[2].body = "Concise prose here."
    payload = agents["reviewer"].deterministic(valid_brief, draft, 1, [])
    assert payload["metrics"]["sentences"] > 0
    assert payload["metrics"]["avg_sentence_words"] > 0


def test_reviewer_flags_filler_and_duplicates(agents, valid_brief):
    draft = Draft.from_brief(valid_brief)
    draft.sections[0].body = "This is is very basically really important to note."
    for section in draft.sections[1:]:
        section.body = "Clean prose without padding."
    payload = agents["reviewer"].deterministic(valid_brief, draft, 1, [])
    findings = coerce_findings(payload["findings"], "reviewer", 1)
    assert any(f.category == "clarity" for f in findings)


def test_architect_detects_missing_and_extra_sections(agents, valid_brief):
    draft = Draft.from_brief(valid_brief)
    draft.sections = draft.sections[:2]
    draft.sections.append(Section(id="rogue", title="Rogue", body="Off outline."))
    payload = agents["architect"].deterministic(valid_brief, draft, 1, [])
    findings = coerce_findings(payload["findings"], "architect", 1)
    assert any("absent from the draft" in f.claim for f in findings)
    assert any("outside the outline" in f.claim for f in findings)


def test_architect_detects_order_divergence(agents, valid_brief):
    draft = Draft.from_brief(valid_brief)
    for section in draft.sections:
        section.body = "Content."
    draft.sections.reverse()
    payload = agents["architect"].deterministic(valid_brief, draft, 1, [])
    assert payload["target_order"] == valid_brief.section_ids


def test_synthesizer_merges_writer_output_into_a_new_draft(agents, valid_brief):
    draft = Draft.from_brief(valid_brief)
    writer = agents["writer"]
    tester = agents["tester"]
    reviewer = agents["reviewer"]
    architect = agents["architect"]

    class _Fake:
        def __init__(self, agent_id, payload=None, findings=None):
            self.agent_id = agent_id
            self.payload = payload or {}
            self.findings = findings or []

    writer_payload = writer.deterministic(valid_brief, draft, 1, [])
    filled = Draft.from_brief(valid_brief)
    for item in writer_payload["sections"]:
        filled.section_map()[item["id"]].body = item["body"]

    reports = [
        _Fake("writer", payload={"sections": writer_payload["sections"]}),
        _Fake("tester", findings=coerce_findings(
            tester.deterministic(valid_brief, filled, 1, [])["findings"], "tester", 1)),
        _Fake("reviewer", findings=coerce_findings(
            reviewer.deterministic(valid_brief, filled, 1, [])["findings"], "reviewer", 1)),
        _Fake("architect", findings=coerce_findings(
            architect.deterministic(valid_brief, filled, 1, [])["findings"], "architect", 1)),
    ]
    payload = agents["synthesizer"].deterministic(valid_brief, draft, 1, reports)
    new_draft = build_draft(payload, valid_brief, draft.version)

    assert new_draft is not None
    assert new_draft.version == 1
    assert [s.id for s in new_draft.sections] == valid_brief.section_ids
    assert all(s.body.strip() for s in new_draft.sections)
    assert payload["applied"]


def test_synthesizer_restores_missing_outline_sections(agents, valid_brief):
    draft = Draft.from_brief(valid_brief)
    for section in draft.sections:
        section.body = "Existing content."
    draft.sections = draft.sections[:1]

    class _Fake:
        def __init__(self, agent_id):
            self.agent_id = agent_id
            self.payload = {}
            self.findings = []

    reports = [_Fake("writer"), _Fake("tester"), _Fake("reviewer"), _Fake("architect")]
    payload = agents["synthesizer"].deterministic(valid_brief, draft, 1, reports)
    new_draft = build_draft(payload, valid_brief, draft.version)
    assert [s.id for s in new_draft.sections] == valid_brief.section_ids


def test_writer_skips_when_there_is_no_work(valid_brief, agents):
    draft = Draft.from_brief(valid_brief)
    for section in draft.sections:
        section.body = "Prose. " * 300
    writer = agents["writer"]
    reason = writer.should_skip(valid_brief, draft, [], [])
    assert reason is not None
    report = writer.run(
        __import__("articlegen.memory", fromlist=["AssembledContext"]).AssembledContext(text="c", tokens=1),
        valid_brief, draft, 2,
    )
    assert report.tokens_in == 0 and report.tokens_out == 0
    assert any("skipped" in note for note in report.notes)
    assert not writer.should_skip(valid_brief, Draft.from_brief(valid_brief), [], [])


def test_filler_cleanup_does_not_change_url_scheme_case():
    body = "It is important to note that the source is useful. https://example.org/paper."

    cleaned = _apply_text_fixes(body, {"remove_fillers"})

    assert "https://example.org/paper" in cleaned
    assert "Https://" not in cleaned


def test_mechanical_polish_preserves_paragraphs_and_lines(valid_brief, agents):
    draft = Draft.from_brief(valid_brief)
    payload = agents["writer"].deterministic(valid_brief, draft, 1, [])
    for item in payload["sections"]:
        body = item["body"]
        polished, ops = __import__("articlegen.agents.writer", fromlist=["apply_mechanical_polish"]).apply_mechanical_polish(body)
        # Convergence: polishing an already-polished body changes nothing.
        again, _ = __import__("articlegen.agents.writer", fromlist=["apply_mechanical_polish"]).apply_mechanical_polish(polished)
        assert again == polished
        # Paragraph count can only grow, never merge.
        assert polished.count("\n\n") >= body.count("\n\n")
        # The gate's 400-character line limit is respected.
        assert all(len(line) <= 420 for line in polished.split("\n\n"))


def test_sections_open_on_distinct_sentences(valid_brief, agents):
    draft = Draft.from_brief(valid_brief)
    payload = agents["writer"].deterministic(valid_brief, draft, 1, [])
    openings = [item["body"].split(".")[0] for item in payload["sections"]]
    assert len(openings) == len(valid_brief.sections)
    assert len({s.lower() for s in openings}) == len(openings), openings


def test_writer_delta_block_hides_unchanged_sections(valid_brief):
    from articlegen.memory import writer_delta_block
    draft = Draft.from_brief(valid_brief)
    for section in draft.sections:
        section.body = "Established prose that needs no further work this run. " * 40
    block = writer_delta_block(valid_brief, draft, [], max_tokens=2000)
    assert block.count("unchanged,") == len(draft.sections)
    assert "Established prose" not in block


def test_agent_run_marks_degraded_when_provider_fails(valid_brief):
    class _BrokenClient(OfflineClient):
        available = True

        def complete(self, system, user, *, max_output_tokens=1000):
            raise ResponseParseError("no JSON in response")

    agent = build_agents(_BrokenClient(), TokenBudget.default())[0]
    from articlegen.memory import AssembledContext
    report = agent.run(AssembledContext(text="context", tokens=10), valid_brief,
                       Draft.from_brief(valid_brief), 1)
    assert report.status == AgentStatus.DEGRADED
    assert any("fell back" in note for note in report.notes)


def test_parse_json_tolerates_fences_and_preamble():
    assert parse_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('prefix {"b": [1, 2]} suffix') == {"b": [1, 2]}
    with pytest.raises(ResponseParseError):
        parse_json("no json here at all")
