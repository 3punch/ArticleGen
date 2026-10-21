from articlegen.memory import Block, RollingMemory, assemble, working_draft_block
from articlegen.models import AgentReport, Draft
from articlegen.tokens import TokenBudget, count_tokens, truncate_to_tokens


def test_count_tokens_is_deterministic_and_monotonic():
    text = "hello world " * 50
    assert count_tokens(text) == count_tokens(text)
    assert count_tokens(text + "more text") > count_tokens(text)
    assert count_tokens("") == 0


def test_code_is_weighted_denser_than_prose():
    prose = "a" * 400
    code = "```python\n" + "x" * 388 + "\n```"
    assert count_tokens(code) > count_tokens(prose)


def test_truncate_to_tokens_respects_the_budget():
    text = "word " * 2000
    clipped = truncate_to_tokens(text, 100)
    assert count_tokens(clipped) <= 100
    assert "[...truncated by token budget...]" in clipped


def test_truncate_to_tokens_is_a_noop_under_budget():
    text = "short text"
    assert truncate_to_tokens(text, 500) == text


def test_assemble_includes_all_blocks_with_no_eviction():
    """With no token ceiling, every non-empty block is included in full."""
    blocks = [
        Block(1, "rules", "rule one. " * 10),
        Block(3, "draft", "draft body. " * 60),
        Block(5, "history", "old history. " * 60),
    ]
    result = assemble(blocks)
    assert "rules" in result.included
    assert "draft" in result.included
    assert "history" in result.included
    assert result.evicted == []
    assert result.truncated == []
    assert result.breached_cap is False


def test_assemble_keeps_required_blocks_in_full():
    """Required blocks are never truncated or dropped for size."""
    blocks = [Block(1, "rules", "important rule. " * 200, required=True)]
    result = assemble(blocks)
    assert "rules" in result.included
    assert result.truncated == []
    assert result.breached_cap is False


def test_rolling_memory_is_bounded_by_entry_count_not_tokens():
    """Rolling memory is capped by entry count, not by a token budget."""
    memory = RollingMemory(max_entries=5)
    for iteration in range(30):
        memory.absorb(f"Iteration {iteration} entry")
    assert len(memory.entries) == 5
    assert memory.entries[-1] == "Iteration 29 entry"


def test_working_draft_overlays_writer_delta_without_mutating_master(valid_brief):
    import json
    from articlegen.memory import working_draft_block

    draft = Draft.from_brief(valid_brief)
    for section in draft.sections:
        section.body = f"Original text for {section.id}."
    reports = [AgentReport(agent_id="writer", role="Writer", iteration=2, payload={
        "sections": [{"id": "mechanism", "body": "New model-written mechanism."}]
    })]
    working = json.loads(working_draft_block(draft, reports))
    assert [s["id"] for s in working["sections"]] == valid_brief.section_ids
    assert working["sections"][0]["body"] == draft.sections[0].body
    assert working["sections"][1]["body"] == "New model-written mechanism."
    assert working["sections"][2]["body"] == draft.sections[2].body
    assert draft.sections[1].body == "Original text for mechanism."
    assert draft.version == 0
