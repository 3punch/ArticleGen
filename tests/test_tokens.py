import pytest

from articlegen.memory import Block, RollingMemory, assemble, summarise_text
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


def test_truncate_respects_the_ceiling():
    text = "word " * 2000
    clipped = truncate_to_tokens(text, 100)
    assert count_tokens(clipped) <= 100
    assert "[...truncated by token budget...]" in clipped


def test_truncate_is_a_noop_under_budget():
    text = "short text"
    assert truncate_to_tokens(text, 500) == text


def test_assemble_evicts_lowest_priority_blocks_first():
    budget = TokenBudget(context_hard_cap=400)
    blocks = [
        Block(1, "rules", "rule one. " * 10, required=True),
        Block(3, "draft", "draft body. " * 60),
        Block(5, "history", "old history. " * 60),
    ]
    result = assemble(blocks, budget)
    assert count_tokens(result.text) <= budget.context_hard_cap
    assert "rules" in result.included
    assert "history" in result.evicted


def test_assemble_truncates_required_blocks_rather_than_dropping_them():
    budget = TokenBudget(context_hard_cap=120)
    blocks = [Block(1, "rules", "important rule. " * 200, required=True)]
    result = assemble(blocks, budget)
    assert "rules" in result.included
    assert "rules" in result.truncated
    assert result.breached_cap is True
    assert count_tokens(result.text) <= budget.context_hard_cap + 40


def test_rolling_memory_stays_within_its_budget_across_many_iterations():
    memory = RollingMemory(max_tokens=200, max_entries=5)
    for iteration in range(30):
        memory.absorb(
            f"Iteration {iteration} raised a blocker in citations and a major readability "
            f"problem, applied four fixes and deferred two findings for later review."
        )
    rendered = memory.render()
    assert len(memory.entries) <= 5
    assert count_tokens(rendered) <= 260  # budget plus folding markers


def test_summarise_text_prefers_high_signal_sentences():
    text = (
        "Filler sentence about nothing in particular at all. "
        "A blocker was found in the citations section. "
        "Another sentence that adds no signal whatsoever. "
        "The run must never re-pass full history."
    )
    summary = summarise_text(text, 40)
    assert "blocker" in summary.lower() or "never" in summary.lower()
    assert count_tokens(summary) <= 40
