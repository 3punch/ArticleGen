"""Tests for the resilient (retry + fallback) LLM client."""

import pytest

from articlegen.llm import LLMClient, LLMResponse, ProviderError, ResilientClient


class _Fake(LLMClient):
    def __init__(self, label, available=True, side_effect=None, response=None):
        self.engine_name = label
        self.model = label
        self._available = available
        self.side_effect = side_effect
        self.response = response
        self.calls = 0

    @property
    def available(self):
        return self._available

    def complete(self, system, user, *, max_output_tokens=1000):
        self.calls += 1
        if self.side_effect is not None:
            return self.side_effect()
        return self.response


def _raise_429():
    raise ProviderError("HTTP 429: model at concurrency capacity")


def test_retries_then_succeeds():
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise ProviderError("429 capacity")
        return LLMResponse(text="done", engine="a")

    primary = _Fake("a", side_effect=flaky)
    rc = ResilientClient([(primary, "a")], max_retries=3, base_backoff=0.0)
    resp = rc.complete("s", "u")
    assert resp.text == "done"
    assert primary.calls == 3


def test_falls_back_on_persistent_failure():
    primary = _Fake("a", side_effect=_raise_429)
    secondary = _Fake("b", response=LLMResponse(text="from-b", engine="b"))
    rc = ResilientClient([(primary, "a"), (secondary, "b")], max_retries=1, base_backoff=0.0)
    resp = rc.complete("s", "u")
    assert resp.text == "from-b"
    assert primary.calls == 2  # initial attempt + one retry
    assert secondary.calls == 1


def test_all_fail_raises_last_error():
    a = _Fake("a", side_effect=_raise_429)
    b = _Fake("b", side_effect=_raise_429)
    rc = ResilientClient([(a, "a"), (b, "b")], max_retries=0, base_backoff=0.0)
    with pytest.raises(ProviderError):
        rc.complete("s", "u")
    assert a.calls == 1 and b.calls == 1


def test_skips_unavailable_candidates():
    primary = _Fake("a", available=False)
    secondary = _Fake("b", response=LLMResponse(text="ok", engine="b"))
    rc = ResilientClient([(primary, "a"), (secondary, "b")])
    assert rc.available is True
    resp = rc.complete("s", "u")
    assert resp.text == "ok"
    assert primary.calls == 0


def test_no_candidates_raises():
    rc = ResilientClient([])
    assert rc.available is False
    with pytest.raises(ProviderError):
        rc.complete("s", "u")
