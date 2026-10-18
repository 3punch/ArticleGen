"""Provider resolution and the Dahl (OpenAI-compatible) network client.

The network clients cannot reach live APIs from CI, so these tests pin the request
shape against a fake ``urlopen``: URL, Bearer auth, JSON payload and response
parsing. If Dahl's contract drifts, this is where it shows up first.
"""

import json

import pytest

from articlegen.llm import (
    DahlClient,
    OfflineClient,
    OpenAIClient,
    ProviderError,
    resolve_client,
)


class _FakeResponse:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._raw

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def clear_provider_env(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DAHL_API_KEY",
                "DAHL_MODEL", "DAHL_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def captured_request(monkeypatch):
    """Replace urlopen with a recorder serving a canned chat-completions reply."""
    state = {"request": None, "timeout": None}
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "Hello back!"}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }

    def fake_urlopen(request, timeout=None):
        state["request"] = request
        state["timeout"] = timeout
        return _FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return state


# -- DahlClient request shape --------------------------------------------


def test_dahl_client_posts_openai_compatible_chat_completions(captured_request):
    client = DahlClient("test-key")
    response = client.complete("You are helpful.", "Hello!", max_output_tokens=64)

    request = captured_request["request"]
    assert request.full_url == "https://inference.dahl.global/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert request.get_header("Content-type") == "application/json"

    body = json.loads(request.data.decode("utf-8"))
    assert body["model"] == "zai-org/GLM-5.3-Flash"
    assert body["max_tokens"] == 64
    assert body["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]

    assert response.text == "Hello back!"
    assert response.tokens_in == 7
    assert response.tokens_out == 3
    assert response.engine == "dahl:zai-org/GLM-5.3-Flash"
    assert client.available is True
    assert client.name() == "dahl"


def test_dahl_client_accepts_custom_model_and_base_url(captured_request):
    client = DahlClient("k", model="vendor/other-model",
                        base_url="https://dahl.example.internal/v1/")
    client.complete("s", "u")

    request = captured_request["request"]
    assert request.full_url == "https://dahl.example.internal/v1/chat/completions"
    body = json.loads(request.data.decode("utf-8"))
    assert body["model"] == "vendor/other-model"


def test_openai_engine_tag_is_namespaced(captured_request):
    response = OpenAIClient("k").complete("s", "u")
    assert response.engine.startswith("openai:")


def test_dahl_http_error_is_wrapped_as_provider_error(monkeypatch):
    import urllib.error

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 401, "unauthorized", {},
            _FakeResponse({"error": "bad key"}),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ProviderError) as excinfo:
        DahlClient("bad-key").complete("s", "u")
    assert "dahl HTTP 401" in str(excinfo.value)


# -- resolve_client ------------------------------------------------------


def test_resolve_client_selects_dahl_when_key_is_set(clear_provider_env, monkeypatch):
    monkeypatch.setenv("DAHL_API_KEY", "dahl-secret")
    client = resolve_client("dahl")
    assert isinstance(client, DahlClient)
    assert client.model == "zai-org/GLM-5.3-Flash"
    assert client.base_url == "https://inference.dahl.global/v1"


def test_resolve_client_dahl_without_key_degrades_to_offline(clear_provider_env):
    client = resolve_client("dahl")
    assert isinstance(client, OfflineClient)
    assert client.available is False


def test_resolve_client_auto_prefers_anthropic_then_openai_then_dahl(
    clear_provider_env, monkeypatch
):
    monkeypatch.setenv("DAHL_API_KEY", "d")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert resolve_client("auto").engine_name == "anthropic"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert resolve_client("auto").engine_name == "openai"
    monkeypatch.delenv("OPENAI_API_KEY")
    assert resolve_client("auto").engine_name == "dahl"
    monkeypatch.delenv("DAHL_API_KEY")
    assert resolve_client("auto").engine_name == "offline"


def test_resolve_client_dahl_honours_model_and_base_url_env(
    clear_provider_env, monkeypatch
):
    monkeypatch.setenv("DAHL_API_KEY", "k")
    monkeypatch.setenv("DAHL_MODEL", "env/model-id")
    monkeypatch.setenv("DAHL_BASE_URL", "https://proxy.internal/v1")
    client = resolve_client("dahl")
    assert client.model == "env/model-id"
    assert client.base_url == "https://proxy.internal/v1"


def test_resolve_client_explicit_model_beats_dahl_env(clear_provider_env, monkeypatch):
    monkeypatch.setenv("DAHL_API_KEY", "k")
    monkeypatch.setenv("DAHL_MODEL", "env/model-id")
    client = resolve_client("dahl", "cli/model-id")
    assert client.model == "cli/model-id"


def test_resolve_client_never_leaks_to_another_provider(clear_provider_env,
                                                        monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert isinstance(resolve_client("dahl"), OfflineClient)
    assert isinstance(resolve_client("openai"), OfflineClient)
