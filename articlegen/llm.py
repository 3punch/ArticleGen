"""LLM provider abstraction.

Four engines behind one interface:

* ``AnthropicClient`` / ``OpenAIClient`` / ``DahlClient`` - real calls over HTTPS,
  standard library only (``urllib``), so the package keeps zero runtime dependencies.
* ``OfflineClient`` - the default when no API key is present. It reports
  ``available = False`` and the agents fall back to their deterministic engines.

``DahlClient`` talks to the Dahl inference API
(``https://inference.dahl.global/v1``), which is OpenAI-compatible chat
completions with Bearer authentication.

**Verification status:** the offline path is exercised by the test suite and by a
full end-to-end run. The network clients are *not* callable from this sandbox
(egress is limited to GitHub/PyPI/npm), so their request shape is implemented but
unverified against the live APIs.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


class ProviderError(RuntimeError):
    pass


class ResponseParseError(ProviderError):
    pass


@dataclass
class LLMResponse:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    engine: str = "unknown"


class LLMClient:
    engine_name = "base"
    available = False

    def complete(self, system: str, user: str, *, max_output_tokens: int = 1000) -> LLMResponse:
        raise NotImplementedError

    def name(self) -> str:
        return self.engine_name


class OfflineClient(LLMClient):
    """Signals that agents must use their deterministic engines."""

    engine_name = "offline"
    available = False

    def complete(self, system: str, user: str, *, max_output_tokens: int = 1000) -> LLMResponse:
        raise ProviderError(
            "no LLM provider configured; agents will use their deterministic engines"
        )


class _HTTPClient(LLMClient):
    def __init__(self, api_key: str, model: str, base_url: str, timeout: int = 120) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.available = bool(api_key)

    def _post(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise ProviderError(f"{self.engine_name} HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"{self.engine_name} network error: {exc.reason}") from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.engine_name} returned non-JSON: {raw[:200]}") from exc


class AnthropicClient(_HTTPClient):
    engine_name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5",
                 base_url: str = "https://api.anthropic.com/v1", timeout: int = 120) -> None:
        super().__init__(api_key, model, base_url, timeout)

    def complete(self, system: str, user: str, *, max_output_tokens: int = 1000) -> LLMResponse:
        data = self._post(
            f"{self.base_url}/messages",
            {
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            {
                "model": self.model,
                "max_tokens": max_output_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        chunks = data.get("content") or []
        text = "".join(chunk.get("text", "") for chunk in chunks if chunk.get("type") == "text")
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            tokens_in=int(usage.get("input_tokens", 0) or 0),
            tokens_out=int(usage.get("output_tokens", 0) or 0),
            engine=f"anthropic:{self.model}",
        )


class OpenAIClient(_HTTPClient):
    engine_name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1", timeout: int = 120) -> None:
        super().__init__(api_key, model, base_url, timeout)

    def complete(self, system: str, user: str, *, max_output_tokens: int = 1000) -> LLMResponse:
        data = self._post(
            f"{self.base_url}/chat/completions",
            {
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            },
            {
                "model": self.model,
                "max_tokens": max_output_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        choices = data.get("choices") or []
        text = choices[0]["message"]["content"] if choices else ""
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text or "",
            tokens_in=int(usage.get("prompt_tokens", 0) or 0),
            tokens_out=int(usage.get("completion_tokens", 0) or 0),
            engine=f"{self.engine_name}:{self.model}",
        )


class DahlClient(OpenAIClient):
    """Dahl inference API - OpenAI-compatible chat completions.

    Endpoints like ``https://inference.dahl.global/v1/chat/completions`` speak the
    standard OpenAI wire format with ``Authorization: Bearer`` auth, so this is the
    OpenAI client with a different origin and default model. The model catalog
    rotates; ``DAHL_MODEL`` / ``DAHL_BASE_URL`` override the defaults at resolve
    time (see :func:`resolve_client`).
    """

    engine_name = "dahl"
    DEFAULT_BASE_URL = "https://inference.dahl.global/v1"
    DEFAULT_MODEL = "zai-org/GLM-5.3-Flash"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL, timeout: int = 120) -> None:
        super().__init__(api_key, model, base_url, timeout)


def resolve_client(provider: str = "auto", model: Optional[str] = None) -> LLMClient:
    """Pick an engine from explicit choice or environment.

    ``auto`` prefers Anthropic, then OpenAI, then Dahl, then offline. An explicitly
    requested provider whose key is missing returns an unavailable client rather
    than raising, so a CI run degrades to deterministic output instead of failing.

    Dahl reads ``DAHL_API_KEY`` and optionally ``DAHL_MODEL`` and ``DAHL_BASE_URL``
    (defaults: ``zai-org/GLM-5.3-Flash`` and ``https://inference.dahl.global/v1``).
    """
    provider = (provider or "auto").lower()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    dahl_key = os.environ.get("DAHL_API_KEY", "").strip()

    if provider in ("auto", "anthropic") and anthropic_key:
        return AnthropicClient(anthropic_key, model or "claude-sonnet-4-5")
    if provider in ("auto", "openai") and openai_key:
        return OpenAIClient(openai_key, model or "gpt-4o-mini")
    if provider in ("auto", "dahl") and dahl_key:
        dahl_model = (model or os.environ.get("DAHL_MODEL", "").strip()
                      or DahlClient.DEFAULT_MODEL)
        dahl_base = (os.environ.get("DAHL_BASE_URL", "").strip()
                     or DahlClient.DEFAULT_BASE_URL)
        return DahlClient(dahl_key, dahl_model, dahl_base)
    return OfflineClient()


# -- response parsing -----------------------------------------------------

_FENCE_BLOCK = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def parse_json(text: str) -> Union[Dict[str, Any], List[Any]]:
    """Tolerantly extract the first JSON value from a model response.

    Models routinely wrap JSON in fences or preface it with commentary. Guessing at a
    broken payload is worse than failing, so a genuine parse failure raises and the
    agent marks itself DEGRADED.
    """
    if not text or not text.strip():
        raise ResponseParseError("empty response")

    candidates: List[str] = []
    stripped = text.strip()
    candidates.append(stripped)
    for match in _FENCE_BLOCK.finditer(text):
        candidates.append(match.group(1).strip())

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(value, (dict, list)):
                return value

    # Last resort: the first balanced {...} or [...] span.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = stripped.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == open_ch:
                depth += 1
            elif char == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(stripped[start:index + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, (dict, list)):
                        return value
                    break

    raise ResponseParseError(f"no JSON value found in response: {stripped[:200]}")


def coerce_findings(raw: Any, agent: str, iteration: int) -> List[Any]:
    """Normalise a findings list from a model payload into Finding objects."""
    from .models import Finding, Severity

    if not isinstance(raw, list):
        return []
    findings: List[Any] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or item.get("issue") or "").strip()
        action = str(item.get("action") or item.get("recommendation") or "").strip()
        if not claim:
            continue
        findings.append(
            Finding(
                agent=agent,
                severity=Severity.parse(item.get("severity", "minor"), default=Severity.MINOR),
                category=str(item.get("category", "general")).strip() or "general",
                section=str(item.get("section", "")).strip(),
                claim=claim,
                action=action or "Review and resolve this finding.",
                evidence=str(item.get("evidence", "")).strip(),
                auto_fixable=bool(item.get("auto_fixable", False)),
                fix=item.get("fix"),
                iteration=iteration,
            )
        )
    return findings
