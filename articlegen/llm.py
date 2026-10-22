"""LLM provider abstraction.

Three engines behind one interface:

* ``AnthropicClient`` / ``OpenAIClient`` / ``DahlClient`` - real calls over HTTPS,
  standard library only (``urllib``), so the package keeps zero runtime dependencies.
* ``OfflineClient`` - a sentinel used when no provider key is configured. The
  orchestrator stops the run; it is never an article-generation engine.

``DahlClient`` talks to the Dahl inference API
(``https://inference.dahl.global/v1``), which is OpenAI-compatible chat
completions with Bearer authentication.

The request shape is verified against stubbed HTTP responses in the test suite.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


DEFAULT_LLM_TIMEOUT_SECONDS = 300
LLM_TIMEOUT_ENV = "ARTICLEGEN_LLM_TIMEOUT_SECONDS"


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
    finish_reason: str = ""


class LLMClient:
    engine_name = "base"
    available = False

    def complete(self, system: str, user: str, *, max_output_tokens: int = 1000) -> LLMResponse:
        raise NotImplementedError

    def name(self) -> str:
        return self.engine_name


class OfflineClient(LLMClient):
    """Explicit offline harness for tests. Not used as implicit ``auto`` fallback."""

    engine_name = "offline"
    available = False

    def complete(self, system: str, user: str, *, max_output_tokens: int = 1000) -> LLMResponse:
        raise ProviderError(
            "no LLM provider configured; article generation requires a Dahl/Anthropic/OpenAI key"
        )


class _HTTPClient(LLMClient):
    def __init__(self, api_key: str, model: str, base_url: str,
                 timeout: int = DEFAULT_LLM_TIMEOUT_SECONDS) -> None:
        if timeout <= 0:
            raise ValueError("LLM request timeout must be a positive number of seconds")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.available = bool(api_key)

    def _timeout_error(self) -> ProviderError:
        return ProviderError(
            f"{self.engine_name} request timed out after {self.timeout} seconds; "
            f"increase {LLM_TIMEOUT_ENV} if this model needs longer to respond"
        )

    def _post(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise ProviderError(f"{self.engine_name} HTTP {exc.code}: {detail}") from exc
        except TimeoutError as exc:
            # urllib can expose socket read timeouts directly, particularly while
            # waiting for the first response byte. Keep them in the provider error
            # path so an agent is marked DEGRADED instead of crashing the UI.
            raise self._timeout_error() from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError) or "timed out" in str(exc.reason).lower():
                raise self._timeout_error() from exc
            raise ProviderError(f"{self.engine_name} network error: {exc.reason}") from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{self.engine_name} returned non-JSON: {raw[:200]}") from exc


class AnthropicClient(_HTTPClient):
    engine_name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5",
                 base_url: str = "https://api.anthropic.com/v1",
                 timeout: int = DEFAULT_LLM_TIMEOUT_SECONDS) -> None:
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


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Drop a thinking trace, keeping it only when it is the entire reply.

    Reasoning models sometimes wrap the answer in ``<think>`` and sometimes
    spend the whole completion there. If anything remains outside the tags,
    that is the answer. If not, the inner text is the only thing left to parse.
    """
    if not text:
        return ""
    inners = [part.strip() for part in _THINK_BLOCK.findall(text) if part and part.strip()]
    stripped = _THINK_BLOCK.sub("", text).strip()
    if stripped:
        return stripped
    return "\n".join(inners).strip()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or "")
    return str(value)


def message_text(message: Dict[str, Any]) -> str:
    """Read the assistant answer from an OpenAI-compatible message.

    GLM-family servers often leave ``content`` null and put the only JSON in
    ``reasoning_content``. Use that fallback only when the visible content has
    no JSON of its own.
    """
    if not isinstance(message, dict):
        return ""
    content = strip_think(_coerce_text(message.get("content")))
    reasoning = strip_think(_coerce_text(message.get("reasoning_content") or message.get("reasoning")))
    if content and ("{" in content or "[" in content):
        return content
    if not content.strip() and ("{" in reasoning or "[" in reasoning):
        return reasoning
    return content or reasoning


def _parameter_rejected(exc: ProviderError) -> bool:
    text = str(exc).lower()
    if "http 400" not in text and "http 422" not in text:
        return False
    if "not currently offered" in text:
        return False
    return True


class OpenAIClient(_HTTPClient):
    engine_name = "openai"
    #: Dahl's default GLM model thinks unless this is set. OpenAI chat models do not.
    reasoning_controls = False

    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1",
                 timeout: int = DEFAULT_LLM_TIMEOUT_SECONDS) -> None:
        super().__init__(api_key, model, base_url, timeout)
        self._drop_reasoning_controls = False

    def _extra_body(self) -> Dict[str, Any]:
        return {}

    def complete(self, system: str, user: str, *, max_output_tokens: int = 1000) -> LLMResponse:
        return self._complete_chat(system, user, max_output_tokens)

    def _complete_chat(self, system: str, user: str, max_output_tokens: int) -> LLMResponse:
        extras: Dict[str, Any] = {}
        if self.reasoning_controls and not self._drop_reasoning_controls:
            extras = self._extra_body()
        payload = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **extras,
        }
        try:
            data = self._post(
                f"{self.base_url}/chat/completions",
                {
                    "content-type": "application/json",
                    "authorization": f"Bearer {self.api_key}",
                },
                payload,
            )
        except ProviderError as exc:
            if extras and _parameter_rejected(exc):
                self._drop_reasoning_controls = True
                return self._complete_chat(system, user, max_output_tokens)
            raise
        return self._response_from(data)

    def _response_from(self, data: Dict[str, Any]) -> LLMResponse:
        choices = data.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        usage = data.get("usage") or {}
        return LLMResponse(
            text=message_text(message),
            tokens_in=int(usage.get("prompt_tokens", 0) or 0),
            tokens_out=int(usage.get("completion_tokens", 0) or 0),
            engine=f"{self.engine_name}:{self.model}",
            finish_reason=str(choice.get("finish_reason") or ""),
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
    reasoning_controls = True
    DEFAULT_BASE_URL = "https://inference.dahl.global/v1"
    DEFAULT_MODEL = "zai-org/GLM-5.3-Flash"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL,
                 timeout: int = DEFAULT_LLM_TIMEOUT_SECONDS) -> None:
        super().__init__(api_key, model, base_url, timeout)

    def _extra_body(self) -> Dict[str, Any]:
        # GLM-5.3-Flash defaults to reasoning_effort=max. A few thousand
        # max_tokens is then spent inside <think> and content comes back empty.
        # "low" is the lightest level the model accepts; anything else falls
        # back to max. Dropped automatically if a gateway rejects the fields.
        return {
            "reasoning_effort": "low",
            "chat_template_kwargs": {
                "reasoning_effort": "low",
                "clear_thinking": True,
            },
        }


def _resolve_timeout(timeout: Optional[int]) -> int:
    """Resolve and validate the HTTP socket timeout from an arg or environment."""
    if timeout is None:
        raw = os.environ.get(LLM_TIMEOUT_ENV, "").strip()
        if not raw:
            return DEFAULT_LLM_TIMEOUT_SECONDS
        try:
            timeout = int(raw)
        except ValueError as exc:
            raise ValueError(f"{LLM_TIMEOUT_ENV} must be a positive integer") from exc
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError(f"{LLM_TIMEOUT_ENV} must be a positive integer")
    return timeout


def resolve_client(provider: str = "auto", model: Optional[str] = None,
                   api_key: Optional[str] = None, base_url: Optional[str] = None,
                   dahl_api_key: Optional[str] = None, dahl_model: Optional[str] = None,
                   dahl_base_url: Optional[str] = None,
                   timeout: Optional[int] = None) -> LLMClient:
    """Resolve a provider from explicit settings first, then environment.

    An empty UI field does not mask a configured environment variable. In ``auto``
    mode, a Dahl key explicitly entered in the UI takes priority; otherwise the
    environment priority is Anthropic, OpenAI, then Dahl. Missing credentials return
    an unavailable ``OfflineClient`` sentinel so the orchestrator can explain the
    problem without generating a fallback article. HTTP requests use a 300-second
    socket timeout by default; pass ``timeout`` or set
    ``ARTICLEGEN_LLM_TIMEOUT_SECONDS`` to override it.
    """
    provider = (provider or "auto").strip().lower()
    explicit_key = (api_key or "").strip() or None
    explicit_model = (model or "").strip() or None
    explicit_base = (base_url or "").strip() or None
    explicit_dahl_key = (dahl_api_key or "").strip() or None
    explicit_dahl_model = (dahl_model or "").strip() or None
    explicit_dahl_base = (dahl_base_url or "").strip() or None

    # A provider-specific key in the UI wins over generic environment auto-detect.
    if provider == "dahl" or (provider == "auto" and explicit_dahl_key):
        key = explicit_dahl_key or (explicit_key if provider == "dahl" else None) \
            or os.environ.get("DAHL_API_KEY", "").strip()
        resolved_model = (
            (explicit_model if provider == "dahl" else None)
            or explicit_dahl_model
            or os.environ.get("DAHL_MODEL", "").strip()
            or DahlClient.DEFAULT_MODEL
        )
        resolved_base = (
            explicit_dahl_base
            or (explicit_base if provider == "dahl" else None)
            or os.environ.get("DAHL_BASE_URL", "").strip()
            or DahlClient.DEFAULT_BASE_URL
        )
        if key:
            return DahlClient(
                key, resolved_model, resolved_base,
                timeout=_resolve_timeout(timeout),
            )
        return OfflineClient()

    if provider in ("auto", "anthropic"):
        key = explicit_key or os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if key:
            return AnthropicClient(
                key, explicit_model or "claude-sonnet-4-5",
                explicit_base or "https://api.anthropic.com/v1",
                timeout=_resolve_timeout(timeout),
            )
        if provider == "anthropic":
            return OfflineClient()

    if provider in ("auto", "openai"):
        key = explicit_key or os.environ.get("OPENAI_API_KEY", "").strip()
        if key:
            return OpenAIClient(
                key, explicit_model or "gpt-4o-mini",
                explicit_base or "https://api.openai.com/v1",
                timeout=_resolve_timeout(timeout),
            )
        if provider == "openai":
            return OfflineClient()

    if provider in ("auto", "dahl"):
        key = explicit_dahl_key or explicit_key or os.environ.get("DAHL_API_KEY", "").strip()
        if key:
            return DahlClient(
                key,
                explicit_model or explicit_dahl_model
                or os.environ.get("DAHL_MODEL", "").strip()
                or DahlClient.DEFAULT_MODEL,
                explicit_dahl_base or explicit_base
                or os.environ.get("DAHL_BASE_URL", "").strip()
                or DahlClient.DEFAULT_BASE_URL,
                timeout=_resolve_timeout(timeout),
            )

    # ``offline`` is supported only as an internal testing sentinel. It has no
    # drafting implementation, and the orchestrator refuses to start a run with it.
    return OfflineClient()


# -- resilient client (retry + model/provider fallback) -------------------


def _label_for(client: LLMClient) -> str:
    """Short, stable identifier for a client (used to dedupe fallback candidates)."""
    model = getattr(client, "model", "") or ""
    return f"{client.engine_name}:{model}" if model else client.engine_name


def _parse_retry_after(text: str) -> Optional[int]:
    """Best-effort extraction of a ``Retry-After``-style delay from an error body."""
    for pattern in (r"retry after\s+(\d+)", r"retry-after[:\s]+(\d+)"):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _split_fallback(spec: str, family: str) -> Tuple[str, str]:
    """Split a ``provider:model`` fallback spec; default the family to the primary."""
    spec = spec.strip()
    if ":" in spec:
        fam, model = spec.split(":", 1)
        fam, model = fam.strip().lower(), model.strip()
    else:
        fam, model = family, spec
    if fam not in ("dahl", "anthropic", "openai"):
        fam = family
    return fam, model


def _build_fallback(family: str, model: str, provider: str, api_key: Optional[str],
                    base_url: Optional[str], dahl_api_key: Optional[str],
                    dahl_base_url: Optional[str],
                    timeout: Optional[int]) -> LLMClient:
    if family == "dahl":
        return resolve_client(provider="dahl", dahl_api_key=dahl_api_key,
                             dahl_model=model, dahl_base_url=dahl_base_url,
                             timeout=timeout)
    if family == "anthropic":
        return resolve_client(provider="anthropic", api_key=api_key, model=model,
                             base_url=base_url, timeout=timeout)
    if family == "openai":
        return resolve_client(provider="openai", api_key=api_key, model=model,
                             base_url=base_url, timeout=timeout)
    return resolve_client(provider=provider, model=model, api_key=api_key,
                         base_url=base_url, dahl_api_key=dahl_api_key,
                         dahl_model=model if provider == "dahl" else None,
                         dahl_base_url=dahl_base_url, timeout=timeout)


class ResilientClient(LLMClient):
    """Retry-with-fallback wrapper around one or more underlying ``LLMClient``s.

    A single model at capacity (HTTP 429) must not degrade an entire run - which is
    exactly what happened to the ``erdos2`` run. On a :class:`ProviderError` the
    client retries the *current* candidate with exponential backoff (honouring a
    ``Retry-After`` hint when one is present in the error text), then moves on to the
    next candidate. If every candidate fails the last error is raised so the caller
    can mark the agent ``DEGRADED`` instead of inventing output.
    """

    def __init__(self, candidates: Sequence[Tuple[LLMClient, str]],
                 *, max_retries: int = 3, base_backoff: float = 2.0) -> None:
        self._candidates = [(c, lbl) for c, lbl in candidates if c is not None]
        self.max_retries = max(0, int(max_retries))
        self.base_backoff = float(base_backoff)
        self.available = any(getattr(c, "available", False) for c, _ in self._candidates)

    @property
    def engine_name(self) -> str:
        return "resilient"

    def name(self) -> str:
        if self._candidates:
            return f"resilient[{self._candidates[0][1]}]"
        return "resilient[none]"

    def complete(self, system: str, user: str, *, max_output_tokens: int = 1000) -> "LLMResponse":
        last_exc: Optional[ProviderError] = None
        for client, _label in self._candidates:
            if not getattr(client, "available", False):
                continue
            attempt = 0
            while True:
                try:
                    return client.complete(system, user, max_output_tokens=max_output_tokens)
                except ProviderError as exc:
                    last_exc = exc
                    if attempt >= self.max_retries:
                        break
                    attempt += 1
                    delay = min(
                        _parse_retry_after(str(exc)) or self.base_backoff * (2 ** (attempt - 1)),
                        30.0,
                    )
                    time.sleep(delay)
        if last_exc is not None:
            raise last_exc
        raise ProviderError(
            "no LLM provider configured; article generation requires a "
            "Dahl/Anthropic/OpenAI key"
        )


def resolve_resilient_client(provider: str = "auto", model: Optional[str] = None,
                             api_key: Optional[str] = None, base_url: Optional[str] = None,
                             dahl_api_key: Optional[str] = None,
                             dahl_model: Optional[str] = None,
                             dahl_base_url: Optional[str] = None,
                             timeout: Optional[int] = None,
                             fallback_models: Optional[Sequence[str]] = None) -> LLMClient:
    """Resolve a :class:`ResilientClient` with the primary plus fallback models.

    The primary is resolved exactly as :func:`resolve_client` would. Fallback models
    come from ``fallback_models`` (e.g. ``["deepseek-ai/DeepSeek-V4-Flash-0731",
    "MiniMaxAI/MiniMax-M2.7"]``) or, when that is ``None``, from the
    ``ARTICLEGEN_FALLBACK_MODELS`` environment variable (comma-separated). Each entry
    may be ``model`` (same family as the primary) or ``provider:model`` (e.g.
    ``anthropic:claude-...``). Candidates that lack credentials are skipped, and the
    primary is always first.
    """
    primary = resolve_client(
        provider=provider, model=model, api_key=api_key, base_url=base_url,
        dahl_api_key=dahl_api_key, dahl_model=dahl_model, dahl_base_url=dahl_base_url,
        timeout=timeout,
    )
    if fallback_models is None:
        raw = os.environ.get("ARTICLEGEN_FALLBACK_MODELS", "").strip()
        fallback_models = [m.strip() for m in raw.split(",") if m.strip()]

    candidates: List[Tuple[LLMClient, str]] = [(primary, _label_for(primary))]
    seen = {_label_for(primary)}
    family = primary.engine_name  # "dahl" | "anthropic" | "openai" | "offline"
    for spec in fallback_models or []:
        fam, mdl = _split_fallback(spec, family)
        try:
            fallback = _build_fallback(
                fam, mdl, provider, api_key, base_url, dahl_api_key, dahl_base_url,
                timeout,
            )
        except Exception:  # noqa: BLE001 - a bad fallback must not break the run
            continue
        if not getattr(fallback, "available", False):
            continue
        label = _label_for(fallback)
        if label in seen:
            continue
        seen.add(label)
        candidates.append((fallback, label))
    return ResilientClient(candidates)


# -- response parsing -----------------------------------------------------

_FENCE_BLOCK = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def salvage_truncated_json(text: str) -> Optional[Dict[str, Any]]:
    """Close a length-truncated JSON object enough to recover completed fields.

    Only used after a normal parse fails. A repaired object that has neither
    sections nor findings is rejected so a half-written key cannot become a draft.
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    fragment = text[start:].rstrip()
    if fragment.endswith("}"):
        return None

    def _closers(source: str) -> tuple:
        stack: List[str] = []
        in_string = False
        escape = False
        for char in source:
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
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char in "}]" and stack and stack[-1] == char:
                stack.pop()
        return in_string, stack

    in_string, stack = _closers(fragment)
    if in_string:
        fragment += '"'
    # A dangling comma or colon cannot be closed into a value.
    fragment = re.sub(r"[,:]\s*$", "", fragment)
    _, stack = _closers(fragment)
    fragment += "".join(reversed(stack))
    try:
        value = json.loads(fragment)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    if not any(key in value for key in ("sections", "findings", "draft", "article")):
        return None
    return value


def parse_json(text: str) -> Union[Dict[str, Any], List[Any]]:
    """Tolerantly extract the first JSON value from a model response.

    Models routinely wrap JSON in fences or preface it with commentary. Guessing at a
    broken payload is worse than failing, so a genuine parse failure raises and the
    agent marks itself DEGRADED.
    """
    text = strip_think(text or "")
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
