"""Normalise the section shapes models actually return.

Writers and synthesizers are asked for ``{"sections": [{"id", "body"}]}``, but
hosted models routinely use titles as ids, ``content`` instead of ``body``, or
a fenced Markdown article. Those replies still contain the prose; dropping them
is what leaves the master draft empty.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..llm import strip_think

_HEADING = re.compile(r"(?m)^(#{1,3})\s+(.+?)\s*$")
_ID_KEYS = ("id", "section_id", "sectionId", "section", "title")
_BODY_KEYS = ("body", "content", "text", "markdown", "prose")


def _fold(value: str) -> str:
    folded = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    for prefix in ("the-", "a-", "an-"):
        if folded.startswith(prefix):
            folded = folded[len(prefix):]
    return folded


def match_section_id(raw: Any, brief) -> Optional[str]:
    """Map a model id or heading onto an outline id. Unknown values stay unknown."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    token = raw.strip()
    if token in brief.section_ids:
        return token
    folded = _fold(token)
    for section in brief.sections:
        if folded == _fold(section.id) or folded == _fold(section.title):
            return section.id
    return None


def _text_from(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [part.strip() for part in value if isinstance(part, str) and part.strip()]
        return "\n\n".join(parts)
    return ""


def section_body(item: Dict[str, Any]) -> str:
    """Read a section body without turning objects into fake prose."""
    if not isinstance(item, dict):
        return ""
    for key in _BODY_KEYS:
        text = _text_from(item.get(key))
        if text.strip():
            return text
        if key == "body" and isinstance(item.get(key), str):
            return item[key]
    return ""


def _looks_like_sections(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    if not all(isinstance(item, dict) for item in value):
        return False
    return any(any(key in item for key in ("id", "section_id", "sectionId", "body", "content", "title"))
               for item in value)


def extract_section_list(payload: Any) -> List[Any]:
    """Find the section array inside the payload shapes models actually emit."""
    if isinstance(payload, list):
        return payload if _looks_like_sections(payload) else []
    if not isinstance(payload, dict):
        return []
    sections = payload.get("sections")
    if isinstance(sections, list):
        return sections
    for key in ("revisions", "draft", "article", "document", "output", "result", "payload"):
        value = payload.get(key)
        if isinstance(value, list) and _looks_like_sections(value):
            return value
        if isinstance(value, dict):
            nested = extract_section_list(value)
            if nested:
                return nested
    return []


def iter_mapped_sections(payload: Any, brief) -> List[Dict[str, str]]:
    """Return outline-aligned ``{id, title, body}`` items. Unknown ids are dropped."""
    mapped: List[Dict[str, str]] = []
    seen = set()
    for item in extract_section_list(payload):
        if not isinstance(item, dict):
            continue
        raw = next((item.get(key) for key in _ID_KEYS if isinstance(item.get(key), str) and item.get(key).strip()), "")
        section_id = match_section_id(raw, brief)
        if not section_id or section_id in seen:
            continue
        seen.add(section_id)
        title = item.get("title") if isinstance(item.get("title"), str) else ""
        mapped.append({"id": section_id, "title": title.strip(), "body": section_body(item)})
    return mapped


def sections_from_markdown(text: str, brief) -> List[Dict[str, str]]:
    """Salvage heading-aligned prose when a model ignores the JSON contract."""
    cleaned = strip_think(text or "")
    matches = list(_HEADING.finditer(cleaned))
    if not matches:
        return []
    found: List[Dict[str, str]] = []
    seen = set()
    for index, match in enumerate(matches):
        title = match.group(2).strip()
        section_id = match_section_id(title, brief)
        if not section_id or section_id in seen:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        body = cleaned[match.end():end].strip()
        if not body:
            continue
        seen.add(section_id)
        found.append({"id": section_id, "title": title, "body": body})
    return found
