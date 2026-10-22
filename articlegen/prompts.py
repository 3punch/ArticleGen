"""Prompt template loading and rendering.

Deliberately dependency-free: ``{{ placeholder }}`` substitution plus one block
extractor. No template engine, so there is nothing to install and nothing that can
silently reinterpret a prompt file.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}")

DEFAULT_PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")
DEFAULT_RULES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "claude.md")

AGENT_TEMPLATES: Dict[str, str] = {
    "writer": "01_writer.md",
    "tester": "02_tester.md",
    "reviewer": "03_reviewer.md",
    "architect": "04_architect.md",
    "synthesizer": "05_synthesizer.md",
    "planner": "06_planner.md",
    "scientific_reviewer": "07_scientific_reviewer.md",
    "reviser": "08_reviser.md",
    "brief_writer": "09_brief_writer.md",
}


class PromptError(RuntimeError):
    """Raised when a required template or rules file cannot be loaded."""


def _read(path: str) -> str:
    if not os.path.exists(path):
        raise PromptError(f"prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def load_template(agent_id: str, prompt_dir: Optional[str] = None) -> str:
    filename = AGENT_TEMPLATES.get(agent_id)
    if filename is None:
        raise PromptError(f"no template registered for agent '{agent_id}'")
    return _read(os.path.join(prompt_dir or DEFAULT_PROMPT_DIR, filename))


def render(template: str, variables: Dict[str, Any]) -> str:
    """Substitute ``{{ key }}`` placeholders; unknown keys become an explicit marker.

    Leaving an unresolved placeholder as ``{{ key }}`` is a trap - an LLM will happily
    invent a value for it. Rendering it as ``<missing:key>`` makes the gap visible.
    """

    def replace(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key not in variables:
            return f"<missing:{key}>"
        return str(variables[key])

    return _PLACEHOLDER.sub(replace, template)


def rules_digest(rules_path: Optional[str] = None) -> str:
    """Extract the fenced 'Agent Digest' block from claude.md.

    Only the digest enters agent context. Sending the whole constitution on every
    call would multiply its cost by (agents x iterations) for no benefit.
    """
    path = rules_path or DEFAULT_RULES_PATH
    text = _read(path)
    match = re.search(r"##\s*Agent Digest\s*\n+>.*?```\n(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fall back to the first fenced block after the digest heading, if formatting drifts.
    match = re.search(r"##\s*Agent Digest[\s\S]*?```([a-z]*)\n(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(2).strip()
    raise PromptError(f"could not locate an 'Agent Digest' fenced block in {path}")


def missing_placeholders(rendered: str) -> list:
    return re.findall(r"<missing:([a-zA-Z0-9_.]+)>", rendered)
