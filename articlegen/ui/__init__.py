"""Web dashboard for the ArticleGen pipeline.

``state`` and ``builder`` are Streamlit-free and unit-tested directly; ``render``
is the thin Streamlit layer on top. ``dashboard`` is the Streamlit script itself;
it is package data so an installed ``articlegen ui`` has something to launch.
"""

import os

from .builder import (
    SECTION_TEMPLATE,
    SectionParseError,
    build_brief_dict,
    changelog_rows,
    brief_to_text,
    format_issues,
    parse_sections,
    sections_to_text,
    validate,
)
from .state import AGENT_LABELS, AGENT_ORDER, AgentView, IterationView, TrackerState

#: Absolute path to the Streamlit script. Resolved from this file's location rather
#: than the repo root so it works identically from a checkout and from site-packages.
DASHBOARD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py")

__all__ = [
    "DASHBOARD_PATH",
    "AGENT_LABELS",
    "AGENT_ORDER",
    "AgentView",
    "IterationView",
    "SECTION_TEMPLATE",
    "SectionParseError",
    "TrackerState",
    "brief_to_text",
    "changelog_rows",
    "build_brief_dict",
    "format_issues",
    "parse_sections",
    "sections_to_text",
    "validate",
]
