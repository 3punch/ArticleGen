"""Agent 2 - Tester / Validator.

The deterministic engine here runs *real* checks whose results are true or false, not
simulated verdicts: Python blocks are actually compiled, anchors are actually resolved
against actual headings, and numeric claims are actually cross-checked for internal
contradiction. What it cannot do without a provider is judge whether a factual claim is
true in the world - so it flags unverifiable claims for removal instead of asserting
anything about them (rule R9).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import urlparse

from ..llm import coerce_findings
from ..models import AgentReport, AgentStatus, Brief, Draft, Finding, Severity
from .base import BaseAgent

PLACEHOLDER_RE = re.compile(
    r"\b(TODO|TBD|FIXME|XXX|Lorem\s+ipsum|fill\s+me\s+in)\b", re.IGNORECASE
)
FENCE_LINE_RE = re.compile(r"^\s*```(.*)$", re.MULTILINE)
CODE_BLOCK_RE = re.compile(r"```([^\n]*)\n(.*?)```", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
ANCHOR_RE = re.compile(r"^#[a-z0-9][a-z0-9-]*$", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)\]>]+")
CITATION_RE = re.compile(r"\[\d+\]|\([A-Z][A-Za-z]+,\s*\d{4}\)|https?://")
# NB: no trailing \b - '%' and 'ms' are followed by non-word characters, so a word
# boundary after them never matches. A negative lookahead is the correct guard.
NUMERIC_CLAIM_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|ms|milliseconds|seconds|minutes|GB|MB|users|requests|"
    r"per\s+second)(?![A-Za-z])",
    re.IGNORECASE,
)
ABSOLUTE_RE = re.compile(r"\b(always|never|guaranteed|proven|100%|undoubtedly|certainly)\b", re.IGNORECASE)
METRIC_RE = re.compile(
    r"(?P<metric>[A-Za-z][A-Za-z\- ]{2,28}?)\s+(?:is|are|equals|reaches|stands at)\s+"
    r"(?P<value>-?\d[\d,\.]*\s*%?)",
)
MAX_PER_CATEGORY = 6

#: Determiners stripped before comparing metric names, so "The cache latency" and
#: "cache latency" are recognised as the same quantity.
_LEADING_STOPWORDS = ("the", "a", "an", "this", "that", "these", "those", "our", "its", "measured")


def _normalise_metric(raw: str) -> str:
    words = re.sub(r"[^a-z0-9 ]+", " ", raw.lower()).split()
    while words and words[0] in _LEADING_STOPWORDS:
        words.pop(0)
    return " ".join(words)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class Tester(BaseAgent):
    agent_id = "tester"
    role = "Tester/Validator"
    draft_required = True

    def deterministic(self, brief: Brief, draft: Draft, iteration: int,
                      peer_reports: List[AgentReport]) -> Dict[str, Any]:
        findings: List[Finding] = []

        def add(finding: Finding, category_count: Dict[str, int]) -> None:
            if category_count.get(finding.category, 0) >= MAX_PER_CATEGORY:
                return
            category_count[finding.category] = category_count.get(finding.category, 0) + 1
            findings.append(finding)

        counts: Dict[str, int] = {}

        # 1. Structural integrity: fences and empty sections.
        for section in draft.sections:
            if not section.body.strip():
                add(Finding(self.agent_id, Severity.BLOCKER, "logic", section.id,
                            f"section '{section.title}' is empty; the deliverable is incomplete",
                            "The Writer must produce content for this section.",
                            iteration=iteration), counts)
                continue

            fences = FENCE_LINE_RE.findall(section.body)
            if len(fences) % 2:
                add(Finding(self.agent_id, Severity.BLOCKER, "code", section.id,
                            f"unbalanced code fences ({len(fences)} markers) make the block unreadable",
                            "Close the open fence with a matching ``` line.",
                            evidence=section.body[:80], iteration=iteration), counts)

            if PLACEHOLDER_RE.search(section.body):
                add(Finding(self.agent_id, Severity.BLOCKER, "completeness", section.id,
                            "section still contains an unfinished placeholder",
                            "Replace the placeholder with finished content.",
                            evidence=PLACEHOLDER_RE.search(section.body).group(0),
                            iteration=iteration), counts)

        # 2. Code accuracy: actually compile Python blocks.
        for section in draft.sections:
            for index, (lang, body) in enumerate(CODE_BLOCK_RE.findall(section.body), start=1):
                lang = lang.strip()
                if not lang:
                    add(Finding(self.agent_id, Severity.MAJOR, "code", section.id,
                                f"code block {index} has no language tag",
                                "Tag the fence (e.g. ```python) so it can be validated and highlighted.",
                                iteration=iteration), counts)
                    continue
                if lang.lower() in ("python", "py", "python3") and body.strip():
                    try:
                        compile(body, f"<{section.id}:block{index}>", "exec")
                    except SyntaxError as exc:
                        add(Finding(self.agent_id, Severity.BLOCKER, "code", section.id,
                                    f"code block {index} does not compile: {exc.msg} (line {exc.lineno})",
                                    "Fix the syntax error; broken code cannot ship.",
                                    evidence=(exc.text or "").strip()[:80],
                                    auto_fixable=False, iteration=iteration), counts)

        # 3. Anchors: resolve against real headings in the assembled document.
        full_text = draft.to_markdown()
        slugs = {_slug(match.group(2).strip()) for match in HEADING_RE.finditer(full_text)}
        for section in draft.sections:
            for text, target in LINK_RE.findall(section.body):
                target = target.strip()
                if not target:
                    add(Finding(self.agent_id, Severity.MAJOR, "links", section.id,
                                f"link '{text[:40]}' has an empty destination",
                                "Supply a target for the link.", iteration=iteration), counts)
                elif ANCHOR_RE.match(target) and target[1:] not in slugs:
                    add(Finding(self.agent_id, Severity.MINOR, "links", section.id,
                                f"internal anchor '{target}' resolves to no heading",
                                "Point the anchor at an existing heading or remove it.",
                                auto_fixable=True,
                                fix={"type": "drop_anchor", "section": section.id, "target": target},
                                iteration=iteration), counts)

        # 4. Unverifiable numeric claims (R9): flagged, never corrected by invention.
        for section in draft.sections:
            for line in section.body.splitlines():
                if CITATION_RE.search(line):
                    continue
                for match in NUMERIC_CLAIM_RE.findall(line):
                    add(Finding(self.agent_id, Severity.MAJOR, "fabrication", section.id,
                                f"numeric claim '{match.strip()}' has no supporting citation",
                                "Add a source, or remove the figure. Do not substitute another number.",
                                evidence=line.strip()[:100], iteration=iteration), counts)
                    break

        # 5. Internal contradiction: same metric asserted with different values.
        metrics: Dict[str, Dict[str, List[str]]] = {}
        for section in draft.sections:
            for match in METRIC_RE.finditer(section.body):
                metric = _normalise_metric(match.group("metric"))
                value = re.sub(r"\s+", "", match.group("value"))
                if not metric:
                    continue
                metrics.setdefault(metric, {}).setdefault(value, []).append(section.id)
        for metric, values in metrics.items():
            if len(values) > 1:
                locations = ", ".join(sorted({sid for ids in values.values() for sid in ids}))
                add(Finding(self.agent_id, Severity.MAJOR, "logic", locations,
                            f"'{metric}' is asserted with conflicting values: {', '.join(sorted(values))}",
                            "Reconcile to a single value with a source, or state the conditions that differ.",
                            auto_fixable=False, iteration=iteration), counts)

        # 6. Absolutist language without support.
        for section in draft.sections:
            for match in ABSOLUTE_RE.findall(section.body):
                add(Finding(self.agent_id, Severity.NOTE, "fabrication", section.id,
                            f"absolute claim '{match}' is unfalsifiable as written",
                            "Qualify the claim or cite evidence for it.",
                            evidence=match, iteration=iteration), counts)

        # 7. Malformed citation URLs.
        for url in URL_RE.findall(full_text):
            parsed = urlparse(url.rstrip(".,;"))
            if not parsed.netloc or "." not in parsed.netloc:
                add(Finding(self.agent_id, Severity.MAJOR, "citations", "document",
                            f"malformed citation URL '{url[:60]}'",
                            "Use a fully qualified http(s) URL.", iteration=iteration), counts)

        # 8. Objective and constraint coverage.
        haystack = full_text.lower()
        for constraint in brief.constraints:
            keywords = [w for w in re.findall(r"[a-z]{5,}", constraint.lower())]
            if keywords and not any(word in haystack for word in keywords):
                add(Finding(self.agent_id, Severity.MAJOR, "completeness", "document",
                            f"brief constraint appears unaddressed: {constraint[:70]}",
                            "Cover the constraint explicitly, or record why it is out of scope.",
                            iteration=iteration), counts)

        if brief.requires_citations and not CITATION_RE.search(full_text):
            add(Finding(self.agent_id, Severity.MAJOR, "citations", "document",
                        "brief requires citations but none are present",
                        "Add numbered references or source URLs for factual claims.",
                        iteration=iteration), counts)

        blockers = len([f for f in findings if f.severity == Severity.BLOCKER])
        return {
            "findings": [f.to_dict() for f in findings],
            "summary": (
                f"{len(findings)} issue(s) found ({blockers} blocking); "
                "checks: fences, code compile, anchors, numeric support, contradictions, coverage"
            ),
        }

    def compose(self, *, payload: Dict[str, Any], brief: Brief, draft: Draft, iteration: int,
                status: AgentStatus, notes: List[str], engine: str, tokens_in: int,
                tokens_out: int, peer_reports: List[AgentReport]) -> AgentReport:
        raw = payload.get("findings")
        if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "severity" in raw[0]:
            findings = coerce_findings(raw, self.agent_id, iteration)
        else:
            findings = [Finding.from_dict(f) for f in (raw or []) if isinstance(f, dict)]
        return self.base_report(
            iteration=iteration, status=status, notes=notes, engine=engine,
            tokens_in=tokens_in, tokens_out=tokens_out, findings=findings,
            summary=str(payload.get("summary") or f"{len(findings)} verification issue(s)"),
            payload={"summary": payload.get("summary", "")},
        )
