"""Review a manuscript the pipeline did not write.

Two layers, kept deliberately separate:

* **Deterministic** -- mechanical checks with reproducible evidence: citation
  markers that resolve to no reference entry, entries never cited, duplicate or
  malformed DOIs, empty sections, placeholders, repeated sentences, and abstract
  numbers absent from the body. Readiness is computed **only** from this layer.
* **Advisory** -- one bounded model call. Every issue must quote the manuscript
  verbatim; an issue without a locatable quote is discarded. Advisory findings
  never change readiness and never produce an accept/reject verdict.

The separation exists because a false accusation ("this reference is fabricated")
does more damage than a missed issue. The tool reports absence of verification,
never fabrication, and prefers "could not check" over a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .llm import LLMClient, ProviderError, ResponseParseError, parse_json
from .manuscript import (
    Manuscript,
    normalize_for_match,
    repeated_sentences,
    word_count,
)
from .prompts import PromptError, load_template

SEVERITY_ORDER = ("blocker", "major", "minor", "note")
SEVERITY_LABELS = {
    "blocker": "Blocker",
    "major": "Major",
    "minor": "Minor",
    "note": "Note",
}

MODEL_DIMENSIONS = (
    "contribution_focus",
    "evidence_and_citations",
    "methods_and_reproducibility",
    "analysis_and_claims",
    "structure_and_clarity",
    "limitations_and_scope",
)
_MODEL_SEVERITIES = {"critical", "major", "moderate", "minor"}

MAX_MODEL_CHARS = 60_000

_PLACEHOLDER_RE = re.compile(r"\b(?:TODO|TBD|FIXME|XXX)\b|lorem ipsum", re.IGNORECASE)
_DOI_IN_TEXT_RE = re.compile(r"\bdoi\b", re.IGNORECASE)

DISCLAIMER = (
    "This is a mechanical and AI-assisted screening, not peer review. It does not "
    "verify factual truth, novelty, statistical validity, or whether a cited work "
    "supports the claim it is attached to."
)

LIMITATIONS = (
    "It does not verify factual truth, novelty, statistical validity, or whether a "
    "cited work supports the claim it is attached to.",
    "A DOI absent from Crossref is reported as unverified, not as fabricated; index "
    "coverage is incomplete and a network error is recorded as an error.",
    "Only the supplied text is inspected: no PDF layout, figures, tables, equations "
    "or supplementary files.",
    "Model findings are advisory. They are validated only for verbatim quotation, "
    "they never change readiness, and they are not a publication decision.",
    "Readiness reflects mechanical checks only; it is not a submission recommendation.",
)

READINESS_RULE = (
    "major_revision if any blocker or three or more major findings; minor_revision if "
    "any major or four or more minor findings; otherwise no_blocking_findings. "
    "Advisory model findings never change this value."
)


@dataclass
class Finding:
    check_id: str
    severity: str
    title: str
    detail: str
    location: str = ""
    evidence: str = ""
    action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "location": self.location,
            "evidence": self.evidence,
            "action": self.action,
        }


@dataclass
class AdvisoryIssue:
    dimension: str
    severity: str
    section: str
    quote: str
    claim: str
    action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "severity": self.severity,
            "section": self.section,
            "quote": self.quote,
            "claim": self.claim,
            "action": self.action,
        }


@dataclass
class ModelReviewOutcome:
    ran: bool = False
    engine: str = ""
    summary: str = ""
    strengths: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    dropped_quotes: int = 0
    dropped_issues: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ran": self.ran,
            "engine": self.engine,
            "summary": self.summary,
            "strengths": list(self.strengths),
            "limitations": list(self.limitations),
            "dropped_quotes": self.dropped_quotes,
            "dropped_issues": self.dropped_issues,
            "notes": list(self.notes),
        }


@dataclass
class ManuscriptReview:
    source: str
    title: str
    readiness: str
    readiness_reason: str
    stats: Dict[str, Any]
    findings: List[Finding] = field(default_factory=list)
    advisory: List[AdvisoryIssue] = field(default_factory=list)
    model: ModelReviewOutcome = field(default_factory=ModelReviewOutcome)
    checks_run: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        counts = {severity: 0 for severity in SEVERITY_ORDER}
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        counts["advisory"] = len(self.advisory)
        return counts

    @property
    def blockers(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "blocker"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "readiness": self.readiness,
            "readiness_reason": self.readiness_reason,
            "readiness_rule": READINESS_RULE,
            "counts": self.counts,
            "stats": self.stats,
            "checks_run": list(self.checks_run),
            "findings": [finding.to_dict() for finding in self.findings],
            "advisory_issues": [issue.to_dict() for issue in self.advisory],
            "model": self.model.to_dict(),
            "notes": list(self.notes),
            "limits": list(LIMITATIONS),
            "disclaimer": DISCLAIMER,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Manuscript review: {self.title or 'Untitled manuscript'}",
            "",
            f"**Source:** `{self.source or '(text)'}` · "
            f"**Words:** {self.stats.get('words', 0)} · "
            f"**Sections:** {self.stats.get('sections', 0)} · "
            f"**Reference entries:** {self.stats.get('references', 0)} · "
            f"**Citation markers:** {self.stats.get('citation_markers', 0)}",
            "",
            "## Readiness (mechanical checks only)",
            "",
            f"**{self.readiness}** — {self.readiness_reason}",
            "",
            f"> Rule: {READINESS_RULE}",
            "",
            "## Deterministic findings",
            "",
            "| Severity | Count |",
            "|---|---:|",
        ]
        lines += [f"| {SEVERITY_LABELS[s]} | {self.counts[s]} |" for s in SEVERITY_ORDER]
        lines.append("")

        if not self.findings:
            lines.extend(["- No issue was found by the checks that ran.", ""])
        for severity in SEVERITY_ORDER:
            group = [f for f in self.findings if f.severity == severity]
            if not group:
                continue
            lines.extend([f"### {SEVERITY_LABELS[severity]}", ""])
            for finding in group:
                location = f" @{finding.location}" if finding.location else ""
                lines.append(f"- **`{finding.check_id}`**{location} — {finding.title}")
                if finding.detail:
                    lines.append(f"  - {finding.detail}")
                if finding.evidence:
                    evidence = finding.evidence.replace("\n", " ")
                    if len(evidence) > 400:
                        evidence = evidence[:397].rstrip() + "..."
                    lines.append(f"  - Evidence: `{evidence}`")
                if finding.action:
                    lines.append(f"  - **Action:** {finding.action}")
            lines.append("")

        lines += [
            "## Advisory findings from the language model",
            "",
            f"_{len(self.advisory)} issue(s); these do not affect readiness._",
            "",
        ]
        if self.model.summary:
            lines.extend([self.model.summary, ""])
        if not self.advisory:
            lines.extend(["- No admissible model finding.", ""])
        for issue in self.advisory:
            section = f" @{issue.section}" if issue.section else ""
            lines += [
                f"### [{issue.severity} · {issue.dimension}]{section}",
                "",
                f"> {issue.quote}",
                "",
                f"**Problem:** {issue.claim}",
                "",
                f"**Action:** {issue.action}",
                "",
            ]
        for heading, items in (("Strengths reported by the model", self.model.strengths),
                               ("Model-stated limitations", self.model.limitations)):
            if items:
                lines.extend([f"**{heading}**", ""])
                lines.extend([f"- {item}" for item in items])
                lines.append("")

        lines.extend(["## Checks that ran", ""])
        lines.append(", ".join(f"`{check}`" for check in self.checks_run) or "none")
        lines.append("")
        if self.notes or self.model.notes:
            lines.extend(["## Run notes", ""])
            lines.extend([f"- {note}" for note in [*self.notes, *self.model.notes]])
            lines.append("")
        lines.extend(["## What this review does not do", ""])
        lines.extend([f"- {item}" for item in LIMITATIONS])
        lines.extend(["", f"> {DISCLAIMER}", ""])
        return "\n".join(lines)


def _finding(check_id: str, severity: str, title: str, detail: str, *,
             location: str = "", evidence: str = "", action: str = "") -> Finding:
    return Finding(check_id, severity, title, detail, location, evidence, action)


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def compute_readiness(findings: Sequence[Finding]) -> Tuple[str, str]:
    counts = {severity: 0 for severity in SEVERITY_ORDER}
    for finding in findings:
        counts[finding.severity] += 1
    if counts["blocker"]:
        return ("major_revision",
                f"{counts['blocker']} blocker finding(s) must be resolved first.")
    if counts["major"] >= 3:
        return ("major_revision", f"{counts['major']} major findings; the manuscript "
                                  "needs substantial revision.")
    if counts["major"]:
        return ("minor_revision",
                f"{counts['major']} major finding(s) require attention.")
    if counts["minor"] >= 4:
        return ("minor_revision", f"{counts['minor']} minor findings accumulate.")
    return ("no_blocking_findings",
            "No blocker or major mechanical finding was detected; this is not an "
            "editorial decision.")


def _check_text(manuscript: Manuscript, findings: List[Finding],
                checks: List[str]) -> None:
    checks.append("text.placeholder")
    for match in list(_PLACEHOLDER_RE.finditer(manuscript.text))[:5]:
        findings.append(_finding(
            "text.placeholder", "blocker",
            "Placeholder marker left in the manuscript",
            f"'{match.group(0)}' appears in the text.",
            location=f"line {_line_of(manuscript.text, match.start())}",
            evidence=match.group(0),
            action="Replace the placeholder with the intended content or remove it.",
        ))

    checks.append("text.repeated_sentences")
    repeated = repeated_sentences(
        "\n\n".join(section.body for section in manuscript.sections))
    for sentence, count in repeated[:3]:
        findings.append(_finding(
            "text.repeated_sentence", "minor",
            f"A sentence appears {count} times",
            "Copy-pasted text usually marks a section that was not edited after a move.",
            location="document",
            evidence=sentence[:240],
            action="Keep the sentence where it belongs and remove the repetition.",
        ))


def _check_structure(manuscript: Manuscript, findings: List[Finding],
                     checks: List[str], notes: List[str]) -> None:
    checks.append("structure.sections")
    if (len(manuscript.sections) == 1 and manuscript.sections[0].id == "document"
            and manuscript.reference_heading_line is None):
        notes.append("No headings were found; structural checks treated the file as "
                     "one document section.")
    for section in manuscript.sections:
        if section.word_count == 0 and len(manuscript.sections) > 1:
            findings.append(_finding(
                "structure.empty_section", "major",
                f"Section '{section.title}' has no body",
                "A heading with no text below it leaves the outline incomplete.",
                location=f"{section.title} (line {section.start_line})",
                evidence=section.title,
                action="Write the section or delete the heading.",
            ))

    checks.append("structure.duplicate_titles")
    titles: Dict[str, List[str]] = {}
    for section in manuscript.sections:
        titles.setdefault(section.title.casefold(), []).append(section.title)
    for repeated in (values for values in titles.values() if len(values) > 1):
        findings.append(_finding(
            "structure.duplicate_titles", "minor",
            f"Section title '{repeated[0]}' is repeated",
            f"{len(repeated)} sections share this title, so references to it are "
            "ambiguous.",
            location=repeated[0],
            evidence=repeated[0],
            action="Give each section a distinct title.",
        ))


def _check_citations(manuscript: Manuscript, findings: List[Finding],
                     checks: List[str]) -> None:
    checks.extend(["citations.numeric_markers", "citations.author_year"])
    references = manuscript.references
    citations = manuscript.citations
    if citations and not references:
        findings.append(_finding(
            "citations.no_reference_list", "major",
            "In-text citations exist but no reference list was found",
            f"{len(citations)} citation marker(s) point at a reference section that "
            "could not be located.",
            location="document",
            action="Add a 'References' (or 'Bibliography') section containing every "
                   "cited work.",
        ))
    if manuscript.reference_heading_line is not None and not references:
        findings.append(_finding(
            "references.empty_section", "major",
            "The reference section contains no entries",
            "A reference heading was found, but nothing below it parsed as an entry.",
            location=f"line {manuscript.reference_heading_line}",
            action="List the cited works, or remove the empty heading.",
        ))
    if not references:
        return

    numeric_cited = sorted(
        {c.key for c in citations if c.kind == "numeric"}, key=int)
    undefined = [key for key in numeric_cited
                 if key not in manuscript.numeric_reference_keys]
    for key in undefined[:10]:
        sections = sorted({c.section_id for c in citations
                           if c.kind == "numeric" and c.key == key})
        findings.append(_finding(
            "citations.undefined_key", "major",
            f"Citation [{key}] has no matching reference entry",
            f"Marker [{key}] is cited in {', '.join(sections)} but the reference list "
            f"has no [{key}] entry.",
            location=", ".join(sections),
            evidence=f"[{key}]",
            action="Add the corresponding reference entry or correct the marker.",
        ))
    if len(undefined) > 10:
        findings.append(_finding(
            "citations.undefined_key_more", "major",
            f"{len(undefined) - 10} further numeric citations do not resolve",
            "Only the first ten unresolved markers were listed individually.",
            location="document",
            action="Audit the remaining numeric citation markers against the list.",
        ))

    author_year_pairs = sorted(set(manuscript.cited_author_years))
    unresolved = [(surname, year) for surname, year in author_year_pairs
                  if not any(entry.matches(surname, year) for entry in references)]
    for surname, year in unresolved[:10]:
        sections = sorted({c.section_id for c in citations
                           if c.kind == "author-year"
                           and c.key == f"{surname}|{year}"})
        findings.append(_finding(
            "citations.undefined_author_year", "major",
            f"Citation '{surname} ({year})' has no matching reference entry",
            "No reference entry contains this author surname together with this year.",
            location=", ".join(sections),
            evidence=f"{surname} ({year})",
            action="Add the reference entry, correct the year, or correct the author "
                   "name in the citation.",
        ))

    checks.append("references.uncited")
    uncited: List[str] = []
    for entry in references:
        cited = (bool(entry.key) and entry.key in manuscript.cited_numeric_keys) or any(
            entry.matches(surname, year) for surname, year in author_year_pairs)
        if not cited:
            label = f"[{entry.key}] {entry.raw[:60]}" if entry.key else entry.raw[:70]
            uncited.append(label)
    if uncited:
        findings.append(_finding(
            "references.uncited", "minor",
            f"{len(uncited)} reference entr{'y is' if len(uncited) == 1 else 'ies are'} "
            "never cited",
            "An entry that is never cited cannot support a claim in the text.",
            location="reference list",
            evidence=" | ".join(uncited[:8]),
            action="Cite the work where it supports the argument, or remove the entry.",
        ))

    checks.append("references.duplicates")
    seen_dois: Dict[str, int] = {}
    seen_text: Dict[str, int] = {}
    for entry in references:
        if entry.doi:
            doi = entry.doi.casefold()
            if doi in seen_dois:
                findings.append(_finding(
                    "references.duplicate_doi", "major",
                    f"DOI {entry.doi} appears more than once",
                    "Two reference entries resolve to the same DOI.",
                    location=f"line {entry.line}",
                    evidence=entry.raw[:160],
                    action="Merge the duplicate entries and keep one citation key.",
                ))
            seen_dois[doi] = entry.line
        key = re.sub(r"[^a-z0-9 ]+", " ", entry.normalized).strip()
        if key and key in seen_text:
            findings.append(_finding(
                "references.duplicate_entry", "major",
                "A reference entry is repeated",
                "Two entries have identical text after normalisation.",
                location=f"lines {seen_text[key]} and {entry.line}",
                evidence=entry.raw[:160],
                action="Remove the duplicate entry.",
            ))
        seen_text[key] = entry.line

    checks.append("references.doi_format")
    for entry in references:
        if not entry.doi and _DOI_IN_TEXT_RE.search(entry.raw):
            findings.append(_finding(
                "references.malformed_doi", "minor",
                "A DOI-looking string is malformed",
                "The text mentions a DOI but no valid '10.xxxx/...' token was found.",
                location=f"line {entry.line}",
                evidence=entry.raw[:160],
                action="Correct the DOI or remove the broken identifier.",
            ))

    checks.append("references.year")
    missing_years = [entry for entry in references if entry.year is None]
    if missing_years:
        findings.append(_finding(
            "references.missing_year", "note",
            f"{len(missing_years)} reference entr{'y has' if len(missing_years) == 1 else 'ies have'} "
            "no year",
            "A reader cannot date these works from the entry alone.",
            location="reference list",
            evidence=" | ".join(entry.raw[:70] for entry in missing_years[:6]),
            action="Add the publication year where one exists.",
        ))


def _verify_dois(manuscript: Manuscript, findings: List[Finding], notes: List[str],
                 retriever: Any) -> None:
    dois = list(dict.fromkeys(entry.doi for entry in manuscript.references if entry.doi))
    if not dois:
        notes.append("DOI verification was requested, but the reference list has no DOI.")
        return
    client = retriever
    if client is None:
        from .research.sources import CrossrefRetriever
        client = CrossrefRetriever(timeout=12)
    resolved = 0
    checked = 0
    for doi in dois[:25]:
        checked += 1
        try:
            result = client.lookup_doi(doi)
        except Exception as exc:  # noqa: BLE001 - a provider outage is data
            notes.append(f"DOI {doi} could not be verified: "
                         f"{type(exc).__name__}: {exc}")
            continue
        if result.ok and result.sources:
            resolved += 1
            continue
        error = (result.error or "").casefold()
        if "404" in error or "not found" in error:
            entry = next(e for e in manuscript.references if e.doi == doi)
            findings.append(_finding(
                "references.doi_unverified", "major",
                f"DOI {doi} was not found in Crossref",
                "Crossref has no record for this identifier. This is absence of "
                "verification, not proof that the reference is fabricated.",
                location=f"line {entry.line}",
                evidence=entry.raw[:160],
                action="Check the DOI (and the entry) against the publisher record.",
            ))
        else:
            notes.append(f"DOI {doi} could not be verified: "
                         f"{result.error or 'no record returned'}")
    notes.append(f"Crossref lookup resolved {resolved} of {checked} DOI(s) checked.")


def _check_abstract(manuscript: Manuscript, findings: List[Finding],
                    checks: List[str]) -> None:
    checks.append("abstract.present")
    if not manuscript.abstract_heading_present:
        findings.append(_finding(
            "abstract.missing", "note",
            "No abstract section was found",
            "The review could not compare an abstract against the body.",
            location="document",
            action="Add an 'Abstract' heading if the venue requires one.",
        ))
        return
    if not manuscript.abstract.strip():
        findings.append(_finding(
            "abstract.empty", "note",
            "The abstract heading is present but empty",
            "An 'Abstract' heading was found, but nothing below it parsed as text.",
            location="abstract",
            action="Write the abstract or remove the empty heading.",
        ))
        return

    abstract_words = word_count(manuscript.abstract)
    checks.append("abstract.length")
    if abstract_words > 350:
        findings.append(_finding(
            "abstract.too_long", "minor",
            f"The abstract is {abstract_words} words",
            "Many venues cap abstracts near 250-300 words; check the target venue.",
            location="abstract",
            evidence=manuscript.abstract[:240],
            action="Trim the abstract to the venue limit.",
        ))
    elif abstract_words < 60:
        findings.append(_finding(
            "abstract.too_short", "minor",
            f"The abstract is only {abstract_words} words",
            "A very short abstract may omit the question, method or main result.",
            location="abstract",
            evidence=manuscript.abstract[:240],
            action="State the problem, method and main finding explicitly.",
        ))

    if word_count(manuscript.body_without_abstract) < 20:
        # Without a body there is nothing to compare the abstract against; flagging
        # every number would be a false positive.
        return
    checks.append("abstract.numbers_supported")
    unsupported = sorted(set(manuscript.abstract_numbers) - set(manuscript.body_numbers))
    for token in unsupported[:10]:
        findings.append(_finding(
            "abstract.number_not_in_body", "major",
            f"'{token}' appears in the abstract but not in the body",
            "A reported quantity that the body never states cannot be checked against "
            "the analysis; this may be a summary typo or an unsupported claim.",
            location="abstract",
            evidence=token,
            action="Align the abstract with a number stated in the body, or add the "
                   "supporting result to the body.",
        ))


def deterministic_findings(manuscript: Manuscript, *, verify_dois: bool = False,
                           retriever: Any = None
                           ) -> Tuple[List[Finding], List[str], List[str]]:
    """Run every mechanical check and return ``(findings, checks_run, notes)``."""
    findings: List[Finding] = []
    checks: List[str] = []
    notes: List[str] = []

    _check_text(manuscript, findings, checks)
    _check_structure(manuscript, findings, checks, notes)
    _check_citations(manuscript, findings, checks)
    if verify_dois:
        checks.append("references.doi_lookup")
        _verify_dois(manuscript, findings, notes, retriever)
    _check_abstract(manuscript, findings, checks)

    if not manuscript.citations and not manuscript.references:
        notes.append("No citation markers or reference entries were found; citation "
                     "checks had nothing to resolve.")

    findings.sort(key=lambda f: (SEVERITY_ORDER.index(f.severity), f.check_id))
    return findings, checks, notes


class ModelReviewer:
    """One bounded model call whose findings must quote the manuscript."""

    def __init__(self, client: Optional[LLMClient], *, prompt_dir: Optional[str] = None,
                 model_review: bool = True) -> None:
        self.client = client
        self.prompt_dir = prompt_dir
        self.enabled = bool(model_review)
        self.available = bool(self.enabled and client is not None
                              and getattr(client, "available", False))

    def _user_prompt(self, manuscript: Manuscript, *, article_type: Optional[str],
                     audience: Optional[str]) -> str:
        truncated = len(manuscript.text) > MAX_MODEL_CHARS
        stats = manuscript.stats()
        return "\n".join([
            f"ARTICLE TYPE: {article_type or 'not specified'}",
            f"INTENDED AUDIENCE: {audience or 'not specified'}",
            f"STRUCTURE FACTS (orientation only, not findings): {stats['sections']} "
            f"section(s), {stats['references']} reference entr(y/ies), "
            f"{manuscript.word_count} words.",
            "If the manuscript is marked TRUNCATED, review only the visible text and "
            "state that in limitations.",
            "",
            "MANUSCRIPT (untrusted content begins)"
            + (" [TRUNCATED]" if truncated else ""),
            manuscript.text[:MAX_MODEL_CHARS],
            "MANUSCRIPT (untrusted content ends)",
            "",
            "Return only the JSON object specified by the system prompt.",
        ])

    def review(self, manuscript: Manuscript, *, article_type: Optional[str] = None,
               audience: Optional[str] = None, max_issues: int = 10
               ) -> Tuple[List[AdvisoryIssue], ModelReviewOutcome]:
        outcome = ModelReviewOutcome()
        if not self.enabled:
            outcome.notes.append("Model review was disabled; deterministic checks only.")
            return [], outcome
        if not self.available:
            outcome.notes.append(
                "No model provider is configured; deterministic checks only. Set "
                "ANTHROPIC_API_KEY, OPENAI_API_KEY or DAHL_API_KEY to add advisory findings."
            )
            return [], outcome

        try:
            system = load_template("manuscript_reviewer", self.prompt_dir)
        except PromptError as exc:
            outcome.notes.append(f"Review prompt unavailable: {exc}")
            return [], outcome

        if len(manuscript.text) > MAX_MODEL_CHARS:
            outcome.notes.append(
                f"The manuscript was truncated for the model call at {MAX_MODEL_CHARS} "
                "characters; issues in the remainder were not assessable."
            )
        try:
            response = self.client.complete(
                system,
                self._user_prompt(manuscript, article_type=article_type,
                                  audience=audience),
                max_output_tokens=4096,
            )
            payload = parse_json(response.text)
        except (ProviderError, ResponseParseError, ValueError) as exc:
            outcome.notes.append(
                f"Model review unavailable: {type(exc).__name__}: {exc}. "
                "Only deterministic findings are reported."
            )
            return [], outcome
        except Exception as exc:  # noqa: BLE001 - a review must not crash the command
            outcome.notes.append(
                f"Model review failed: {type(exc).__name__}: {exc}. "
                "Only deterministic findings are reported."
            )
            return [], outcome

        if not isinstance(payload, dict):
            outcome.notes.append("Model review returned no JSON object; ignored.")
            return [], outcome

        outcome.ran = True
        outcome.engine = getattr(response, "engine", "") or self.client.name()
        summary = payload.get("summary")
        if isinstance(summary, str):
            outcome.summary = summary.strip()[:2000]
        outcome.strengths = _string_list(payload.get("strengths"))[:8]
        outcome.limitations = _string_list(payload.get("limitations"))[:8]
        if any(key in payload for key in ("readiness", "decision", "verdict",
                                          "recommendation")):
            # A model verdict would be the least defensible output in the report.
            outcome.notes.append(
                "A model-provided decision/verdict was ignored; readiness is computed "
                "only from deterministic checks."
            )

        raw_issues = payload.get("issues")
        if not isinstance(raw_issues, list):
            outcome.notes.append("Model review contained no 'issues' array.")
            return [], outcome

        issues: List[AdvisoryIssue] = []
        seen: set = set()
        for item in raw_issues:
            if len(issues) >= max_issues:
                break
            if not isinstance(item, dict):
                outcome.dropped_issues += 1
                continue
            quote = item.get("quote")
            if not isinstance(quote, str) or len(normalize_for_match(quote)) < 10:
                outcome.dropped_issues += 1
                continue
            # The quote is the admissibility test: without a verbatim location the
            # finding is unverifiable, so it is discarded rather than shown.
            if normalize_for_match(quote) not in manuscript.normalized_text:
                outcome.dropped_quotes += 1
                continue
            claim = item.get("claim")
            action = item.get("action")
            if not isinstance(claim, str) or not claim.strip() \
                    or not isinstance(action, str) or not action.strip():
                outcome.dropped_issues += 1
                continue
            key = (normalize_for_match(quote), claim.strip().casefold())
            if key in seen:
                outcome.dropped_issues += 1
                continue
            seen.add(key)
            dimension = str(item.get("dimension") or "").strip().casefold()
            severity = str(item.get("severity") or "").strip().casefold()
            issues.append(AdvisoryIssue(
                dimension=dimension if dimension in MODEL_DIMENSIONS
                else "analysis_and_claims",
                severity=severity if severity in _MODEL_SEVERITIES else "moderate",
                section=str(item.get("section") or "").strip()[:120],
                quote=quote.strip(),
                claim=claim.strip()[:1200],
                action=action.strip()[:1200],
            ))
        return issues, outcome


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def review_manuscript(manuscript: Manuscript, *, client: Optional[LLMClient] = None,
                      model_review: bool = True, article_type: Optional[str] = None,
                      audience: Optional[str] = None, verify_dois: bool = False,
                      retriever: Any = None, max_issues: int = 10,
                      prompt_dir: Optional[str] = None) -> ManuscriptReview:
    findings, checks, notes = deterministic_findings(
        manuscript, verify_dois=verify_dois, retriever=retriever)
    reviewer = ModelReviewer(client, prompt_dir=prompt_dir, model_review=model_review)
    advisory, model = reviewer.review(
        manuscript, article_type=article_type, audience=audience,
        max_issues=max(0, int(max_issues)))
    if model.dropped_quotes:
        model.notes.append(
            f"{model.dropped_quotes} model issue(s) were discarded because their quote "
            "does not appear verbatim in the manuscript."
        )
    readiness, reason = compute_readiness(findings)
    return ManuscriptReview(
        source=manuscript.source,
        title=manuscript.title or "Untitled manuscript",
        readiness=readiness,
        readiness_reason=reason,
        stats=manuscript.stats(),
        findings=findings,
        advisory=advisory,
        model=model,
        checks_run=checks,
        notes=notes,
    )
