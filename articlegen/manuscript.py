"""Import an existing manuscript for review.

This is the boundary for documents the pipeline did not write. Parsing is
deliberately conservative: it extracts only what is literally present (headings,
reference entries, citation markers, numeric tokens) and never repairs, rewrites
or normalises an author's content. A review can therefore quote the manuscript
and point at a real location.

Plain text is accepted as well as Markdown. When no headings exist the whole
document is one section, so structural checks report "not applicable" instead of
inventing a structure.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_BOLD_REFERENCE_RE = re.compile(
    r"^\*\*\s*(references?|bibliography|works cited|literature cited|منابع|مراجع)\s*\*\*\s*:?\s*$",
    re.IGNORECASE,
)
_REFERENCE_HEADING_RE = re.compile(
    r"^(references?|bibliography|works cited|literature cited|منابع|مراجع)$",
    re.IGNORECASE,
)
_ENTRY_RE = re.compile(r"^(?:\[(\d+)\]|(\d+)[.)]|[-*•])\s+(.*)$")
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)")
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_NUMERIC_CITE_RE = re.compile(r"\[(\d+(?:\s*[-,–]\s*\d+)*)\](?!\()")
_PAREN_RE = re.compile(r"\(([^()]{1,200})\)")
_NARRATIVE_RE = re.compile(
    r"\b([A-Z][A-Za-zÀ-ÿ'’\-]{1,})(?:\s+et\s+al\.?)?\s*\(\s*((?:19|20)\d{2})[a-z]?\s*\)"
)
_CHUNK_RE = re.compile(
    r"^(?P<names>[A-Z][A-Za-zÀ-ÿ'’\-\.\s&]+?)\s*,?\s*(?P<year>(?:19|20)\d{2})[a-z]?$"
)
_NUMBER_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(%|percent\b)?", re.IGNORECASE
)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\S+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")

_ABSTRACT_TITLES = {"abstract", "summary", "executive summary"}

# Plain-text files have no ``#`` markers. Only a standalone line whose complete text is
# a standard section title becomes a heading, so a sentence that merely mentions
# "references" is never treated as a section boundary.
_PLAIN_HEADINGS = frozenset({
    "abstract", "summary", "executive summary",
    "introduction", "background", "related work", "literature review",
    "state of the art",
    "methods", "method", "methodology", "materials and methods", "approach",
    "experimental setup", "experiment", "experiments",
    "results", "findings", "results and discussion",
    "discussion", "analysis", "evaluation",
    "conclusion", "conclusions", "conclusion and future work", "future work",
    "limitations", "threats to validity",
    "references", "bibliography", "works cited", "literature cited",
    "acknowledgements", "acknowledgments", "appendix", "appendices",
})
_PLAIN_HEADING_TEXT_RE = re.compile(r"[A-Za-z][A-Za-z &/\-]{0,59}")


def normalize_for_match(text: str) -> str:
    """Whitespace-collapsed, casefolded text used for verbatim quote checks."""
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def strip_code(text: str) -> str:
    """Remove fenced and inline code before reading citations or numbers.

    Code blocks routinely contain bracketed numbers, years and percentages that
    are not claims about the manuscript; parsing them as citations produces false
    findings.
    """
    without_fences = _CODE_FENCE_RE.sub(" ", text or "")
    return _INLINE_CODE_RE.sub(" ", without_fences)


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def slugify(value: str, fallback: str = "section") -> str:
    token = _SLUG_RE.sub("-", (value or "").casefold()).strip("-")
    return token or fallback


@dataclass
class ManuscriptSection:
    id: str
    title: str
    heading_level: int
    start_line: int
    body: str

    @property
    def word_count(self) -> int:
        return word_count(self.body)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "heading_level": self.heading_level,
            "start_line": self.start_line,
            "word_count": self.word_count,
        }


@dataclass
class ReferenceEntry:
    raw: str
    line: int
    key: Optional[str] = None
    year: Optional[int] = None
    doi: str = ""
    tokens: Set[str] = field(default_factory=set)

    @property
    def normalized(self) -> str:
        return re.sub(r"\s+", " ", self.raw).strip().casefold()

    def matches(self, surname: str, year: str) -> bool:
        if not surname or surname.casefold() not in self.tokens:
            return False
        if self.year is None:
            return True
        return str(self.year) == str(year)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line": self.line,
            "key": self.key,
            "year": self.year,
            "doi": self.doi,
            "raw": self.raw,
        }


@dataclass
class Citation:
    kind: str  # "numeric" | "author-year"
    key: str   # "3" or "smith|2020"
    raw: str
    section_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "key": self.key, "raw": self.raw,
                "section_id": self.section_id}


@dataclass
class Manuscript:
    text: str
    source: str = ""
    title: str = ""
    sections: List[ManuscriptSection] = field(default_factory=list)
    references: List[ReferenceEntry] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    abstract: str = ""
    abstract_heading_present: bool = False
    body_without_abstract: str = ""
    normalized_text: str = ""
    reference_heading_line: Optional[int] = None

    @property
    def word_count(self) -> int:
        return word_count(self.text)

    @property
    def abstract_numbers(self) -> Dict[str, int]:
        return numeric_tokens(self.abstract)

    @property
    def body_numbers(self) -> Dict[str, int]:
        return numeric_tokens(self.body_without_abstract)

    @property
    def numeric_reference_keys(self) -> Set[str]:
        return {entry.key for entry in self.references if entry.key}

    @property
    def cited_numeric_keys(self) -> Set[str]:
        return {c.key for c in self.citations if c.kind == "numeric"}

    @property
    def cited_author_years(self) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for citation in self.citations:
            if citation.kind != "author-year":
                continue
            surname, _, year = citation.key.partition("|")
            pairs.append((surname, year))
        return pairs

    def stats(self) -> Dict[str, Any]:
        return {
            "words": self.word_count,
            "sections": len(self.sections),
            "references": len(self.references),
            "citation_markers": len(self.citations),
            "abstract_words": word_count(self.abstract),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "stats": self.stats(),
            "sections": [s.to_dict() for s in self.sections],
            "references": [r.to_dict() for r in self.references],
            "citations": [c.to_dict() for c in self.citations],
        }


def _fenced_line_numbers(lines: Sequence[str]) -> Set[int]:
    fenced: Set[int] = set()
    inside = False
    for index, line in enumerate(lines):
        if line.strip().startswith("```"):
            inside = not inside
            fenced.add(index)
            continue
        if inside:
            fenced.add(index)
    return fenced


def _clean_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip("*_ ")).strip()


def _plain_heading(line: str) -> Optional[str]:
    """A standalone standard section title, with no Markdown marker."""
    candidate = line.strip().strip("*_` ").rstrip(":").strip()
    if not candidate or candidate.casefold() not in _PLAIN_HEADINGS:
        return None
    if not _PLAIN_HEADING_TEXT_RE.fullmatch(candidate):
        return None
    return _clean_heading(candidate)


def _primary_surname(names: str) -> str:
    primary = re.split(r",|&|\band\b|\bet\s+al\b", names, maxsplit=1)[0]
    tokens = re.findall(r"[A-Za-zÀ-ÿ'’\-]+", primary)
    return tokens[0].casefold() if tokens else ""


def _expand_numeric_marker(marker: str, limit: int = 100) -> List[str]:
    keys: List[str] = []
    for part in re.split(r"\s*,\s*", marker):
        if not part:
            continue
        bounds = re.split(r"\s*[-–]\s*", part)
        if len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
            start, end = int(bounds[0]), int(bounds[1])
            if start <= end and end - start <= limit:
                keys.extend(str(value) for value in range(start, end + 1))
                continue
        if part.isdigit():
            keys.append(part)
    return keys


def _extract_citations(section_id: str, body: str) -> List[Citation]:
    text = strip_code(body)
    found: List[Citation] = []
    for match in _NUMERIC_CITE_RE.finditer(text):
        for key in _expand_numeric_marker(match.group(1)):
            found.append(Citation("numeric", key, match.group(0), section_id))
    for group in _PAREN_RE.finditer(text):
        for chunk in group.group(1).split(";"):
            parsed = _CHUNK_RE.match(chunk.strip())
            if not parsed:
                continue
            surname = _primary_surname(parsed.group("names"))
            if surname:
                found.append(Citation(
                    "author-year", f"{surname}|{parsed.group('year')}",
                    chunk.strip(), section_id))
    for match in _NARRATIVE_RE.finditer(text):
        found.append(Citation(
            "author-year", f"{match.group(1).casefold()}|{match.group(2)}",
            match.group(0), section_id))
    return found


def _dedupe_citations(citations: Sequence[Citation]) -> List[Citation]:
    seen: Set[Tuple[str, str, str]] = set()
    unique: List[Citation] = []
    for citation in citations:
        key = (citation.kind, citation.key, citation.section_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(citation)
    return unique


def _reference_entry(raw: str, line: int, key: Optional[str] = None) -> ReferenceEntry:
    doi_match = _DOI_RE.search(raw)
    doi = doi_match.group(1).rstrip(".,;:)]}") if doi_match else ""
    year_match = _YEAR_RE.search(raw)
    tokens = {token.casefold() for token in re.findall(r"[A-Za-zÀ-ÿ'’\-]{2,}", raw)}
    return ReferenceEntry(
        raw=raw.strip(),
        line=line,
        key=key,
        year=int(year_match.group(1)) if year_match else None,
        doi=doi,
        tokens=tokens,
    )


def _parse_references(lines: Sequence[str], heading_index: int) -> List[ReferenceEntry]:
    body = [(index, lines[index]) for index in range(heading_index + 1, len(lines))]
    non_blank = [(index, line) for index, line in body if line.strip()]
    if not non_blank:
        return []
    has_markers = any(_ENTRY_RE.match(line.strip()) for _, line in non_blank)
    if not has_markers:
        # Plain paragraphs: one paragraph per entry, separated by blank lines.
        entries: List[ReferenceEntry] = []
        buffer: List[str] = []
        start_line = non_blank[0][0]
        for index, line in body:
            if not line.strip():
                if buffer:
                    entries.append(_reference_entry(" ".join(buffer), start_line + 1))
                    buffer = []
                continue
            if not buffer:
                start_line = index
            buffer.append(line.strip())
        if buffer:
            entries.append(_reference_entry(" ".join(buffer), start_line + 1))
        return entries

    entries = []
    current_raw: Optional[str] = None
    current_line = 0
    current_key: Optional[str] = None
    for index, line in non_blank:
        match = _ENTRY_RE.match(line.strip())
        if match:
            if current_raw is not None:
                entries.append(_reference_entry(current_raw, current_line, current_key))
            current_raw = match.group(3).strip()
            current_line = index + 1
            current_key = match.group(1) or match.group(2)
        elif current_raw is not None:
            current_raw = f"{current_raw} {line.strip()}"
        else:
            current_raw = line.strip()
            current_line = index + 1
            current_key = None
    if current_raw is not None:
        entries.append(_reference_entry(current_raw, current_line, current_key))
    return [entry for entry in entries if entry.raw]


def numeric_tokens(text: str) -> Dict[str, int]:
    """Numeric tokens that a reader would treat as a reported quantity.

    Years and single digits are ignored: ``2020`` is usually prose, and a lone
    ``3`` is too often a list marker or "three". A trailing percent is preserved
    so ``37%`` and ``37`` are not silently equated.
    """
    counts: Dict[str, int] = {}
    clean = _NUMERIC_CITE_RE.sub(" ", strip_code(text or ""))
    for match in _NUMBER_RE.finditer(clean):
        raw, unit = match.group(1), (match.group(2) or "").casefold()
        value = raw.replace(",", "")
        if not unit and "." not in value and value.isdigit() and 1900 <= int(value) <= 2099:
            continue
        if not unit and len(value.replace(".", "")) < 2:
            continue
        token = f"{value}%" if unit else value
        counts[token] = counts.get(token, 0) + 1
    return counts


def parse_manuscript(text: str, source: str = "") -> Manuscript:
    if not isinstance(text, str):
        raise ValueError("manuscript must be text")
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    if not cleaned.strip():
        raise ValueError("manuscript is empty")

    lines = cleaned.split("\n")
    fenced = _fenced_line_numbers(lines)
    headings: List[Tuple[int, int, str]] = []
    saw_atx_heading = False
    for index, line in enumerate(lines):
        if index in fenced:
            continue
        match = _HEADING_RE.match(line)
        if match:
            saw_atx_heading = True
            headings.append((index, len(match.group(1)), _clean_heading(match.group(2))))
            continue
        bold = _BOLD_REFERENCE_RE.match(line.strip())
        if bold:
            headings.append((index, 2, _clean_heading(bold.group(1))))

    if not saw_atx_heading:
        # Plain text: promote standard section lines (Abstract, References, ...) to
        # headings so the reference list is not swallowed into one document section.
        seen = {index for index, _, _ in headings}
        for index, line in enumerate(lines):
            if index in fenced or index in seen:
                continue
            plain = _plain_heading(line)
            if plain:
                headings.append((index, 2, plain))
        headings.sort()

    title = ""
    title_index: Optional[int] = None
    if headings:
        first_index, first_level, first_title = headings[0]
        next_index = headings[1][0] if len(headings) > 1 else len(lines)
        between = "\n".join(lines[first_index + 1:next_index]).strip()
        # A leading H1 is a document title only when no prose follows it directly.
        # Otherwise '# Introduction' would be consumed as the title and its body
        # would disappear from every check.
        if (first_level == 1 and first_index <= 5 and len(headings) > 1
                and word_count(between) < 20):
            title = first_title
            title_index = first_index

    reference_heading = next(
        ((index, title_text) for index, _, title_text in headings
         if _REFERENCE_HEADING_RE.match(title_text)),
        None,
    )
    reference_index = reference_heading[0] if reference_heading else None
    sections_end = reference_index if reference_index is not None else len(lines)

    section_headings = [
        item for item in headings
        if item[0] < sections_end and item[0] != title_index
        and item[0] != reference_index
    ]

    sections: List[ManuscriptSection] = []
    used_ids: Set[str] = set()
    for position, (index, level, heading_title) in enumerate(section_headings):
        next_index = (section_headings[position + 1][0]
                      if position + 1 < len(section_headings) else sections_end)
        body = "\n".join(lines[index + 1:next_index]).strip()
        section_id = slugify(heading_title)
        suffix = 2
        while section_id in used_ids:
            section_id = f"{slugify(heading_title)}-{suffix}"
            suffix += 1
        used_ids.add(section_id)
        sections.append(ManuscriptSection(
            id=section_id, title=heading_title, heading_level=level,
            start_line=index + 1, body=body,
        ))

    if not sections:
        start = (title_index + 1) if title_index is not None else 0
        body = "\n".join(lines[start:sections_end]).strip()
        if not body.strip():
            body = cleaned
        sections = [ManuscriptSection(
            id="document", title=title or "Document", heading_level=1,
            start_line=start + 1, body=body,
        )]

    references = (_parse_references(lines, reference_index)
                  if reference_index is not None else [])

    citations: List[Citation] = []
    for section in sections:
        citations.extend(_extract_citations(section.id, section.body))

    abstract_section = next(
        (section for section in sections
         if section.title.casefold() in _ABSTRACT_TITLES
         or section.title.casefold().startswith("abstract")),
        None,
    )
    abstract = abstract_section.body if abstract_section else ""
    abstract_ids = {abstract_section.id} if abstract_section else set()
    body_parts = [section.body for section in sections if section.id not in abstract_ids]

    return Manuscript(
        text=cleaned,
        source=source,
        title=title,
        sections=sections,
        references=references,
        citations=_dedupe_citations(citations),
        abstract=abstract,
        abstract_heading_present=abstract_section is not None,
        body_without_abstract="\n\n".join(body_parts),
        normalized_text=normalize_for_match(cleaned),
        reference_heading_line=(reference_index + 1) if reference_index is not None else None,
    )


def load_manuscript(path: str) -> Manuscript:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    label = os.path.basename(path)
    manuscript = parse_manuscript(text, source=label)
    if not manuscript.title:
        manuscript.title = os.path.splitext(label)[0].replace("_", " ").strip()
    return manuscript


def repeated_sentences(text: str, *, minimum_words: int = 12,
                       limit: int = 20) -> List[Tuple[str, int]]:
    """Sentences of at least ``minimum_words`` words that occur more than once."""
    counts: Dict[str, int] = {}
    display: Dict[str, str] = {}
    order: List[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(strip_code(text or "")):
        stripped = sentence.strip()
        if word_count(stripped) < minimum_words:
            continue
        key = normalize_for_match(stripped)
        if key not in counts:
            counts[key] = 0
            display[key] = stripped
            order.append(key)
        counts[key] += 1
    repeated = [(display[key], counts[key]) for key in order if counts[key] > 1]
    repeated.sort(key=lambda item: (-item[1], item[0]))
    return repeated[:limit]
