"""Source retrieval, verification, claim mapping and bibliography generation.

Design rules enforced here:

* A ``SourceRecord`` is the only thing that may become a reference. The Writer is
  given verified records and citation keys; it is never asked to invent a reference.
* Retrieval failure is reported as failure. There is no "plausible-looking" fallback
  record, and a user-supplied record is labelled ``user-provided`` until an index
  actually confirms it.
* Citation resolution is deterministic: numbered markers must fall inside the
  generated bibliography, author-year markers must match a record's first author and
  year, and every reference-list entry must correspond to a library record.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..models import CitationStyle
from .models import (
    ReviewSeverity,
    SourceRecord,
    normalise_doi,
)

DEFAULT_RETRIEVAL_TIMEOUT = 12
CROSSREF_BASE = "https://api.crossref.org/works"
_HEADING_RE = re.compile(
    r"(?m)^(#{1,6})\s*(references?|bibliography|works cited|literature cited)\s*$",
    re.IGNORECASE,
)
_NUMBERED_ENTRY_RE = re.compile(r"(?m)^\s*\[(\d+)\]\s*(.+?)(?=^\s*\[\d+\]|\Z)", re.DOTALL)
_PLAIN_NUMBERED_RE = re.compile(r"(?m)^\s*(\d+)\.\s+(.+?)(?=^\s*\d+\.\s+|\Z)", re.DOTALL)
_CITATION_NUMBER_RE = re.compile(r"\[(\d+(?:\s*[-,]\s*\d+)*)\](?!\()")
_AUTHOR_YEAR_IN_TEXT_RE = re.compile(
    r"\(([A-Z][A-Za-zÀ-ÿ'\-]+)(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][A-Za-zÀ-ÿ'\-]+)?,?\s*(\d{4})[a-z]?\)"
)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    ok: bool
    sources: List[SourceRecord] = field(default_factory=list)
    error: str = ""
    query: str = ""
    provider: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "provider": self.provider,
            "query": self.query,
            "error": self.error,
            "sources": [s.to_dict() for s in self.sources],
        }


class SourceRetriever:
    """Interface for a scholarly index. Implementations must never invent records."""

    name = "none"
    verifies = False

    def available(self) -> bool:
        return self.verifies

    def search(self, query: str, limit: int = 10) -> RetrievalResult:
        return RetrievalResult(False, [], "source retrieval is not implemented",
                               query=query, provider=self.name)

    def lookup_doi(self, doi: str) -> RetrievalResult:
        return RetrievalResult(False, [], "DOI lookup is not implemented",
                               query=doi, provider=self.name)


class NullSourceRetriever(SourceRetriever):
    """Explicitly disabled retrieval. Failure is reported, never papered over."""

    name = "disabled"
    verifies = False

    def search(self, query: str, limit: int = 10) -> RetrievalResult:
        return RetrievalResult(False, [], "scholarly source retrieval is disabled "
                               "for this run", query=query, provider=self.name)

    def lookup_doi(self, doi: str) -> RetrievalResult:
        return RetrievalResult(False, [], "scholarly source retrieval is disabled "
                               "for this run", query=doi, provider=self.name)


class StaticSourceRetriever(SourceRetriever):
    """Local index used for tests and for deployments without network access.

    Records are treated as index-verified *only* when the caller marks them so;
    otherwise they stay ``user-provided`` and the gate reports them as unverified.
    """

    name = "local-index"
    verifies = True

    def __init__(self, records: Sequence[SourceRecord], *, mark_verified: bool = True,
                 name: str = "local-index") -> None:
        self.name = name
        self.records = [SourceRecord.from_dict(r.to_dict()) for r in records]
        for record in self.records:
            if mark_verified and record.verification != "user-provided":
                record.verification = "verified-index"
                record.retrieved_via = record.retrieved_via or name

    def available(self) -> bool:
        return True

    def search(self, query: str, limit: int = 10) -> RetrievalResult:
        tokens = {t for t in re.findall(r"[a-z]{3,}", query.lower())}
        scored: List[Tuple[int, SourceRecord]] = []
        for record in self.records:
            haystack = f"{record.title} {record.abstract} {record.venue}".lower()
            score = sum(1 for token in tokens if token in haystack)
            if score:
                scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].title))
        return RetrievalResult(True, [r for _, r in scored[:limit]],
                               query=query, provider=self.name)

    def lookup_doi(self, doi: str) -> RetrievalResult:
        wanted = normalise_doi(doi)
        for record in self.records:
            if record.doi and normalise_doi(record.doi) == wanted:
                return RetrievalResult(True, [record], query=doi, provider=self.name)
        return RetrievalResult(False, [], f"DOI {doi} not found in {self.name}",
                               query=doi, provider=self.name)


class CrossrefRetriever(SourceRetriever):
    """Crossref REST API client (standard library only).

    Network problems, HTTP errors and malformed responses are surfaced as failed
    ``RetrievalResult`` objects with a message for the user. They never produce a
    synthetic record.
    """

    name = "crossref"
    verifies = True

    def __init__(self, timeout: int = DEFAULT_RETRIEVAL_TIMEOUT,
                 mailto: Optional[str] = None) -> None:
        self.timeout = max(1, int(timeout))
        self.mailto = mailto or os.environ.get("ARTICLEGEN_CROSSREF_MAILTO", "")

    def _user_agent(self) -> str:
        suffix = f" (mailto:{self.mailto})" if self.mailto else ""
        return f"ArticleGen/0.1 scholarly-retrieval{suffix}"

    def _get(self, url: str) -> Dict[str, Any]:
        request = urllib.request.Request(url, headers={
            "User-Agent": self._user_agent(),
            "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Crossref HTTP {exc.code} for {url}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Crossref network error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Crossref request timed out after {self.timeout}s"
            ) from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Crossref returned a non-JSON response") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Crossref returned an unexpected payload shape")
        return payload

    def search(self, query: str, limit: int = 10) -> RetrievalResult:
        params = {
            "query.bibliographic": query,
            "rows": str(max(1, min(50, limit))),
            "select": "DOI,title,author,issued,container-title,publisher,type,abstract,URL",
        }
        url = CROSSREF_BASE + "?" + urllib.parse.urlencode(params)
        try:
            payload = self._get(url)
        except RuntimeError as exc:
            return RetrievalResult(False, [], str(exc), query=query, provider=self.name)
        items = ((payload.get("message") or {}).get("items") or [])
        records = [self._record_from_item(item) for item in items if isinstance(item, dict)]
        return RetrievalResult(True, [r for r in records if r.title],
                               query=query, provider=self.name)

    def lookup_doi(self, doi: str) -> RetrievalResult:
        token = normalise_doi(doi)
        if not token:
            return RetrievalResult(False, [], "empty DOI", query=doi, provider=self.name)
        url = f"{CROSSREF_BASE}/{urllib.parse.quote(token)}"
        try:
            payload = self._get(url)
        except RuntimeError as exc:
            return RetrievalResult(False, [], str(exc), query=doi, provider=self.name)
        item = payload.get("message") or {}
        if not isinstance(item, dict):
            return RetrievalResult(False, [], "Crossref returned no work record",
                                   query=doi, provider=self.name)
        record = self._record_from_item(item)
        record.verification = "verified-index"
        record.retrieved_via = self.name
        return RetrievalResult(True, [record], query=doi, provider=self.name)

    def _record_from_item(self, item: Dict[str, Any]) -> SourceRecord:
        titles = item.get("title") or []
        title = str(titles[0]) if isinstance(titles, list) and titles else str(titles or "")
        venues = item.get("container-title") or []
        venue = str(venues[0]) if isinstance(venues, list) and venues else str(venues or "")
        authors: List[str] = []
        for author in item.get("author") or []:
            if not isinstance(author, dict):
                continue
            given = str(author.get("given") or "").strip()
            family = str(author.get("family") or "").strip()
            name = f"{given} {family}".strip() or str(author.get("name") or "").strip()
            if name:
                authors.append(name)
        year = None
        issued = item.get("issued") or {}
        parts = (issued.get("date-parts") or [[None]])[0] if isinstance(issued, dict) else []
        if parts:
            try:
                year = int(parts[0])
            except (TypeError, ValueError):
                year = None
        doi = normalise_doi(str(item.get("DOI") or ""))
        return SourceRecord(
            title=title.strip(),
            authors=authors,
            year=year,
            venue=venue.strip(),
            doi=doi,
            url=str(item.get("URL") or (f"https://doi.org/{doi}" if doi else "")).strip(),
            publisher=str(item.get("publisher") or "").strip(),
            source_type=str(item.get("type") or "journal-article"),
            retrieved_via=self.name,
            verification="verified-index",
            abstract=re.sub(r"<[^>]+>", "", str(item.get("abstract") or "")).strip(),
        )


class UserSourceRetriever(SourceRetriever):
    """Retriever for records the user supplied with the brief.

    It can *find* those records, but it cannot verify them against an external
    index, so it reports ``verifies = False``.
    """

    name = "user-provided"
    verifies = False

    def __init__(self, records: Sequence[SourceRecord]) -> None:
        self.records = list(records)

    def available(self) -> bool:
        return True

    def search(self, query: str, limit: int = 10) -> RetrievalResult:
        tokens = {t for t in re.findall(r"[a-z]{3,}", query.lower())}
        hits = [r for r in self.records
                if any(token in f"{r.title} {r.abstract}".lower() for token in tokens)]
        return RetrievalResult(True, hits[:limit], query=query, provider=self.name)

    def lookup_doi(self, doi: str) -> RetrievalResult:
        wanted = normalise_doi(doi)
        hits = [r for r in self.records if normalise_doi(r.doi) == wanted]
        return RetrievalResult(bool(hits), hits, query=doi, provider=self.name)


def resolve_retriever(policy: str = "auto", *, timeout: int = DEFAULT_RETRIEVAL_TIMEOUT,
                      user_sources: Optional[Sequence[SourceRecord]] = None
                      ) -> SourceRetriever:
    """Build a retriever from a brief/config policy string."""
    token = (policy or "auto").strip().lower()
    if token in ("none", "disabled", "off", "local-only"):
        return NullSourceRetriever()
    if token in ("user", "user-provided", "brief"):
        return UserSourceRetriever(user_sources or [])
    if token in ("static", "fixture"):
        return StaticSourceRetriever(user_sources or [], mark_verified=False,
                                     name="fixture-index")
    if token in ("crossref", "auto", ""):
        return CrossrefRetriever(timeout=timeout)
    raise ValueError(f"unknown source_retrieval policy '{policy}'")


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


@dataclass
class RetrievalOutcome:
    attempted: bool = False
    provider: str = ""
    queries: List[str] = field(default_factory=list)
    retrieved: int = 0
    errors: List[str] = field(default_factory=list)
    available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempted": self.attempted,
            "provider": self.provider,
            "queries": list(self.queries),
            "retrieved": self.retrieved,
            "errors": list(self.errors),
            "available": self.available,
        }


@dataclass
class SourceVerificationReport:
    total: int = 0
    verified: int = 0
    user_provided: int = 0
    unverified: int = 0
    metadata_incomplete: int = 0
    unusable: int = 0
    retrieval: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def usable(self) -> int:
        return self.verified + self.user_provided

    @property
    def complete(self) -> bool:
        """True only when every usable record was verified against an index."""
        return self.total > 0 and self.verified == self.total and not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "verified": self.verified,
            "user_provided": self.user_provided,
            "unverified": self.unverified,
            "metadata_incomplete": self.metadata_incomplete,
            "unusable": self.unusable,
            "complete": self.complete,
            "retrieval": self.retrieval,
            "errors": list(self.errors),
            "notes": list(self.notes),
        }


def _surname(name: str) -> str:
    tokens = [t for t in re.split(r"\s+", name.strip()) if t]
    return tokens[-1] if tokens else name.strip()


def _initials(name: str) -> str:
    tokens = [t for t in re.split(r"\s+", name.strip()) if t]
    if len(tokens) <= 1:
        return ""
    letters = []
    for token in tokens[:-1]:
        cleaned = re.sub(r"[^A-Za-zÀ-ÿ]", "", token)
        if cleaned:
            letters.append(cleaned[0].upper() + ".")
    return " ".join(letters)


def _normalise_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", title.lower()).strip()


def _tokens(text: str) -> set:
    return {t for t in _normalise_title(text).split() if len(t) > 2}


class SourceLibrary:
    """Ordered, deduplicated source records plus claim links and bibliography."""

    def __init__(self, style: CitationStyle = CitationStyle.APA) -> None:
        self.style = style
        self.records: List[SourceRecord] = []
        self.retrieval = RetrievalOutcome()
        self.errors: List[str] = []
        self.notes: List[str] = []
        self._counter = 0

    # -- ingestion --------------------------------------------------------
    def add(self, record: SourceRecord) -> SourceRecord:
        """Add or merge a record. Duplicate DOIs/titles do not create fake entries."""
        incoming = SourceRecord.from_dict(record.to_dict())
        existing = self.find(incoming)
        if existing is not None:
            self._merge(existing, incoming)
            return existing
        self._counter += 1
        incoming.source_id = incoming.source_id or f"S{self._counter}"
        self.records.append(incoming)
        return incoming

    def _merge(self, existing: SourceRecord, incoming: SourceRecord) -> None:
        for attribute in ("title", "venue", "publisher", "url", "abstract"):
            if not getattr(existing, attribute) and getattr(incoming, attribute):
                setattr(existing, attribute, getattr(incoming, attribute))
        if not existing.authors and incoming.authors:
            existing.authors = list(incoming.authors)
        if existing.year is None and incoming.year is not None:
            existing.year = incoming.year
        if not existing.doi and incoming.doi:
            existing.doi = incoming.doi
        if (incoming.verification == "verified-index"
                and existing.verification != "verified-index"):
            existing.verification = incoming.verification
            existing.retrieved_via = incoming.retrieved_via
        for claim_id in incoming.supports:
            if claim_id not in existing.supports:
                existing.supports.append(claim_id)
        for note in incoming.notes:
            if note not in existing.notes:
                existing.notes.append(note)

    def find(self, record: SourceRecord) -> Optional[SourceRecord]:
        doi = normalise_doi(record.doi)
        title = _normalise_title(record.title)
        for existing in self.records:
            if doi and normalise_doi(existing.doi) == doi:
                return existing
            if title and _normalise_title(existing.title) == title and \
                    existing.year in (None, record.year, record.year is None):
                return existing
        return None

    def get(self, source_id: str) -> Optional[SourceRecord]:
        return next((r for r in self.records if r.source_id == source_id), None)

    def from_brief_sources(self, raw_sources: Iterable[Dict[str, Any]]) -> None:
        for item in raw_sources:
            record = SourceRecord.from_dict(item)
            if not record.retrieved_via or record.retrieved_via == "none":
                record.retrieved_via = "user-provided"
            if record.verification == "unverified":
                record.verification = "user-provided"
            record.notes.append("supplied by the user with the brief; not index-verified")
            self.add(record)

    # -- retrieval and verification --------------------------------------
    def retrieve(self, queries: Sequence[str], retriever: SourceRetriever,
                 *, limit_per_query: int = 5, max_sources: int = 20) -> RetrievalOutcome:
        outcome = RetrievalOutcome(provider=retriever.name,
                                   available=bool(retriever.available()))
        for query in [q for q in dict.fromkeys(queries) if q and q.strip()]:
            if len(self.records) >= max_sources:
                break
            outcome.attempted = True
            outcome.queries.append(query)
            try:
                result = retriever.search(query, limit=limit_per_query)
            except Exception as exc:  # noqa: BLE001 - a provider outage is data, not a crash
                outcome.errors.append(
                    f"{retriever.name} search failed: {type(exc).__name__}: {exc}")
                continue
            if not result.ok:
                outcome.errors.append(result.error or f"{retriever.name} search failed")
                continue
            for record in result.sources:
                if len(self.records) >= max_sources:
                    break
                if result.provider:
                    record.retrieved_via = result.provider
                    if retriever.verifies:
                        record.verification = "verified-index"
                self.add(record)
                outcome.retrieved += 1
        self.retrieval = outcome
        self.errors.extend(outcome.errors)
        return outcome

    def verify(self, retriever: Optional[SourceRetriever] = None) -> SourceVerificationReport:
        """Verify every record against an index; never rewrite a record into one."""
        if retriever is None:
            retriever = NullSourceRetriever()
        errors: List[str] = []
        for record in self.records:
            if record.verification == "verified-index":
                continue
            if not retriever.available():
                record.verification = ("user-provided" if record.retrieved_via == "user-provided"
                                       else "unverified")
                errors.append(
                    f"{record.source_id or record.title[:40]}: cannot verify against an "
                    f"index ({retriever.name} unavailable); record remains unverified"
                )
                continue
            confirmation = None
            if record.doi:
                result = retriever.lookup_doi(record.doi)
                if result.ok and result.sources:
                    confirmation = result.sources[0]
                elif result.error:
                    errors.append(f"DOI lookup failed for {record.doi}: {result.error}")
            if confirmation is None and record.title:
                result = retriever.search(record.title, limit=3)
                if result.ok:
                    confirmation = self._match_search_result(record, result.sources)
                elif result.error:
                    errors.append(f"title search failed for {record.title[:40]}: {result.error}")
            if confirmation is None:
                record.verification = "unverified"
                record.notes.append("not found in the configured scholarly index")
                continue
            self._merge(record, confirmation)
            record.verification = "verified-index"
            record.retrieved_via = retriever.name
            if not record.doi and confirmation.doi:
                record.doi = confirmation.doi
            if not record.url and confirmation.url:
                record.url = confirmation.url
        report = self.report(retriever)
        report.errors = errors
        return report

    @staticmethod
    def _match_search_result(record: SourceRecord, candidates: Sequence[SourceRecord]
                             ) -> Optional[SourceRecord]:
        wanted = _tokens(record.title)
        if not wanted:
            return None
        for candidate in candidates:
            overlap = wanted & _tokens(candidate.title)
            if len(overlap) / max(1, len(wanted)) >= 0.8:
                if record.year and candidate.year and abs(record.year - candidate.year) > 1:
                    continue
                return candidate
        return None

    def report(self, retriever: Optional[SourceRetriever] = None) -> SourceVerificationReport:
        report = SourceVerificationReport(total=len(self.records))
        for record in self.records:
            if record.verification == "verified-index":
                report.verified += 1
            elif record.verification == "user-provided":
                report.user_provided += 1
            elif record.metadata_issues():
                report.metadata_incomplete += 1
            else:
                report.unverified += 1
            if not record.is_usable:
                report.unusable += 1
        report.retrieval = dict(self.retrieval.to_dict())
        report.errors = list(self.errors)
        report.notes = list(self.notes)
        if retriever is not None:
            report.retrieval.setdefault("provider", retriever.name)
        return report

    # -- bibliographic output --------------------------------------------
    def ordered_records(self) -> List[SourceRecord]:
        usable = [r for r in self.records if r.is_usable]
        if self.style == CitationStyle.APA:
            return sorted(usable, key=lambda r: (_surname(r.authors[0]) if r.authors else "",
                                                 r.year or 0, r.title.lower()))
        return list(usable)

    def bibliography(self, style: Optional[CitationStyle] = None) -> str:
        style = style or self.style
        records = self.ordered_records()
        if not records:
            return ""
        lines: List[str] = []
        for index, record in enumerate(records, start=1):
            lines.append(format_bibliography_entry(record, style, index))
        return "\n\n".join(lines)

    def numbered_index(self) -> Dict[int, SourceRecord]:
        return {i: record for i, record in enumerate(self.ordered_records(), start=1)}

    def citation_key_for(self, record: SourceRecord) -> str:
        records = self.ordered_records()
        if self.style == CitationStyle.APA:
            return record.citation_key
        for index, candidate in enumerate(records, start=1):
            if candidate is record:
                return f"[{index}]"
        return record.citation_key

    # -- resolution -------------------------------------------------------
    def resolve_citations(self, draft_text: str) -> "CitationResolution":
        body, references_text = split_references(draft_text)
        # Fenced code is not prose: an array literal such as ``[4, 8, 16, 32]`` is
        # not a set of numeric citations.
        body = re.sub(r"```.*?```", " ", body, flags=re.DOTALL)
        resolution = CitationResolution()
        index = self.numbered_index()
        records = list(index.values())
        resolution.bibliography_entries = len(index)

        for match in _CITATION_NUMBER_RE.finditer(body):
            for number in _expand_numbers(match.group(1)):
                resolution.numeric_markers.append(number)
                if number in index:
                    resolution.resolved.append(("numeric", number, index[number].source_id))
                else:
                    resolution.unresolved.append(
                        f"citation [{number}] has no bibliography entry"
                    )
        for match in _AUTHOR_YEAR_IN_TEXT_RE.finditer(body):
            surname, year = match.group(1), int(match.group(2))
            resolution.author_year_markers.append(f"{surname} {year}")
            hit = next((r for r in records if r.authors and r.year == year
                        and _surname(r.authors[0]).lower() == surname.lower()), None)
            if hit is not None:
                resolution.resolved.append(("author_year", f"{surname} {year}", hit.source_id))
            else:
                resolution.unresolved.append(
                    f"citation ({surname}, {year}) matches no verified source"
                )

        if (self.style != CitationStyle.APA
                and _AUTHOR_YEAR_IN_TEXT_RE.search(body)
                and not resolution.numeric_markers):
            resolution.notes.append(
                f"{self.style.value.upper()} citations should use numbered markers; "
                "author-year markers were found without any numbered citation"
            )

        entries = parse_bibliography_entries(references_text)
        resolution.parsed_entries = [text for _, text in entries]
        for _, entry_text in entries:
            matched = self._match_entry(entry_text, records)
            if matched is None:
                resolution.fabricated_entries.append(entry_text[:160])
            else:
                resolution.matched_entries.append(matched.source_id)
        if not references_text.strip() and (resolution.numeric_markers
                                            or resolution.author_year_markers):
            resolution.unresolved.append(
                "in-text citations exist but the manuscript has no References section"
            )
        used = {source_id for _, _, source_id in resolution.resolved}
        resolution.unused_entries = [
            f"[{number}] {record.title[:60]}"
            for number, record in index.items() if record.source_id not in used
        ]
        return resolution

    @staticmethod
    def _match_entry(entry_text: str, records: Sequence[SourceRecord]) -> Optional[SourceRecord]:
        doi_match = re.search(r"10\.\d{4,9}/\S+", entry_text)
        if doi_match:
            doi = normalise_doi(doi_match.group(0))
            for record in records:
                if record.doi and normalise_doi(record.doi) == doi:
                    return record
        entry_tokens = _tokens(entry_text)
        best: Optional[Tuple[float, SourceRecord]] = None
        for record in records:
            title_tokens = _tokens(record.title)
            if not title_tokens:
                continue
            overlap = len(entry_tokens & title_tokens) / len(title_tokens)
            year_ok = bool(re.search(rf"\b{record.year}\b", entry_text)) if record.year else True
            score = overlap + (0.25 if year_ok else 0)
            if best is None or score > best[0]:
                best = (score, record)
        if best and best[0] >= 0.75:
            return best[1]
        return None

    # -- context ----------------------------------------------------------
    def context_block(self, max_records: int = 12, max_abstract_chars: int = 320) -> str:
        """Compact verified evidence for the Writer/Reviewer context packet."""
        records = self.ordered_records()[:max_records]
        if not records:
            return ("no verified sources available - do not cite anything; label the "
                    "manuscript as unverified")
        lines = [
            "Only these verified sources may be cited. Use the exact key shown.",
            "Never invent a DOI, author, venue, page range or quotation.",
        ]
        for index, record in enumerate(records, start=1):
            authors = ", ".join(record.authors[:3])
            if len(record.authors) > 3:
                authors += ", et al."
            abstract = re.sub(r"\s+", " ", record.abstract)[:max_abstract_chars]
            key = f"[{index}]" if self.style != CitationStyle.APA else record.citation_key
            lines.append(
                f"- key={key} source_id={record.source_id} | {authors} ({record.year}). "
                f"{record.title}. {record.venue}."
                + (f" doi:{record.doi}" if record.doi else "")
                + (f" | indexed abstract: {abstract}" if abstract else "")
            )
        return "\n".join(lines)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "style": self.style.value,
            "records": [r.to_dict() for r in self.records],
            "retrieval": self.retrieval.to_dict(),
            "errors": list(self.errors),
            "notes": list(self.notes),
            "bibliography": self.bibliography(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceLibrary":
        try:
            style = CitationStyle.parse(data.get("style", "apa"))
        except ValueError:
            style = CitationStyle.APA
        library = cls(style)
        for item in data.get("records") or []:
            library.add(SourceRecord.from_dict(item))
        retrieval = data.get("retrieval") or {}
        library.retrieval = RetrievalOutcome(
            attempted=bool(retrieval.get("attempted")),
            provider=str(retrieval.get("provider") or ""),
            queries=list(retrieval.get("queries") or []),
            retrieved=int(retrieval.get("retrieved") or 0),
            errors=list(retrieval.get("errors") or []),
            available=bool(retrieval.get("available")),
        )
        library.errors = list(data.get("errors") or [])
        library.notes = list(data.get("notes") or [])
        return library


@dataclass
class CitationResolution:
    numeric_markers: List[int] = field(default_factory=list)
    author_year_markers: List[str] = field(default_factory=list)
    resolved: List[Tuple[str, Any, str]] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    unused_entries: List[str] = field(default_factory=list)
    fabricated_entries: List[str] = field(default_factory=list)
    matched_entries: List[str] = field(default_factory=list)
    parsed_entries: List[str] = field(default_factory=list)
    bibliography_entries: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def has_citations(self) -> bool:
        return bool(self.numeric_markers or self.author_year_markers)

    @property
    def ok(self) -> bool:
        return not self.unresolved and not self.fabricated_entries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numeric_markers": list(self.numeric_markers),
            "author_year_markers": list(self.author_year_markers),
            "resolved": [{"kind": k, "marker": m, "source_id": s} for k, m, s in self.resolved],
            "unresolved": list(self.unresolved),
            "unused_entries": list(self.unused_entries),
            "fabricated_entries": list(self.fabricated_entries),
            "matched_entries": list(self.matched_entries),
            "bibliography_entries": self.bibliography_entries,
            "ok": self.ok,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expand_numbers(marker: str) -> List[int]:
    numbers: List[int] = []
    for part in marker.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-", 1)
            try:
                start, end = int(bounds[0]), int(bounds[1])
            except ValueError:
                continue
            if 0 < end - start <= 100:
                numbers.extend(range(start, end + 1))
        else:
            try:
                numbers.append(int(part))
            except ValueError:
                continue
    return numbers


def split_references(markdown: str) -> Tuple[str, str]:
    """Split a manuscript into body and reference-list text."""
    matches = list(_HEADING_RE.finditer(markdown or ""))
    if not matches:
        return markdown or "", ""
    match = matches[-1]
    return markdown[:match.start()], markdown[match.end():]


def parse_bibliography_entries(references_text: str) -> List[Tuple[int, str]]:
    """Parse either ``[n] entry`` or plain numbered / paragraph entries."""
    text = (references_text or "").strip()
    if not text:
        return []
    numbered = [(int(m.group(1)), re.sub(r"\s+", " ", m.group(2)).strip())
                for m in _NUMBERED_ENTRY_RE.finditer(text)]
    if numbered:
        return numbered
    plain = [(int(m.group(1)), re.sub(r"\s+", " ", m.group(2)).strip())
             for m in _PLAIN_NUMBERED_RE.finditer(text)]
    if plain:
        return plain
    paragraphs = [re.sub(r"\s+", " ", block).strip()
                  for block in re.split(r"\n\s*\n", text) if block.strip()]
    return [(index, block) for index, block in enumerate(paragraphs, start=1)]


def _apa_authors(authors: Sequence[str]) -> str:
    formatted: List[str] = []
    for author in authors:
        surname = _surname(author)
        initials = _initials(author)
        formatted.append(f"{surname}, {initials}".rstrip(", "))
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"


def _ieee_authors(authors: Sequence[str]) -> str:
    names = []
    for author in authors:
        initials = _initials(author)
        surname = _surname(author)
        names.append(f"{initials} {surname}".strip())
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _vancouver_authors(authors: Sequence[str]) -> str:
    names = []
    for author in authors:
        initials = _initials(author).replace(". ", "").replace(".", "")
        names.append(f"{_surname(author)} {initials}".strip())
    return ", ".join(names)


def format_bibliography_entry(record: SourceRecord, style: CitationStyle,
                              index: int = 1) -> str:
    year = record.year or "n.d."
    title = record.title.strip().rstrip(".")
    venue = record.venue.strip()
    doi = normalise_doi(record.doi)
    link = f"https://doi.org/{doi}" if doi else record.url
    authors = record.authors or ["Anonymous"]
    if style == CitationStyle.IEEE:
        parts = [f'[{index}] {_ieee_authors(authors)},', f'"{title},"']
        if venue:
            parts.append(f"{venue},")
        parts.append(f"{year}.")
        if doi:
            parts.append(f"doi: {doi}.")
        elif link:
            parts.append(f"{link}.")
        return " ".join(parts)
    if style == CitationStyle.VANCOUVER:
        parts = [f"{index}. {_vancouver_authors(authors)}. {title}."]
        if venue:
            parts.append(f"{venue}.")
        parts.append(f"{year}.")
        if doi:
            parts.append(f"doi:{doi}")
        elif link:
            parts.append(link)
        return " ".join(parts).strip()
    if style == CitationStyle.PLAIN:
        parts = [f"[{index}] {_apa_authors(authors)} ({year}). {title}."]
        if venue:
            parts.append(f"{venue}.")
        if link:
            parts.append(link)
        return " ".join(parts)
    # APA
    parts = [f"{_apa_authors(authors)} ({year}). {title}."]
    if venue:
        parts.append(f"{venue}.")
    if link:
        parts.append(link)
    return " ".join(parts)


def citation_issue_severity(resolution: CitationResolution) -> ReviewSeverity:
    if resolution.fabricated_entries or resolution.unresolved:
        return ReviewSeverity.CRITICAL
    if resolution.unused_entries:
        return ReviewSeverity.MINOR
    return ReviewSeverity.MINOR
