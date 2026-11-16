"""Generate a structured abstract, title options and keywords from a manuscript.

Every generated number and proper name is checked against the source text before it
is returned ("fact lock"). A sentence that introduces an ungrounded fact is dropped
and reported; a title or keyword is rejected the same way. Nothing is silently
"fixed" and the source is never rewritten, so a wrong summary cannot quietly replace
a right article.

Fact-lock limits: numbers are checked for any script. Proper-name detection is
capitalisation based, so it applies to Latin-script names; a Persian or Arabic name
has no case marking and is not checked as a name (its numbers still are). A single
capitalised word at the start of a sentence is treated as sentence case, so only
multi-word names and acronyms are locked there. The report states these limits
instead of implying full verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .llm import LLMClient, ProviderError, ResponseParseError, parse_json
from .manuscript import Manuscript, normalize_for_match, word_count
from .prompts import PromptError, load_template

ABSTRACT_SECTIONS: Tuple[Tuple[str, str], ...] = (
    ("background", "Background"),
    ("objective", "Objective"),
    ("methods", "Methods"),
    ("results", "Results"),
    ("conclusions", "Conclusions"),
)

MAX_SOURCE_CHARS = 60_000
MAX_ABSTRACT_WORDS = 350

_NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(%)?")
_CAP_WORD = r"[A-ZÀ-ÖØ-Þ][\wÀ-ÿ'’\-]*"
_NAME_RE = re.compile(rf"(?<![\w]){_CAP_WORD}(?:\s+{_CAP_WORD})*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

#: Capitalised words that are ordinary prose (sentence starters, section labels), not
#: candidate proper names. Without this list every sentence would look like a name.
_NON_NAMES = frozenset({
    "the", "this", "these", "that", "those", "a", "an", "we", "our", "its", "it",
    "in", "on", "for", "to", "from", "with", "by", "as", "at", "however", "although",
    "while", "using", "here", "finally", "overall", "therefore", "thus", "first",
    "second", "third", "background", "objective", "objectives", "methods", "method",
    "results", "conclusions", "conclusion", "study", "studies", "data", "analysis",
    "article", "paper", "manuscript", "author", "authors", "purpose", "aim", "aims",
    "limitations", "reliability", "performance", "accuracy", "efficiency", "safety",
    "quality", "cost", "findings", "evidence", "participants", "users", "systems",
    "models", "further", "future", "additionally", "moreover", "consequently",
})


class AbstractError(ValueError):
    """The source, provider response, or fact lock made an output unsafe to return."""


@dataclass
class FactViolation:
    field: str
    kind: str  # "number" | "name"
    token: str
    sentence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "kind": self.kind, "token": self.token,
                "sentence": self.sentence}


@dataclass
class AbstractPackage:
    source: str
    title: str
    abstract: Dict[str, str] = field(default_factory=dict)
    titles: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    violations: List[FactViolation] = field(default_factory=list)
    dropped_sentences: List[Dict[str, Any]] = field(default_factory=list)
    rejected_titles: List[str] = field(default_factory=list)
    rejected_keywords: List[str] = field(default_factory=list)
    engine: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def fact_lock_ok(self) -> bool:
        return not self.violations

    @property
    def abstract_word_count(self) -> int:
        return sum(word_count(text) for text in self.abstract.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "fact_lock_ok": self.fact_lock_ok,
            "abstract": dict(self.abstract),
            "abstract_word_count": self.abstract_word_count,
            "titles": list(self.titles),
            "keywords": list(self.keywords),
            "violations": [violation.to_dict() for violation in self.violations],
            "dropped_sentences": list(self.dropped_sentences),
            "rejected_titles": list(self.rejected_titles),
            "rejected_keywords": list(self.rejected_keywords),
            "engine": self.engine,
            "notes": list(self.notes),
            "limitations": [
                "Only the supplied manuscript was used; no external facts were checked.",
                "Numbers are checked in any script; proper-name locking is "
                "capitalisation-based and covers Latin-script names only. A single "
                "capitalised word at the start of a sentence is treated as sentence "
                "case, so only multi-word names and acronyms are locked there.",
                "A passed fact lock means the generated text introduced no ungrounded "
                "number or name; it is not a guarantee that every claim is correct.",
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Abstract package: {self.title or 'Untitled manuscript'}",
            "",
            f"**Source:** `{self.source or '(text)'}` · "
            f"**Fact lock:** {'PASSED' if self.fact_lock_ok else 'FAILED'} · "
            f"**Abstract words:** {self.abstract_word_count}",
            "",
            "## Structured abstract",
            "",
        ]
        for key, label in ABSTRACT_SECTIONS:
            if key not in self.abstract:
                continue
            lines.extend([f"### {label}", "", self.abstract[key], ""])
        if not self.abstract:
            lines.extend(["_No section survived the fact lock._", ""])

        lines.extend(["## Title options", ""])
        lines.extend([f"{index}. {title}" for index, title in enumerate(self.titles, 1)]
                     or ["_No title passed the fact lock._"])
        lines.append("")

        lines.extend(["## Keywords", ""])
        lines.append(", ".join(f"`{keyword}`" for keyword in self.keywords)
                     or "_No keyword passed the fact lock._")
        lines.append("")

        lines.extend(["## Fact lock", ""])
        if self.fact_lock_ok:
            lines.extend(["- Every number and Latin-script proper name in the output "
                          "appears in the source text.", ""])
        else:
            lines.extend(["| Field | Kind | Token | Source sentence |",
                          "|---|---|---|---|"])
            for violation in self.violations:
                sentence = violation.sentence.replace("|", "\\|").replace("\n", " ")
                if len(sentence) > 160:
                    sentence = sentence[:157] + "..."
                lines.append(f"| {violation.field} | {violation.kind} | "
                             f"`{violation.token}` | {sentence} |")
            lines.append("")
            if self.dropped_sentences:
                lines.extend(["**Quarantined sentences**", ""])
                for item in self.dropped_sentences:
                    lines.append(f"- [{item['field']}] {item['sentence']}")
                lines.append("")
            if self.rejected_titles:
                lines.extend(["**Rejected titles**", ""])
                lines.extend([f"- {title}" for title in self.rejected_titles])
                lines.append("")
            if self.rejected_keywords:
                lines.extend(["**Rejected keywords**", ""])
                lines.extend([f"- {keyword}" for keyword in self.rejected_keywords])
                lines.append("")

        if self.notes:
            lines.extend(["## Notes", ""])
            lines.extend([f"- {note}" for note in self.notes])
            lines.append("")

        lines.extend([
            "## What this does not do", "",
            "- It does not check facts against external sources; only against the "
            "supplied text.",
            "- Name locking covers Latin-script capitalised names; numbers are checked "
            "in any script. Sentence-initial single words are treated as sentence "
            "case; multi-word names and acronyms are locked there.",
            "- A passed lock means no ungrounded number or name was introduced, not "
            "that every claim is correct.",
            "",
        ])
        return "\n".join(lines)


def _normalise(text: str) -> str:
    value = normalize_for_match(text)
    return re.sub(r"\s+", " ", value.replace("–", "-").replace("—", "-")
                  .replace("’", "'"))


def _numbers(text: str) -> List[str]:
    tokens: List[str] = []
    for match in _NUMBER_RE.finditer(text or ""):
        value = match.group(1).replace(",", "")
        token = f"{value}%" if match.group(2) else value
        if token not in tokens:
            tokens.append(token)
    return tokens


def _sentence_starts(text: str) -> set:
    starts = {0}
    for match in _SENTENCE_RE.finditer(text or ""):
        starts.add(match.end())
    return starts


def _names(text: str) -> List[str]:
    names: List[str] = []
    starts = _sentence_starts(text)
    for match in _NAME_RE.finditer(text or ""):
        words = [word for word in match.group(0).split()
                 if _normalise(word).strip(".,:;") not in _NON_NAMES]
        if len(words) == 1 and match.start() in starts and not words[0].isupper():
            # Sentence-case capitalisation is grammar, not evidence of a proper name.
            continue
        candidate = re.sub(r"[.,:;]+$", "", " ".join(words)).strip()
        candidate = re.sub(r"'s$", "", candidate)
        if not candidate or _normalise(candidate) in _NON_NAMES:
            continue
        if candidate not in names:
            names.append(candidate)
    return names


def unsupported_facts(text: str, source: str) -> List[Tuple[str, str]]:
    """Return the numbers and names in ``text`` that the source does not contain."""
    source_numbers = set(_numbers(source))
    source_names = _normalise(source)
    found: List[Tuple[str, str]] = []
    for token in _numbers(text):
        if token not in source_numbers:
            found.append(("number", token))
    for name in _names(text):
        if _normalise(name) not in source_names:
            found.append(("name", name))
    return found


def _record(violations: List[FactViolation], seen: set, field_name: str, kind: str,
            token: str, sentence: str = "") -> None:
    key = (field_name, kind, token)
    if key in seen:
        return
    seen.add(key)
    violations.append(FactViolation(field_name, kind, token, sentence[:300]))


def _lock_sentences(text: str, field_name: str, source: str,
                    violations: List[FactViolation], seen: set,
                    dropped: List[Dict[str, Any]]) -> str:
    kept: List[str] = []
    for sentence in _SENTENCE_RE.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        bad = unsupported_facts(sentence, source)
        if bad:
            for kind, token in bad:
                _record(violations, seen, field_name, kind, token, sentence)
            dropped.append({"field": field_name, "sentence": sentence[:300],
                            "tokens": [token for _, token in bad]})
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


def _lock_value(value: str, field_name: str, source: str,
                violations: List[FactViolation], seen: set) -> Optional[str]:
    bad = unsupported_facts(value, source)
    for kind, token in bad:
        _record(violations, seen, field_name, kind, token, value)
    return None if bad else value


def _source_text(manuscript: Manuscript) -> str:
    # The document title is part of the article even though it is not a section body;
    # without it a generated title that reuses the paper's own name is rejected.
    parts = [manuscript.title] if manuscript.title else []
    parts.extend(section.body for section in manuscript.sections)
    body = "\n".join(parts).strip()
    return body or manuscript.text


def _user_prompt(manuscript: Manuscript, source: str, *, article_type: Optional[str],
                 audience: Optional[str], max_titles: int) -> str:
    headings = ", ".join(section.title for section in manuscript.sections) or "(none)"
    stats = manuscript.stats()
    return "\n".join([
        f"TITLE: {manuscript.title or '(not stated)'}",
        f"ARTICLE TYPE: {article_type or 'not specified'}",
        f"INTENDED AUDIENCE: {audience or 'not specified'}",
        f"STRUCTURE: {headings}",
        f"LENGTH: {stats['words']} words",
        f"TITLE OPTIONS REQUESTED: {max_titles}",
        "Every number and proper name you use must appear verbatim in the source "
        "below. Do not introduce a year, percentage, sample size, institution or "
        "product name that is not there.",
        "",
        "SOURCE (untrusted content begins)",
        source[:MAX_SOURCE_CHARS],
        "SOURCE (untrusted content ends)",
        "",
        "Return only the JSON object specified by the system prompt.",
    ])


def generate_abstract(manuscript: Manuscript, *, client: Optional[LLMClient],
                      article_type: Optional[str] = None,
                      audience: Optional[str] = None, max_titles: int = 5,
                      prompt_dir: Optional[str] = None) -> AbstractPackage:
    if client is None or not getattr(client, "available", False):
        raise AbstractError(
            "abstract generation needs an LLM API key: set ANTHROPIC_API_KEY, "
            "OPENAI_API_KEY or DAHL_API_KEY, or pass --provider/--model.")

    source = _source_text(manuscript)
    if word_count(source) < 30:
        raise AbstractError("the source text is too short to summarise safely")

    try:
        system = load_template("abstract_writer", prompt_dir)
    except PromptError as exc:
        raise AbstractError(f"abstract prompt unavailable: {exc}") from exc

    try:
        response = client.complete(
            system,
            _user_prompt(manuscript, source, article_type=article_type,
                         audience=audience, max_titles=max_titles),
            max_output_tokens=2048,
        )
        payload = parse_json(response.text)
    except (ProviderError, ResponseParseError, ValueError) as exc:
        raise AbstractError(f"abstract generation failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface, never crash the command
        raise AbstractError(
            f"abstract generation failed: {type(exc).__name__}: {exc}") from exc

    if not isinstance(payload, dict):
        raise AbstractError("the model returned no JSON object")
    raw_abstract = payload.get("abstract")
    if not isinstance(raw_abstract, dict):
        raise AbstractError("the model reply is missing the 'abstract' object")

    package = AbstractPackage(
        source=manuscript.source,
        title=manuscript.title or "Untitled manuscript",
        engine=getattr(response, "engine", "") or client.name(),
    )
    violations: List[FactViolation] = []
    seen: set = set()

    for key, _label in ABSTRACT_SECTIONS:
        value = raw_abstract.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        locked = _lock_sentences(value, key, source, violations, seen,
                                 package.dropped_sentences)
        if locked:
            package.abstract[key] = locked
        elif value.strip():
            package.notes.append(f"The {key} section had no sentence that passed the "
                                 "fact lock.")

    if not package.abstract:
        raise AbstractError("no abstract section survived the fact lock; reduce the "
                            "request or supply a longer source")

    raw_titles = payload.get("titles")
    if isinstance(raw_titles, list):
        for item in raw_titles:
            if len(package.titles) >= max(1, max_titles):
                break
            if not isinstance(item, str) or not item.strip():
                continue
            value = item.strip()
            locked = _lock_value(value, "title", source, violations, seen)
            if locked:
                package.titles.append(locked)
            else:
                package.rejected_titles.append(value)

    raw_keywords = payload.get("keywords")
    if isinstance(raw_keywords, list):
        for item in raw_keywords:
            if len(package.keywords) >= 8:
                break
            if not isinstance(item, str) or not item.strip():
                continue
            value = item.strip()
            locked = _lock_value(value, "keyword", source, violations, seen)
            if not locked:
                package.rejected_keywords.append(value)
            elif locked.casefold() not in {k.casefold() for k in package.keywords}:
                package.keywords.append(locked)

    raw_notes = payload.get("notes")
    if isinstance(raw_notes, list):
        package.notes.extend(note.strip() for note in raw_notes
                             if isinstance(note, str) and note.strip())
    package.violations = violations
    if package.abstract_word_count > MAX_ABSTRACT_WORDS:
        package.notes.append(
            f"The structured abstract is {package.abstract_word_count} words; most "
            f"venues cap it near 250-300.")
    if len(source) > MAX_SOURCE_CHARS:
        package.notes.append(
            f"The source was truncated for the model call at {MAX_SOURCE_CHARS} "
            "characters.")
    return package
