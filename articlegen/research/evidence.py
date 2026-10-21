"""Claim / evidence ledger and evidence-language analysis.

The ledger is the bridge between prose and provenance:

* every checkable claim gets a ``ClaimRecord`` whose ``kind`` says whether it is an
  established fact (cited), a result of *this* study (executed), an external finding,
  a hypothesis, an interpretation, an assumption, future work, or unverified;
* ``support_status`` is only ``verified`` when the claim points at a verified source
  or an executed result;
* the deterministic language detectors below find the failure modes that models
  reproduce most often - planned work described in the past tense, hypotheses stated
  as findings, and completion claims with no executed experiment behind them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..models import Draft
from .models import (
    ClaimRecord,
    EvidenceKind,
    ExperimentRun,
    ExperimentStatus,
    ResultRecord,
    SourceRecord,
    classify_section_role,
)
from .sources import SourceLibrary, split_references

#: Sentences that assert something checkable. Prose that does not match any of these
#: is not turned into a claim, so the ledger stays readable instead of indexing every
#: sentence in the article.
CLAIM_CUE_RE = re.compile(
    r"\b(show|shows|shown|demonstrate|demonstrates|found|find|finds|report|reports|"
    r"measure|measured|observe|observed|achiev|outperform|reduce|reduces|increase|"
    r"improve|improves|equal|equals|improv|guarantee|prove|proves|proven|establish|"
    r"confirm|confirms|suggest|suggests|indicate|indicates|estimate|estimated|"
    r"yield|yields|remain|remains|require|requires|is|are|was|were|has|have)\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:%|ms|s\b|seconds?|minutes?|hours?|MB|GB|x\b)?")
CITATION_RE = re.compile(r"\[(\d+(?:\s*[-,]\s*\d+)*)\](?!\()")
AUTHOR_YEAR_RE = re.compile(
    r"\(([A-Z][A-Za-zÀ-ÿ'\-]+)(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][A-Za-zÀ-ÿ'\-]+)?,?\s*(\d{4})[a-z]?\)"
)

HYPOTHESIS_RE = re.compile(
    r"\b(we\s+hypothesi[sz]e|hypothesis|conjecture|conjectur\w+|we\s+speculate|"
    r"might|may\s+(?:be|indicate|suggest)|could\s+(?:be|indicate)|it\s+is\s+plausible)\b",
    re.IGNORECASE,
)
INTERPRETATION_RE = re.compile(
    r"\b(we\s+interpret|this\s+suggests|these\s+results\s+suggest|we\s+infer|"
    r"interpreted\s+as|consistent\s+with|we\s+attribute|likely\s+due\s+to)\b",
    re.IGNORECASE,
)
ASSUMPTION_RE = re.compile(r"\b(we\s+assume|assuming|assumption|given\s+that|"
                           r"provided\s+that|under\s+the\s+assumption)\b", re.IGNORECASE)
FUTURE_RE = re.compile(
    r"\b(future\s+work|in\s+future|next\s+steps?|we\s+plan\s+to|will\s+be\s+"
    r"(?:run|conducted|tested|evaluated|reported)|remains?\s+to\s+be|"
    r"is\s+left\s+(?:for|to)\s+future)\b",
    re.IGNORECASE,
)
PENDING_RE = re.compile(
    r"\b(results?\s+(?:are\s+)?pending|pending\s+(?:results?|experiments?)|"
    r"not\s+yet\s+(?:run|executed|measured|completed)|will\s+be\s+"
    r"(?:run|executed|conducted|measured|reported)|to\s+be\s+(?:run|executed|"
    r"conducted|measured)|expected\s+(?:results?|outcomes?|to\s+observe)|"
    r"we\s+expect\s+that|planned\s+(?:experiments?|runs?|study)|"
    r"experiments?\s+(?:is|are)\s+(?:planned|forthcoming))\b",
    re.IGNORECASE,
)
COMPLETION_RE = re.compile(
    r"\b(we\s+(?:ran|executed|conducted|performed|measured|implemented|tested)|"
    r"our\s+(?:experiments?|tests?|benchmarks?|implementation|analysis)\s+"
    r"(?:shows?|showed|confirms?|confirmed|found|produced|passed)|"
    r"experiments?\s+(?:was|were)\s+(?:run|executed|conducted|completed)|"
    r"results?\s+(?:were|was)\s+(?:obtained|measured|recorded)|"
    r"all\s+tests?\s+(?:pass|passed)|the\s+test\s+suite\s+(?:passes|passed)|"
    r"we\s+obtained|we\s+observed|we\s+found)\b",
    re.IGNORECASE,
)
NOVELTY_RE = re.compile(
    r"\b(first\s+(?:study|paper|work|to\s+\w+)|the\s+first\s+\w+|novel\s+"
    r"(?:algorithm|method|approach|framework|contribution)|no\s+(?:prior|previous)\s+"
    r"(?:work|study|research)|unprecedented|state\s+of\s+the\s+art|"
    r"has\s+not\s+been\s+(?:studied|addressed|considered))\b",
    re.IGNORECASE,
)
PROOF_RE = re.compile(
    r"\b(we\s+prove|proof\s+of|theorem|lemma|corollary|guarantees?\s+that|"
    r"for\s+all\s+(?:n|graphs?|inputs?|sizes?)|in\s+general|universally|"
    r"mathematically\s+proven|establishes?\s+that)\b",
    re.IGNORECASE,
)
PERFORMANCE_RE = re.compile(
    r"\b(faster|slower|speedup|speed-up|throughput|latency|runtime|"
    r"impractical|infeasible|scales?\s+(?:to|as|linearly|quadratically)|"
    r"practical\s+up\s+to|becomes?\s+(?:too\s+slow|impractical)|"
    r"orders?\s+of\s+magnitude)\b",
    re.IGNORECASE,
)
COMPLEXITY_RE = re.compile(
    r"(?:\bO\s*\([^)]{1,40}\))|(?:\b(?:theta|Ω|omega)\s*\([^)]{1,40}\))|"
    r"\b(linear|quadratic|cubic|polynomial|exponential|logarithmic|constant)[\s-]+"
    r"(?:time|space|memory|complexity|running\s+time)\b",
    re.IGNORECASE,
)
FUZZY_RE = re.compile(
    r"\b(close\s+to\s+fail\w*|near(?:ly)?\s+fail\w*|significantly\s+"
    r"(?:better|worse|faster|slower|higher|lower)|substantially|"
    r"efficient|efficiently|robust|robustly|scalable|practical|"
    r"reasonable|competitive|state[- ]of[- ]the[- ]art)\b",
    re.IGNORECASE,
)
DEFINITION_CUE_RE = re.compile(
    r"\b(we\s+define|is\s+defined\s+as|defined\s+as|operationali[sz]\w*|"
    r"measured\s+as|we\s+measure|metric\s+is|operational\s+definition)\b",
    re.IGNORECASE,
)


def strip_code(text: str) -> str:
    return re.sub(r"```.*?```", " ", text or "", flags=re.DOTALL)


def split_sentences(text: str) -> List[str]:
    cleaned = strip_code(text)
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", cleaned)
    return [re.sub(r"\s+", " ", part).strip()
            for part in parts if len(part.strip()) > 20]


def section_text(draft: Draft, roles: Iterable[str]) -> str:
    wanted = set(roles)
    chunks = []
    for section in draft.sections:
        if classify_section_role(section.id, section.title) in wanted:
            chunks.append(section.body)
    return "\n\n".join(chunks)


def section_texts(draft: Draft) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    for section in draft.sections:
        role = classify_section_role(section.id, section.title)
        texts[role] = texts.get(role, "") + "\n\n" + section.body
    return texts


def detect_pending_experiments(text: str) -> List[str]:
    return [m.group(0) for m in PENDING_RE.finditer(text or "")]


def detect_completion_claims(text: str) -> List[str]:
    return [m.group(0) for m in COMPLETION_RE.finditer(text or "")]


def detect_novelty_claims(text: str) -> List[str]:
    return [m.group(0) for m in NOVELTY_RE.finditer(text or "")]


def detect_proof_claims(text: str) -> List[str]:
    return [m.group(0) for m in PROOF_RE.finditer(text or "")]


def detect_performance_claims(text: str) -> List[str]:
    return [m.group(0) for m in PERFORMANCE_RE.finditer(text or "")]


def detect_complexity_claims(text: str) -> List[str]:
    return [m.group(0) for m in COMPLEXITY_RE.finditer(text or "")]


def detect_fuzzy_terms(text: str) -> List[str]:
    return [m.group(0) for m in FUZZY_RE.finditer(text or "")]


def detect_hypothesis_language(text: str) -> List[str]:
    return [m.group(0) for m in HYPOTHESIS_RE.finditer(text or "")]


def detect_future_language(text: str) -> List[str]:
    return [m.group(0) for m in FUTURE_RE.finditer(text or "")]


@dataclass
class EvidenceLedger:
    claims: List[ClaimRecord] = field(default_factory=list)
    results: List[ResultRecord] = field(default_factory=list)
    runs: List[ExperimentRun] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    _counter: int = 0

    # -- population -------------------------------------------------------
    def add_claim(self, claim: ClaimRecord) -> ClaimRecord:
        self._counter += 1
        if not claim.claim_id:
            claim.claim_id = f"C{self._counter}"
        self.claims.append(claim)
        return claim

    def add_result(self, result: ResultRecord) -> ResultRecord:
        if not result.result_id:
            result.result_id = f"R{len(self.results) + 1}"
        self.results.append(result)
        return result

    def add_run(self, run: ExperimentRun) -> ExperimentRun:
        self.runs.append(run)
        return run

    def result_metric_index(self) -> Dict[str, List[ResultRecord]]:
        index: Dict[str, List[ResultRecord]] = {}
        for result in self.results:
            key = re.sub(r"[^a-z0-9]+", " ", result.metric.lower()).strip()
            if key:
                index.setdefault(key, []).append(result)
        return index

    def links_for_claim(self, claim_id: str) -> List[ResultRecord]:
        return [r for r in self.results if r.result_id in
                next((c.result_ids for c in self.claims if c.claim_id == claim_id), [])]

    # -- queries ----------------------------------------------------------
    def executed_results(self) -> List[ResultRecord]:
        return [r for r in self.results if r.provenance == "executed"]

    def unsupported(self) -> List[ClaimRecord]:
        return [c for c in self.claims if c.support_status in ("unsupported", "unverified")
                and c.kind in (EvidenceKind.ESTABLISHED_FACT, EvidenceKind.STUDY_RESULT,
                               EvidenceKind.EXTERNAL_FINDING)]

    def by_kind(self, kind: EvidenceKind) -> List[ClaimRecord]:
        return [c for c in self.claims if c.kind == kind]

    def summary(self) -> Dict[str, Any]:
        kinds = {kind.value: 0 for kind in EvidenceKind}
        for claim in self.claims:
            kinds[claim.kind.value] += 1
        return {
            "claims": len(self.claims),
            "kinds": kinds,
            "results": len(self.results),
            "executed_results": len(self.executed_results()),
            "runs": len(self.runs),
            "successful_runs": len([r for r in self.runs
                                    if r.status == ExperimentStatus.OK]),
            "notes": list(self.notes),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "results": [r.to_dict() for r in self.results],
            "runs": [r.to_dict() for r in self.runs],
            "notes": list(self.notes),
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceLedger":
        ledger = cls()
        for item in data.get("claims") or []:
            ledger.claims.append(ClaimRecord.from_dict(item))
        for item in data.get("results") or []:
            ledger.results.append(ResultRecord.from_dict(item))
        for item in data.get("runs") or []:
            ledger.runs.append(ExperimentRun.from_dict(item))
        ledger.notes = list(data.get("notes") or [])
        ledger._counter = len(ledger.claims)
        return ledger


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------


def _section_roles(draft: Draft) -> Dict[str, str]:
    return {s.id: classify_section_role(s.id, s.title) for s in draft.sections}


def classify_claim(sentence: str, *, role: str, library: SourceLibrary,
                   result_index: Dict[str, List[ResultRecord]]) -> ClaimRecord:
    """Assign an epistemic kind and provenance to one extracted sentence."""
    lower = sentence.lower()
    numeric_markers = [int(n) for m in CITATION_RE.finditer(sentence)
                       for n in _expand(m.group(1))]
    author_year = bool(AUTHOR_YEAR_RE.search(sentence))
    resolved_sources: List[str] = []
    unresolved_citation = False
    index = library.numbered_index()
    for number in numeric_markers:
        if number in index:
            resolved_sources.append(index[number].source_id)
        else:
            unresolved_citation = True
    if author_year:
        for match in AUTHOR_YEAR_RE.finditer(sentence):
            surname, year = match.group(1).lower(), int(match.group(2))
            hit = next((r for r in library.ordered_records()
                        if r.year == year and r.authors
                        and r.authors[0].split()[-1].lower() == surname), None)
            if hit is not None:
                resolved_sources.append(hit.source_id)
            else:
                unresolved_citation = True

    linked_results: List[ResultRecord] = []
    for metric_key, results in result_index.items():
        name_tokens = set(metric_key.split())
        if metric_key and (metric_key in lower or (name_tokens and
                          len(name_tokens & set(re.findall(r"[a-z0-9]+", lower)))
                          >= max(1, len(name_tokens) - 1))) and NUMBER_RE.search(sentence):
            linked_results.extend(results)

    if HYPOTHESIS_RE.search(sentence):
        kind = EvidenceKind.HYPOTHESIS
    elif FUTURE_RE.search(sentence):
        kind = EvidenceKind.FUTURE_WORK
    elif ASSUMPTION_RE.search(sentence):
        kind = EvidenceKind.ASSUMPTION
    elif INTERPRETATION_RE.search(sentence) or (
            re.search(r"\b(suggests?|indicates?|implies?)\b", lower)
            and not linked_results):
        kind = EvidenceKind.INTERPRETATION
    elif linked_results:
        kind = EvidenceKind.STUDY_RESULT
    elif resolved_sources and re.search(
            r"\b(previous|prior|reported|et\s+al|in\s+\d{4}|literature|study|studies)\b",
            lower):
        kind = EvidenceKind.EXTERNAL_FINDING
    elif resolved_sources:
        kind = EvidenceKind.ESTABLISHED_FACT
    elif numeric_markers or author_year:
        kind = EvidenceKind.ESTABLISHED_FACT
    elif role in ("results", "experiments") and NUMBER_RE.search(sentence):
        kind = EvidenceKind.STUDY_RESULT
    elif role in ("results", "experiments") and detect_completion_claims(sentence):
        kind = EvidenceKind.STUDY_RESULT
    elif SCOPE_RE.search(sentence):
        kind = EvidenceKind.INTERPRETATION
    elif role in ("introduction", "background", "discussion", "conclusion") \
            and CONTRIBUTION_RE.search(sentence):
        kind = EvidenceKind.INTERPRETATION
    else:
        kind = EvidenceKind.UNVERIFIED

    if kind == EvidenceKind.STUDY_RESULT:
        if linked_results:
            support_status = "verified"
        elif role in ("results", "experiments") and detect_completion_claims(sentence):
            support_status = "unsupported"
        else:
            support_status = "unsupported" if NUMBER_RE.search(sentence) else "unverified"
    elif kind in (EvidenceKind.ESTABLISHED_FACT, EvidenceKind.EXTERNAL_FINDING):
        support_status = "verified" if resolved_sources and not unresolved_citation else "unsupported"
    elif kind == EvidenceKind.HYPOTHESIS:
        support_status = "not_required"
    elif kind in (EvidenceKind.INTERPRETATION, EvidenceKind.ASSUMPTION,
                  EvidenceKind.FUTURE_WORK):
        support_status = "not_required"
    else:
        support_status = "unverified"

    return ClaimRecord(
        claim_id="",
        text=sentence.strip(),
        section_id="",
        kind=kind,
        source_ids=list(dict.fromkeys(resolved_sources)),
        result_ids=[r.result_id for r in linked_results],
        experiment_ids=list(dict.fromkeys(r.experiment_id for r in linked_results)),
        support_status=support_status,
        notes=(["unresolved citation marker"] if unresolved_citation else []),
    )


def _expand(marker: str) -> List[int]:
    numbers: List[int] = []
    for part in marker.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start, end = (int(x) for x in part.split("-", 1))
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


#: Statements about the paper's scope or contribution are not evidence claims: they
#: describe what the work is, not what is true about the world.
SCOPE_RE = re.compile(
    r"\b(limited to|cannot establish|does not (?:prove|establish|apply|show)|"
    r"in this (?:article|paper|study)|this (?:article|paper|study) (?:is|makes|"
    r"contributes|reports)|we (?:do not|make no)|not measured|out of scope|"
    r"beyond the scope|scope of (?:this|the)|no general .{0,30} claim|"
    r"the observations are limited)\b", re.I)
CONTRIBUTION_RE = re.compile(
    r"\bwe (?:contribute|present|propose|introduce|describe|report)\b", re.I)


def _links_recorded_result(sentence: str,
                           result_index: Dict[str, List[ResultRecord]]) -> bool:
    """True when the sentence names a metric that an executed run recorded."""
    lowered = sentence.lower()
    return any(key and key in lowered for key in result_index)


def build_ledger(draft: Draft, library: SourceLibrary,
                 results: Sequence[ResultRecord] = ()) -> EvidenceLedger:
    """Extract checkable claims from the draft and link them to evidence."""
    ledger = EvidenceLedger()
    for result in results:
        ledger.add_result(ResultRecord.from_dict(result.to_dict()))
    result_index = ledger.result_metric_index()
    roles = _section_roles(draft)
    for section in draft.sections:
        role = roles.get(section.id, "other")
        if role == "references":
            # Bibliography entries are records, not prose claims; they are validated
            # against the source library instead.
            continue
        body, _references = split_references(strip_code(section.body))
        for sentence in split_sentences(body):
            if len(sentence.split()) < 4:
                continue
            if not (CLAIM_CUE_RE.search(sentence) or NUMBER_RE.search(sentence)
                    or CITATION_RE.search(sentence) or AUTHOR_YEAR_RE.search(sentence)):
                continue
            # Procedure prose ("the harness is executed with X") is substantiated by
            # the executed code and the recorded parameters, not by a citation or a
            # result. Indexing it as an unsupported claim would bury the real ones.
            if role in ("methods", "algorithm") and not (
                    CITATION_RE.search(sentence) or AUTHOR_YEAR_RE.search(sentence)
                    or _links_recorded_result(sentence, result_index)):
                continue
            claim = classify_claim(sentence, role=role,
                                   library=library, result_index=result_index)
            claim.section_id = section.id
            claim.role = roles.get(section.id, "other")
            ledger.add_claim(claim)
    return ledger


def evidence_context_block(ledger: EvidenceLedger, results: Sequence[ResultRecord],
                           max_results: int = 20, max_facts: int = 10) -> str:
    """Render the executed-results and evidence rules for a drafting context packet."""
    lines = [
        "EVIDENCE RULES (binding):",
        "- A result may be reported only if it appears in EXECUTED RESULTS below.",
        "- Established facts must cite a verified source key; user-supplied or "
        "unverified records may not be presented as established knowledge.",
        "- Hypotheses, assumptions, interpretations and future work must be labelled "
        "as such and must not appear in the Results section as findings.",
        "- Executing a finite test suite is not a proof of a general theorem.",
    ]
    if results:
        lines.append("EXECUTED RESULTS (report these verbatim; do not round or invent):")
        for result in list(results)[:max_results]:
            value = result.value
            unit = f" {result.unit}" if result.unit else ""
            lines.append(
                f"- result_id={result.result_id} experiment={result.experiment_id} "
                f"metric={result.metric} value={value}{unit}"
                + (f" n={result.n}" if result.n is not None else "")
                + " [executed]"
            )
    else:
        lines.append("EXECUTED RESULTS: none. Do not write a Results section that "
                     "reports findings; mark the study as not yet run.")
    if ledger.claims:
        unsupported = ledger.unsupported()[:max_facts]
        if unsupported:
            lines.append("UNSUPPORTED CLAIMS CARRIED FROM THE PREVIOUS DRAFT:")
            for claim in unsupported:
                lines.append(f"- {claim.section_id or '-'}: {claim.text[:160]}")
    return "\n".join(lines)
