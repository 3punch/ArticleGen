"""Quantitative analysis: complexity statements, benchmarks and statistics.

Nothing here decides that a claim is *true*. Each analyser reports what the manuscript
states and what the recorded artefacts can support, so the validators can tell
"claimed and measured" apart from "claimed and unmeasured" - and from "not claimed".

The three categories are kept apart on purpose (Section 7 of the requirements):

* **theory** - asymptotic statements, their input parameters, assumptions and
  worst/average-case qualification;
* **empirical** - timings and resource numbers, together with the hardware, runtime,
  dataset, input sizes, repetitions, methodology and variability needed to interpret
  them;
* **statistics** - whether a comparative claim carries a dispersion/uncertainty
  statement, and whether ambiguous quality terms were operationalised.

Detection is regex-based and every verdict carries the matched sentence as evidence, so
a reviewer can check the analyser itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..models import ArticleType, Brief, Draft
from .models import (
    BenchmarkRecord,
    ExperimentRun,
    ExperimentSpec,
    Measurement,
    classify_section_role,
)

# --- detection patterns ----------------------------------------------------

COMPLEXITY_CLAIM_RE = re.compile(
    r"(?:\bO\([^)]*\)|\bΘ\([^)]*\)|\\Theta\([^)]*\)|\bΩ\([^)]*\)|"
    r"\b(?:polynomial|linear|sub-?linear|quadratic|cubic|logarithmic|exponential)"
    r"[- ](?:time|space)\b|\bruns? in (?:linear|quadratic|cubic|logarithmic|constant)"
    r" time\b)", re.I)
COMPLEXITY_ANALYSIS_RE = re.compile(
    r"(?:\bO\(|\bΘ\(|\bcomplexity\b|\basymptotic\b|\bworst[- ]case\b|"
    r"\baverage[- ]case\b|\bamortized\b|\bupper bound\b|\blower bound\b|"
    r"\binput (?:size|length)\b|\bin terms of\b)", re.I)
INPUT_PARAM_RE = re.compile(
    r"\b(?:where|with|let)\s+n\b|\bn\s*(?:=|is)\s*(?:the|number|size|length|input|"
    r"|vertices|edges|nodes)|"
    r"\binput (?:size|length)\b|\bnotation\b|\bdenote\b", re.I)
MEMORY_RE = re.compile(r"\b(memory|space complexity|auxiliary space|RAM|bytes|"
                       r"linear space|in[- ]place)\b", re.I)
AVERAGE_RE = re.compile(r"\b(average[- ]case|expected (?:time|cost)|on average)\b", re.I)
WORST_RE = re.compile(r"\b(worst[- ]case|upper bound|guarantee[sd]?)\b", re.I)
ASSUMPTION_RE = re.compile(r"\b(assum\w+|provided that|holds? (?:for|when)|"
                           r"under the (?:condition|assumption)|for (?:simple|connected|"
                           r"bipartite|planar|sparse|dense) graphs?)\b", re.I)
TIMING_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|sec(?:ond)?s?|µs|"
                       r"microseconds?|ns|nanoseconds?|minutes?)\b", re.I)
SPEEDUP_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:x|×|times)\s*(faster|slower|speed-?up)\b",
                        re.I)
PERCENT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*%\s*(faster|slower|less|more|improvement|"
                        r"reduction|better|worse)?", re.I)
MEMORY_VALUE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(KB|MB|GB|KiB|MiB|GiB|bytes?)\b")
SEED_RE = re.compile(r"\b(seed|random_state|rng)\s*(?:=|:|of)?\s*(\d+)", re.I)
REPETITION_RE = re.compile(r"\b(\d+)\s*(?:repetitions?|repeats?|runs? per (?:setting|"
                           r"configuration)|trials?)\b", re.I)
STAT_PAIR_RE = re.compile(r"\b(?:mean|median|average)\b[^.]{0,40}?"
                          r"(?:±|\\pm|\+/-|stdev|std\.? dev|standard deviation|"
                          r"confidence interval|CI|IQR|interquartile)\b", re.I)
COMPARATIVE_RE = re.compile(r"\b(faster|slower|better|worse|outperforms?|improves?|"
                            r"significantly|superior|beats?)\b", re.I)
SIGNIFICANCE_RE = re.compile(r"\b(p\s*[<=>]\s*0?\.\d+|p-value|confidence interval|"
                             r"95% CI|t-test|Wilcoxon|Mann[- ]Whitney|bootstrap|"
                             r"effect size|Bonferroni|Holm)\b", re.I)
SEARCH_PROTOCOL_RE = re.compile(r"\b(search (?:string|query|terms?|strategy)|database[s]?|"
                                r"inclusion criteria|exclusion criteria|PRISMA|"
                                r"screening|records identified|query terms)\b", re.I)
REVIEW_TYPE_RE = re.compile(r"\b(systematic review|meta-?analysis|narrative review|"
                            r"literature review|survey)\b", re.I)
SCALABILITY_RE = re.compile(r"\b(scales? (?:linearly|quadratically|well)|scalab\w+|"
                            r"throughput|as n grows|growth rate)\b", re.I)
SAMPLE_SIZE_RE = re.compile(r"\b(power analysis|sample size|n\s*=\s*\d+\s*(?:because|to "
                            r"detect|achieve))\b", re.I)
PARTICIPANTS_RE = re.compile(r"\b(sample|participants?|subjects?|corpus)\b", re.I)

#: Informal quality terms that must be operationalised before they can be evidence.
AMBIGUOUS_TERMS: Tuple[Tuple[str, str], ...] = (
    ("close to failing", "state the exact limit, margin or failure threshold"),
    ("near the limit", "state the limit and the measured distance to it"),
    ("significantly better", "report the test, effect size and uncertainty"),
    ("significantly worse", "report the test, effect size and uncertainty"),
    ("significantly faster", "report the measured difference and uncertainty"),
    ("significantly improves", "report the test, effect size and uncertainty"),
    ("efficient", "state the metric and the baseline it is measured against"),
    ("scalable", "state the input range and how time/memory grow over it"),
    ("robust", "state the perturbations or conditions actually tested"),
    ("state of the art", "cite the comparison and the benchmark"),
    ("competitive", "name the baselines and the metric"),
    ("almost always", "report the observed frequency and the sample"),
    ("proves", "distinguish a proof from a finite case analysis"),
)

DEFINITION_CUES = ("we define", "defined as", "we operationally define", "by \"",
                   "by '", "refers to", "is measured as", "we measure", "criterion",
                   "threshold", "operationalis", "operationaliz", "we take",
                   "in this paper we call")

THEORY_INPUT_PARAM_RE = INPUT_PARAM_RE


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _prose(draft: Draft) -> str:
    return "\n\n".join(section.body for section in draft.sections)


def _prose_without_code(text: str) -> str:
    return re.sub(r"```.*?```", " ", text or "", flags=re.DOTALL)


def _sentences(text: str) -> List[str]:
    cleaned = _prose_without_code(re.sub(r"\s+", " ", text or ""))
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if len(part.strip().split()) >= 3]


def _paragraphs(text: str) -> List[str]:
    cleaned = _prose_without_code(text)
    return [re.sub(r"\s+", " ", block).strip() for block in re.split(r"\n\s*\n", cleaned)
            if block.strip()]


def _role_text(draft: Draft, roles: Iterable[str]) -> str:
    wanted = set(roles)
    return "\n\n".join(section.body for section in draft.sections
                       if classify_section_role(section.id, section.title) in wanted)


def _numbers(text: str) -> List[float]:
    return [float(value) for value in
            re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])", text)]


def _parameter_mentions(text: str) -> List[str]:
    mentions = re.findall(r"\b[OΘ]\(([^)]*)\)", text)
    return list(dict.fromkeys(value.strip() for value in mentions if value.strip()))[:8]


# ---------------------------------------------------------------------------
# The analysis record consumed by the validators
# ---------------------------------------------------------------------------


@dataclass
class ComplexityAnalysis:
    """Everything the validators need to judge quantitative claims.

    Named ``ComplexityAnalysis`` for continuity with the validator call sites; it holds
    the complexity, performance, statistics and terminology views.
    """

    # complexity (theory)
    claims: List[str] = field(default_factory=list)
    analysis_present: bool = False
    missing_elements: List[str] = field(default_factory=list)
    bound_expressions: List[str] = field(default_factory=list)
    input_parameters: List[str] = field(default_factory=list)
    worst_case_stated: bool = False
    average_case_stated: bool = False
    assumptions_stated: bool = False
    memory_stated: bool = False
    memory_claims: List[str] = field(default_factory=list)

    # performance (empirical)
    performance_claims: List[str] = field(default_factory=list)
    benchmark_claims_without_records: List[str] = field(default_factory=list)
    untraceable_numbers: List[str] = field(default_factory=list)
    benchmark_metadata_issues: List[str] = field(default_factory=list)
    benchmark_records: int = 0

    # statistics and terminology
    statistical_gaps: List[str] = field(default_factory=list)
    comparative_claims: List[str] = field(default_factory=list)
    undefined_terms: List[str] = field(default_factory=list)

    # review protocols
    search_strategy_gaps: List[str] = field(default_factory=list)
    mislabelled_review: List[str] = field(default_factory=list)

    @property
    def has_performance_claims(self) -> bool:
        return bool(self.performance_claims)

    @property
    def has_claims(self) -> bool:
        return bool(self.claims or self.memory_claims or self.performance_claims)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


# ---------------------------------------------------------------------------
# Complexity (theory)
# ---------------------------------------------------------------------------


def complexity_result(draft: Draft, article_type: ArticleType) -> ComplexityAnalysis:
    """Collect what the manuscript says about asymptotic cost."""
    body = _prose_without_code(_prose(draft))
    analysis_text = _role_text(draft, ("complexity",)) or _role_text(draft, ("methods",))
    result = ComplexityAnalysis()

    for sentence in _sentences(body):
        if COMPLEXITY_CLAIM_RE.search(sentence):
            result.claims.append(sentence)
        if MEMORY_RE.search(sentence) and re.search(
                r"\b(uses?|requires?|needs?|complexity|space|memory)\b", sentence, re.I):
            result.memory_claims.append(sentence)

    combined = analysis_text or body
    result.bound_expressions = _parameter_mentions(combined)
    result.input_parameters = [
        sentence[:180] for sentence in _sentences(combined)
        if THEORY_INPUT_PARAM_RE.search(sentence)
        and not COMPLEXITY_CLAIM_RE.search(sentence)
    ][:5]
    result.worst_case_stated = bool(WORST_RE.search(combined))
    result.average_case_stated = bool(AVERAGE_RE.search(combined))
    result.assumptions_stated = bool(ASSUMPTION_RE.search(combined))
    result.memory_stated = bool(MEMORY_RE.search(combined))

    missing: List[str] = []
    if result.claims and not result.input_parameters:
        missing.append("input-size parameters used by the complexity statement")
    if result.claims and not (result.worst_case_stated or result.average_case_stated):
        missing.append("worst-case and average-case qualification")
    if result.claims and not result.assumptions_stated:
        missing.append("assumptions under which the bound holds")
    if (result.claims or result.memory_claims) and not result.memory_stated:
        missing.append("memory/space statement")
    result.missing_elements = missing

    # "Analysis present" means one paragraph contains the parameters, the bounds, the
    # case qualifier and the assumptions together - not four facts scattered in prose.
    for paragraph in _paragraphs(combined):
        if (COMPLEXITY_ANALYSIS_RE.search(paragraph)
                and THEORY_INPUT_PARAM_RE.search(paragraph)
                and (WORST_RE.search(paragraph) or AVERAGE_RE.search(paragraph))
                and MEMORY_RE.search(paragraph)):
            result.analysis_present = True
            break
    if not result.analysis_present and not result.claims:
        result.analysis_present = False
    return result


# ---------------------------------------------------------------------------
# Performance claims against recorded measurements
# ---------------------------------------------------------------------------


def known_measurement_values(records: Sequence[BenchmarkRecord]) -> List[float]:
    values: List[float] = []
    for record in records:
        for measurement in record.metrics:
            try:
                values.append(float(measurement.value))
            except (TypeError, ValueError):
                continue
    return values


def _performance_sentences(text: str) -> List[str]:
    found: List[str] = []
    for sentence in _sentences(text):
        if TIMING_RE.search(sentence) or SPEEDUP_RE.search(sentence) \
                or MEMORY_VALUE_RE.search(sentence) or PERCENT_RE.search(sentence) \
                or SCALABILITY_RE.search(sentence):
            found.append(sentence)
    return found


def _number_supported(value: float, known: Sequence[float],
                      tolerance: float = 0.02) -> bool:
    for candidate in known:
        if abs(value - candidate) <= max(tolerance * abs(candidate), 1e-9):
            return True
    return False


def _claim_numbers(sentence: str) -> List[float]:
    """Numbers that represent a measurement, not a count or an exponent."""
    numbers: List[float] = []
    for match in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])", sentence):
        token = match.group(1)
        value = float(token)
        if "." not in token and value <= 100:
            continue
        numbers.append(value)
    return numbers


def performance_analysis(records: Sequence[BenchmarkRecord],
                         text: str) -> Tuple[List[str], List[str]]:
    """Return (claims with no recorded measurement, untraceable numbers)."""
    claims_without_records: List[str] = []
    untraceable: List[str] = []
    known = known_measurement_values(records)
    for sentence in _performance_sentences(text):
        numbers = _claim_numbers(sentence)
        if not records:
            claims_without_records.append(sentence[:250])
            for value in numbers[:3]:
                untraceable.append(
                    f"{value:g} (stated in: \"{sentence[:110]}\") - no recorded "
                    "measurement")
            continue
        missing = [value for value in numbers if not _number_supported(value, known)]
        if missing:
            claims_without_records.append(sentence[:250])
            for value in missing[:3]:
                untraceable.append(
                    f"{value:g} (stated in: \"{sentence[:110]}\") - not among the "
                    "recorded measurements")
    return claims_without_records, untraceable


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def statistics_gaps(draft: Draft, article_type: ArticleType,
                    records: Sequence[BenchmarkRecord]) -> Tuple[List[str], List[str]]:
    body = _prose_without_code(_prose(draft))
    comparative = [sentence for sentence in _sentences(body)
                   if COMPARATIVE_RE.search(sentence)]
    uncertainty = bool(STAT_PAIR_RE.search(body) or SIGNIFICANCE_RE.search(body))
    gaps: List[str] = []
    if comparative and not uncertainty:
        gaps.append("comparative claim without dispersion or uncertainty: \""
                    + comparative[0][:140] + "\"")
    if records and not SEED_RE.search(body):
        gaps.append("deterministic seeds are not reported for the measured runs")
    if len(records) > 1 and not REPETITION_RE.search(body):
        gaps.append("the number of repetitions is not reported")
    if article_type == ArticleType.EMPIRICAL and PARTICIPANTS_RE.search(body) \
            and not SAMPLE_SIZE_RE.search(body):
        gaps.append("sample size is not justified")
    return gaps, comparative


# ---------------------------------------------------------------------------
# Terminology
# ---------------------------------------------------------------------------


def undefined_terms(draft: Draft, extra_terms: Sequence[str] = ()) -> List[str]:
    body = _prose(draft).lower()
    used: List[str] = []
    for term, _guidance in AMBIGUOUS_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", body):
            used.append(term)
    for term in extra_terms:
        token = str(term).strip().lower()
        if token and re.search(r"\b" + re.escape(token) + r"\b", body):
            if token not in used:
                used.append(token)

    undefined: List[str] = []
    for term in used:
        sentences = [sentence.lower() for sentence in _sentences(body)
                     if term in sentence]
        if not sentences:
            continue
        described = False
        for sentence in sentences:
            if any(cue in sentence for cue in DEFINITION_CUES) and (
                    term in sentence or _cue_near(sentence, term)):
                described = True
                break
        # A numeric threshold stated next to the term operationalises it.
        if not described:
            for sentence in sentences:
                numbers = _numbers(sentence)
                if numbers and any(word in sentence for word in
                                   ("threshold", "limit", "margin", "when", "below",
                                    "above", "less than", "more than")):
                    described = True
                    break
        if not described:
            undefined.append(term)
    return undefined


def _cue_near(sentence: str, term: str) -> bool:
    index = sentence.find(term)
    if index < 0:
        return False
    for cue in DEFINITION_CUES:
        position = sentence.find(cue)
        if position >= 0 and abs(position - index) <= 120:
            return True
    return False


# ---------------------------------------------------------------------------
# Review protocol
# ---------------------------------------------------------------------------


def review_protocol_gaps(draft: Draft, article_type: ArticleType
                         ) -> Tuple[List[str], List[str]]:
    body = _prose(draft)
    protocol_present = bool(SEARCH_PROTOCOL_RE.search(body))
    gaps: List[str] = []
    mislabelled: List[str] = []
    if article_type == ArticleType.SYSTEMATIC_REVIEW and not protocol_present:
        gaps.append("systematic review without a documented search strategy "
                    "(databases, query terms, inclusion/exclusion criteria)")
    mentions_review = bool(REVIEW_TYPE_RE.search(body))
    if article_type not in (ArticleType.SYSTEMATIC_REVIEW, ArticleType.NARRATIVE_REVIEW) \
            and re.search(r"\bsystematic review\b|\bmeta-?analysis\b", body, re.I) \
            and not protocol_present:
        mislabelled.append(
            "the manuscript calls itself a systematic review or meta-analysis but "
            "reports no search protocol, databases or inclusion criteria")
    return gaps, mislabelled


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def analyse(draft: Draft, article_type: ArticleType,
            benchmarks: Sequence[BenchmarkRecord] = (),
            extra_terms: Sequence[str] = ()) -> ComplexityAnalysis:
    """Run every quantitative analyser once and return the merged record."""
    result = complexity_result(draft, article_type)
    body = _prose_without_code(_prose(draft))

    result.benchmark_records = len(benchmarks)
    missing_records, untraceable = performance_analysis(benchmarks, body)
    result.performance_claims = _performance_sentences(body)
    result.benchmark_claims_without_records = missing_records
    result.untraceable_numbers = untraceable

    seen: List[str] = []
    for record in benchmarks:
        for issue in record.metadata_issues():
            if issue not in seen:
                seen.append(issue)
    result.benchmark_metadata_issues = seen

    result.statistical_gaps, result.comparative_claims = statistics_gaps(
        draft, article_type, benchmarks)
    result.undefined_terms = undefined_terms(draft, extra_terms)
    result.search_strategy_gaps, result.mislabelled_review = review_protocol_gaps(
        draft, article_type)
    return result


# ---------------------------------------------------------------------------
# Benchmark records from executed runs
# ---------------------------------------------------------------------------

_RUNTIME_KEY_RE = re.compile(r"(time|sec|duration|elapsed|latency|runtime|ns|µs|us|ms)",
                             re.I)
_MEMORY_KEY_RE = re.compile(r"(mem|rss|heap|bytes|peak)", re.I)


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    clean = sorted(float(value) for value in values)
    if not clean:
        return {}
    count = len(clean)
    mean = sum(clean) / count
    variance = (sum((value - mean) ** 2 for value in clean) / (count - 1)
                if count > 1 else 0.0)
    middle = count // 2
    median = clean[middle] if count % 2 else (clean[middle - 1] + clean[middle]) / 2
    return {"count": count, "mean": round(mean, 6), "median": round(median, 6),
            "stdev": round(variance ** 0.5, 6), "min": clean[0], "max": clean[-1]}


def build_benchmark_records(runs: Sequence[ExperimentRun],
                            specs: Sequence[ExperimentSpec],
                            brief: Optional[Brief] = None) -> List[BenchmarkRecord]:
    """Promote executed metrics into auditable benchmark records.

    Only numbers produced by a real run appear. Environment fields come from the
    recorded run environment; anything absent stays absent and is reported by
    ``BenchmarkRecord.metadata_issues``, so a pre-run (non-benchmark) experiment does
    not masquerade as a reproducible measurement study.
    """
    spec_map = {spec.experiment_id: spec for spec in specs}
    supplied = dict(getattr(brief, "benchmark", {}) or {}) if brief is not None else {}
    records: List[BenchmarkRecord] = []
    for run in runs:
        if not run.status.succeeded:
            continue
        numeric = {key: value for key, value in run.metrics.items()
                   if isinstance(value, (int, float)) and not key.endswith("_unit")}
        if not numeric:
            continue
        spec = spec_map.get(run.experiment_id)
        timed = any(_RUNTIME_KEY_RE.search(key) for key in numeric)
        if spec is not None and not (spec.benchmark or run.repetitions > 1 or timed):
            continue
        if spec is None and not timed:
            continue

        env = dict(run.environment or {})
        runtime_version = ""
        if env.get("implementation"):
            runtime_version = str(env["implementation"])
            if env.get("python_version"):
                runtime_version += f" {env['python_version']}"
        elif env.get("python_version"):
            runtime_version = f"Python {env['python_version']}"

        metrics = []
        for key, value in numeric.items():
            unit = str(run.metrics.get(f"{key}_unit") or "")
            if not unit and _RUNTIME_KEY_RE.search(key):
                unit = "s"
            elif not unit and _MEMORY_KEY_RE.search(key):
                unit = "bytes"
            metrics.append(Measurement(name=key, value=float(value), unit=unit))

        per_metric: Dict[str, List[float]] = {}
        for repetition in run.repetition_metrics:
            for key, value in repetition.items():
                if isinstance(value, (int, float)) and not key.endswith("_unit"):
                    per_metric.setdefault(key, []).append(float(value))
        variability = {key: _stats(values) for key, values in per_metric.items()
                       if len(values) > 1}

        dataset = spec.dataset if spec is not None else ""
        parameters = dict(spec.parameters or {}) if spec is not None else {}
        records.append(BenchmarkRecord(
            benchmark_id=f"bench-{run.run_id}",
            experiment_id=run.experiment_id,
            run_ids=[run.run_id],
            description=(spec.description if spec is not None else "") or "",
            dataset=dataset,
            dataset_ids=[dataset] if dataset else [],
            input_sizes=list(parameters.get("input_sizes")
                             or parameters.get("sizes") or []),
            input_characteristics=str(parameters.get("input_characteristics") or ""),
            hardware=str(env.get("machine") or env.get("processor")
                         or supplied.get("hardware") or ""),
            operating_system=str(env.get("platform") or supplied.get("os") or ""),
            language=str(env.get("language") or (spec.language if spec else "") or ""),
            runtime_version=runtime_version or str(supplied.get("runtime_version") or ""),
            dependencies=[str(dep) for dep in (env.get("dependencies") or [])],
            repetitions=max(int(run.repetitions or 1), len(run.repetition_metrics)),
            timing_method=str(supplied.get("timing_method") or ""),
            memory_method=str(supplied.get("memory_method") or ""),
            metrics=metrics,
            variability=variability,
            failed_runs=0,
            timeouts=0,
            exclusions=[],
        ))
    return records
