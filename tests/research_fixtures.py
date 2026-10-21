"""Deterministic fixtures for the research-integrity tests.

Everything here is built from code and records, never from a live model or network
call. The two anchor manuscripts are:

* ``good_draft`` - a rigorous computational study whose claims the recorded run and
  verified sources actually support. It must not be rejected.
* ``erdos_draft`` - the failing Erdős–Gyárfás style article: overclaiming prose, a
  hard-coded size domain, unmeasured timings, claimed-but-unrun tests and a proof
  claim. It must be rejected, class by class.

The tests mutate copies of the good manuscript to reproduce one failure class at a
time, so a failure is attributable rather than a property of the fixture.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from articlegen.llm import LLMClient, LLMResponse
from articlegen.memory import MemoryStore
from articlegen.models import ArticleType, Brief, CitationStyle, Draft, Section
from articlegen.research import analysis, codecheck, evidence, quality, validators
from articlegen.research.codecheck import SubprocessExecutor
from articlegen.research.models import (ExperimentRun, ExperimentSpec, ReviewReport,
                                        SourceRecord)
from articlegen.research.planning import derive_plan, validate_plan
from articlegen.research.report import apply_bibliography
from articlegen.research.sources import SourceLibrary, StaticSourceRetriever

GOOD_CODE = '''"""Enumerate generated graph sizes and count the ones checked."""


def checked_sizes(max_exponent=5):
    """Sizes 2**k for 2 <= k <= max_exponent, generated rather than listed."""
    return [2 ** k for k in range(2, max_exponent + 1)]


def count_checked(max_exponent=5):
    sizes = checked_sizes(max_exponent)
    assert all(size > 0 for size in sizes)
    return len(sizes)


if __name__ == "__main__":
    checked = count_checked()
    print("METRIC: cycles_checked=%d count" % checked)'''

GOOD_TESTS = '''import unittest
from artifact import count_checked, checked_sizes


class T(unittest.TestCase):
    def test_count(self):
        self.assertEqual(count_checked(5), 4)

    def test_sizes(self):
        self.assertEqual(checked_sizes(5), [4, 8, 16, 32])
'''

#: The Erdős–Gyárfás regression listing: the domain is listed, not generated.
ERDOS_CODE = '''def has_monochromatic_cycle(n, k=4):
    return n in (4, 8, 16, 32)


if __name__ == "__main__":
    checked = 0
    for n in (4, 8, 16, 32):
        checked += 1 if has_monochromatic_cycle(n) else 0
    print("METRIC: cycles_checked=%d count" % checked)
'''

ERDOS_TESTS = '''import unittest
from artifact import has_monochromatic_cycle


class T(unittest.TestCase):
    def test_four(self):
        self.assertTrue(has_monochromatic_cycle(4))

    def test_five(self):
        self.assertFalse(has_monochromatic_cycle(5))
'''


# ---------------------------------------------------------------------------
# Briefs, tasks and plans
# ---------------------------------------------------------------------------


def study_brief(vector: str = "good", article_type: ArticleType = ArticleType.COMPUTATIONAL
                ) -> Brief:
    overclaiming = vector == "erdos"
    return Brief(
        task_id=f"research-{vector}",
        title=("A Parameterised Enumeration Harness for Cycle Checks" if not overclaiming
               else "Two-Direction Traversal for Monochromatic Cycles"),
        objective=(
            "Provide an executable enumeration harness for monochromatic cycle checks "
            "and report exactly which sizes were examined."
            if not overclaiming else
            "Show that the two-direction traversal eliminates every monochromatic cycle "
            "in both directions for every n."
        ),
        audience="graph theorists",
        target_words=900,
        article_type=article_type,
        citation_style=CitationStyle.IEEE,
        research_question=(
            "Which graph sizes can the parameterised harness examine, and what does the "
            "analysis cover?" if not overclaiming else
            "Does the traversal eliminate all monochromatic cycles for every n?"
        ),
        contribution_type="routine_implementation",
        requires_experiments=True,
        sections=[
            Section(id="introduction", title="Introduction", purpose="frame",
                    word_target=120),
            Section(id="methods", title="Methods", purpose="describe", word_target=120),
            Section(id="algorithm", title="Algorithm", purpose="code", word_target=120),
            Section(id="results", title="Results", purpose="report", word_target=120),
            Section(id="discussion", title="Discussion", purpose="interpret",
                    word_target=120),
            Section(id="limitations", title="Limitations", purpose="limits",
                    word_target=120),
            Section(id="conclusion", title="Conclusion", purpose="conclude",
                    word_target=120),
        ],
    )


def source_records(count: int = 3) -> List[SourceRecord]:
    base = [
        SourceRecord(title="Monochromatic cycles in edge-coloured complete graphs",
                     authors=["Paul Erdos"], year=1981, venue="Journal of Graph Theory",
                     doi="10.1002/jgt.3190050307"),
        SourceRecord(title="Colourings of complete graphs and cycles",
                     authors=["Andras Gyarfas"], year=1992, venue="Combinatorica",
                     doi="10.1007/bf01204730"),
        SourceRecord(title="Cycle enumeration in dense graphs",
                     authors=["Jane Doe"], year=2004, venue="SIAM Journal on Computing",
                     doi="10.1137/s0097539703434111"),
    ]
    return base[:count]


def verified_library(count: int = 3,
                     style: CitationStyle = CitationStyle.IEEE) -> SourceLibrary:
    """A library whose records were 'index-verified' by a fixture retriever."""
    library = SourceLibrary(style)
    records = source_records(count)
    library.from_brief_sources([record.to_dict() for record in records])
    retriever = StaticSourceRetriever(records, mark_verified=True)
    library.retrieve(["monochromatic cycle"], retriever, max_sources=count)
    library.verify(retriever)
    return library


def unverified_library(style: CitationStyle = CitationStyle.IEEE) -> SourceLibrary:
    """Records the author supplied, which no index confirmed."""
    library = SourceLibrary(style)
    library.from_brief_sources([record.to_dict() for record in source_records()])
    return library


def clean_review(run_models: bool = True) -> "ReviewReport":
    """A review that ran and found nothing - the reviewer's own output shape."""
    return ReviewReport(issues=[], notes=["fixture reviewer: no findings"],
                        model_review_ran=run_models)


def good_plan(brief: Brief, library: Optional[SourceLibrary] = None) -> Any:
    plan = derive_plan(brief, origin="fixture")
    plan.problem = brief.objective
    plan.background = ("Monochromatic cycle problems in edge-coloured complete graphs are "
                       "a classical topic; published bounds concern existence rather than "
                       "an implementation harness.")
    plan.gap = ("Implementations of cycle-checking procedures rarely state the size domain "
                "they enumerate, so reported checks cannot be interpreted.")
    plan.gap_evidence = ["[1] reports asymptotic existence results",
                         "[2] reports no size domain"]
    plan.contribution = ("A parameterised harness that generates the examined sizes and a "
                         "recorded run whose measured output is reported.")
    plan.methodology = ("Implement a generator for the examined sizes, execute the harness "
                        "and record the measured count; give a correctness argument for the "
                        "mapping from exponents to sizes and analyse time complexity in "
                        "terms of the maximum exponent.")
    plan.methodology_justification = ("Generation from an exponent keeps the size domain "
                                      "explicit, so the reported count can be traced to the "
                                      "code.")
    plan.required_materials = ["Python 3 runtime", "graph size generator"]
    plan.required_experiments = ["run the harness for the default exponent range"]
    plan.evaluation_criteria = ["edges of the harness return the generated sizes"]
    plan.limitations = ["The enumeration covers a finite exponent range."]
    plan.failure_modes = ["Sizes outside the exponent range are not examined."]
    plan.source_queries = ["monochromatic cycle edge coloured complete graph"]
    plan.outline = [{"id": section.id, "title": section.title,
                     "purpose": section.purpose, "word_target": section.word_target}
                    for section in brief.sections]
    plan.experiments = [spec_payload("exp-1", GOOD_CODE, GOOD_TESTS)]
    validate_plan(plan, brief, library=library or verified_library(),
                  retrieval_attempted=True, retrieval_available=True)
    return plan


def erdos_plan(brief: Brief, library: Optional[SourceLibrary] = None) -> Any:
    plan = derive_plan(brief, origin="fixture")
    plan.problem = brief.objective
    plan.background = ("Monochromatic cycles are classical; published results are "
                       "asymptotic rather than algorithmic.")
    plan.gap = ""  # the failing article never establishes a gap
    plan.gap_evidence = []
    plan.contribution = "A two-direction traversal that eliminates every monochromatic cycle."
    plan.novelty_claims = ["the first traversal that eliminates cycles in both directions"]
    plan.methodology = "Traverse in both directions and verify small cases."
    plan.methodology_justification = ""
    plan.required_materials = []
    plan.required_experiments = ["verify sizes 4, 8, 16 and 32"]
    plan.evaluation_criteria = ["the checker reports the tested sizes"]
    plan.limitations = []
    plan.failure_modes = []
    plan.source_queries = ["monochromatic cycle"]
    plan.outline = [{"id": section.id, "title": section.title,
                     "purpose": section.purpose, "word_target": section.word_target}
                    for section in brief.sections]
    plan.experiments = [spec_payload("exp-1", ERDOS_CODE, ERDOS_TESTS)]
    validate_plan(plan, brief, library=library or verified_library(),
                  retrieval_attempted=True, retrieval_available=True)
    return plan


def spec_payload(experiment_id: str, code: str, tests: str = "",
                 **extra: Any) -> Dict[str, Any]:
    payload = {
        "experiment_id": experiment_id,
        "title": "harness",
        "description": "counts generated sizes",
        "language": "python",
        "code": code,
        "tests": tests,
        "metrics": ["cycles_checked"],
        "seed": 0,
        "repetitions": 1,
        "parameters": {"max_exponent": 5},
        "section_id": "results",
    }
    payload.update(extra)
    return payload


def good_spec() -> ExperimentSpec:
    return ExperimentSpec.from_dict(spec_payload("exp-1", GOOD_CODE, GOOD_TESTS))


def erdos_spec() -> ExperimentSpec:
    return ExperimentSpec.from_dict(spec_payload("exp-1", ERDOS_CODE, ERDOS_TESTS))


def execute(spec: ExperimentSpec, timeout: int = 30) -> ExperimentRun:
    return SubprocessExecutor(timeout=timeout).run(spec)


# ---------------------------------------------------------------------------
# Drafts
# ---------------------------------------------------------------------------


def good_draft(brief: Brief, code: str = GOOD_CODE) -> Draft:
    code = code.rstrip("\n")
    return Draft(task_id=brief.task_id, version=1, title=brief.title, sections=[
        Section(id="introduction", title="Introduction", purpose="", word_target=120, body=(
            "Monochromatic cycle problems in edge-coloured complete graphs are a "
            "classical topic [1], and later work studies cycle structure in colourings "
            "[2], while enumeration techniques are surveyed elsewhere [3]. "
            "Implementations rarely state the size domain they enumerate, which makes "
            "reported checks hard to interpret.\n\nWe contribute a parameterised harness "
            "that generates the examined sizes from an exponent and reports the measured "
            "count. The scope of this article is the harness and one recorded run.")),
        Section(id="methods", title="Methods", purpose="", word_target=120, body=(
            "Let n denote the maximum exponent used to generate the sizes. "
            "The worst-case time complexity is O(n^2) under the assumption of an "
            "adjacency matrix representation, and memory usage is O(n^2) because the "
            "matrix has n rows when n sizes are generated.\n\nThe harness is executed "
            "with the parameter max_exponent = 5 and the artificial seed = 0 so the run "
            "is reproducible from the recorded parameters. One repetition is recorded.")),
        Section(id="algorithm", title="Algorithm", purpose="", word_target=120, body=(
            "The listing below is the exact code that was executed for the reported "
            "run:\n\n```python\n" + code + "\n```\n"
            "The asserted non-negativity guard covers the generated sizes.")),
        Section(id="results", title="Results", purpose="", word_target=120, body=(
            "The executed harness returned cycles_checked = 4 for the generated range, "
            "which matches the four sizes produced by checked_sizes with max_exponent = "
            "5. The unit tests executed with the run also passed.")),
        Section(id="discussion", title="Discussion", purpose="", word_target=120, body=(
            "The measured count is consistent with the mapping described in the methods, "
            "so the contribution of this article is a traceable harness rather than a new "
            "cyclability result. The observations are limited to the single recorded "
            "configuration.")),
        Section(id="limitations", title="Limitations", purpose="", word_target=120, body=(
            "The enumeration covers a finite range of exponents, so these results cannot "
            "establish any statement about larger sizes. The complexity discussion "
            "assumes an adjacency-matrix representation and was not measured empirically; "
            "memory behaviour is therefore analysed rather than observed. Findings are "
            "limited to the recorded configuration and one repetition.")),
        Section(id="conclusion", title="Conclusion", purpose="", word_target=120, body=(
            "The harness generates the examined sizes and the recorded run returned four "
            "checked sizes. This is a reproducibility contribution; no general "
            "cyclability claim is made, and the limitation to a finite exponent range "
            "stands.")),
    ])


def erdos_draft(brief: Brief) -> Draft:
    """The failing article: every Section-1 failure class in one manuscript."""
    draft = good_draft(brief, code=ERDOS_CODE)
    sections = draft.section_map()
    sections["introduction"].body = (
        "Monochromatic cycles in edge-coloured complete graphs are a classical topic "
        "[1]. We describe a two-direction traversal and claim it eliminates every "
        "monochromatic cycle for all n, in both directions.")
    sections["results"].body = (
        "The traversal eliminated monochromatic cycles in both directions for every "
        "size we examined. The algorithm ran in 12.3 seconds and was 2.5x faster than "
        "the baseline. We tested five cases, and the method was close to failing.")
    sections["discussion"].body = (
        "This proves the conjecture holds for all n. The results are complete.")
    sections["limitations"].body = "None."
    sections["conclusion"].body = (
        "The traversal eliminates all monochromatic cycles for every n and proves the "
        "general claim.")
    return draft


def replace_section(draft: Draft, section_id: str, body: str) -> Draft:
    """Return a deep copy with one section body replaced."""
    clone = copy.deepcopy(draft)
    target = clone.section_map()[section_id]
    target.body = body
    return clone


# ---------------------------------------------------------------------------
# Validation bundle
# ---------------------------------------------------------------------------


def validate(draft: Draft, brief: Brief, *, plan=None, library: Optional[SourceLibrary] = None,
             runs: Sequence[ExperimentRun] = (), specs: Sequence[ExperimentSpec] = (),
             review=None, retrieval: Optional[Dict[str, Any]] = None,
             benchmarks: Optional[Sequence[Any]] = None,
             apply_bibliography_first: bool = True) -> Dict[str, Any]:
    """Run the deterministic stack exactly as the pipeline does.

    ``apply_bibliography_first=False`` keeps a hand-written references section so the
    tests can exercise fabricated entries; the pipeline itself always regenerates it.
    """
    library = library if library is not None else verified_library()
    if apply_bibliography_first:
        apply_bibliography(draft, library)
    results = validators.results_from_runs(list(runs))
    ledger = evidence.build_ledger(draft, library, results)
    for run in runs:
        ledger.add_run(run)
    code_analysis = codecheck.analyse_code(
        draft, list(runs), article_type_research=True, specs=list(specs))
    if benchmarks is None:
        benchmarks = analysis.build_benchmark_records(list(runs), list(specs), brief)
    report = validators.validate_manuscript(
        draft, brief, plan=plan, library=library, ledger=ledger, runs=list(runs),
        specs=list(specs), benchmarks=list(benchmarks), review=review,
        code_analysis=code_analysis,
        retrieval=retrieval if retrieval is not None else {
            "attempted": True, "available": True},
        include_review=True,
    )
    gate = quality.evaluate_quality_gate(
        brief=brief, draft=draft, validation=report, review=review, library=library,
        runs=list(runs), specs=list(specs), plan=plan,
    )
    return {"validation": report, "gate": gate, "ledger": ledger,
            "code": code_analysis, "library": library, "draft": draft,
            "benchmarks": list(benchmarks)}


def status_of(bundle: Dict[str, Any], check_id: str) -> str:
    check = bundle["validation"].get(check_id)
    assert check is not None, f"check {check_id} was not run; have " \
        f"{[c.check_id for c in bundle['validation'].checks]}"
    return check.status.value


# ---------------------------------------------------------------------------
# Fake provider for pipeline / orchestrator tests
# ---------------------------------------------------------------------------


class FakeResearchClient(LLMClient):
    """Provider stub: planner, reviewer and reviser answers, no network."""

    engine_name = "fake-research"
    available = True

    def __init__(self, plan_payload: Optional[Dict[str, Any]] = None,
                 reviewer_payload: Optional[Dict[str, Any]] = None,
                 reviser_payload: Optional[Dict[str, Any]] = None,
                 planner_text: Optional[str] = None,
                 raise_on_planner: bool = False) -> None:
        self.plan_payload = plan_payload
        self.reviewer_payload = reviewer_payload or {"issues": []}
        self.reviser_payload = reviser_payload or {"sections": [], "addressed": [],
                                                   "notes": []}
        self.planner_text = planner_text
        self.raise_on_planner = raise_on_planner
        self.calls: List[str] = []

    def complete(self, system: str, user: str, *, max_output_tokens: int = 1000
                 ) -> LLMResponse:
        if "# Research Planner" in system:
            self.calls.append("planner")
            if self.raise_on_planner:
                from articlegen.llm import ProviderError

                raise ProviderError("planner provider unavailable (fixture)")
            text = self.planner_text if self.planner_text is not None else json.dumps(
                self.plan_payload or {})
            return LLMResponse(text=text, tokens_in=10, tokens_out=10,
                               engine=self.engine_name)
        if "# Scientific Reviewer" in system:
            self.calls.append("reviewer")
            return LLMResponse(text=json.dumps(self.reviewer_payload), tokens_in=10,
                               tokens_out=5, engine=self.engine_name)
        if "# Targeted Reviser" in system:
            self.calls.append("reviser")
            return LLMResponse(text=json.dumps(self.reviser_payload), tokens_in=10,
                               tokens_out=5, engine=self.engine_name)
        # Writer/Tester/Reviewer/Architect/Synthesizer fixtures are supplied by the
        # caller when the orchestrator is exercised.
        return LLMResponse(text=json.dumps({"notes": []}), tokens_in=1, tokens_out=1,
                           engine=self.engine_name)


def memory_store(tmp_path, task_id: str) -> MemoryStore:
    return MemoryStore(str(tmp_path), task_id)
