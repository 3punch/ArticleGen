"""Orchestration of the research stages around the writing loop.

Order of operations:

1. **Plan** - a validated ten-part research plan (model planner, deterministic
   fallback that leaves unknowns unknown).
2. **Sources** - retrieve and verify bibliographic records, link them to the plan.
3. **Experiments** - execute the plan's experiments *before* drafting, so the Writer
   receives recorded results instead of inventing them.
4. *(the existing Writer -> Tester -> Reviewer -> Architect -> Synthesizer loop)*
5. **Code verification** - execute displayed listings that were not already executed
   by a planned experiment, and compare hashes.
6. **Peer review and targeted revision** - model + deterministic review, bounded
   revision pass, re-validation after every change.
7. **Validation and quality gate** - deterministic checks, three-outcome gate, and a
   labelled export.

The pipeline never invents a plan element, source record or result to fill a gap.
Anything unavailable is recorded as unavailable and carried into the gate.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..events import EventBus, EventType
from ..llm import LLMClient, ProviderError, ResponseParseError, parse_json
from ..memory import MemoryStore
from ..models import ArticleType, Brief, Draft, Section
from ..prompts import PromptError, load_template
from ..tokens import count_tokens, truncate_to_tokens
from . import analysis as analysis_module
from . import codecheck, evidence, quality, report, review, validators
from .codecheck import SubprocessExecutor
from .models import (
    BenchmarkRecord,
    CheckStatus,
    ContributionType,
    EvidenceKind,
    ExperimentRun,
    ExperimentSpec,
    ExperimentStatus,
    GateStatus,
    QualityGateResult,
    ResearchPlan,
    ReviewReport,
    SourceRecord,
    ValidationReport,
    classify_section_role,
    profile_for,
)
from .planning import (derive_plan, outline_additions, plan_from_payload, plan_summary_block,
                       validate_plan)
from .sources import (
    NullSourceRetriever,
    SourceLibrary,
    SourceRetriever,
    UserSourceRetriever,
    resolve_retriever,
)

#: Patterns refused before execution. This is a guard against an obviously destructive
#: model-generated listing, not a security boundary - the executor is a subprocess.
_REFUSE_PATTERNS = (
    "shutil.rmtree", "os.system", "os.remove", "os.rmdir", "subprocess.",
    "socket.", "urllib.request", "requests.", "eval(", "exec(", "while True",
)


@dataclass
class ResearchOutcome:
    enabled: bool = False
    plan: Optional[ResearchPlan] = None
    library: SourceLibrary = field(default_factory=SourceLibrary)
    runs: List[ExperimentRun] = field(default_factory=list)
    specs: List[ExperimentSpec] = field(default_factory=list)
    benchmarks: List[BenchmarkRecord] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)
    ledger: Optional[evidence.EvidenceLedger] = None
    code_analysis: Optional[codecheck.CodeAnalysis] = None
    validation: Optional[ValidationReport] = None
    review: Optional[ReviewReport] = None
    gate: Optional[QualityGateResult] = None
    retrieval: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    brief_additions: List[str] = field(default_factory=list)
    tokens: int = 0
    draft: Optional[Draft] = None

    @property
    def quality_status(self) -> str:
        return self.gate.status.value if self.gate is not None else ""

    def to_dict(self, *, brief: Optional[Brief] = None,
                markdown: str = "") -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "notes": list(self.notes),
            "tokens": self.tokens,
            "brief_additions": list(self.brief_additions),
            "plan": self.plan.to_dict() if self.plan else None,
            "sources": self.library.to_dict(),
            "retrieval": self.retrieval,
            "experiments": {
                "specs": [spec.to_dict() for spec in self.specs],
                "runs": [run.to_dict() for run in self.runs],
            },
            "results": [r.to_dict() if hasattr(r, "to_dict") else r
                        for r in self.results],
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "evidence": self.ledger.to_dict() if self.ledger else None,
            "code_analysis": self.code_analysis.to_dict() if self.code_analysis else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "review": self.review.to_dict() if self.review else None,
            "quality_gate": self.gate.to_dict() if self.gate else None,
            "markdown": markdown,
        }


def _looks_unsafe(code: str) -> Optional[str]:
    lowered = code.lower()
    for pattern in _REFUSE_PATTERNS:
        if pattern.lower() in lowered:
            return pattern
    return None


class ResearchPipeline:
    def __init__(self, *, client: LLMClient, store: MemoryStore, brief: Brief,
                 enabled: bool, model_planning: bool = True,
                 model_review: bool = True, source_retrieval: Optional[str] = None,
                 max_sources: Optional[int] = None,
                 review_iterations: Optional[int] = None,
                 experiment_timeout: Optional[int] = None,
                 execute_experiments: Optional[bool] = None,
                 citation_style=None, prompt_dir: Optional[str] = None,
                 events: Optional[EventBus] = None,
                 retriever: Optional[SourceRetriever] = None,
                 executor: Optional[SubprocessExecutor] = None) -> None:
        self.client = client
        self.store = store
        self.brief = brief
        self.outcome = ResearchOutcome(enabled=enabled)
        self.prompt_dir = prompt_dir
        self.events = events or EventBus()
        self.retriever = retriever
        self.executor = executor or SubprocessExecutor(
            timeout=int(experiment_timeout or brief.experiment_timeout or 30))
        self.model_planning = model_planning and bool(getattr(client, "available", False))
        self.model_review = model_review
        self.source_policy = source_retrieval or brief.source_retrieval or "auto"
        self.max_sources = int(max_sources or brief.max_sources or 20)
        self.review_iterations = max(0, int(review_iterations if review_iterations is not None
                                            else brief.review_iterations))
        self.execute_experiments = (brief.execute_experiments
                                    if execute_experiments is None else execute_experiments)
        self.citation_style = citation_style or brief.citation_style
        self.profile = profile_for(brief.article_type)
        self._context: str = ""

    # -- helpers ----------------------------------------------------------
    def _emit(self, stage: str, status: str, summary: str, **payload: Any) -> None:
        self.events.emit(EventType.RESEARCH_STAGE, task_id=self.brief.task_id,
                         message=summary, stage=stage, status=status,
                         summary=summary, **payload)

    def _note(self, message: str) -> None:
        if message not in self.outcome.notes:
            self.outcome.notes.append(message)

    def _persist(self, name: str, payload: Dict[str, Any]) -> None:
        try:
            self.store.write_research_json(name, payload)
        except OSError as exc:  # pragma: no cover - disk failure is surfaced, not hidden
            self._note(f"could not persist {name}: {exc}")

    def _usage(self, response) -> int:
        spent = int(getattr(response, "tokens_in", 0)) + int(getattr(response, "tokens_out", 0))
        self.outcome.tokens += spent
        return spent

    # -- stage 1: plan ----------------------------------------------------
    def _planner_user_prompt(self) -> str:
        brief = self.brief
        outline = "\n".join(
            f"- {section.id} | {section.title} | {section.purpose} | "
            f"{section.word_target} words"
            for section in brief.sections
        )
        profile = self.profile
        return "\n".join([
            f"task_id: {brief.task_id}",
            f"title: {brief.title}",
            f"objective: {brief.objective}",
            f"audience: {brief.audience}",
            f"article_type: {brief.article_type.value}",
            f"venue: {brief.venue or '(not specified)'}",
            f"citation_style: {brief.citation_style.value}",
            f"research_question_from_user: {brief.research_question or '(not supplied)'}",
            f"contribution_from_user: {brief.contribution_type or '(not supplied)'}",
            f"novelty_claim_requested: {'yes' if brief.novelty_claim else 'no'}",
            f"constraints: {'; '.join(brief.constraints) or '(none)'}",
            f"target_words: {brief.target_words}",
            "required_section_roles: " + ", ".join(profile.required_roles),
            "requires_experiments: " + ("yes" if profile.requires_experiments else "no"),
            "requires_correctness_argument: "
            + ("yes" if profile.requires_correctness_argument else "no"),
            "requires_complexity_analysis: "
            + ("yes" if profile.requires_complexity_analysis else "no"),
            "outline:",
            outline,
        ])

    def _build_plan(self) -> ResearchPlan:
        brief = self.brief
        plan: Optional[ResearchPlan] = None
        if self.outcome.enabled and self.model_planning:
            try:
                system = load_template("planner", self.prompt_dir)
                response = self.client.complete(
                    system, self._planner_user_prompt(), max_output_tokens=6144)
                self._usage(response)
                payload = parse_json(response.text)
                if isinstance(payload, dict) and payload:
                    plan = plan_from_payload(payload, brief, origin="model")
                else:
                    self._note("planner returned no usable JSON; the plan was not "
                               "populated by a model")
            except (PromptError, ProviderError, ResponseParseError, ValueError) as exc:
                self._note(f"model planning unavailable: {type(exc).__name__}: {exc}")
            except Exception as exc:  # noqa: BLE001 - planning must not abort a run
                self._note(f"model planning failed: {type(exc).__name__}: {exc}")
        if plan is None:
            plan = derive_plan(brief, origin="deterministic")
            if self.outcome.enabled:
                self._note(
                    "no model-generated plan was available; only brief-derived plan "
                    "fields are present and missing research elements are reported"
                )
        self.outcome.plan = plan
        return plan

    def _augment_brief(self, plan: ResearchPlan) -> None:
        """Add required-role and references sections before drafting.

        The gate already accepted the brief. These additions are structural, not
        content: they give the Writer a place to put a required section and a place
        for the verified bibliography (which is written from records, not by the model).
        """
        if not self.outcome.enabled:
            return
        additions = outline_additions(plan, self.brief)
        existing_roles = {classify_section_role(s.id, s.title) for s in self.brief.sections}
        if "references" not in existing_roles and self.outcome.library.records:
            bibliography = self.outcome.library.bibliography()
            if bibliography:
                additions.append(Section(
                    id="references", title="References",
                    purpose=("List only the verified source records retrieved for this "
                             "run; the bibliography is generated from those records."),
                    word_target=120,
                ))
        added: List[str] = []
        for section in additions:
            if len(self.brief.sections) >= 12:
                self._note(f"skipped outline addition '{section.id}': 12-section maximum")
                continue
            if section.id in self.brief.section_ids:
                continue
            self.brief.sections.append(section)
            added.append(section.id)
        if added:
            self.outcome.brief_additions = added
            self._note("outline extended with required section(s): " + ", ".join(added))

    # -- stage 2: sources -------------------------------------------------
    def _queries(self, plan: ResearchPlan) -> List[str]:
        queries = [q for q in plan.source_queries if q.strip()]
        if not queries and plan.research_question:
            queries.append(plan.research_question)
        if not queries:
            queries.append(self.brief.title)
        if plan.gap:
            queries.append(plan.gap[:180])
        for material in plan.required_materials:
            if len(material) > 20 and len(queries) < 5:
                queries.append(material[:180])
        return list(dict.fromkeys(q.strip() for q in queries if q.strip()))[:6]

    def _collect_sources(self, plan: ResearchPlan) -> None:
        library = SourceLibrary(self.citation_style)
        self.outcome.library = library
        library.from_brief_sources(self.brief.sources)
        needs_sources = self.profile.requires_sources or self.brief.requires_citations
        if not needs_sources:
            self.outcome.retrieval = {
                "attempted": False, "available": False, "provider": "not-required",
                "queries": [], "retrieved": 0, "errors": [],
            }
            self._persist("sources.json", library.to_dict())
            return

        retriever = self.retriever
        if retriever is None:
            try:
                retriever = resolve_retriever(
                    self.source_policy, timeout=12,
                    user_sources=[SourceRecord.from_dict(r.to_dict()) for r in library.records],
                )
            except ValueError as exc:
                self._note(f"invalid source retrieval policy: {exc}")
                retriever = NullSourceRetriever()
        queries = self._queries(plan)
        if retriever.available():
            library.retrieve(queries, retriever, limit_per_query=4,
                             max_sources=self.max_sources)
        else:
            self._note(
                f"source retrieval is unavailable ({retriever.name}); no bibliography "
                "will be fabricated"
            )
        if retriever.verifies:
            library.verify(retriever)
        else:
            library.verify(None)
        self.outcome.retrieval = dict(library.retrieval.to_dict())
        self.outcome.retrieval["available"] = bool(retriever.available())
        self.outcome.retrieval["provider"] = retriever.name
        self._persist("sources.json", library.to_dict())
        source_report = library.report()
        self._emit("sources", "done" if library.records else "incomplete",
                   f"{len(library.records)} source record(s); {source_report.verified} "
                   f"index-verified, {source_report.user_provided} user-provided",
                   records=len(library.records), verified=source_report.verified,
                   user_provided=source_report.user_provided,
                   unverified=source_report.unverified,
                   provider=retriever.name, queries=len(queries))

    # -- stage 3: experiments --------------------------------------------
    def _specs_from_plan(self, plan: ResearchPlan) -> List[ExperimentSpec]:
        specs: List[ExperimentSpec] = []
        seen = set()
        for index, item in enumerate(plan.experiments, start=1):
            if not isinstance(item, dict):
                continue
            spec = ExperimentSpec.from_dict(item)
            # Leading/trailing blank lines are formatting, not code: normalising them
            # keeps "displayed == executed" a statement about the program.
            spec.code = spec.code.strip("\n")
            spec.tests = spec.tests.strip("\n")
            if not spec.experiment_id:
                spec.experiment_id = f"exp-{index}"
            if spec.experiment_id in seen:
                spec.experiment_id = f"{spec.experiment_id}-{index}"
            seen.add(spec.experiment_id)
            specs.append(spec)
        return specs

    def _run_spec(self, spec: ExperimentSpec) -> ExperimentRun:
        unsafe = _looks_unsafe(spec.code)
        if unsafe:
            run = ExperimentRun(
                experiment_id=spec.experiment_id,
                run_id=f"{spec.experiment_id}-refused",
                status=ExperimentStatus.SKIPPED,
                code_sha256=codecheck.hash_code(spec.code),
                error=(f"execution refused: the code contains '{unsafe}', which the "
                       "research executor will not run. The listing is unverified."),
                environment=codecheck.execution_environment(),
                started_at=codecheck.utc_now(),
                finished_at=codecheck.utc_now(),
            )
            return run
        return self.executor.run(spec, timeout=int(self.brief.experiment_timeout or 30))

    def _execute_plan_experiments(self, plan: ResearchPlan) -> None:
        specs = self._specs_from_plan(plan)
        self.outcome.specs = specs
        if not specs:
            if self.profile.requires_experiments or self.brief.requires_experiments:
                self._note("the plan requires experiments but specifies no executable "
                           "experiment; no result will be reported")
            self.outcome.runs = []
            self._persist("experiments.json",
                          {"specs": [], "runs": [], "note": "no experiment specified"})
            return
        runs: List[ExperimentRun] = []
        if not self.execute_experiments:
            for spec in specs:
                runs.append(ExperimentRun(
                    experiment_id=spec.experiment_id,
                    run_id=f"{spec.experiment_id}-not-run",
                    status=ExperimentStatus.NOT_RUN,
                    code_sha256=codecheck.hash_code(spec.code),
                    error="experiment execution is disabled for this run",
                ))
            self._note("experiment execution is disabled; the manuscript cannot be a "
                       "completed study")
        else:
            for spec in specs:
                run = self._run_spec(spec)
                runs.append(run)
                self._emit(
                    "experiment", run.status.value,
                    f"experiment {spec.experiment_id}: {run.status.value}",
                    experiment_id=spec.experiment_id, exit_code=run.exit_code,
                    tests_run=run.tests_run, tests_passed=run.tests_passed,
                )
        self.outcome.runs = runs
        self.outcome.results = validators.results_from_runs(
            [run for run in runs if run.status == ExperimentStatus.OK])
        self.outcome.ledger = evidence.EvidenceLedger()
        for run in runs:
            self.outcome.ledger.add_run(run)
        for result in self.outcome.results:
            self.outcome.ledger.add_result(result)
        self._persist("experiments.json", {
            "specs": [spec.to_dict() for spec in specs],
            "runs": [run.to_dict() for run in runs],
            "results": [r.to_dict() for r in self.outcome.results],
        })

    # -- context for the drafting loop -----------------------------------
    def context_block(self, max_tokens: Optional[int] = None) -> str:
        """The evidence packet handed to the writing agents.

        ``max_tokens`` defaults to ``None``: the pipeline applies no token ceiling
        (see ``articlegen.tokens``). A caller may still request a limit explicitly.
        """
        if not self.outcome.enabled:
            return ""
        plan = self.outcome.plan
        parts = ["RESEARCH MODE: use only the evidence below. Never invent a source, "
                 "result, metric or proof."]
        if plan is not None:
            parts.append("RESEARCH PLAN\n" + plan_summary_block(plan))
        parts.append("SOURCE KEYS (the only citable records)\n"
                     + self.outcome.library.context_block(max_records=10,
                                                          max_abstract_chars=180))
        parts.append("EXECUTED RESULTS (the only reportable results)\n"
                     + evidence.evidence_context_block(
                         self.outcome.ledger or evidence.EvidenceLedger(),
                         self.outcome.results, max_results=14))
        if self.outcome.specs:
            code_parts = []
            for spec in self.outcome.specs[:2]:
                if spec.code.strip():
                    code_parts.append(
                        f"Experiment {spec.experiment_id} (this exact code was executed; "
                        f"a displayed listing must match it):\n```python\n"
                        f"{spec.code[:2200]}\n```"
                    )
            if code_parts:
                parts.append("EXECUTED CODE\n" + "\n\n".join(code_parts))
        text = "\n\n".join(parts)
        self._context = (truncate_to_tokens(text, max_tokens)
                         if max_tokens is not None else text)
        return self._context

    # -- stage 5: verify displayed code ----------------------------------
    def _verify_displayed_code(self, draft: Draft) -> None:
        analysis = codecheck.analyse_code(
            draft, self.outcome.runs,
            article_type_research=self.profile.research, specs=self.outcome.specs)
        executed = {run.code_sha256 for run in self.outcome.runs if run.code_sha256}
        planned = [spec for spec in self.outcome.specs if spec.code.strip()]
        attempted: set = set(executed)
        executed_this_pass = 0
        for block in analysis.executable_blocks:
            if block.sha256 in attempted:
                continue
            if block.language.strip().lower() not in ("python", "py", "python3"):
                continue
            if len(block.body) > 200_000 or executed_this_pass >= 6:
                self._note(
                    f"displayed listing {block.section_id}#{block.index} was not executed "
                    "(size/limit); it remains unverified"
                )
                continue
            # If this exact code was already specified (for example a refused plan
            # experiment), do not execute it a second time.
            same_code = next((s for s in planned
                              if s.code.strip("\n").strip() == block.body.strip()), None)
            if same_code is not None and codecheck.hash_code(same_code.code) == block.sha256:
                # Byte-identical to code that already ran: nothing new to verify.
                continue
            attempted.add(block.sha256)
            executed_this_pass += 1
            spec = ExperimentSpec(
                experiment_id=f"listing-{block.section_id}-{block.index}",
                title=f"displayed listing in {block.section_id}",
                description="Executed to verify that the displayed code runs.",
                language="python",
                code=block.body,
                section_id=block.section_id,
            )
            run = self._run_spec(spec)
            self.outcome.runs.append(run)
            self.outcome.specs.append(spec)
            self._emit("code_execution", run.status.value,
                       f"displayed listing {block.section_id}#{block.index}: {run.status.value}",
                       experiment_id=spec.experiment_id)
        self.outcome.results = validators.results_from_runs(
            [run for run in self.outcome.runs if run.status == ExperimentStatus.OK])
        if self.outcome.ledger is None:
            self.outcome.ledger = evidence.EvidenceLedger()
        else:
            existing = {(r.experiment_id, r.metric) for r in self.outcome.ledger.results}
            for result in self.outcome.results:
                if (result.experiment_id, result.metric) not in existing:
                    self.outcome.ledger.add_result(result)
        for run in self.outcome.runs:
            if run not in self.outcome.ledger.runs:
                self.outcome.ledger.add_run(run)

    # -- stage 6-7: review, validation, gate -----------------------------
    def complete(self, draft: Draft) -> Draft:
        if not self.outcome.enabled:
            return draft
        self._verify_displayed_code(draft)
        report.apply_bibliography(draft, self.outcome.library)
        self.outcome.benchmarks = analysis_module.build_benchmark_records(
            self.outcome.runs, self.outcome.specs, self.brief)

        reviewer = review.ScientificReviewer(
            self.client if (self.model_review and getattr(self.client, "available", False))
            else None,
            prompt_dir=self.prompt_dir, model_review=self.model_review)
        reviser = review.TargetedReviser(reviewer.client, prompt_dir=self.prompt_dir)
        self._emit("peer_review", "running", "scientific review started")
        revised, validation, review_report = review.run_review_loop(
            draft, self.brief, plan=self.outcome.plan,
            library=self.outcome.library, ledger=self.outcome.ledger,
            runs=self.outcome.runs, specs=self.outcome.specs,
            benchmarks=self.outcome.benchmarks,
            retrieval=self.outcome.retrieval,
            code_analysis=None, reviewer=reviewer, reviser=reviser,
            max_iterations=self.review_iterations,
        )
        draft = revised
        report.apply_bibliography(draft, self.outcome.library)
        blocking = review_report.blocking()
        self._emit(
            "peer_review", "done",
            f"{len(review_report.issues)} finding(s), {len(blocking)} unresolved "
            f"blocking; {review_report.iterations_run} revision iteration(s)",
            issues=[issue.to_dict() for issue in blocking[:15]],
            severity_counts=review_report.severity_counts(),
            iterations=review_report.iterations_run,
            model_review_ran=review_report.model_review_ran,
            notes=review_report.notes[:6],
        )
        # Recompute evidence and code analysis against the final text.
        self.outcome.ledger = evidence.build_ledger(
            draft, self.outcome.library, self.outcome.results)
        for run in self.outcome.runs:
            self.outcome.ledger.add_run(run)
        self.outcome.code_analysis = codecheck.analyse_code(
            draft, self.outcome.runs,
            article_type_research=self.profile.research, specs=self.outcome.specs)
        self.outcome.review = review_report
        self.outcome.validation = validators.validate_manuscript(
            draft, self.brief, plan=self.outcome.plan,
            library=self.outcome.library, ledger=self.outcome.ledger,
            runs=self.outcome.runs, specs=self.outcome.specs,
            benchmarks=self.outcome.benchmarks, review=review_report,
            code_analysis=self.outcome.code_analysis,
            retrieval=self.outcome.retrieval, include_review=True,
        )
        validation_summary = self.outcome.validation.summary()
        problems = self.outcome.validation.problems()
        self._emit(
            "validation", "done" if not problems else "incomplete",
            f"{validation_summary['pass']} passed, {validation_summary['fail']} failed, "
            f"{validation_summary['unverified']} unverified",
            counts=validation_summary,
            failing=[{"check": c.check_id, "severity": c.severity.value,
                      "details": c.details[:240], "effect": c.gate_effect}
                     for c in problems[:20]],
        )
        self.outcome.gate = quality.evaluate_quality_gate(
            brief=self.brief, draft=draft, validation=self.outcome.validation,
            review=review_report, library=self.outcome.library,
            runs=self.outcome.runs, specs=self.outcome.specs,
            plan=self.outcome.plan,
        )
        self.outcome.draft = draft
        self._persist("validation.json", self.outcome.validation.to_dict())
        self._persist("review.json", review_report.to_dict())
        self._persist("quality_gate.json", self.outcome.gate.to_dict())
        self._persist("plan.json", self.outcome.plan.to_dict() if self.outcome.plan else {})
        self._persist("benchmarks.json", {
            "benchmarks": [b.to_dict() for b in self.outcome.benchmarks]})
        self._persist("evidence.json",
                      self.outcome.ledger.to_dict() if self.outcome.ledger else {})
        if self.outcome.code_analysis:
            self._persist("code_analysis.json", self.outcome.code_analysis.to_dict())
        try:
            self.store.write_research_text(
                "revision_summary.md", review.revision_summary_markdown(review_report))
            markdown = report.manuscript_with_appendix(
                draft, brief=self.brief, plan=self.outcome.plan,
                library=self.outcome.library, runs=self.outcome.runs,
                specs=self.outcome.specs, validation=self.outcome.validation,
                review=review_report, gate=self.outcome.gate,
                benchmarks=self.outcome.benchmarks,
            )
            self.store.write_research_text("report.md", markdown)
            self.store.write_research_text("draft.md", draft.to_markdown())
        except OSError as exc:
            self._note(f"could not write research report: {exc}")
        self._emit(
            "quality_gate", self.outcome.gate.status.value,
            f"quality gate: {self.outcome.gate.status.label}",
            reasons=self.outcome.gate.reasons[:6],
            critical=len(self.outcome.gate.critical_failures),
            states=self.outcome.gate.states.to_dict(),
        )
        return draft

    # -- entry point ------------------------------------------------------
    def begin(self) -> None:
        """Run plan, source and experiment stages before the writing loop."""
        plan = self._build_plan()
        self._collect_sources(plan)
        validate_plan(
            plan, self.brief, library=self.outcome.library,
            retrieval_attempted=bool(self.outcome.retrieval.get("attempted")),
            retrieval_available=bool(self.outcome.retrieval.get("available")),
        )
        self._augment_brief(plan)
        self._execute_plan_experiments(plan)
        self._persist("plan.json", plan.to_dict())
        self._emit(
            "plan", "done" if plan.validated else "incomplete",
            f"research plan {'validated' if plan.validated else 'incomplete'}: "
            f"{plan.decision}",
            decision=plan.decision, validated=plan.validated,
            article_type=plan.article_type.value,
            contribution_type=plan.contribution_type.value,
            research_question=plan.research_question[:280],
            contribution=plan.contribution[:280],
            decision_reasons=plan.decision_reasons[:5],
            missing_elements=plan.missing_elements()[:8],
        )

    def export_markdown(self) -> str:
        if not self.outcome.enabled or self.outcome.draft is None:
            return self.outcome.draft.to_markdown() if self.outcome.draft else ""
        return report.manuscript_with_appendix(
            self.outcome.draft, brief=self.brief, plan=self.outcome.plan,
            library=self.outcome.library, runs=self.outcome.runs,
            specs=self.outcome.specs, validation=self.outcome.validation,
            review=self.outcome.review, gate=self.outcome.gate,
            benchmarks=self.outcome.benchmarks,
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.outcome.to_dict(brief=self.brief, markdown=self.export_markdown())
