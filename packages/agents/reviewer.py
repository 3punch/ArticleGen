"""Evidence-bearing ProjectSpec reviewer with deterministic static/test checks.

The reviewer never owns workflow control. Automatable assessments are made from
bounded source snapshots and sandbox audit records; subjective assessments are only
recommendations and remain explicitly pending a human decision.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from projectforge_schemas.models import (
    HumanReviewRecommendation,
    ProjectSpec,
    ReviewCriterionAssessment,
    ReviewerAgentResponse,
    ReviewerHumanResponse,
)

from .config import ModelConfig
from .errors import AgentOutputValidationError
from .langgraph_runner import LangGraphModelCall
from .litellm_client import ChatModelClient, LiteLLMChatClient, ModelCompletion


_PROMPT = Path(__file__).with_name("prompts").joinpath("reviewer_v1.txt").read_text(
    encoding="utf-8",
)
_STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "then", "than",
    "when", "where", "which", "while", "must", "should", "have", "has", "had",
    "are", "was", "were", "will", "can", "may", "all", "each", "every", "only",
    "match", "matches", "matched", "according", "against", "based", "provided",
    "expected", "independently", "computed", "calculated", "fixture", "value", "values",
    "result", "results", "output", "required", "correctly", "clear", "actionable",
    "produce", "produces", "return", "returns", "contain", "contains", "verify",
    "check", "checks", "checked", "test", "tests", "using", "both", "from", "also",
    "into", "more", "less", "over", "under", "each", "should", "ensure", "every",
}
_TEST_TERMS = {
    "test", "tests", "pytest", "unittest", "assert", "asserts", "assertion", "run",
    "execute", "execution", "coverage", "runner",
}
_STATIC_TERMS = {"syntax", "parse", "parsing", "compile", "compilation", "static", "lint"}


class ReviewerAgent:
    """Combine deterministic checks with an LLM's non-binding human-review advice."""

    agent_name = "reviewer"
    prompt_version = "reviewer_v1"

    def __init__(self, *, client: ChatModelClient | None = None,
                 config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig.from_env()
        self.client = client or LiteLLMChatClient(self.config)
        self.model_call = LangGraphModelCall(self.client)

    def run(
        self,
        project_spec: ProjectSpec | Mapping[str, Any],
        *,
        repository_files: Mapping[str, str],
        test_runs: Sequence[Mapping[str, Any]],
        timeout_seconds: int | None = None,
    ) -> tuple[ReviewerAgentResponse, ModelCompletion | None]:
        spec = ProjectSpec.model_validate(project_spec)
        automated_items = [item for item in spec.rubric_items if item.type == "automatable"]
        human_items = [item for item in spec.rubric_items if item.type == "human-judgment"]
        python_files = {
            path: content for path, content in repository_files.items()
            if Path(path).suffix.casefold() in {".py", ".pyi"}
        }
        syntax_errors = self._python_syntax_errors(python_files)
        automated = [
            self._assess_automatable(item, repository_files, test_runs, syntax_errors)
            for item in automated_items
        ]
        human, completion = self._review_human_items(
            spec,
            human_items,
            repository_files=repository_files,
            test_runs=test_runs,
            timeout_seconds=timeout_seconds,
        )
        human_by_criterion = {item.rubric_item.criterion.casefold(): item for item in human}
        criteria = [
            (human_by_criterion[item.criterion.casefold()] if item.type == "human-judgment"
             else self._assessment_by_criterion(automated, item.criterion))
            for item in spec.rubric_items
        ]
        criteria_evidence = [
            f"{assessment.rubric_item.criterion}: {assessment.status}; "
            + " ".join(assessment.evidence[:2])
            for assessment in criteria
        ]
        automated_passed = all(item.status == "passed" for item in automated)
        summary = (
            f"{sum(item.status == 'passed' for item in automated)} of {len(automated)} "
            "automatable rubric item(s) passed the recorded checks. "
            f"{len(human)} human-judgment item(s) remain explicitly pending human review."
        )
        if human and completion is not None:
            summary += f" Reviewer recommendation summary: {human[0].rationale[:500]}"
        response = ReviewerAgentResponse(
            summary=summary,
            criteria=criteria,
            automatable_checks_passed=automated_passed,
            human_review_required=bool(human_items),
            evidence=criteria_evidence or ["No rubric evidence was produced."],
        )
        return response, completion

    def build_human_review_prompts(
        self,
        project_spec: ProjectSpec | Mapping[str, Any],
        human_criteria: Sequence[Any],
        *,
        repository_files: Mapping[str, str],
        test_runs: Sequence[Mapping[str, Any]],
    ) -> tuple[str, str]:
        spec = ProjectSpec.model_validate(project_spec)
        schema = json.dumps(ReviewerHumanResponse.model_json_schema(), ensure_ascii=False)
        safe_file_context: dict[str, str] = {}
        total_chars = 0
        priority = sorted(
            repository_files,
            key=lambda path: (
                0 if path.startswith("tests/") else
                1 if Path(path).name.casefold() in {"readme.md", "index.html"} else 2,
                path,
            ),
        )
        for path in priority:
            excerpt = repository_files[path][:2000]
            if total_chars + len(path) + len(excerpt) > 8_000:
                break
            safe_file_context[path] = excerpt
            total_chars += len(path) + len(excerpt)
        user_payload = {
            "project_spec": spec.model_dump(mode="json"),
            "human_judgment_criteria": [item.model_dump(mode="json") for item in human_criteria],
            "repository_files": safe_file_context,
            "sandbox_test_runs": [self._bounded_test_run(run) for run in test_runs],
        }
        return (
            f"{_PROMPT}\n\nExpected JSON Schema:\n{schema}",
            json.dumps(user_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )

    def _review_human_items(
        self,
        spec: ProjectSpec,
        human_items: Sequence[Any],
        *,
        repository_files: Mapping[str, str],
        test_runs: Sequence[Mapping[str, Any]],
        timeout_seconds: int | None,
    ) -> tuple[list[ReviewCriterionAssessment], ModelCompletion | None]:
        if not human_items:
            return [], None
        system_prompt, user_prompt = self.build_human_review_prompts(
            spec,
            human_items,
            repository_files=repository_files,
            test_runs=test_runs,
        )
        completion = self.model_call.invoke(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name=self.agent_name,
            timeout_seconds=timeout_seconds,
        )
        try:
            result = ReviewerHumanResponse.model_validate(json.loads(completion.content))
            expected = {item.criterion.casefold() for item in human_items}
            actual = {item.criterion.casefold() for item in result.recommendations}
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise ValueError(
                    f"human recommendations must exactly match the rubric (missing={missing}, extra={extra})"
                )
        except (json.JSONDecodeError, ValueError) as exc:
            raise AgentOutputValidationError(
                f"Reviewer agent returned invalid human-review output: {exc}",
                completion=completion,
            ) from exc
        recommendations = {
            item.criterion.casefold(): item for item in result.recommendations
        }
        assessments: list[ReviewCriterionAssessment] = []
        for rubric_item in human_items:
            recommendation: HumanReviewRecommendation = recommendations[rubric_item.criterion.casefold()]
            evidence = [*recommendation.evidence]
            if repository_files:
                evidence.append(
                    "Human inspection is mandatory; bounded repository evidence includes: "
                    + ", ".join(list(sorted(repository_files))[:20])
                )
            else:
                evidence.append("Human inspection is mandatory; no repository files were available to inspect.")
            assessments.append(ReviewCriterionAssessment(
                rubric_item=rubric_item,
                status="human_review_required",
                recommendation=recommendation.recommendation,
                rationale=recommendation.rationale,
                evidence=evidence[:200],
            ))
        return assessments, completion

    def _system_prompt(self) -> str:
        schema = json.dumps(ReviewerHumanResponse.model_json_schema(), ensure_ascii=False)
        return f"{_PROMPT}\n\nExpected JSON Schema:\n{schema}"

    @classmethod
    def _assess_automatable(
        cls,
        rubric_item: Any,
        repository_files: Mapping[str, str],
        test_runs: Sequence[Mapping[str, Any]],
        syntax_errors: Sequence[str],
    ) -> ReviewCriterionAssessment:
        evidence: list[str] = []
        if syntax_errors:
            evidence.append("Python static parse failed: " + "; ".join(syntax_errors[:10]))
            return ReviewCriterionAssessment(
                rubric_item=rubric_item, status="failed", evidence=evidence,
                rationale="At least one Python source or test file has a syntax error.",
            )

        python_paths = sorted(
            path for path in repository_files
            if Path(path).suffix.casefold() in {".py", ".pyi"}
        )
        if python_paths:
            evidence.append(
                f"AST static parse passed for {len(python_paths)} Python source/test file(s): "
                + ", ".join(python_paths[:20])
            )
        else:
            evidence.append("No Python files were present for AST static parsing.")

        coverage_paths = [
            path for path in repository_files
            if Path(path).name.casefold() in {"coverage.xml", ".coverage"}
            or path.casefold().startswith("htmlcov/")
        ]
        if coverage_paths:
            evidence.append("Coverage artifact(s) found: " + ", ".join(coverage_paths[:10]))
        else:
            evidence.append(
                "Coverage measurement was not recorded by the isolated test runs; no coverage pass is claimed."
            )

        test_related = cls._is_test_check(rubric_item)
        runs = [run for run in test_runs if str(run.get("phase", "TEST")).upper() == "TEST"]
        test_paths = sorted({
            str(path)
            for run in runs
            for path in run.get("tested_files", [])
            if isinstance(path, str)
        })
        test_files = {
            path: content for path, content in repository_files.items()
            if path.startswith("tests/") or Path(path).name.casefold().startswith("test_")
            or Path(path).name.casefold().endswith("_test.py")
            or ".test." in Path(path).name.casefold()
            or ".spec." in Path(path).name.casefold()
        }
        if test_paths:
            evidence.append("Sandbox test run(s) discovered: " + ", ".join(test_paths[:30]))

        if test_related:
            if not runs:
                evidence.append("No persisted sandbox test run was available to verify this criterion.")
                return ReviewCriterionAssessment(
                    rubric_item=rubric_item, status="not_verified", evidence=evidence,
                    rationale="The criterion calls for test or execution evidence, but none was recorded.",
                )
            failed_runs = [run for run in runs if run.get("passed") is not True]
            if failed_runs:
                evidence.extend(cls._test_run_evidence(failed_runs, passed=False))
                return ReviewCriterionAssessment(
                    rubric_item=rubric_item, status="failed", evidence=evidence,
                    rationale="One or more persisted sandbox test runs failed or timed out.",
                )
            evidence.extend(cls._test_run_evidence(runs, passed=True))
            matched = cls._matching_test_anchors(rubric_item, test_files)
            if not matched:
                evidence.append(
                    "Passing sandbox tests were recorded, but no criterion-specific terms were found "
                    "in the bounded test sources; this rubric item remains unverified."
                )
                return ReviewCriterionAssessment(
                    rubric_item=rubric_item, status="not_verified", evidence=evidence,
                    rationale="Passing tests could not be tied to this specific rubric criterion.",
                )
            evidence.append(
                "Criterion-specific test-source anchors matched: " + ", ".join(matched) + "."
            )
            return ReviewCriterionAssessment(
                rubric_item=rubric_item, status="passed", evidence=evidence,
                rationale="Recorded sandbox runs passed and test source contains criterion-specific anchors.",
            )

        path_evidence = cls._path_check(rubric_item, repository_files)
        if path_evidence is not None:
            path, exists = path_evidence
            evidence.append(f"Rubric path '{path}' {'exists' if exists else 'is missing'} in the repository.")
            return ReviewCriterionAssessment(
                rubric_item=rubric_item,
                status="passed" if exists else "failed",
                evidence=evidence,
                rationale="The required named path was checked against the repository snapshot.",
            )

        combined_text = "\n".join(repository_files.values()).casefold()
        anchors = cls._anchors(rubric_item.criterion)
        matched = [anchor for anchor in anchors if anchor in combined_text]
        if anchors and len(matched) >= min(2, len(anchors)):
            evidence.append(
                "Criterion-specific repository text anchors matched: " + ", ".join(matched[:20]) + "."
            )
            return ReviewCriterionAssessment(
                rubric_item=rubric_item, status="passed", evidence=evidence,
                rationale="The rubric's concrete identifiers were found in the project source snapshot.",
            )
        evidence.append(
            "No reproducible criterion-specific static check could be derived from the rubric wording."
        )
        return ReviewCriterionAssessment(
            rubric_item=rubric_item, status="not_verified", evidence=evidence,
            rationale="The Reviewer did not silently convert a human judgment into an automated pass.",
        )

    @staticmethod
    def _python_syntax_errors(repository_files: Mapping[str, str]) -> list[str]:
        errors: list[str] = []
        for path, content in sorted(repository_files.items()):
            if Path(path).suffix.casefold() not in {".py", ".pyi"}:
                continue
            try:
                ast.parse(content, filename=path)
            except (SyntaxError, ValueError) as exc:
                line = getattr(exc, "lineno", None)
                errors.append(f"{path}:{line or '?'}: {getattr(exc, 'msg', str(exc))}")
        return errors

    @staticmethod
    def _is_test_check(rubric_item: Any) -> bool:
        words = set(re.findall(
            r"[a-z][a-z0-9_-]*",
            f"{rubric_item.criterion} {rubric_item.check_method}".casefold(),
        ))
        return bool(words & _TEST_TERMS)

    @classmethod
    def _matching_test_anchors(
        cls,
        rubric_item: Any,
        test_files: Mapping[str, str],
    ) -> list[str]:
        text = "\n".join(test_files.values()).casefold()
        anchors = cls._anchors(rubric_item.criterion)
        return [anchor for anchor in anchors if anchor in text]

    @staticmethod
    def _anchors(value: str) -> list[str]:
        candidates = re.findall(r"[a-z][a-z0-9_-]{2,}", value.casefold())
        return list(dict.fromkeys(
            token for token in candidates
            if token not in _STOP_WORDS and len(token) >= 3
        ))

    @staticmethod
    def _path_check(
        rubric_item: Any,
        repository_files: Mapping[str, str],
    ) -> tuple[str, bool] | None:
        text = f"{rubric_item.criterion} {rubric_item.check_method}"
        paths = re.findall(r"`([^`]+)`", text)
        paths += re.findall(
            r"(?<![A-Za-z0-9_./-])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\.[A-Za-z0-9]{1,8})(?![A-Za-z0-9_./-])",
            text,
        )
        for path in paths:
            normalized = path.strip().lstrip("./")
            if (not normalized or normalized.startswith("/") or ".." in Path(normalized).parts
                    or "\\" in normalized):
                continue
            if Path(normalized).suffix:
                return normalized, normalized in repository_files
        return None

    @staticmethod
    def _test_run_evidence(
        runs: Sequence[Mapping[str, Any]],
        *,
        passed: bool,
    ) -> list[str]:
        evidence: list[str] = []
        for run in runs[:30]:
            task_id = str(run.get("task_id") or "unknown task")
            exit_code = run.get("exit_code", "unknown")
            timed_out = bool(run.get("timed_out", False))
            evidence.append(
                f"Sandbox task {task_id}: {'passed' if passed else 'failed'}; "
                f"exit_code={exit_code}, timed_out={timed_out}."
            )
        return evidence

    @staticmethod
    def _bounded_test_run(run: Mapping[str, Any]) -> dict[str, Any]:
        keys = (
            "task_id", "phase", "passed", "runtime", "command", "tested_files",
            "exit_code", "timed_out", "duration_seconds", "evidence",
        )
        bounded: dict[str, Any] = {}
        for key in keys:
            if key in run:
                value = run[key]
                if isinstance(value, list):
                    bounded[key] = value[:50]
                elif isinstance(value, str):
                    bounded[key] = value[:4000]
                else:
                    bounded[key] = value
        return bounded

    @staticmethod
    def _assessment_by_criterion(
        assessments: Sequence[ReviewCriterionAssessment],
        criterion: str,
    ) -> ReviewCriterionAssessment:
        for assessment in assessments:
            if assessment.rubric_item.criterion.casefold() == criterion.casefold():
                return assessment
        raise ValueError(f"no automated assessment exists for rubric item '{criterion}'")
