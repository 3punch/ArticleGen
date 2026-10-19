import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from articlegen.llm import LLMClient, LLMResponse  # noqa: E402
from articlegen.models import Brief, Section  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO_ROOT, "examples")


class TestLLMClient(LLMClient):
    """Deterministic provider stub for exercising the orchestration plumbing.

    This lives exclusively in the test suite. Production has no static drafting
    fallback; all generated bodies below are responses from this fake LLM client.
    """

    engine_name = "test-llm"
    available = True

    @staticmethod
    def _sections(user: str):
        outline = re.search(r"(?m)^outline:\s*\n((?:^[ \t]*- .*(?:\n|$))*)", user)
        if not outline:
            return []
        sections = []
        for line in outline.group(1).splitlines():
            fields = line.strip().lstrip("-").strip().split(" | ", 3)
            if len(fields) < 4:
                continue
            section_id, title, target_text, purpose = fields
            match = re.search(r"target\s+(\d+)w", target_text)
            target = int(match.group(1)) if match else 80
            sections.append((section_id.strip(), title.strip(), target, purpose.strip()))
        return sections

    @staticmethod
    def _body(title: str, purpose: str, target: int) -> str:
        sentence = f"{title} explains {purpose} in this test-model response."
        sentence_size = len(sentence.split())
        sentences = []
        words = 0
        while words < max(50, target):
            sentences.append(sentence)
            words += sentence_size
        paragraphs = [" ".join(sentences[index:index + 4])
                      for index in range(0, len(sentences), 4)]
        return "\n\n".join(paragraphs)

    def complete(self, system: str, user: str, *, max_output_tokens: int = 1000) -> LLMResponse:
        sections = self._sections(user)
        if "# Agent 1 — Writer" in system:
            payload = {
                "sections": [
                    {"id": section_id, "body": self._body(title, purpose, target)}
                    for section_id, title, target, purpose in sections
                ],
                "notes": [],
                "summary": "test LLM produced brief-specific section content",
            }
        elif "# Agent 5 — Synthesizer" in system:
            title = re.search(r"(?m)^title:\s*(.+)$", user)
            payload = {
                "title": title.group(1).strip() if title else "Test article",
                "sections": [
                    {"id": section_id, "title": section_title,
                     "body": self._body(section_title, purpose, target)}
                    for section_id, section_title, target, purpose in sections
                ],
                "applied": [], "deferred": [], "rejected": [],
                "summary": "test LLM assembled the complete outline",
            }
        elif "# Agent 3 — Reviewer" in system:
            payload = {
                "findings": [{
                    "severity": "major", "category": "readability", "section": "document",
                    "claim": "test fixture records a review item so iteration charts are exercised",
                    "action": "Resolve this fixture-only finding in a later test iteration.",
                }],
                "metrics": {"avg_sentence_words": 12},
                "summary": "test LLM emitted its fixture-only review finding",
            }
        else:
            payload = {"findings": [], "summary": "test LLM found no issues"}
        text = json.dumps(payload, ensure_ascii=False)
        return LLMResponse(text=text, tokens_in=100, tokens_out=len(text.split()),
                           engine=f"{self.engine_name}:fixture")


@pytest.fixture
def model_client() -> TestLLMClient:
    return TestLLMClient()


@pytest.fixture
def valid_brief() -> Brief:
    return Brief(
        task_id="test-article",
        title="Testing the ArticleGen Pipeline",
        objective=(
            "Demonstrate that the orchestrated loop runs end to end under a test model "
            "and that the change log records every iteration."
        ),
        audience="engineers maintaining the pipeline",
        target_words=600,
        sections=[
            Section(id="introduction", title="Introduction",
                    purpose="frame the problem being solved", word_target=150),
            Section(id="mechanism", title="Mechanism",
                    purpose="describe how the loop converges", word_target=250),
            Section(id="limits", title="Limits",
                    purpose="name what the approach does not handle", word_target=200),
        ],
    )


@pytest.fixture
def example_brief_dict() -> dict:
    with open(os.path.join(EXAMPLES, "brief_example.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)
