import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from articlegen.models import Brief, Section  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO_ROOT, "examples")


@pytest.fixture
def valid_brief() -> Brief:
    return Brief(
        task_id="test-article",
        title="Testing the ArticleGen Pipeline",
        objective=(
            "Demonstrate that the orchestrated loop runs end to end and converges "
            "without credentials, and that the change log records every iteration."
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
