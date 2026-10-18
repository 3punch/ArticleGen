from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError

from projectforge_api.db import build_engine, make_session_factory
from projectforge_api.orm import (
    AgentRunRow,
    Base,
    BudgetLedgerRow,
    Project,
    ProjectSpecRow,
    ReviewResultRow,
)


def make_database():
    engine = build_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine, make_session_factory(engine)


def add_project(session, project_id):
    project = Project(id=project_id, name="Persistence test")
    project.spec = ProjectSpecRow(
        project_id=project_id,
        title="Test spec",
        description="Persist a normalized assignment description.",
        deliverables=["report"],
        constraints=[],
        tech_stack_hints=["Python"],
        rubric_items=[{
            "criterion": "Correctness",
            "type": "automatable",
            "check_method": "Run tests",
        }],
    )
    project.budget = BudgetLedgerRow(
        project_id=project_id,
        token_cap=1000,
        spend_cap=Decimal("2.50"),
        used_tokens=0,
        used_spend=Decimal("0"),
    )
    session.add(project)
    session.flush()
    return project


def test_agent_run_and_review_result_round_trip():
    engine, factory = make_database()
    project_id = uuid4()
    with factory() as session:
        add_project(session, project_id)
        agent_run = AgentRunRow(
            agent_name="reviewer",
            input_payload={"task": "review"},
            output_payload={"summary": "Reviewed"},
            tokens_used=91,
            cost=Decimal("0.0031"),
            timestamp=datetime.now(timezone.utc),
            project_id=project_id,
        )
        review = ReviewResultRow(
            project_id=project_id,
            rubric_item={"criterion": "Correctness", "type": "automatable",
                         "check_method": "Run tests"},
            rubric_type="automatable",
            status="passed",
            passed=True,
            evidence=["pytest exit code 0"],
            reviewer_notes="All checks passed.",
        )
        session.add_all([agent_run, review])
        session.commit()
        assert session.get(AgentRunRow, agent_run.id).tokens_used == 91
        assert session.get(ReviewResultRow, review.id).evidence == ["pytest exit code 0"]
    engine.dispose()


def test_database_constraints_never_store_a_human_judgment_as_machine_passed():
    engine, factory = make_database()
    with factory() as session:
        project_id = uuid4()
        add_project(session, project_id)
        session.add(ReviewResultRow(
            project_id=project_id,
            rubric_item={"criterion": "Clarity", "type": "human-judgment",
                         "check_method": "Human review"},
            rubric_type="human-judgment",
            status="passed",
            passed=True,
            evidence=["The model approved it."],
            reviewer_notes="invalid machine pass",
        ))
        with pytest.raises(IntegrityError):
            session.commit()
    engine.dispose()


def test_database_constraints_reject_negative_ledger_and_agent_usage():
    engine, factory = make_database()
    with factory() as session:
        project_id = uuid4()
        add_project(session, project_id)
        with pytest.raises(IntegrityError):
            session.execute(update(BudgetLedgerRow).where(
                BudgetLedgerRow.project_id == project_id
            ).values(token_cap=-1))
            session.commit()
        session.rollback()

        session.add(AgentRunRow(
            agent_name="coder", input_payload={}, output_payload={}, tokens_used=-1,
            cost=Decimal("0"), timestamp=datetime.now(timezone.utc), project_id=project_id,
        ))
        with pytest.raises(IntegrityError):
            session.commit()
    engine.dispose()
