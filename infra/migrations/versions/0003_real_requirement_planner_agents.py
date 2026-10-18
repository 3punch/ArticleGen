"""Persist raw briefs, clarification gates, and atomic provider budget reservations.

Revision ID: 0003_real_requirement_planner_agents
Revises: 0002_workflow_fsm
Create Date: 2026-10-17
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003_real_requirement_planner_agents"
down_revision = "0002_workflow_fsm"
branch_labels = None
depends_on = None


UUID = sa.Uuid(as_uuid=True)
JSON_DOC = postgresql.JSONB(astext_type=sa.Text())
WORKFLOW_STATES = (
    "'INTAKE', 'SPEC', 'PLAN', 'AWAITING_PLAN_APPROVAL', 'AWAITING_CLARIFICATION', "
    "'BUILD_LOOP', 'INTEGRATION', 'REVIEW', 'AWAITING_FINAL_APPROVAL', 'PACKAGE', "
    "'DONE', 'FAILED', 'ESCALATED'"
)


def upgrade() -> None:
    op.add_column("projects", sa.Column("raw_brief_text", sa.Text(), nullable=True))
    op.add_column("projects", sa.Column("rubric_text", sa.Text(), nullable=True))

    op.add_column("budget_ledgers", sa.Column(
        "reserved_tokens", sa.BigInteger(), server_default="0", nullable=False,
    ))
    op.add_column("budget_ledgers", sa.Column(
        "reserved_spend", sa.Numeric(16, 8), server_default="0", nullable=False,
    ))
    op.create_check_constraint(
        "ck_budget_reserved_tokens_nonnegative", "budget_ledgers", "reserved_tokens >= 0",
    )
    op.create_check_constraint(
        "ck_budget_reserved_spend_nonnegative", "budget_ledgers", "reserved_spend >= 0",
    )

    op.create_table(
        "budget_reservations",
        sa.Column("id", UUID, primary_key=True, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("workflow_id", UUID, nullable=True),
        sa.Column("reserved_tokens", sa.BigInteger(), nullable=False),
        sa.Column("reserved_spend", sa.Numeric(16, 8), nullable=False),
        sa.Column("actual_tokens", sa.BigInteger(), nullable=True),
        sa.Column("actual_spend", sa.Numeric(16, 8), nullable=True),
        sa.Column("state", sa.String(length=16), server_default="reserved", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("reserved_tokens >= 0",
                           name="ck_budget_reservations_tokens_nonnegative"),
        sa.CheckConstraint("reserved_spend >= 0",
                           name="ck_budget_reservations_spend_nonnegative"),
        sa.CheckConstraint("state IN ('reserved', 'settled', 'released', 'expired')",
                           name="ck_budget_reservations_state"),
        sa.ForeignKeyConstraint(["project_id"], ["budget_ledgers.project_id"],
                                ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflow_runs.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_budget_reservations_project_state_expiry", "budget_reservations",
                    ["project_id", "state", "expires_at"])

    op.add_column("agent_runs", sa.Column("budget_reservation_id", UUID, nullable=True))
    op.add_column("agent_runs", sa.Column("provider", sa.String(length=80), nullable=True))
    op.add_column("agent_runs", sa.Column("model", sa.String(length=200), nullable=True))
    op.add_column("agent_runs", sa.Column("idempotency_key", sa.String(length=255), nullable=True))
    op.create_unique_constraint(
        "uq_agent_runs_idempotency_key", "agent_runs", ["idempotency_key"],
    )
    op.create_foreign_key(
        "fk_agent_runs_budget_reservation_id_budget_reservations",
        "agent_runs", "budget_reservations", ["budget_reservation_id"], ["id"],
        ondelete="SET NULL",
    )

    op.drop_constraint("ck_workflow_runs_state", "workflow_runs", type_="check")
    op.create_check_constraint(
        "ck_workflow_runs_state", "workflow_runs", f"state IN ({WORKFLOW_STATES})",
    )
    op.alter_column("workflow_runs", "task_graph_id", existing_type=UUID, nullable=True)

    op.drop_constraint("ck_workflow_events_to_state", "workflow_events", type_="check")
    op.create_check_constraint(
        "ck_workflow_events_to_state", "workflow_events", f"to_state IN ({WORKFLOW_STATES})",
    )


def downgrade() -> None:
    op.execute(
        "UPDATE workflow_events SET to_state = 'ESCALATED' WHERE to_state = 'AWAITING_CLARIFICATION'"
    )
    op.drop_constraint("ck_workflow_events_to_state", "workflow_events", type_="check")
    op.create_check_constraint(
        "ck_workflow_events_to_state", "workflow_events",
        "to_state IN ('INTAKE', 'SPEC', 'PLAN', 'AWAITING_PLAN_APPROVAL', 'BUILD_LOOP', "
        "'INTEGRATION', 'REVIEW', 'AWAITING_FINAL_APPROVAL', 'PACKAGE', 'DONE', 'FAILED', 'ESCALATED')",
    )
    op.execute(
        "UPDATE workflow_runs SET state = 'ESCALATED' WHERE state = 'AWAITING_CLARIFICATION'"
    )
    op.drop_constraint("ck_workflow_runs_state", "workflow_runs", type_="check")
    op.create_check_constraint(
        "ck_workflow_runs_state", "workflow_runs",
        "state IN ('INTAKE', 'SPEC', 'PLAN', 'AWAITING_PLAN_APPROVAL', 'BUILD_LOOP', "
        "'INTEGRATION', 'REVIEW', 'AWAITING_FINAL_APPROVAL', 'PACKAGE', 'DONE', 'FAILED', 'ESCALATED')",
    )
    op.alter_column("workflow_runs", "task_graph_id", existing_type=UUID, nullable=False)

    op.drop_constraint("fk_agent_runs_budget_reservation_id_budget_reservations",
                       "agent_runs", type_="foreignkey")
    op.drop_constraint("uq_agent_runs_idempotency_key", "agent_runs", type_="unique")
    op.drop_column("agent_runs", "idempotency_key")
    op.drop_column("agent_runs", "model")
    op.drop_column("agent_runs", "provider")
    op.drop_column("agent_runs", "budget_reservation_id")
    op.drop_index("ix_budget_reservations_project_state_expiry", table_name="budget_reservations")
    op.drop_table("budget_reservations")
    op.drop_constraint("ck_budget_reserved_spend_nonnegative", "budget_ledgers", type_="check")
    op.drop_constraint("ck_budget_reserved_tokens_nonnegative", "budget_ledgers", type_="check")
    op.drop_column("budget_ledgers", "reserved_spend")
    op.drop_column("budget_ledgers", "reserved_tokens")
    op.drop_column("projects", "rubric_text")
    op.drop_column("projects", "raw_brief_text")
