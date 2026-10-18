"""Persist human-review feedback and per-workflow rubric assessments.

Revision ID: 0005_review_packaging
Revises: 0004_build_loop_context
Create Date: 2026-10-18
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005_review_packaging"
down_revision = "0004_build_loop_context"
branch_labels = None
depends_on = None


UUID = sa.Uuid(as_uuid=True)
JSON_DOC = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.add_column(
        "task_nodes",
        sa.Column("review_feedback", JSON_DOC,
                  server_default=sa.text("'[]'::jsonb"), nullable=False),
    )

    op.add_column(
        "review_results",
        sa.Column("workflow_id", UUID, nullable=True),
    )
    op.add_column(
        "review_results",
        sa.Column("rubric_type", sa.String(length=24),
                  server_default="automatable", nullable=False),
    )
    op.add_column(
        "review_results",
        sa.Column("status", sa.String(length=32),
                  server_default="not_verified", nullable=False),
    )
    op.add_column(
        "review_results",
        sa.Column("recommendation", sa.String(length=24), nullable=True),
    )
    op.alter_column(
        "review_results", "passed", existing_type=sa.Boolean(), nullable=True,
    )
    op.create_foreign_key(
        "fk_review_results_workflow_id_workflow_runs",
        "review_results", "workflow_runs", ["workflow_id"], ["id"],
        ondelete="SET NULL",
    )
    # Earlier stub records treated every rubric item as a machine verdict. Preserve
    # objective history but make subjective legacy rows explicitly unverified, never
    # a pass, until a real human review is recorded.
    op.execute(
        "UPDATE review_results SET rubric_type = CASE "
        "WHEN rubric_item->>'type' = 'human-judgment' THEN 'human-judgment' "
        "ELSE 'automatable' END"
    )
    op.execute(
        "UPDATE review_results SET status = CASE "
        "WHEN rubric_type = 'human-judgment' THEN 'not_verified' "
        "WHEN passed IS TRUE THEN 'passed' ELSE 'failed' END"
    )
    op.execute(
        "UPDATE review_results SET passed = NULL WHERE rubric_type = 'human-judgment'"
    )
    op.create_check_constraint(
        "ck_review_results_rubric_type", "review_results",
        "rubric_type IN ('automatable', 'human-judgment')",
    )
    op.create_check_constraint(
        "ck_review_results_status", "review_results",
        "status IN ('passed', 'failed', 'not_verified', 'human_review_required')",
    )
    op.create_check_constraint(
        "ck_review_results_verdict_consistency", "review_results",
        "(status = 'passed' AND rubric_type = 'automatable' AND passed IS TRUE) OR "
        "(status = 'failed' AND rubric_type = 'automatable' AND passed IS FALSE) OR "
        "(status = 'not_verified' AND passed IS NULL) OR "
        "(status = 'human_review_required' AND rubric_type = 'human-judgment' AND passed IS NULL)",
    )
    op.create_index("ix_review_results_workflow", "review_results", ["workflow_id"])


def downgrade() -> None:
    op.drop_index("ix_review_results_workflow", table_name="review_results")
    op.drop_constraint(
        "ck_review_results_verdict_consistency", "review_results", type_="check",
    )
    op.drop_constraint("ck_review_results_verdict_consistency", "review_results", type_="check")
    op.drop_constraint("ck_review_results_status", "review_results", type_="check")
    op.drop_constraint("ck_review_results_rubric_type", "review_results", type_="check")
    op.drop_constraint(
        "fk_review_results_workflow_id_workflow_runs", "review_results", type_="foreignkey",
    )
    op.drop_column("review_results", "recommendation")
    op.drop_column("review_results", "status")
    op.drop_column("review_results", "rubric_type")
    op.drop_column("review_results", "workflow_id")
    # The pre-Phase-6 schema did not permit NULL here. Restore a conservative false
    # value for legacy subjective records before reapplying NOT NULL.
    op.execute("UPDATE review_results SET passed = FALSE WHERE passed IS NULL")
    op.alter_column(
        "review_results", "passed", existing_type=sa.Boolean(), nullable=False,
    )
    op.drop_column("task_nodes", "review_feedback")
