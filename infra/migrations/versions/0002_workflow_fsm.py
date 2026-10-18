"""Add durable workflow FSM, audit events, and task resource declarations.

Revision ID: 0002_workflow_fsm
Revises: 0001_projectforge_foundations
Create Date: 2026-10-17
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_workflow_fsm"
down_revision = "0001_projectforge_foundations"
branch_labels = None
depends_on = None


UUID = sa.Uuid(as_uuid=True)
JSON_DOC = postgresql.JSONB(astext_type=sa.Text())
WORKFLOW_STATES = (
    "'INTAKE', 'SPEC', 'PLAN', 'AWAITING_PLAN_APPROVAL', 'BUILD_LOOP', "
    "'INTEGRATION', 'REVIEW', 'AWAITING_FINAL_APPROVAL', 'PACKAGE', 'DONE', "
    "'FAILED', 'ESCALATED'"
)


def upgrade() -> None:
    op.add_column(
        "task_nodes",
        sa.Column("read_set", JSON_DOC, server_default=sa.text("'[]'::jsonb"), nullable=False),
    )
    op.add_column(
        "task_nodes",
        sa.Column("write_set", JSON_DOC, server_default=sa.text("'[]'::jsonb"), nullable=False),
    )
    op.add_column(
        "task_nodes",
        sa.Column("timeout_seconds", sa.Integer(), server_default="300", nullable=False),
    )
    op.create_check_constraint(
        "ck_task_nodes_timeout_positive", "task_nodes", "timeout_seconds >= 1",
    )

    op.create_table(
        "workflow_runs",
        sa.Column("id", UUID, primary_key=True, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("task_graph_id", UUID, nullable=False),
        sa.Column("state", sa.String(length=32), server_default="INTAKE", nullable=False),
        sa.Column("max_retries", sa.Integer(), server_default="3", nullable=False),
        sa.Column("timeout_seconds", sa.Integer(), server_default="3600", nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("result_payload", JSON_DOC, server_default=sa.text("'{}'::jsonb"),
                  nullable=False),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_owner", sa.String(length=80), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.CheckConstraint(f"state IN ({WORKFLOW_STATES})", name="ck_workflow_runs_state"),
        sa.CheckConstraint("max_retries BETWEEN 0 AND 3", name="ck_workflow_runs_max_retries"),
        sa.CheckConstraint("timeout_seconds >= 1", name="ck_workflow_runs_timeout_positive"),
        sa.UniqueConstraint("task_graph_id", name="uq_workflow_run_task_graph"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["task_graph_id"], ["task_graphs.id"], ondelete="RESTRICT"),
    )
    op.create_index("ix_workflow_runs_project_created", "workflow_runs",
                    ["project_id", "created_at"])
    op.create_index("ix_workflow_runs_state_updated", "workflow_runs", ["state", "updated_at"])
    op.create_index("ix_workflow_runs_lease_expiry", "workflow_runs", ["lease_expires_at"])

    op.add_column("agent_runs", sa.Column("workflow_id", UUID, nullable=True))
    op.create_foreign_key(
        "fk_agent_runs_workflow_id_workflow_runs", "agent_runs", "workflow_runs",
        ["workflow_id"], ["id"], ondelete="SET NULL",
    )
    op.create_index("ix_agent_runs_workflow_timestamp", "agent_runs",
                    ["workflow_id", "timestamp"])

    op.create_table(
        "workflow_events",
        sa.Column("workflow_id", UUID, nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("from_state", sa.String(length=32), nullable=True),
        sa.Column("to_state", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.String(length=2000), nullable=False),
        sa.Column("payload", JSON_DOC, server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.CheckConstraint("sequence >= 1", name="ck_workflow_events_sequence_positive"),
        sa.CheckConstraint(f"to_state IN ({WORKFLOW_STATES})", name="ck_workflow_events_to_state"),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflow_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("workflow_id", "sequence"),
    )
    op.create_index("ix_workflow_events_created", "workflow_events",
                    ["workflow_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_agent_runs_workflow_timestamp", table_name="agent_runs")
    op.drop_constraint("fk_agent_runs_workflow_id_workflow_runs", "agent_runs",
                       type_="foreignkey")
    op.drop_column("agent_runs", "workflow_id")
    op.drop_index("ix_workflow_events_created", table_name="workflow_events")
    op.drop_table("workflow_events")
    op.drop_index("ix_workflow_runs_lease_expiry", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_state_updated", table_name="workflow_runs")
    op.drop_index("ix_workflow_runs_project_created", table_name="workflow_runs")
    op.drop_table("workflow_runs")
    op.drop_constraint("ck_task_nodes_timeout_positive", "task_nodes", type_="check")
    op.drop_column("task_nodes", "timeout_seconds")
    op.drop_column("task_nodes", "write_set")
    op.drop_column("task_nodes", "read_set")
