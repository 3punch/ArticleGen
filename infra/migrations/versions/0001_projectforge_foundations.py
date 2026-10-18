"""Create ProjectForge Phase 1 persistence tables.

Revision ID: 0001_projectforge_foundations
Revises:
Create Date: 2026-10-17
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_projectforge_foundations"
down_revision = None
branch_labels = None
depends_on = None


UUID = sa.Uuid(as_uuid=True)
JSON_DOC = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    # The extension is part of the fixed Postgres stack; vector columns are added
    # only when a later phase introduces persisted embeddings.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "projects",
        sa.Column("id", UUID, primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )

    op.create_table(
        "project_specs",
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("deliverables", JSON_DOC, nullable=False),
        sa.Column("constraints", JSON_DOC, nullable=False),
        sa.Column("tech_stack_hints", JSON_DOC, nullable=False),
        sa.Column("rubric_items", JSON_DOC, nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("project_id"),
    )

    op.create_table(
        "budget_ledgers",
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("token_cap", sa.BigInteger(), nullable=False),
        sa.Column("spend_cap", sa.Numeric(16, 8), nullable=False),
        sa.Column("used_tokens", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("used_spend", sa.Numeric(16, 8), server_default="0", nullable=False),
        sa.CheckConstraint("token_cap >= 0", name="ck_budget_token_cap_nonnegative"),
        sa.CheckConstraint("used_tokens >= 0", name="ck_budget_used_tokens_nonnegative"),
        sa.CheckConstraint("spend_cap >= 0", name="ck_budget_spend_cap_nonnegative"),
        sa.CheckConstraint("used_spend >= 0", name="ck_budget_used_spend_nonnegative"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("project_id"),
    )

    op.create_table(
        "task_graphs",
        sa.Column("id", UUID, primary_key=True, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("project_id", "version", name="uq_task_graph_project_version"),
        sa.CheckConstraint("version >= 1", name="ck_task_graphs_version_positive"),
    )
    op.create_index("ix_task_graphs_project_created", "task_graphs",
                    ["project_id", "created_at"])

    op.create_table(
        "task_nodes",
        sa.Column("graph_id", UUID, nullable=False),
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("dependencies", JSON_DOC, nullable=False),
        sa.Column("status", sa.String(length=16), server_default="pending", nullable=False),
        sa.Column("acceptance_criteria", JSON_DOC, nullable=False),
        sa.Column("retries", sa.Integer(), server_default="0", nullable=False),
        sa.Column("assigned_agent", sa.String(length=32), nullable=False),
        sa.CheckConstraint("retries >= 0", name="ck_task_nodes_retries_nonnegative"),
        sa.CheckConstraint(
            "status IN ('pending', 'ready', 'running', 'blocked', 'completed', 'failed', 'cancelled')",
            name="ck_task_nodes_status",
        ),
        sa.CheckConstraint(
            "assigned_agent IN ('requirement', 'planner', 'coder', 'tester', 'reviewer', 'debugger', 'packager')",
            name="ck_task_nodes_assigned_agent",
        ),
        sa.ForeignKeyConstraint(["graph_id"], ["task_graphs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("graph_id", "id"),
    )
    op.create_index("ix_task_nodes_graph_status", "task_nodes", ["graph_id", "status"])

    op.create_table(
        "task_edges",
        sa.Column("graph_id", UUID, nullable=False),
        sa.Column("source_id", sa.String(length=64), nullable=False),
        sa.Column("target_id", sa.String(length=64), nullable=False),
        sa.CheckConstraint("source_id <> target_id", name="ck_task_edges_not_self_referential"),
        sa.ForeignKeyConstraint(["graph_id"], ["task_graphs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["graph_id", "source_id"], ["task_nodes.graph_id", "task_nodes.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["graph_id", "target_id"], ["task_nodes.graph_id", "task_nodes.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("graph_id", "source_id", "target_id"),
    )

    op.create_table(
        "agent_runs",
        sa.Column("id", UUID, primary_key=True, nullable=False),
        sa.Column("agent_name", sa.String(length=32), nullable=False),
        sa.Column("input", JSON_DOC, nullable=False),
        sa.Column("output", JSON_DOC, nullable=False),
        sa.Column("tokens_used", sa.BigInteger(), nullable=False),
        sa.Column("cost", sa.Numeric(16, 8), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.CheckConstraint("tokens_used >= 0", name="ck_agent_runs_tokens_nonnegative"),
        sa.CheckConstraint("cost >= 0", name="ck_agent_runs_cost_nonnegative"),
        sa.CheckConstraint(
            "agent_name IN ('requirement', 'planner', 'coder', 'tester', 'reviewer', 'debugger', 'packager')",
            name="ck_agent_runs_agent_name",
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_agent_runs_project_timestamp", "agent_runs", ["project_id", "timestamp"])

    op.create_table(
        "review_results",
        sa.Column("id", UUID, primary_key=True, nullable=False),
        sa.Column("project_id", UUID, nullable=False),
        sa.Column("rubric_item", JSON_DOC, nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("evidence", JSON_DOC, nullable=False),
        sa.Column("reviewer_notes", sa.Text(), server_default="", nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_review_results_project", "review_results", ["project_id"])


def downgrade() -> None:
    op.drop_index("ix_review_results_project", table_name="review_results")
    op.drop_table("review_results")
    op.drop_index("ix_agent_runs_project_timestamp", table_name="agent_runs")
    op.drop_table("agent_runs")
    op.drop_table("task_edges")
    op.drop_index("ix_task_nodes_graph_status", table_name="task_nodes")
    op.drop_table("task_nodes")
    op.drop_index("ix_task_graphs_project_created", table_name="task_graphs")
    op.drop_table("task_graphs")
    op.drop_table("budget_ledgers")
    op.drop_table("project_specs")
    op.drop_table("projects")
