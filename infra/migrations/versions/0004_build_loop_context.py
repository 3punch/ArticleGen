"""Add per-task runtime metadata and pgvector-backed ProjectSpec context chunks.

Revision ID: 0004_build_loop_context
Revises: 0003_real_requirement_planner_agents
Create Date: 2026-10-18
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004_build_loop_context"
down_revision = "0003_real_requirement_planner_agents"
branch_labels = None
depends_on = None


class Vector384(sa.types.UserDefinedType):
    """Migration-local SQL type so Alembic does not require a Python pgvector package."""

    cache_ok = True

    def get_col_spec(self, **kw) -> str:
        return "vector(384)"


def upgrade() -> None:
    # The foundation migration enables the extension; this is idempotent for
    # databases that were provisioned manually before applying the full chain.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("task_nodes", sa.Column("runtime", sa.String(length=16), nullable=True))
    op.create_check_constraint(
        "ck_task_nodes_runtime", "task_nodes",
        "runtime IS NULL OR runtime IN ('python', 'node')",
    )

    op.create_table(
        "project_context_chunks",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("project_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("chunk_kind", sa.String(length=32), nullable=False),
        sa.Column("chunk_key", sa.String(length=160), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector384(), nullable=False),
        sa.Column(
            "metadata", postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"), nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.CheckConstraint("ordinal >= 0", name="ck_project_context_chunk_ordinal_nonnegative"),
        sa.CheckConstraint(
            "chunk_kind IN ('title', 'description', 'deliverable', 'constraint', "
            "'tech_stack_hint', 'rubric_item')",
            name="ck_project_context_chunk_kind",
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id", "chunk_kind", "chunk_key", "ordinal",
            name="uq_project_context_chunk_source",
        ),
    )
    op.create_index(
        "ix_project_context_chunks_project", "project_context_chunks",
        ["project_id", "chunk_kind"],
    )
    op.create_index(
        "ix_project_context_chunks_embedding_hnsw", "project_context_chunks", ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_project_context_chunks_embedding_hnsw",
                  table_name="project_context_chunks")
    op.drop_index("ix_project_context_chunks_project",
                  table_name="project_context_chunks")
    op.drop_table("project_context_chunks")
    op.drop_constraint("ck_task_nodes_runtime", "task_nodes", type_="check")
    op.drop_column("task_nodes", "runtime")
