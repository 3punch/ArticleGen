"""ProjectSpec chunking, deterministic offline embeddings, and pgvector retrieval.

The embedding implementation uses signed feature hashing so indexing never sends
project data to a second provider or introduces an unapproved dependency. Production
PostgreSQL uses its native ``vector(384)`` column/operator; SQLite is a small offline
contract-test fallback and ranks the same normalized vectors in Python.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid5, NAMESPACE_URL

from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session, sessionmaker

from projectforge_api.orm import ProjectContextChunkRow, VECTOR_DIMENSIONS
from projectforge_schemas.models import ProjectSpec, RetrievedContextChunk


_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_MAX_CHUNK_CHARS = 1400


@dataclass(frozen=True)
class SpecChunk:
    """Canonical unembedded source chunk used for indexing and unit tests."""

    id: UUID
    project_id: UUID
    chunk_kind: str
    chunk_key: str
    ordinal: int
    content: str
    metadata: dict[str, Any]
    embedding: list[float]


def _split_description(value: str, *, limit: int = _MAX_CHUNK_CHARS) -> list[str]:
    text_value = " ".join(value.split())
    if len(text_value) <= limit:
        return [text_value]
    sentences = re.split(r"(?<=[.!?])\s+", text_value)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:limit])
            sentence = sentence[limit:]
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) > limit and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [text_value[:limit]]


def _signed_feature(feature: str) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    index = int.from_bytes(digest[:4], "little") % VECTOR_DIMENSIONS
    sign = 1.0 if digest[4] & 1 else -1.0
    return index, sign


def embed_text(value: str) -> list[float]:
    """Create a stable, normalized 384-dimensional lexical feature vector."""
    tokens = [item.casefold() for item in _TOKEN_RE.findall(value)]
    values = [0.0] * VECTOR_DIMENSIONS
    features: list[tuple[str, float]] = [(f"w:{token}", 1.0) for token in tokens]
    features.extend(
        (f"b:{left}_{right}", 0.8)
        for left, right in zip(tokens, tokens[1:])
    )
    # Character n-grams improve matching on singular/plural and inflected terms
    # without relying on an external model or a language-specific stemmer.
    for token in tokens:
        padded = f"^{token}$"
        for size in (3, 4):
            features.extend(
                (f"c{size}:{padded[index:index + size]}", 0.12)
                for index in range(max(0, len(padded) - size + 1))
            )
    for feature, weight in features:
        index, sign = _signed_feature(feature)
        values[index] += sign * weight
    norm = math.sqrt(sum(item * item for item in values))
    if norm == 0:
        values[0] = 1.0
        return values
    return [item / norm for item in values]


def _cosine(left: list[float], right: list[float]) -> float:
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right))))


def _chunk(
    project_id: UUID,
    kind: str,
    key: str,
    ordinal: int,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> SpecChunk:
    stable_key = f"projectforge:project-spec:{project_id}:{kind}:{key}:{ordinal}"
    normalized = content.strip()
    return SpecChunk(
        id=uuid5(NAMESPACE_URL, stable_key),
        project_id=project_id,
        chunk_kind=kind,
        chunk_key=key,
        ordinal=ordinal,
        content=normalized,
        metadata={"source": "project_spec", **(metadata or {})},
        embedding=embed_text(normalized),
    )


def chunk_project_spec(project_id: UUID, project_spec: ProjectSpec | dict[str, Any]) -> list[SpecChunk]:
    """Split only the normalized specification into small, individually retrievable facts."""
    spec = ProjectSpec.model_validate(project_spec)
    chunks: list[SpecChunk] = [
        _chunk(project_id, "title", "title", 0, spec.title),
    ]
    for ordinal, part in enumerate(_split_description(spec.description)):
        chunks.append(_chunk(project_id, "description", "description", ordinal, part))
    chunks.extend(
        _chunk(project_id, "deliverable", f"deliverable:{index}", 0, value,
               {"position": index})
        for index, value in enumerate(spec.deliverables)
    )
    chunks.extend(
        _chunk(project_id, "constraint", f"constraint:{index}", 0, value,
               {"position": index})
        for index, value in enumerate(spec.constraints)
    )
    chunks.extend(
        _chunk(project_id, "tech_stack_hint", f"tech_stack_hint:{index}", 0, value,
               {"position": index})
        for index, value in enumerate(spec.tech_stack_hints)
    )
    chunks.extend(
        _chunk(
            project_id,
            "rubric_item",
            f"rubric_item:{index}",
            0,
            f"Criterion: {item.criterion}\nCheck method: {item.check_method}",
            {"position": index, "rubric_type": item.type},
        )
        for index, item in enumerate(spec.rubric_items)
    )
    return chunks


class PgVectorContextRetriever:
    """Persist and retrieve project-specific ProjectSpec excerpts from pgvector."""

    def __init__(self, session_factory: sessionmaker[Session], *, default_top_k: int = 5) -> None:
        if not 1 <= default_top_k <= 20:
            raise ValueError("default_top_k must be between 1 and 20")
        self.session_factory = session_factory
        self.default_top_k = default_top_k

    def replace_project_spec(
        self,
        session: Session,
        project_id: UUID,
        project_spec: ProjectSpec | dict[str, Any],
    ) -> int:
        """Atomically replace the project spec's chunks in the caller's transaction."""
        chunks = chunk_project_spec(project_id, project_spec)
        session.execute(
            delete(ProjectContextChunkRow).where(ProjectContextChunkRow.project_id == project_id)
        )
        session.add_all([
            ProjectContextChunkRow(
                id=chunk.id,
                project_id=chunk.project_id,
                chunk_kind=chunk.chunk_kind,
                chunk_key=chunk.chunk_key,
                ordinal=chunk.ordinal,
                content=chunk.content,
                embedding=chunk.embedding,
                metadata_payload=chunk.metadata,
            )
            for chunk in chunks
        ])
        return len(chunks)

    def retrieve(
        self,
        project_id: UUID,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[RetrievedContextChunk]:
        """Return at most ``top_k`` relevant chunks, never the entire ProjectSpec."""
        limit = self.default_top_k if top_k is None else top_k
        if not 1 <= limit <= 20:
            raise ValueError("top_k must be between 1 and 20")
        query_vector = embed_text(query)
        with self.session_factory() as session:
            dialect = session.get_bind().dialect.name
            if dialect == "postgresql":
                rows = self._retrieve_postgres(session, project_id, query_vector, limit)
            else:
                rows = self._retrieve_python(session, project_id, query_vector, limit)
        return [
            RetrievedContextChunk.model_validate({
                "chunk_id": row["id"],
                "project_id": row["project_id"],
                "chunk_kind": row["chunk_kind"],
                "chunk_key": row["chunk_key"],
                "ordinal": row["ordinal"],
                "content": row["content"],
                "similarity": row["similarity"],
                "metadata": row["metadata"] or {},
            })
            for row in rows
        ]

    @staticmethod
    def _retrieve_postgres(
        session: Session, project_id: UUID, query_vector: list[float], limit: int,
    ) -> list[dict[str, Any]]:
        vector_literal = json.dumps(query_vector, separators=(",", ":"))
        result = session.execute(text("""
            SELECT id, project_id, chunk_kind, chunk_key, ordinal, content, metadata,
                   1 - (embedding <=> CAST(:embedding AS vector(384))) AS similarity
            FROM project_context_chunks
            WHERE project_id = :project_id
            ORDER BY embedding <=> CAST(:embedding AS vector(384)), chunk_kind, chunk_key, ordinal
            LIMIT :limit
        """), {"embedding": vector_literal, "project_id": project_id, "limit": limit})
        return [dict(row) for row in result.mappings().all()]

    @staticmethod
    def _retrieve_python(
        session: Session, project_id: UUID, query_vector: list[float], limit: int,
    ) -> list[dict[str, Any]]:
        chunks = session.scalars(
            select(ProjectContextChunkRow).where(ProjectContextChunkRow.project_id == project_id)
        ).all()
        scored = [
            {
                "id": row.id,
                "project_id": row.project_id,
                "chunk_kind": row.chunk_kind,
                "chunk_key": row.chunk_key,
                "ordinal": row.ordinal,
                "content": row.content,
                "metadata": row.metadata_payload or {},
                "similarity": _cosine(query_vector, row.embedding),
            }
            for row in chunks
        ]
        scored.sort(key=lambda row: (
            -row["similarity"], row["chunk_kind"], row["chunk_key"], row["ordinal"],
        ))
        return scored[:limit]
