"""Deterministic topological task scheduler with conservative resource locking."""

from __future__ import annotations

from projectforge_schemas.models import TaskGraph, TaskNode


class TaskScheduler:
    def __init__(self, *, max_parallel_tasks: int = 4) -> None:
        if max_parallel_tasks < 1:
            raise ValueError("max_parallel_tasks must be at least one")
        self.max_parallel_tasks = max_parallel_tasks

    @staticmethod
    def _paths_overlap(left: str, right: str) -> bool:
        return left == right or left.startswith(f"{right}/") or right.startswith(f"{left}/")

    @classmethod
    def resource_conflict(cls, left: TaskNode, right: TaskNode) -> bool:
        # An undeclared workspace claim is ambiguous and therefore exclusive.
        if not (left.read_set or left.write_set) or not (right.read_set or right.write_set):
            return True
        left_writes = left.write_set
        right_writes = right.write_set
        return any(
            cls._paths_overlap(left_path, right_path)
            for left_path in left_writes
            for right_path in (*right.read_set, *right_writes)
        ) or any(
            cls._paths_overlap(left_path, right_path)
            for left_path in left.read_set
            for right_path in right_writes
        )

    def next_batch(
        self,
        graph: TaskGraph,
        *,
        completed_ids: set[str],
        pending_ids: set[str] | None = None,
    ) -> list[TaskNode]:
        """Choose the next stable set of dependency-ready, resource-safe tasks."""
        nodes = {node.id: node for node in graph.nodes}
        pending = set(nodes) - completed_ids if pending_ids is None else set(pending_ids)
        ready = sorted(
            node_id for node_id in pending
            if set(nodes[node_id].dependencies) <= completed_ids
        )
        batch: list[TaskNode] = []
        for node_id in ready:
            candidate = nodes[node_id]
            if all(not self.resource_conflict(candidate, selected) for selected in batch):
                batch.append(candidate)
            if len(batch) >= self.max_parallel_tasks:
                break
        return batch

    def batches(self, graph: TaskGraph) -> list[list[TaskNode]]:
        """Return stable ready batches; dependency edges and resource locks are honored."""
        pending = {node.id for node in graph.nodes}
        completed: set[str] = set()
        batches: list[list[TaskNode]] = []
        while pending:
            batch = self.next_batch(graph, completed_ids=completed, pending_ids=pending)
            if not batch:
                raise ValueError("task graph has unresolved dependencies or a resource conflict")
            batches.append(batch)
            completed.update(node.id for node in batch)
            pending.difference_update(node.id for node in batch)
        return batches
