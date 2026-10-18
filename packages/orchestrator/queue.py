"""RQ dispatch adapter; Redis/RQ imports stay lazy for API-only installs."""

from __future__ import annotations

from uuid import UUID, uuid4


class RQDispatcher:
    def __init__(self, redis_url: str, *, queue_name: str = "projectforge") -> None:
        if not redis_url.strip():
            raise ValueError("redis_url must not be empty")
        self.redis_url = redis_url
        self.queue_name = queue_name

    def enqueue_workflow(self, workflow_id: UUID, *, timeout_seconds: int) -> str:
        try:
            from redis import Redis
            from rq import Queue
        except ImportError as exc:  # helpful, explicit failure for API-only installs
            raise RuntimeError(
                "RQ dispatch requires the projectforge-worker extra; install .[projectforge-worker]"
            ) from exc
        connection = Redis.from_url(self.redis_url)
        queue = Queue(self.queue_name, connection=connection)
        job_id = f"workflow-{workflow_id}-{uuid4().hex}"
        job = queue.enqueue(
            "projectforge_orchestrator.worker.run_workflow_job",
            str(workflow_id),
            job_id=job_id,
            job_timeout=timeout_seconds + 30,
            result_ttl=3600,
            failure_ttl=86_400,
        )
        return job.id
