"""RQ worker entrypoint for durable ProjectForge workflow jobs."""

from __future__ import annotations

import os
from uuid import UUID


def run_workflow_job(workflow_id: str) -> dict[str, object]:
    from projectforge_api.db import build_engine, configured_database_url, make_session_factory
    from projectforge_orchestrator.engine import PostgresFSMEngine
    from projectforge_agents.suite import ProjectForgeAgentSuite

    database_url = configured_database_url()
    if not database_url:
        raise RuntimeError("PROJECTFORGE_DATABASE_URL must be configured for the workflow worker")
    engine = build_engine(database_url)
    try:
        factory = make_session_factory(engine)
        result = PostgresFSMEngine(
            factory, agents=ProjectForgeAgentSuite(session_factory=factory),
        ).drive(UUID(workflow_id))
        return result.model_dump(mode="json")
    finally:
        engine.dispose()


def main() -> None:
    try:
        from redis import Redis
        from rq import Queue, Worker
    except ImportError as exc:
        raise SystemExit(
            "Install the projectforge-worker extra to run the RQ worker."
        ) from exc
    redis_url = os.environ.get("PROJECTFORGE_REDIS_URL")
    if not redis_url:
        raise SystemExit("PROJECTFORGE_REDIS_URL must be configured for the workflow worker")
    connection = Redis.from_url(redis_url)
    queue_name = os.environ.get("PROJECTFORGE_RQ_QUEUE", "projectforge")
    queue = Queue(queue_name, connection=connection)
    Worker([queue], connection=connection).work()


if __name__ == "__main__":
    main()
