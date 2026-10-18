"""Database wiring. Production uses Postgres; SQLite is supported for isolated tests."""

from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


def build_engine(database_url: str, *, echo: bool = False) -> Engine:
    """Create an engine with safe SQLite test defaults and foreign-key enforcement."""
    kwargs = {"pool_pre_ping": True, "echo": echo}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in database_url:
            kwargs["poolclass"] = StaticPool
    engine = create_engine(database_url, **kwargs)

    if database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def configured_database_url(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve the production database URL without embedding credentials."""
    return explicit or os.environ.get("PROJECTFORGE_DATABASE_URL")
