"""Local API entry point; defaults to loopback because authentication is not configured."""

from __future__ import annotations

import os
import sys


def main() -> int:
    if not os.environ.get("PROJECTFORGE_DATABASE_URL"):
        print("error: set PROJECTFORGE_DATABASE_URL before starting the API", file=sys.stderr)
        return 2
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - guarded by the optional extra
        print("error: install the projectforge-api extra to run the API", file=sys.stderr)
        return 2

    uvicorn.run(
        "projectforge_api.main:app",
        host=os.environ.get("PROJECTFORGE_HOST", "127.0.0.1"),
        port=int(os.environ.get("PROJECTFORGE_PORT", "8000")),
        reload=False,
    )
    return 0
