"""Thin launcher so `streamlit run app.py` keeps working from a checkout.

The real script lives at ``articlegen/ui/dashboard.py`` so it ships as package data
with ``pip install articlegen[ui]``. Without that, an installed ``articlegen ui`` has
no script to launch - the dashboard would only ever run from a source checkout.

Streamlit executes the target file as ``__main__``, so this shim forwards to it with
``runpy`` rather than importing it (importing would run the app twice).
"""

from __future__ import annotations

import os
import runpy

from articlegen.ui import DASHBOARD_PATH

if __name__ == "__main__":
    if not os.path.exists(DASHBOARD_PATH):
        raise SystemExit(f"dashboard script not found at {DASHBOARD_PATH}")
    runpy.run_path(DASHBOARD_PATH, run_name="__main__")
