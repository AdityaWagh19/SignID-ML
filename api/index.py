"""
Vercel serverless entry point for the FastAPI backend.
Vercel discovers this file because it lives in /api/ at the repo root.
"""
import sys
from pathlib import Path

# Make the repo root importable so `from src.xxx import yyy` works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.compat  # noqa: F401 — register sklearn version compatibility shims

from app.api import app  # noqa: E402 — re-export the FastAPI ASGI app
