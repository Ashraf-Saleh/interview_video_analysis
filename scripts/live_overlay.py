
"""Run live camera overlay.

Usage:
    uvicorn api.main:app --reload  # (separate, for API)
    python scripts/live_overlay.py  # (to see camera overlay window)

Press 'q' to quit the window.
"""
from core.config import Settings
from core.live import run_live_overlay

if __name__ == '__main__':
    s = Settings()
    run_live_overlay(s)
