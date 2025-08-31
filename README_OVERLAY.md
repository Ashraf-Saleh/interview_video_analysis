
# Overlay Additions

Files:
- core/visual.py — helpers to draw face rectangles, emotions, and NO_FACE / MULTIPLE_FACES flags; and annotate a video
- core/live_overlay_snippet.py — the `run_live_overlay` function to paste into the bottom of `core/live.py`
- tests/test_visual.py — unit test that mocks DeepFace and writes an annotated video
- tests/test_live_overlay.py — unit test that mocks camera/GUI/DeepFace and runs the live overlay loop
- scripts/live_overlay.py — run live overlay from your webcam (press 'q' to quit)

## How to integrate
1) Copy `core/visual.py` into your project.
2) Open `core/live.py` and paste the content of `core/live_overlay_snippet.py` at the bottom.
3) Copy the two test files into your `tests/` folder.
4) Copy `scripts/live_overlay.py` into your `scripts/` folder.
5) Run tests: `pytest -q tests/test_visual.py tests/test_live_overlay.py`
6) Try live: `python scripts/live_overlay.py`
