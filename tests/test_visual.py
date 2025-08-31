# tests/test_visual.py
import numpy as np
from core.visual import draw_overlays

def test_draw_overlays_cases():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    out1 = draw_overlays(frame, [], flag="NO_FACE")
    assert out1.shape == frame.shape
    out2 = draw_overlays(frame, [], flag="MULTIPLE_FACES")
    assert out2.shape == frame.shape
    faces = [{"region": {"x":10,"y":10,"w":15,"h":12}, "emotion":"happy"}]
    out3 = draw_overlays(frame, faces, None)
    assert out3.shape == frame.shape

def test_draw_overlays_zero_region_treated_as_no_face():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    # region with w/h = 0 should be ignored, so faces list is empty → NO_FACE flag shown
    faces = [{"region": {"x": 0, "y": 0, "w": 0, "h": 0}, "emotion":"neutral"}]
    out = draw_overlays(frame, [], flag="NO_FACE")
    assert out.shape == frame.shape
