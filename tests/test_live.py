# tests/test_live.py
import sys, types, numpy as np
from core.config import Settings
import core.live as live

class DummyCap:
    def __init__(self): self.calls = 0
    def isOpened(self): return True
    def read(self):
        self.calls += 1
        if self.calls > 5: return False, None
        return True, np.zeros((32,32,3), dtype=np.uint8)
    def release(self): pass

class DummyDF:
    @staticmethod
    def analyze(frame, actions, enforce_detection, detector_backend):
        # Simulate no face (zero region) on first, then single neutral face
        if np.sum(frame) == 0:
            return [{"region":{"x":0,"y":0,"w":0,"h":0}, "dominant_emotion":"neutral"}]
        return [{"region":{"x":5,"y":5,"w":20,"h":20}, "dominant_emotion":"neutral"}]

def test_run_live_overlay(monkeypatch):
    monkeypatch.setattr(live.cv2, "VideoCapture", lambda idx: DummyCap())
    monkeypatch.setitem(sys.modules, "deepface", types.SimpleNamespace(DeepFace=DummyDF))
    monkeypatch.setattr(live.cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(live.cv2, "waitKey", lambda d: ord("q"))  # exit immediately

    s = Settings()
    s.LIVE_EMOTION_INTERVAL = 0.01
    live.run_live_overlay(s, camera_index=0)
