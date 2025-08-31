
import sys, types
import numpy as np
from core.config import Settings
import core.live as live

class DummyCap:
    def __init__(self):
        self.i = 0
        self.frame = np.zeros((32,32,3), dtype=np.uint8)
    def isOpened(self): return True
    def read(self):
        self.i += 1
        if self.i > 10:
            return False, None
        return True, self.frame.copy()
    def release(self): pass

class DummyDF:
    @staticmethod
    def analyze(frame, actions, enforce_detection, detector_backend):
        if np.sum(frame) == 0:
            return []
        return [{"region": {"x":8,"y":8,"w":10,"h":10}, "dominant_emotion": "neutral"}]

def test_run_live_overlay_monkeypatch(monkeypatch):
    monkeypatch.setattr(live.cv2, 'VideoCapture', lambda idx: DummyCap())
    monkeypatch.setitem(sys.modules, 'deepface', types.SimpleNamespace(DeepFace=DummyDF))
    monkeypatch.setattr(live.cv2, 'imshow', lambda *a, **k: None)
    calls = {'n': 0}
    def fake_waitKey(delay):
        calls['n'] += 1
        return ord('q') if calls['n'] > 3 else -1
    monkeypatch.setattr(live.cv2, 'waitKey', fake_waitKey)

    s = Settings()
    s.LIVE_EMOTION_INTERVAL = 0.01
    live.run_live_overlay(s, camera_index=0)
