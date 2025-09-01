import types
import numpy as np
import core.live as live


class DummyCap:
    def __init__(self):
        self.calls = 0
    def isOpened(self):
        return True
    def read(self):
        self.calls += 1
        if self.calls > 2:
            return False, None
        return True, np.zeros((64, 64, 3), dtype=np.uint8)
    def release(self):
        pass


def test_overlay_multi_faces_no_emotion(monkeypatch):
    # Arrange: two faces detected -> analyze for emotion must NOT be called
    analyze_called = {"n": 0}

    class DF:
        @staticmethod
        def extract_faces(img_path=None, detector_backend=None, enforce_detection=None, align=None):
            return [
                {"facial_area": {"x": 5, "y": 5, "w": 20, "h": 20}},
                {"facial_area": {"x": 30, "y": 8, "w": 18, "h": 18}},
            ]
        @staticmethod
        def analyze(*args, **kwargs):
            analyze_called["n"] += 1
            return []

    monkeypatch.setattr(live.cv2, "VideoCapture", lambda idx: DummyCap())
    monkeypatch.setattr(live.cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(live.cv2, "waitKey", lambda d: ord("q"))  # exit after first frame
    monkeypatch.setitem(__import__('sys').modules, "deepface", types.SimpleNamespace(DeepFace=DF))

    s = live.Settings()
    live.run_live_overlay(s, camera_index=0)

    # Assert: emotion analyze should not be called when faces > 1
    assert analyze_called["n"] == 0


def test_overlay_single_face_triggers_emotion(monkeypatch):
    # Arrange: one face detected -> analyze on crop should be called exactly once
    analyze_called = {"n": 0}

    class DF:
        @staticmethod
        def extract_faces(img_path=None, detector_backend=None, enforce_detection=None, align=None):
            return [{"facial_area": {"x": 10, "y": 10, "w": 30, "h": 30}}]
        @staticmethod
        def analyze(*args, **kwargs):
            analyze_called["n"] += 1
            return [{"dominant_emotion": "neutral", "emotion": {"neutral": 0.9}}]

    monkeypatch.setattr(live.cv2, "VideoCapture", lambda idx: DummyCap())
    monkeypatch.setattr(live.cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(live.cv2, "waitKey", lambda d: ord("q"))
    monkeypatch.setitem(__import__('sys').modules, "deepface", types.SimpleNamespace(DeepFace=DF))

    s = live.Settings()
    live.run_live_overlay(s, camera_index=0)

    # Assert: emotion computed exactly once on the crop
    assert analyze_called["n"] == 1

