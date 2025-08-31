import numpy as np, cv2, sys, types
import core.emotion as emotion_mod
from core.models import EmotionEntry

class DummyDeepFace:
    @staticmethod
    def analyze(frame, actions, enforce_detection, detector_backend):
        # Return a single-face result with a dominant emotion
        return [{"dominant_emotion": "happy"}]

def test_analyze_emotions_from_video(monkeypatch, tmp_path):
    # ✅ Inject a fake 'deepface' module so `from deepface import DeepFace` works
    monkeypatch.setitem(sys.modules, "deepface", types.SimpleNamespace(DeepFace=DummyDeepFace))

    # Build a tiny test video (10 frames @ 5fps, 32x32)
    h, w = 32, 32
    path = str(tmp_path / "tiny.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, 5, (w, h))
    assert writer.isOpened(), "OpenCV VideoWriter failed to open (try changing codec to 'MJPG' or 'mp4v')"
    for _ in range(10):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    out = emotion_mod.analyze_emotions_from_video(path, interval_s=1.0)
    assert isinstance(out, list) and all(isinstance(e, EmotionEntry) for e in out)
    # Optional: check at least one 'happy'
    assert any(getattr(e, "emotion", None) == "happy" for e in out)

