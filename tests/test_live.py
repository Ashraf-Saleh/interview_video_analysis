
import time, types, sys
from core.live import LiveAnalyzer
from core.config import Settings
import core.asr as asr


class DummyWModel:
    def transcribe(self, *a, **k): return {"text": "ok", "language": "en"}

class DummyVideoCap:
    def __init__(self, idx):
        self.idx = idx
    def isOpened(self): return True
    def read(self): return True, None
    def get(self, code): return 30.0
    def release(self): pass

class DummyInputStream:
    def __init__(self, callback, channels, samplerate, blocksize):
        self.callback = callback
    def __enter__(self): 
        import numpy as np
        # push a small buffer once
        self.callback(np.zeros((8000,1), dtype='float32'), 8000, None, None)
        return self
    def __exit__(self, *a): pass

def test_LiveAnalyzer():
    import core.live as live
    # patch camera & mic
    # live.cv2.VideoCapture = lambda idx: DummyVideoCap(idx)
    # live.sd.InputStream = lambda **kw: DummyInputStream(**kw)
    monkeypatch.setattr(asr, "_model", DummyWModel())
    monkeypatch.setattr(asr.whisper, "load_model", lambda *a, **k: DummyWModel())
    class DF: 
        @staticmethod
        def analyze(*a, **k): return [{"dominant_emotion": "neutral"}]
    # sys.modules['deepface'] = types.SimpleNamespace(DeepFace=DF)

    s = Settings()
    s.LIVE_EMOTION_INTERVAL = 0.1
    s.LIVE_ASR_INTERVAL = 0.2
    s.LIVE_WINDOW_SECONDS = 0.5
    la = LiveAnalyzer(s)
    la.start()
    time.sleep(0.5)
    snap = la.status()
    la.stop()
    assert (snap is None) or (snap.verdict in ('Confident', 'Not Confident'))
