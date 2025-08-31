
import numpy as np, cv2, sys, types, os
from pathlib import Path
from core.visual import draw_overlays, annotate_video

class DummyDeepFace:
    idx = 0
    @staticmethod
    def analyze(frame, actions, enforce_detection, detector_backend):
        # Sequence: [], 2 faces, then 1 face with emotion
        if DummyDeepFace.idx == 0:
            DummyDeepFace.idx += 1
            return []
        elif DummyDeepFace.idx == 1:
            DummyDeepFace.idx += 1
            return [{"region": {"x":5,"y":5,"w":10,"h":10}}, {"region": {"x":20,"y":5,"w":10,"h":10}}]
        else:
            DummyDeepFace.idx += 1
            return [{"region": {"x":15,"y":15,"w":12,"h":12}, "dominant_emotion": "happy"}]

def test_draw_overlays_cases():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    out1 = draw_overlays(frame, [], flag="NO_FACE")
    assert out1.shape == frame.shape
    out2 = draw_overlays(frame, [], flag="MULTIPLE_FACES")
    assert out2.shape == frame.shape
    faces = [{"region": {"x":10,"y":10,"w":15,"h":12}, "emotion":"happy"}]
    out3 = draw_overlays(frame, faces, None)
    assert out3.shape == frame.shape

def test_annotate_video_with_fake_deepface(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'deepface', types.SimpleNamespace(DeepFace=DummyDeepFace))
    h, w = 32, 32
    in_path = str(tmp_path / 'in.avi')
    out_path = str(tmp_path / 'out.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(in_path, fourcc, 5, (w, h))
    for _ in range(6):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()

    res = annotate_video(in_path, out_path, analyze_every_n_frames=1)
    assert os.path.exists(res)
    cap = cv2.VideoCapture(out_path)
    assert cap.isOpened()
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frames >= 5
