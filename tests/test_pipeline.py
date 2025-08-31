
import core.pipeline as pipe
from core.config import Settings

def test_analyze_audio_pipeline(monkeypatch, tmp_path):
    import numpy as np, soundfile as sf
    wav = tmp_path / 'a.wav'
    sf.write(wav, np.zeros(16000, dtype='float32'), 16000)
    monkeypatch.setattr(pipe, "detect_silences", lambda *a, **k: ([], 1.0))
    monkeypatch.setattr(pipe, "transcribe_audio", lambda *a, **k: ("hello world", "en"))
    res = pipe.analyze_audio_pipeline(str(wav), Settings())
    assert 'transcript' in res and 'fluency_analysis' in res

def test_analyze_video_pipeline(monkeypatch, tmp_path):
    vid = tmp_path / 'v.mp4'
    vid.write_bytes(b'fake')
    monkeypatch.setattr(pipe, "extract_audio_from_video", lambda *a, **k: str(tmp_path / 'out.wav'))
    monkeypatch.setattr(pipe, "analyze_emotions_from_video", lambda *a, **k: [])
    monkeypatch.setattr(pipe, "detect_silences", lambda *a, **k: ([], 1.0))
    monkeypatch.setattr(pipe, "transcribe_audio", lambda *a, **k: ("hello", "en"))
    res = pipe.analyze_video_pipeline(str(vid), Settings())
    assert 'emotion_timeline' in res and 'confidence_analysis' in res
