
import numpy as np, soundfile as sf
import core.asr as asr
from core.config import Settings

class DummyModel:
    def transcribe(self, path, **kwargs):  # ← accept fp16 or anything else
        return {"text": "hello world", "language": "en"}

def test_transcribe_audio(monkeypatch, tmp_path):
    import numpy as np, soundfile as sf
    import core.asr as asr
    from core.config import Settings

    monkeypatch.setattr(asr, "_model", None)
    monkeypatch.setattr(asr.whisper, "load_model", lambda *a, **k: DummyModel())

    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(16000, dtype="float32"), 16000)

    text, lang = asr.transcribe_audio(str(wav), Settings())
    assert text == "hello world" and lang == "en"

