
import numpy as np, soundfile as sf
from core.silence import detect_silences

def test_detect_silences(tmp_path):
    sr = 16000
    tone = np.sin(2*np.pi*440*np.linspace(0,1,sr,endpoint=False))*0.1
    sil  = np.zeros(int(sr*0.7))
    y = np.concatenate([tone, sil, tone])
    wav = tmp_path / 't.wav'
    sf.write(wav, y, sr)
    spans, dur = detect_silences(str(wav), sample_rate=sr, min_silence_dur=0.5)
    assert dur > 2.3
    assert any((sp.end - sp.start) >= 0.5 for sp in spans)
