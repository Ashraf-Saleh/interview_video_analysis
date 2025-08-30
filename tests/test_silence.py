import numpy as np, soundfile as sf, os
from core.silence import detect_silences

def test_detect_silences_simple(tmp_path):
    sr = 16000
    # 2s tone, 1s silence, 2s tone
    t1 = np.sin(2*np.pi*440*np.linspace(0,2,2*sr,endpoint=False))*0.1
    s  = np.zeros(sr)
    t2 = np.sin(2*np.pi*440*np.linspace(0,2,2*sr,endpoint=False))*0.1
    y = np.concatenate([t1, s, t2])
    wav = tmp_path / "test.wav"
    sf.write(wav, y, sr)

    spans, dur = detect_silences(str(wav), sample_rate=sr, min_silence_dur=0.5)
    assert dur > 4.9
    assert any(0.9 <= (sp.end - sp.start) <= 1.1 for sp in spans)
