
from core.config import Settings

def test_Settings():
    s = Settings()
    assert s.AUDIO_SAMPLE_RATE >= 8000
    # override via env-like behavior (construct new instance)
    s2 = Settings(AUDIO_SAMPLE_RATE=22050)
    assert s2.AUDIO_SAMPLE_RATE == 22050
