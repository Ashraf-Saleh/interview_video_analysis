from core.fluency import compute_fluency
from core.models import SilenceSpan

def test_compute_fluency_basic():
    transcript = "hello this is a simple test without um or uh"
    silences = [SilenceSpan(start=5.0, end=6.0)]
    fm = compute_fluency(transcript, silences, record_seconds=30.0)
    assert fm.words_per_minute > 0
    assert fm.fluency_score >= 0
