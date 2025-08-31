
from core.fluency import compute_fluency, append_silence_markers
from core.models import SilenceSpan

def test_compute_fluency():
    fm1 = compute_fluency('um hello uh this is fine', [], record_seconds=10)
    assert fm1.filler_word_count >= 2 and 0 <= fm1.fluency_score <= 100
    fm2 = compute_fluency('this is a longer transcript ' * 10, [], record_seconds=30)
    assert fm2.words_per_minute > 0

def test_append_silence_markers():
    t = 'This is a test'
    spans = [SilenceSpan(start=1.0, end=2.0), SilenceSpan(start=3.0, end=3.5)]
    out = append_silence_markers(t, spans)
    assert out.count('[silence') == 2
