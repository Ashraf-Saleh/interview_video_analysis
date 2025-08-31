
from core.models import SilenceSpan, FluencyMetrics, EmotionEntry, ConfidenceResult, AnalysisResponse

def test_models():
    sl = [SilenceSpan(start=1.0, end=1.5)]
    fm = FluencyMetrics(words_per_minute=120.0, filler_word_count=1, rate_variation=0.5, fluency_score=85.0, accent_signal="likely fluent")
    em = [EmotionEntry(time=0.0, emotion="happy")]
    cr = ConfidenceResult(confidence_inferred=True, confidence_score=3, verdict="Confident")
    ar = AnalysisResponse(transcript="hi", emotion_timeline=em, silence_timeline=sl, fluency_analysis=fm, confidence_analysis=cr)
    assert ar.fluency_analysis.fluency_score == 85.0
