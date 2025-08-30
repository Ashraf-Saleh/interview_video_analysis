"""
Confidence inference heuristics.
"""
from __future__ import annotations
from typing import List
from core.models import ConfidenceResult, EmotionEntry, FluencyMetrics

def infer_confidence(fluency: FluencyMetrics, emotions: List[EmotionEntry]) -> ConfidenceResult:
    """
    Infer confidence from fluency metrics and emotion stability.
    """
    score = 0
    if fluency.fluency_score > 70:
        score += 1
    if fluency.filler_word_count < 3:
        score += 1
    if fluency.rate_variation < 2:
        score += 1

    confident = {"happy", "neutral", "angry"}
    nervous = {"fear", "sad", "surprise"}
    c_cnt = sum(1 for e in emotions if e.emotion in confident)
    n_cnt = sum(1 for e in emotions if (e.emotion in nervous) or (e.flag is not None))
    if c_cnt > n_cnt:
        score += 1

    return ConfidenceResult(
        confidence_inferred=score >= 3,
        confidence_score=score,
        verdict="Confident" if score >= 3 else "Not Confident",
    )
