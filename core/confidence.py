"""
Confidence inference heuristics.
"""
from __future__ import annotations
from typing import List, Dict
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


# core/confidence.py


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _safe_float(v, d=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(d)

def compute_confidence(
    fluency: Dict,
    emotion_log: List[Dict],
    silence_log: List[Dict]
) -> Dict:
    """
    Combine tone (audio) + affect (video) into a transparent 0..100 Confidence score.

    Inputs:
      fluency: {
        "words_per_minute": float,
        "filler_word_count": int,
        "rate_variation": float,
        ...
      }
      emotion_log: list of dicts with either
        {"time": float, "emotion": str, "region": {...}} OR
        {"time": float, "flag": "NO_FACE"|"MULTIPLE_FACES"}
      silence_log: list of {"start": float, "end": float}

    Returns:
      {
        "score": int,
        "verdict": "Confident" | "Moderately Confident" | "Needs Improvement",
        "components": {...}
      }
    """
    # --- Tone (audio) ---
    wpm = _safe_float(fluency.get("words_per_minute"), 0.0)
    filler_count = int(fluency.get("filler_word_count", 0))
    rate_var = _safe_float(fluency.get("rate_variation"), 0.0)

    # Pleasant rate peaks near ~135 WPM (110-160 acceptable)
    rate_score = clamp01(1.0 - abs(wpm - 135.0) / 50.0)

    # Normalize fillers roughly by speech time
    # For a simple proxy, use WPM ratio: more words -> more time
    norm_minutes = max(1e-6, wpm / 135.0)  # rough, prevents div-by-zero
    fillers_per_min = filler_count / norm_minutes
    filler_score = clamp01(1.0 - (fillers_per_min / 3.0))  # >=3/min -> 0

    # Long pauses from silence timeline
    long_pauses = sum(
        1 for s in silence_log
        if _safe_float(s.get("end")) - _safe_float(s.get("start")) >= 1.5
    )
    # Estimate duration in minutes (silences + talking – crude but stable)
    approx_minutes = max(
        1e-6,
        (sum(_safe_float(s.get("end")) - _safe_float(s.get("start")) for s in silence_log) / 60.0) + 0.5
    )
    long_pauses_per_min = long_pauses / approx_minutes
    pause_score = clamp01(1.0 - (long_pauses_per_min / 2.0))  # >=2/min -> 0

    # Rate stability: lower std is better
    stability_score = clamp01(1.0 - (rate_var / 3.0))  # >=3s std -> 0

    # --- Affect (video) ---
    emo_labels = [e.get("emotion") for e in emotion_log if "emotion" in e]
    flags = [e.get("flag") for e in emotion_log if "flag" in e]

    face_frames = max(1, len(emo_labels))
    positive_neutral = sum(1 for l in emo_labels if l in ("happy", "neutral"))
    anxiety_frames = sum(1 for l in emo_labels if l in ("fear", "sad"))
    affect_score = positive_neutral / face_frames
    anxiety_score = 1.0 - (anxiety_frames / face_frames)

    # MULTIPLE_FACES penalty
    multi_face_ratio = (flags.count("MULTIPLE_FACES") / max(1, len(emotion_log)))
    multi_face_penalty = clamp01(multi_face_ratio / 0.2)  # up to 0.1 in the weighted formula

    confidence_raw = (
        0.25 * rate_score +
        0.20 * filler_score +
        0.15 * pause_score +
        0.10 * stability_score +
        0.20 * affect_score +
        0.10 * anxiety_score -
        0.10 * multi_face_penalty
    )

    score = int(round(100.0 * clamp01(confidence_raw)))
    verdict = ("Confident" if score >= 80
               else "Moderately Confident" if score >= 60
               else "Needs Improvement")

    return {
        "score": score,
        "verdict": verdict,
        "components": {
            "rate_score": rate_score,
            "fillers_per_min": fillers_per_min,
            "filler_score": filler_score,
            "pause_score": pause_score,
            "stability_score": stability_score,
            "affect_score": affect_score,
            "anxiety_score": anxiety_score,
            "multi_face_penalty": multi_face_penalty,
        }
    }
