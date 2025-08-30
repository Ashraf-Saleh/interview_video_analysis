"""
Fluency metrics from transcript and silence spans.
"""
from __future__ import annotations
import re
import numpy as np
from typing import List
from core.models import SilenceSpan, FluencyMetrics

DEFAULT_FILLERS = r"\b(um|uh|like|you know)\b"

def compute_fluency(
    transcript: str,
    silence_spans: List[SilenceSpan],
    record_seconds: float,
    fillers_regex: str = DEFAULT_FILLERS,
) -> FluencyMetrics:
    """
    Compute WPM, filler count, rate variation, and a heuristic fluency score.

    Args:
        transcript: Full transcript.
        silence_spans: Detected silence spans (seconds).
        record_seconds: Total duration (seconds) to normalize WPM.
        fillers_regex: Regex pattern for filler words.

    Returns:
        FluencyMetrics
    """
    words = len(transcript.split())
    wpm = round(words / (max(record_seconds, 1e-6) / 60.0), 2)

    fillers = re.findall(fillers_regex, transcript, flags=re.IGNORECASE)
    filler_count = len(fillers)

    # Speech segment durations between silences
    speech_durations = []
    prev_end = 0.0
    for span in silence_spans:
        seg = max(0.0, span.start - prev_end)
        if seg > 0:
            speech_durations.append(seg)
        prev_end = span.end
    tail = max(0.0, record_seconds - prev_end)
    if tail > 0:
        speech_durations.append(tail)

    rate_variation = round(np.std(speech_durations), 2) if speech_durations else 0.0

    fluency_score = max(0, min(100, 100 - filler_count * 3 - rate_variation * 5 + wpm * 0.5))
    return FluencyMetrics(
        words_per_minute=wpm,
        filler_word_count=filler_count,
        rate_variation=rate_variation,
        fluency_score=round(fluency_score, 2),
        accent_signal="likely fluent",
    )

def append_silence_markers(transcript: str, silence_spans: List[SilenceSpan]) -> str:
    """
    Append inline silence markers to the transcript.
    """
    out = transcript
    for s in silence_spans:
        out += f" [silence {s.start}s to {s.end}s] "
    return out
