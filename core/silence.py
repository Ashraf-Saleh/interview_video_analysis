"""
Offline energy-based silence detection using librosa RMS.
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import librosa
from core.models import SilenceSpan

def detect_silences(
    audio_path: str,
    sample_rate: int = 16000,
    min_silence_dur: float = 0.35,
    threshold: Optional[float] = None,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> tuple[list[SilenceSpan], float]:
    """
    Detect silence spans from an audio file using RMS energy.
    If `threshold` is None, an automatic threshold is chosen from the RMS distribution.

    Args:
        audio_path: Path to audio file.
        sample_rate: Target sample rate for loading.
        min_silence_dur: Minimum duration (sec) to treat as silence.
        threshold: Optional absolute RMS threshold; if None, auto-computed.
        frame_length: RMS frame length.
        hop_length: Hop length.

    Returns:
        (silence_spans, duration_seconds)
    """
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # RMS energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)

    # Choose threshold
    thr = float(threshold) if threshold is not None else max(np.percentile(rms, 25) * 0.8, 1e-6)

    below = rms < thr
    spans: list[SilenceSpan] = []
    start_idx = None

    for i, is_silent in enumerate(below):
        if is_silent and start_idx is None:
            start_idx = i
        elif (not is_silent) and (start_idx is not None):
            s = times[start_idx]
            e = times[i]
            if (e - s) >= min_silence_dur:
                spans.append(SilenceSpan(start=round(float(s), 2), end=round(float(e), 2)))
            start_idx = None

    # Tail case
    if start_idx is not None:
        s = times[start_idx]
        e = times[-1]
        if (e - s) >= min_silence_dur:
            spans.append(SilenceSpan(start=round(float(s), 2), end=round(float(e), 2)))

    return spans, float(duration)
