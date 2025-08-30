"""
Pipelines orchestrating the analysis stages.
"""
from __future__ import annotations
import os
import uuid
from pathlib import Path
from typing import List

from core.config import Settings
from core.models import AnalysisResponse, SilenceSpan, EmotionEntry
from core.audio_video import extract_audio_from_video
from core.asr import transcribe_audio
from core.silence import detect_silences
from core.fluency import compute_fluency, append_silence_markers
from core.emotion import analyze_emotions_from_video
from core.confidence import infer_confidence

def analyze_video_pipeline(video_path: str, settings: Settings) -> dict:
    """
    Video pipeline: extract audio -> emotions -> ASR -> silence -> fluency -> confidence.
    """
    tmp_dir = Path("temp")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    audio_path = tmp_dir / f"{uuid.uuid4()}.wav"

    extract_audio_from_video(video_path, str(audio_path), sample_rate=settings.AUDIO_SAMPLE_RATE)

    # Emotions on video stream
    emotion_timeline = analyze_emotions_from_video(
        video_path=video_path,
        interval_s=settings.EMOTION_INTERVAL,
        detector_backend="opencv",
    )

    # Silence via RMS
    silence_timeline, duration = detect_silences(
        str(audio_path),
        sample_rate=settings.AUDIO_SAMPLE_RATE,
        min_silence_dur=settings.MIN_SILENCE_DUR,
        threshold=settings.SILENCE_THRESHOLD,
    )

    # ASR
    transcript_text, language = transcribe_audio(str(audio_path), settings)
    transcript_with_markers = append_silence_markers(transcript_text, silence_timeline)

    # Fluency
    seconds = duration if duration > 0 else settings.RECORD_SECONDS
    fluency = compute_fluency(transcript_text, silence_timeline, seconds)

    # Confidence
    confidence = infer_confidence(fluency, emotion_timeline)

    try:
        os.remove(str(audio_path))
    except Exception:
        pass

    resp = AnalysisResponse(
        transcript=transcript_with_markers,
        emotion_timeline=emotion_timeline,
        silence_timeline=silence_timeline,
        fluency_analysis=fluency,
        confidence_analysis=confidence,
    )
    return resp.model_dump()

def analyze_audio_pipeline(audio_path: str, settings: Settings) -> dict:
    """
    Audio-only pipeline: silence -> ASR -> fluency -> confidence.
    """
    silence_timeline, duration = detect_silences(
        audio_path,
        sample_rate=settings.AUDIO_SAMPLE_RATE,
        min_silence_dur=settings.MIN_SILENCE_DUR,
        threshold=settings.SILENCE_THRESHOLD,
    )
    transcript_text, language = transcribe_audio(audio_path, settings)
    transcript_with_markers = append_silence_markers(transcript_text, silence_timeline)

    seconds = duration if duration > 0 else settings.RECORD_SECONDS
    fluency = compute_fluency(transcript_text, silence_timeline, seconds)

    emotions: List[EmotionEntry] = []
    confidence = infer_confidence(fluency, emotions)

    resp = AnalysisResponse(
        transcript=transcript_with_markers,
        emotion_timeline=emotions,
        silence_timeline=silence_timeline,
        fluency_analysis=fluency,
        confidence_analysis=confidence,
    )
    return resp.model_dump()
