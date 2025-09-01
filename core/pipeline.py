# core/pipeline.py
from __future__ import annotations
from typing import Dict, Tuple
import logging
import os

from core.config import Settings
from core.asr import transcribe_audio
from core.audio_video import extract_audio_from_video
from core.emotion import analyze_emotions_from_video
from core.fluency import compute_fluency, append_silence_markers
from core.confidence import compute_confidence
import soundfile as sf

logger = logging.getLogger(__name__)

def analyze_audio_pipeline(audio_path: str, settings: Settings) -> Dict:
    """
    Transcribe audio, compute fluency (tone) metrics, infer confidence using audio-only signals.
    """
    logger.debug(f"[pipeline] analyze_audio_pipeline start audio_path={audio_path}")
    text, lang = transcribe_audio(audio_path, settings)
    # We assume silence detection was done during record; if not, pass [].
    silence_log = []  # or load/compute if you track it
    try:
        info = sf.info(audio_path)
        record_seconds = (info.frames / float(info.samplerate)) if info.samplerate else 0.0
    except Exception:
        record_seconds = 0.0
        logger.exception("[pipeline] failed to read audio info; record_seconds=0")
    fluency = compute_fluency(text, silence_log, record_seconds)
    text_marked = append_silence_markers(text, silence_log)

    # With no emotion timeline, confidence will lean on tone only
    confidence = compute_confidence(fluency.dict(), [], silence_log)

    payload = {
        "transcript": text_marked,
        "emotion_timeline": [],
        "silence_timeline": silence_log,
        "fluency_analysis": fluency.dict(),
        "confidence_analysis": confidence,
    }
    logger.debug("[pipeline] analyze_audio_pipeline finished successfully")
    return payload

def analyze_video_pipeline(video_path: str, settings: Settings) -> Dict:
    """
    Full pipeline for a video answer: extract audio, transcribe, fluency, emotions, confidence.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    logger.debug(f"[pipeline] analyze_video_pipeline start video_path={video_path}")

    # 1) Extract audio
    base, _ = os.path.splitext(video_path)
    audio_path = base + ".wav"
    logger.debug(f"[pipeline] extracting audio -> {audio_path} sr={settings.AUDIO_SAMPLE_RATE}")
    audio_path = extract_audio_from_video(video_path, audio_path, sample_rate=settings.AUDIO_SAMPLE_RATE)

    # 2) Emotions from video (with robust detector/filters)
    logger.debug("[pipeline] analyzing emotions from video frames")
    emotion_log, _fps = analyze_emotions_from_video(video_path, settings)

    # 3) Transcribe + tone metrics
    logger.debug(f"[pipeline] transcribing audio: {audio_path}")
    text, lang = transcribe_audio(audio_path, settings)
    silence_log = []  # or populate if you run silence detection separately
    logger.debug("[pipeline] computing fluency metrics")
    try:
        info = sf.info(audio_path)
        record_seconds = (info.frames / float(info.samplerate)) if info.samplerate else 0.0
    except Exception:
        record_seconds = 0.0
        logger.exception("[pipeline] failed to read audio info; record_seconds=0")
    fluency = compute_fluency(text, silence_log, record_seconds)
    text_marked = append_silence_markers(text, silence_log)

    # 4) Confidence (tone + affect)
    logger.debug("[pipeline] computing confidence from fluency + emotions")
    confidence = compute_confidence(fluency.dict(), emotion_log, silence_log)

    payload = {
        "transcript": text_marked,
        "emotion_timeline": emotion_log,
        "silence_timeline": silence_log,
        "fluency_analysis": fluency.dict(),
        "confidence_analysis": confidence,
    }
    logger.debug("[pipeline] analyze_video_pipeline finished successfully")
    return payload
