"""
Whisper ASR wrapper (lazy-loaded).
"""
from __future__ import annotations
import whisper
from core.config import Settings

_model = None

def _ensure_model(settings: Settings):
    global _model
    if _model is None:
        _model = whisper.load_model(settings.WHISPER_MODEL, device=settings.DEVICE)
    return _model

def transcribe_audio(audio_path: str, settings: Settings) -> tuple[str, str]:
    """
    Transcribe audio with Whisper.

    Args:
        audio_path: Path to audio file.
        settings: Runtime settings.

    Returns:
        (text, language_code)
    """
    model = _ensure_model(settings)
    result = model.transcribe(audio_path)
    return result.get("text", ""), result.get("language", "en")
