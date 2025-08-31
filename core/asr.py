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

    - Uses device from Settings (cpu/cuda)
    - Disables fp16 on CPU to avoid the 'FP16 is not supported on CPU' warning
    """
    model = _ensure_model(settings)

    use_fp16 = settings.DEVICE == "cuda"  # FP16 only makes sense on GPU
    result = model.transcribe(audio_path, fp16=use_fp16)

    return result.get("text", ""), result.get("language", "en")

