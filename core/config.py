"""
Configuration for the analysis pipeline.
"""
from pydantic import BaseModel
import os

class Settings(BaseModel):
    """
    Runtime settings with environment-variable overrides.
    """
    DEVICE: str = os.getenv("DEVICE", "cpu")
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")

    AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    EMOTION_INTERVAL: float = float(os.getenv("EMOTION_INTERVAL", "5"))

    # Silence detection
    MIN_SILENCE_DUR: float = float(os.getenv("MIN_SILENCE_DUR", "0.35"))
    # If empty/None -> auto threshold from RMS distribution
    SILENCE_THRESHOLD: float | None = (
        float(os.getenv("SILENCE_THRESHOLD")) if os.getenv("SILENCE_THRESHOLD") else None
    )

    # Fallback duration if actual duration cannot be read (rare)
    RECORD_SECONDS: float = float(os.getenv("RECORD_SECONDS", "30"))

    # live settings 
    CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
    LIVE_WINDOW_SECONDS: float = float(os.getenv("LIVE_WINDOW_SECONDS", "8"))  # rolling ASR window
    LIVE_ASR_INTERVAL: float = float(os.getenv("LIVE_ASR_INTERVAL", "5"))     # seconds between ASR refresh
    LIVE_EMOTION_INTERVAL: float = float(os.getenv("LIVE_EMOTION_INTERVAL", "2"))  # seconds between emotion samples
