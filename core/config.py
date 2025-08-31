"""
Configuration for the analysis pipeline.
"""
from pydantic import BaseModel
import os

class Settings(BaseModel):
    """
    Runtime settings with environment-variable overrides.
    """
    DEVICE: str = (os.getenv("DEVICE", "cpu") or "cpu")

    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    EMOTION_INTERVAL: float = float(os.getenv("EMOTION_INTERVAL", "5"))
    MIN_SILENCE_DUR: float = float(os.getenv("MIN_SILENCE_DUR", "0.35"))
    SILENCE_THRESHOLD: float | None = (
        float(os.getenv("SILENCE_THRESHOLD")) if os.getenv("SILENCE_THRESHOLD") else None
    )
    RECORD_SECONDS: float = float(os.getenv("RECORD_SECONDS", "30"))
    CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
    LIVE_WINDOW_SECONDS: float = float(os.getenv("LIVE_WINDOW_SECONDS", "8"))
    LIVE_ASR_INTERVAL: float = float(os.getenv("LIVE_ASR_INTERVAL", "5"))
    LIVE_EMOTION_INTERVAL: float = float(os.getenv("LIVE_EMOTION_INTERVAL", "2"))

    def __init__(self, **data):
        super().__init__(**data)
        # Normalize DEVICE: strip comments/extra words, lower-case, validate
        dev = (self.DEVICE or "cpu").strip().split()[0].lower()
        if dev not in ("cpu", "cuda"):
            dev = "cpu"
        object.__setattr__(self, "DEVICE", dev)