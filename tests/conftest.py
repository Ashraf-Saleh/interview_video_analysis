import pytest
from pathlib import Path

# Base data directory relative to repo root
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

@pytest.fixture
def sample_audio_path():
    return DATA_DIR / "sample.wav"

@pytest.fixture
def sample_video_path():
    return DATA_DIR / "sample.mp4"
