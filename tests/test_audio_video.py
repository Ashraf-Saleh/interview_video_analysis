
import pytest
from core.audio_video import extract_audio_from_video

def test_extract_audio_from_video():
    # case: missing file
    with pytest.raises(FileNotFoundError):
        extract_audio_from_video('does_not_exist.mp4', 'out.wav')
