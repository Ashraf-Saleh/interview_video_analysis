"""
Audio/video utilities (FFmpeg extraction).
"""
from __future__ import annotations
import subprocess
import os

def extract_audio_from_video(video_path: str, audio_path: str, sample_rate: int = 16000) -> str:
    """
    Extract mono audio from a video using FFmpeg.

    Args:
        video_path: Input video path.
        audio_path: Output audio file path (WAV recommended).
        sample_rate: Target sample rate.

    Returns:
        str: Output audio path.

    Raises:
        FileNotFoundError: Input file missing.
        RuntimeError: FFmpeg failure.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = ["ffmpeg", "-y", "-i", video_path, "-ar", str(sample_rate), "-ac", "1", "-vn", audio_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {proc.stderr.decode('utf-8', errors='ignore')}")
    return audio_path
