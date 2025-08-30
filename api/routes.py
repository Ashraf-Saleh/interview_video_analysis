"""
REST endpoints for analysis.
"""
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi import WebSocket, WebSocketDisconnect

from core.config import Settings
from core.pipeline import analyze_video_pipeline, analyze_audio_pipeline
from core.live import LiveAnalyzer


router = APIRouter()

@router.post("/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    emotion_interval: float | None = Form(None),
):
    """
    Analyze an interview video: extract audio, run ASR, compute fluency and silence,
    sample emotions from frames, and infer confidence.

    Args:
        file: Uploaded video file.
        emotion_interval: Optional override for seconds between emotion samples.

    Returns:
        JSONResponse: Structured analysis payload.
    """
    settings = Settings()
    if emotion_interval is not None:
        settings.EMOTION_INTERVAL = float(emotion_interval)

    tmp_dir = Path("temp")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    vid_path = tmp_dir / f"{uuid.uuid4()}_{file.filename}"
    with vid_path.open("wb") as f:
        f.write(await file.read())

    try:
        payload = analyze_video_pipeline(str(vid_path), settings=settings)
        return JSONResponse(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            vid_path.unlink(missing_ok=True)
        except Exception:
            pass

@router.post("/analyze/audio")
async def analyze_audio(
    file: UploadFile = File(...),
    min_silence_dur: float | None = Form(None),
    silence_threshold: float | None = Form(None),
):
    """
    Analyze an audio recording: run ASR, compute fluency and silence markers,
    and infer confidence (emotion timeline omitted).

    Args:
        file: Uploaded audio file.
        min_silence_dur: Minimum duration (sec) to consider a span as silence.
        silence_threshold: Optional numeric RMS threshold override.

    Returns:
        JSONResponse: Structured analysis payload.
    """
    settings = Settings()
    if min_silence_dur is not None:
        settings.MIN_SILENCE_DUR = float(min_silence_dur)
    if silence_threshold is not None:
        settings.SILENCE_THRESHOLD = float(silence_threshold)

    tmp_dir = Path("temp")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    audio_path = tmp_dir / f"{uuid.uuid4()}_{file.filename}"
    with audio_path.open("wb") as f:
        f.write(await file.read())

    try:
        payload = analyze_audio_pipeline(str(audio_path), settings=settings)
        return JSONResponse(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass
