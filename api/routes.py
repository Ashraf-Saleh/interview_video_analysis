"""
REST endpoints for analysis.
"""
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi import WebSocket, WebSocketDisconnect
import logging

from core.config import Settings
from core.pipeline import analyze_video_pipeline, analyze_audio_pipeline
from core.live import LiveAnalyzer

import tempfile
import shutil
import os



# app = FastAPI()
live_session = {"running": False}

router = APIRouter()
settings = Settings()
logger = logging.getLogger(__name__)


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
    logger.debug(f"[api] /analyze/video filename={file.filename} emotion_interval={emotion_interval}")
    if emotion_interval is not None:
        settings.EMOTION_INTERVAL = float(emotion_interval)

    # Save to temp file
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    except Exception as e:
        logger.exception("[api] upload save failed")
        raise HTTPException(status_code=400, detail=f"Upload failed: {e}")

    try:
        logger.debug(f"[api] starting analyze_video_pipeline tmp_path={tmp_path}")
        payload = analyze_video_pipeline(tmp_path, settings)
        logger.debug("[api] analyze_video_pipeline completed")
        return JSONResponse(payload)
    except FileNotFoundError as e:
        logger.exception("[api] analyze_video_pipeline file not found")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("[api] analyze_video_pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            logger.warning(f"[api] failed to cleanup tmp file: {tmp_path}")

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
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {e}")

    try:
        payload = analyze_audio_pipeline(tmp_path, settings)
        return JSONResponse(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass





# @app.post("/analyze/audio")
# async def analyze_audio(file: UploadFile = File(...)):
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
#         content = await file.read()
#         tmp.write(content)
#         tmp.flush()
        
#         try:
#             result = analyze_audio_pipeline(tmp.name, {})
#             return JSONResponse(result)
#         finally:
#             os.unlink(tmp.name)

# @app.post("/analyze/video") 
# async def analyze_video(file: UploadFile = File(...)):
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
#         content = await file.read()
#         tmp.write(content)
#         tmp.flush()

#         try:
#             result = analyze_video_pipeline(tmp.name, {})
#             return JSONResponse(result)
#         finally:
#             os.unlink(tmp.name)

@router.post("/live/start")
async def live_start():
    if live_session["running"]:
        return {"status": "already_running"}
    live_session["running"] = True
    return {"status": "started"}

@router.get("/live/status")
async def live_status():
    return {"running": live_session["running"]}

@router.post("/live/stop")
async def live_stop():
    if not live_session["running"]:
        return {"status": "not_running"}
    live_session["running"] = False
    return {"status": "stopped"}
