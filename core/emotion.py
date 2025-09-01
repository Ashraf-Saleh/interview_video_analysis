"""
Emotion sampling from video with DeepFace.
"""
# core/emotion.py
from __future__ import annotations
from typing import List, Dict, Tuple
import logging
import cv2

from core.config import Settings

logger = logging.getLogger(__name__)

def analyze_emotions_from_video(
    video_path: str,
    settings: Settings
) -> Tuple[List[Dict], float]:
    """
    Process video frames at fixed intervals and log facial emotions using DeepFace.

    Returns:
      (emotion_log, fps)
      emotion_log is a list of entries:
        {"time": t, "emotion": <label>, "region": {x,y,w,h}} OR
        {"time": t, "flag": "NO_FACE"|"MULTIPLE_FACES"}
    """
    logger.debug(f"[emotion] open video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval_frames = max(1, int(fps * float(settings.EMOTION_INTERVAL)))
    frame_index = 0
    emotion_log: List[Dict] = []
    logger.debug(f"[emotion] fps={fps} interval_frames={interval_frames}")

    # Lazy import for easier testing and to avoid loading heavy stacks too early
    from deepface import DeepFace

    # Tunables to improve detection robustness
    MIN_BOX = 40          # px; increase to avoid tiny faces
    MIN_DET_CONF = 0.5    # if backend supplies a score

    def _valid_region(r) -> bool:
        reg = (r or {}).get("region") or {}
        w = int(reg.get("w", 0)); h = int(reg.get("h", 0))
        ok_size = (w >= MIN_BOX and h >= MIN_BOX)
        conf = (r.get("face_confidence") or r.get("detector_score") or 1.0)
        try:
            conf = float(conf)
        except Exception:
            conf = 1.0
        ok_conf = (conf >= MIN_DET_CONF)
        return ok_size and ok_conf

    def _best_label(blob: Dict) -> str:
        # Prefer dominant_emotion; fall back to max-prob from dict
        if not isinstance(blob, dict):
            return ""
        dom = blob.get("dominant_emotion")
        if isinstance(dom, str) and dom:
            return dom
        em = blob.get("emotion")
        if isinstance(em, dict) and em:
            try:
                return max(em, key=em.get)
            except Exception:
                return ""
        return ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % interval_frames == 0:
            timestamp = round(frame_index / fps, 2)
            try:
                # 1) detect faces only (OpenCV backend)
                faces = []
                try:
                    dets = DeepFace.extract_faces(
                        img_path=frame,
                        detector_backend="opencv",
                        enforce_detection=False,
                        align=True,
                    )
                    for d in dets or []:
                        fa = d.get("facial_area") or {}
                        blob = {"region": {"x": fa.get("x", 0), "y": fa.get("y", 0), "w": fa.get("w", 0), "h": fa.get("h", 0)},
                                "face_confidence": d.get("confidence", 1.0)}
                        if _valid_region(blob):
                            faces.append(blob)
                except Exception:
                    # Fallback to analyze to get regions if extract not available
                    res = DeepFace.analyze(
                        frame,
                        actions=["emotion"],
                        enforce_detection=False,
                        detector_backend="opencv",
                    )
                    res = res if isinstance(res, list) else [res]
                    for r in res:
                        if _valid_region(r):
                            faces.append(r)
                logger.debug(f"[emotion] t={timestamp}s faces_detected={len(faces)}")

                if len(faces) == 0:
                    emotion_log.append({"time": timestamp, "flag": "NO_FACE"})
                elif len(faces) > 1:
                    emotion_log.append({"time": timestamp, "flag": "MULTIPLE_FACES"})
                else:
                    # 2) single-face confirmed -> compute emotion on the crop only now
                    reg = faces[0].get("region") or {}
                    x, y, w, h = int(reg.get("x", 0)), int(reg.get("y", 0)), int(reg.get("w", 0)), int(reg.get("h", 0))
                    chip = frame[y:y+h, x:x+w]
                    logger.debug(f"[emotion] single face @ t={timestamp}s region=({x},{y},{w},{h}) -> analyzing emotion")
                    try:
                        emo = DeepFace.analyze(
                            chip if chip.size else frame,
                            actions=["emotion"],
                            enforce_detection=False,
                            detector_backend="opencv",
                            align=True,
                        )
                        emo = emo if isinstance(emo, list) else [emo]
                        r0 = emo[0] if emo else {}
                        label = _best_label(r0)
                    except Exception:
                        label = ""
                        logger.exception("[emotion] emotion inference failed; labeling empty")
                    emotion_log.append({
                        "time": timestamp,
                        "emotion": label,
                        "region": reg
                    })
            except Exception:
                # Be resilient: mark as NO_FACE rather than failing the whole run
                logger.exception(f"[emotion] exception at t={timestamp}s; marking NO_FACE")
                emotion_log.append({"time": timestamp, "flag": "NO_FACE"})

        frame_index += 1

    cap.release()
    logger.debug(f"[emotion] finished; entries={len(emotion_log)}")
    return emotion_log, float(fps)
