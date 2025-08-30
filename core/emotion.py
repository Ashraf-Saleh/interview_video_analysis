"""
Emotion sampling from video with DeepFace.
"""
from __future__ import annotations
from typing import List
import cv2
from core.models import EmotionEntry

def analyze_emotions_from_video(
    video_path: str,
    interval_s: float,
    detector_backend: str = "opencv",
) -> List[EmotionEntry]:
    """
    Sample frames each `interval_s` seconds and infer dominant emotion.

    Returns:
        List[EmotionEntry]
    """
    from deepface import DeepFace

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval_frames = max(1, int(fps * interval_s))
    idx = 0
    log: List[EmotionEntry] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval_frames == 0:
            ts = round(idx / fps, 2)
            try:
                result = DeepFace.analyze(
                    frame, actions=["emotion"], enforce_detection=False, detector_backend=detector_backend
                )
                items = result if isinstance(result, list) else [result]
                if not items or (len(items) == 1 and items[0].get("face_detected", 1) == 0):
                    log.append(EmotionEntry(time=ts, flag="NO_FACE"))
                elif len(items) > 1:
                    log.append(EmotionEntry(time=ts, flag="MULTIPLE_FACES"))
                else:
                    emo = items[0].get("dominant_emotion")
                    if emo is None and "emotion" in items[0]:
                        probs = items[0]["emotion"]
                        emo = max(probs, key=probs.get)
                    log.append(EmotionEntry(time=ts, emotion=emo))
            except Exception:
                log.append(EmotionEntry(time=ts, flag="NO_FACE"))
        idx += 1

    cap.release()
    return log
