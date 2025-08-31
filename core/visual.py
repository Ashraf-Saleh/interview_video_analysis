
"""Visualization & video annotation helpers.

- draw_overlays: draw rectangles & labels for faces/emotions OR flags (NO_FACE / MULTIPLE_FACES)
- annotate_video: read a video, analyze frames with DeepFace, and write an annotated video

These utilities are designed to be testable by monkeypatching the DeepFace import.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

def draw_overlays(frame: np.ndarray,
                  faces: List[Dict] | None = None,
                  flag: Optional[str] = None,
                  color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw bounding boxes and labels on a frame.

    Args:
        frame: BGR image
        faces: list of dicts with keys {"region": {x,y,w,h}, "emotion": str}
        flag: optional flag string (e.g., "NO_FACE", "MULTIPLE_FACES")
        color: BGR color for rectangles

    Returns:
        Annotated frame (in-place modified and returned)
    """
    out = frame.copy()
    h, w = out.shape[:2]

    if faces is None:
        faces = []

    if flag == "NO_FACE":
        cv2.putText(out, "NO_FACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return out
    if flag == "MULTIPLE_FACES":
        cv2.putText(out, "MULTIPLE_FACES", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    for face in faces:
        reg = face.get("region") or {}
        x, y, fw, fh = int(reg.get("x", 0)), int(reg.get("y", 0)), int(reg.get("w", 0)), int(reg.get("h", 0))
        # clamp to image bounds
        x = max(0, min(x, w-1)); y = max(0, min(y, h-1))
        fw = max(0, min(fw, w-x)); fh = max(0, min(fh, h-y))

        cv2.rectangle(out, (x, y), (x+fw, y+fh), color, 2)
        label = face.get("emotion") or ""
        if label:
            cv2.putText(out, label, (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return out


def annotate_video(input_path: str,
                   output_path: str,
                   analyze_every_n_frames: int = 5,
                   detector_backend: str = "opencv",
                   enforce_detection: bool = False) -> str:
    """Annotate a video with face windows, emotions, and flags.

    For each frame N, we run DeepFace.analyze(frame, actions=['emotion']) and:
      - NO_FACE: draw flag text
      - MULTIPLE_FACES: draw flag text
      - exactly one face: draw rectangle + emotion label

    NOTE: DeepFace is imported lazily to simplify testing via monkeypatching.
    Returns the path to the annotated video.
    """
    import os
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # robust across platforms for tests
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        # Lazy import DeepFace so tests can monkeypatch sys.modules['deepface']
        from deepface import DeepFace
    except Exception as e:
        cap.release()
        writer.release()
        raise RuntimeError("DeepFace import failed. Ensure deepface/tensorflow stack is installed.") from e

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        flag = None
        faces = []
        if idx % max(1, analyze_every_n_frames) == 0:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'],
                                          enforce_detection=enforce_detection,
                                          detector_backend=detector_backend)
                # DeepFace returns list[dict] or dict depending on version; normalize to list
                if isinstance(result, dict):
                    result = [result]
                if len(result) == 0:
                    flag = "NO_FACE"
                elif len(result) > 1:
                    flag = "MULTIPLE_FACES"
                # build faces list
                for r in result:
                    faces.append({
                        "region": r.get("region") or {},
                        "emotion": (r.get("dominant_emotion") or r.get("emotion") or "")
                    })
            except Exception:
                # on detection error, mark as NO_FACE to keep pipeline robust
                flag = "NO_FACE"

        annotated = draw_overlays(frame, faces, flag)
        writer.write(annotated)
        idx += 1

    cap.release()
    writer.release()
    return output_path
