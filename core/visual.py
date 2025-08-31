
# core/visual.py
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

def draw_overlays(
    frame: np.ndarray,
    faces: List[Dict] | None = None,
    flag: Optional[str] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw rectangles + emotion labels and/or a big flag text.

    faces: list of {"region": {"x","y","w","h"}, "emotion": "happy|..."}
    flag:  "NO_FACE" or "MULTIPLE_FACES" or None
    """
    out = frame.copy()
    h, w = out.shape[:2]
    faces = faces or []

    if flag == "NO_FACE":
        cv2.putText(out, "NO_FACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 2, cv2.LINE_AA)
        return out

    if flag == "MULTIPLE_FACES":
        cv2.putText(out, "MULTIPLE_FACES", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 2, cv2.LINE_AA)

    for f in faces:
        r = f.get("region") or {}
        x, y, fw, fh = int(r.get("x", 0)), int(r.get("y", 0)), int(r.get("w", 0)), int(r.get("h", 0))
        # clamp to image bounds
        x = max(0, min(x, w - 1)); y = max(0, min(y, h - 1))
        fw = max(0, min(fw, w - x)); fh = max(0, min(fh, h - y))
        cv2.rectangle(out, (x, y), (x + fw, y + fh), color, 2)
        label = f.get("emotion") or ""
        if label:
            cv2.putText(out, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2, cv2.LINE_AA)
    return out
