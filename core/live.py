# core/live.py
"""
Live (real-time) analysis utilities.

Captures webcam frames & mic audio locally, produces rolling snapshots:
- Emotion every LIVE_EMOTION_INTERVAL seconds
- ASR+fluency every LIVE_ASR_INTERVAL seconds over the latest LIVE_WINDOW_SECONDS audio
- Confidence from current fluency + recent emotion

This module also provides a live overlay window (run_live_overlay) that draws:
- Face rectangle(s)
- Emotion label (smoothed & probability gated)
- Flags: NO_FACE / MULTIPLE_FACES (debounced with hysteresis)
"""

from __future__ import annotations

import io
import os
import time
import threading
import collections
from typing import Optional, List, Deque

import cv2
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from collections import deque, Counter

# Prevent OpenMP oversubscription on CPU
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from core.config import Settings
from core.fluency import compute_fluency
from core.confidence import infer_confidence
from core.models import LiveSnapshot, LiveFluency, EmotionEntry, SilenceSpan
from core.visual import draw_overlays


# -----------------------------------------------------------------------------
# 🔧 Tuning knobs (speed vs accuracy) - adjust freely without touching code
# -----------------------------------------------------------------------------
DEFAULT_DETECT_EVERY_N = 6     # Detect every N frames (↑ faster, ↓ jitter)
DETECT_WIDTH = 480             # Downscale width for detection (480..960)
TRACKER_TYPE = "KCF"           # MOSSE fastest, KCF balanced, CSRT most stable

# Detection hygiene
MIN_BOX_FRACTION = 0.04        # Min face box ≈ 5% of min(frame_w, frame_h)
CENTER_BIAS = 0.35             # 0..1 weight to prefer faces near center
MULTI_FACE_HYSTERESIS = 2      # Require N consecutive multi-face detections
NO_FACE_HYSTERESIS = 2         # Require N consecutive no-face detections

# Emotion hygiene
EMO_MIN_PROB = 0.55            # Min smoothed probability to accept emotion
EMO_SMOOTH_ALPHA = 0.6         # EMA smoothing factor for emotion probabilities
EMO_STALE_SEC = 0.9            # Re-run crop emotion if older than X seconds
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Small helpers for smoothing and debouncing
# -----------------------------------------------------------------------------
class ProbEMA:
    """Exponential moving average for emotion probability dicts."""
    def __init__(self, alpha: float = 0.5):
        self.alpha = float(alpha)
        self.state: Optional[dict] = None

    def update(self, probs: Optional[dict]) -> Optional[dict]:
        if not probs:
            return self.state
        if self.state is None:
            self.state = {k: float(v) for k, v in probs.items()}
            return self.state
        out: dict[str, float] = {}
        keys = set(self.state.keys()) | set(probs.keys())
        for k in keys:
            pv = float(probs.get(k, 0.0))
            sv = float(self.state.get(k, 0.0))
            out[k] = self.alpha * pv + (1.0 - self.alpha) * sv
        self.state = out
        return self.state

    def top(self) -> tuple[Optional[str], float]:
        if not self.state:
            return None, 0.0
        k = max(self.state, key=self.state.get)
        return k, float(self.state[k])


class FlagHysteresis:
    """Debounce NO_FACE / MULTIPLE_FACES with consecutive confirmations."""
    def __init__(self, needed_no: int = 3, needed_multi: int = 3):
        self.need_no = int(needed_no)
        self.need_multi = int(needed_multi)
        self.no_cnt = 0
        self.multi_cnt = 0
        self.cur: Optional[str] = None

    def step(self, flag: Optional[str]) -> Optional[str]:
        """
        Update debounced flag state.
        - "NO_FACE" / "MULTIPLE_FACES": increment respective counters and latch when met
        - None: keep current counters/state (do not reset between detection frames)
        - "CLEAR": explicit clear (used when a single face is confirmed)
        """
        if flag == "NO_FACE":
            self.no_cnt += 1
            self.multi_cnt = 0
            if self.no_cnt >= self.need_no:
                self.cur = "NO_FACE"
        elif flag == "MULTIPLE_FACES":
            self.multi_cnt += 1
            self.no_cnt = 0
            if self.multi_cnt >= self.need_multi:
                self.cur = "MULTIPLE_FACES"
        elif flag == "CLEAR":
            self.no_cnt = 0
            self.multi_cnt = 0
            self.cur = None
        else:
            # None -> no update; preserve counters and current latched state
            return self.cur
        return self.cur


# -----------------------------------------------------------------------------
# LiveAnalyzer: background threads for audio+video snapshots (no UI overlay)
# -----------------------------------------------------------------------------
class LiveAnalyzer:
    """Collects live audio + periodic emotion to produce rolling snapshots."""
    def __init__(self, settings: Settings):
        self.s = settings
        self._run = False
        self._video_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        self._last_emotion: Optional[EmotionEntry] = None
        self._audio_ring: Deque[np.ndarray] = collections.deque(maxlen=20)  # ~ LIVE_WINDOW_SECONDS of chunks
        self._audio_lock = threading.Lock()
        self._last_snapshot: Optional[LiveSnapshot] = None
        self._started_at: Optional[float] = None

    # ---- lifecycle ----
    def start(self):
        if self._run:
            return
        self._run = True
        self._started_at = time.time()
        self._video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._video_thread.start()
        self._audio_thread.start()

    def stop(self):
        self._run = False

    def status(self) -> LiveSnapshot | None:
        return self._last_snapshot

    # ---- loops ----
    def _video_loop(self):
        """Lightweight periodic emotion sampling to feed snapshots (no overlays)."""
        cap = cv2.VideoCapture(self.s.CAMERA_INDEX)
        if not cap.isOpened():
            while self._run:
                self._last_emotion = EmotionEntry(time=time.time(), flag="NO_FACE")
                time.sleep(self.s.LIVE_EMOTION_INTERVAL)
            return

        next_t = 0.0
        while self._run:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            tnow = time.time()
            if tnow >= next_t:
                try:
                    # Use DeepFace OpenCV backend; first detect faces only
                    from deepface import DeepFace
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
                            faces.append({"x": int(fa.get("x", 0)), "y": int(fa.get("y", 0)),
                                          "w": int(fa.get("w", 0)), "h": int(fa.get("h", 0))})
                    except Exception:
                        # Fallback to analyze for detection if extract_faces unavailable
                        res = DeepFace.analyze(
                            frame,
                            actions=["emotion"],
                            enforce_detection=False,
                            detector_backend="opencv",
                        )
                        res = res if isinstance(res, list) else [res]
                        for r in res:
                            reg = (r or {}).get("region") or {}
                            faces.append({"x": int(reg.get("x", 0)), "y": int(reg.get("y", 0)),
                                          "w": int(reg.get("w", 0)), "h": int(reg.get("h", 0))})

                    if len(faces) == 0:
                        self._last_emotion = EmotionEntry(time=tnow, flag="NO_FACE")
                    elif len(faces) > 1:
                        self._last_emotion = EmotionEntry(time=tnow, flag="MULTIPLE_FACES")
                    else:
                        # Single face confirmed -> compute emotion on the crop
                        x, y, w, h = faces[0]["x"], faces[0]["y"], faces[0]["w"], faces[0]["h"]
                        chip = frame[y:y+h, x:x+w]
                        try:
                            emo_res = DeepFace.analyze(
                                chip if chip.size else frame,
                                actions=["emotion"],
                                enforce_detection=False,
                                detector_backend="opencv",
                                align=True,
                            )
                            emo_res = emo_res if isinstance(emo_res, list) else [emo_res]
                            r0 = emo_res[0] if emo_res else {}
                            emo = r0.get("dominant_emotion")
                            if emo is None and isinstance(r0.get("emotion"), dict):
                                probs = r0["emotion"]
                                emo = max(probs, key=probs.get)
                            self._last_emotion = EmotionEntry(time=tnow, emotion=emo)
                        except Exception:
                            self._last_emotion = EmotionEntry(time=tnow, flag="NO_FACE")
                except Exception:
                    self._last_emotion = EmotionEntry(time=tnow, flag="NO_FACE")

                next_t = tnow + self.s.LIVE_EMOTION_INTERVAL

            time.sleep(0.01)
        cap.release()

    def _audio_loop(self):
        """Continuously capture audio, periodically run ASR -> fluency -> confidence."""
        from core.asr import transcribe_audio  # lazy import to avoid loading torch at module import time

        sr = self.s.AUDIO_SAMPLE_RATE
        chunk_sec = 0.5
        chunk_frames = int(sr * chunk_sec)
        next_asr_t = time.time() + self.s.LIVE_ASR_INTERVAL

        def callback(indata, frames, time_info, status):
            with self._audio_lock:
                self._audio_ring.append(indata.copy().astype(np.float32).reshape(-1))

        with sd.InputStream(callback=callback, channels=1, samplerate=sr, blocksize=chunk_frames):
            while self._run:
                now = time.time()
                if now >= next_asr_t:
                    with self._audio_lock:
                        if len(self._audio_ring) == 0:
                            next_asr_t = now + self.s.LIVE_ASR_INTERVAL
                            time.sleep(0.05)
                            continue
                        audio = np.concatenate(list(self._audio_ring), axis=0)

                    # keep only the last LIVE_WINDOW_SECONDS
                    max_len = int(sr * self.s.LIVE_WINDOW_SECONDS)
                    if audio.shape[0] > max_len:
                        audio = audio[-max_len:]

                    # Whisper prefers a file path; write temp wav
                    tmp_path = f"temp/live_{int(now)}.wav"
                    sf.write(tmp_path, audio, sr)
                    try:
                        text, lang = transcribe_audio(tmp_path, self.s)
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

                    # Quick silence proxy from RMS
                    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
                    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=512)
                    thr = float(np.percentile(rms, 25) * 0.8)
                    silence_spans: List[SilenceSpan] = []
                    start_idx = None
                    for i, v in enumerate(rms < thr):
                        if v and start_idx is None:
                            start_idx = i
                        elif (not v) and (start_idx is not None):
                            s = float(times[start_idx]); e = float(times[i])
                            if (e - s) >= 0.35:
                                silence_spans.append(SilenceSpan(start=round(s, 2), end=round(e, 2)))
                            start_idx = None

                    # Compute fluency
                    fm = compute_fluency(
                        transcript=text,
                        silence_spans=silence_spans,
                        record_seconds=float(audio.shape[0]) / sr,
                    )

                    # Confidence using the latest emotion (live snapshot)
                    emotions = [self._last_emotion] if self._last_emotion is not None else []
                    conf = infer_confidence(fm, emotions)

                    self._last_snapshot = LiveSnapshot(
                        ts=now,
                        emotion=(self._last_emotion.emotion if self._last_emotion else None),
                        emotion_flag=(self._last_emotion.flag if self._last_emotion else None),
                        fluency=LiveFluency(
                            transcript_window=text.strip(),
                            words_per_minute=fm.words_per_minute,
                            filler_word_count=fm.filler_word_count,
                            rate_variation=fm.rate_variation,
                            fluency_score=fm.fluency_score,
                        ),
                        confidence_score=conf.confidence_score,
                        verdict=conf.verdict,
                    )
                    next_asr_t = now + self.s.LIVE_ASR_INTERVAL

                time.sleep(0.02)


# -----------------------------------------------------------------------------
# Live camera overlay (DeepFace + OpenCV path, with smoothing & hysteresis)
# -----------------------------------------------------------------------------
def _mode_nonempty(q: deque[str]) -> Optional[str]:
    items = [x for x in q if x]
    return Counter(items).most_common(1)[0][0] if items else None


def run_live_overlay(settings: Settings, camera_index: Optional[int] = None) -> None:
    """
    Open webcam, detect faces/emotions with DeepFace (OpenCV backend), draw rectangles + flags.

    Flags:
      - NO_FACE if no face
      - MULTIPLE_FACES if >1 face
      - otherwise rectangle + emotion label
    Press 'q' to quit.
    """
    cam_idx = settings.CAMERA_INDEX if camera_index is None else camera_index
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_idx}")

    from deepface import DeepFace  # TF pinned by your requirements

    # --- heuristics to prevent bad boxes (edit if needed) ---
    MIN_BOX_FRAC = 0.02     # box area must be >= 2% of frame
    MAX_BOX_FRAC = 0.60     # box area must be <= 60% of frame (avoid "whole window")
    MIN_AR, MAX_AR = 0.6, 1.8  # aspect ratio (w/h) must be reasonable

    # Secondary fast counter for faces (helps MULTIPLE_FACES/NO_FACE)
    HAAR_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Smoothing buffers
    flag_hist: deque[str] = deque(maxlen=7)
    emo_hist: deque[str] = deque(maxlen=7)

    # Trackers & state
    tracker = None
    tracked_reg = None
    frame_idx = 0
    last_emotion = ""
    last_flag = None

    emo_ema = ProbEMA(alpha=EMO_SMOOTH_ALPHA)
    last_emo_infer_t = 0.0
    flag_debouncer = FlagHysteresis(
        needed_no=NO_FACE_HYSTERESIS,
        needed_multi=MULTI_FACE_HYSTERESIS
    )

    # -------- helpers --------
    def _resize_for_detect(img, target_w=DETECT_WIDTH):
        H, W = img.shape[:2]
        if W <= target_w:
            return img, 1.0
        scale = target_w / float(W)
        new_size = (target_w, int(H * scale))
        small = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return small, scale

    def _to_full_coords(reg_small: dict, scale: float) -> dict:
        return {
            "x": int(reg_small.get("x", 0) / scale),
            "y": int(reg_small.get("y", 0) / scale),
            "w": int(reg_small.get("w", 0) / scale),
            "h": int(reg_small.get("h", 0) / scale),
        }

    def _create_tracker():
        t = TRACKER_TYPE.upper()
        if hasattr(cv2, "legacy"):
            if t == "MOSSE": return cv2.legacy.TrackerMOSSE_create()
            if t == "CSRT":  return cv2.legacy.TrackerCSRT_create()
            return cv2.legacy.TrackerKCF_create()
        return cv2.TrackerKCF_create()

    def _adaptive_min_box(frame_w: int, frame_h: int) -> int:
        base = min(frame_w, frame_h)
        return max(24, int(base * MIN_BOX_FRACTION))

    def _sanitize_and_filter(reg_full: dict, W: int, H: int) -> Optional[dict]:
        """Clamp to frame, drop absurd sizes/aspect ratios."""
        x, y, w, h = reg_full["x"], reg_full["y"], reg_full["w"], reg_full["h"]
        # reject non-positive sizes
        if w <= 0 or h <= 0: return None
        # clamp
        x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
        area = w * h
        area_frac = area / float(W * H)
        if not (MIN_BOX_FRAC <= area_frac <= MAX_BOX_FRAC):
            return None
        ar = w / float(h)
        if not (MIN_AR <= ar <= MAX_AR):
            return None
        return {"x": x, "y": y, "w": w, "h": h}

    def _center_score(rr: dict, W: int, H: int) -> float:
        reg = rr.get("region") or {}
        cx = reg.get("x", 0) + reg.get("w", 0) / 2.0
        cy = reg.get("y", 0) + reg.get("h", 0) / 2.0
        nx = abs(cx - W / 2.0) / (W / 2.0 + 1e-6)
        ny = abs(cy - H / 2.0) / (H / 2.0 + 1e-6)
        dist = (nx * nx + ny * ny) ** 0.5
        area = reg.get("w", 0) * reg.get("h", 0)
        return (1.0 - CENTER_BIAS) * area - CENTER_BIAS * dist * 1e6

    def _iou(a: dict, b: dict) -> float:
        ax0, ay0, aw, ah = a["x"], a["y"], a["w"], a["h"]
        bx0, by0, bw, bh = b["x"], b["y"], b["w"], b["h"]
        ax1, ay1 = ax0 + aw, ay0 + ah
        bx1, by1 = bx0 + bw, by0 + bh
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = aw * ah
        area_b = bw * bh
        return inter / float(area_a + area_b - inter + 1e-6)

    def _dedup_candidates(cands: list[dict], iou_thresh: float = 0.45) -> list[dict]:
        """Non-maximum suppression-ish: keep largest / most central, drop near-duplicates."""
        if not cands:
            return []
        # sort by area descending
        cands = sorted(cands, key=lambda rr: rr["region"]["w"] * rr["region"]["h"], reverse=True)
        kept: list[dict] = []
        for c in cands:
            rc = c["region"]
            dupe = False
            for k in kept:
                if _iou(rc, k["region"]) >= iou_thresh:
                    dupe = True
                    break
            if not dupe:
                kept.append(c)
        return kept

    def _decide_face_count(num_deepface: int, num_haar: int,
                        min_count_for_multi: int = 2,
                        vote_policy: str = "any") -> str | None:
        """
        Decide flag based on two sources.
        - vote_policy="any": if either source >= min_count_for_multi => MULTIPLE_FACES
        - vote_policy="both": require both sources to agree
        Returns "MULTIPLE_FACES", "NO_FACE", or None.
        """
        if num_deepface == 0 and num_haar == 0:
            return "NO_FACE"
        if vote_policy == "any":
            if num_deepface >= min_count_for_multi or num_haar >= min_count_for_multi:
                return "MULTIPLE_FACES"
        else:  # "both"
            if num_deepface >= min_count_for_multi and num_haar >= min_count_for_multi:
                return "MULTIPLE_FACES"
        return None

    # -------- main loop --------
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        do_detect = (frame_idx % DEFAULT_DETECT_EVERY_N == 0) or (tracker is None)

        try:
            if do_detect:
                # 1) Detect on downscaled frame with DeepFace(opencv)
                small, scale = _resize_for_detect(frame)
                H, W = frame.shape[:2]
                MIN_BOX_ADAPT = _adaptive_min_box(W, H)

                result = None
                try:
                    # Stage 1: detection only (no emotion yet)
                    dets = []
                    try:
                        dets = DeepFace.extract_faces(
                            img_path=small,
                            detector_backend="opencv",
                            enforce_detection=False,
                            align=True,
                        )
                    except Exception:
                        # Fallback: analyze to get regions if extract not available
                        tmp = DeepFace.analyze(
                            small,
                            actions=["emotion"],
                            enforce_detection=False,
                            detector_backend="opencv",
                        )
                        tmp = tmp if isinstance(tmp, list) else ([tmp] if isinstance(tmp, dict) else [])
                        dets = []
                        for r in tmp:
                            fa = (r or {}).get("region") or {}
                            dets.append({"facial_area": {"x": fa.get("x", 0), "y": fa.get("y", 0), "w": fa.get("w", 0), "h": fa.get("h", 0)}})

                    # Map & sanitize DeepFace detections to candidates
                    candidates: list[dict] = []
                    for d in dets or []:
                        reg_small = (d or {}).get("facial_area") or {}
                        if not reg_small:
                            continue
                        reg_full = _to_full_coords(reg_small, scale)
                        reg_full = _sanitize_and_filter(reg_full, W, H)
                        if not reg_full:
                            continue
                        if reg_full["w"] < MIN_BOX_ADAPT or reg_full["h"] < MIN_BOX_ADAPT:
                            continue
                        candidates.append({"region": reg_full})
                except Exception:
                    candidates = []

                # 2) Secondary fast face count (Haar) to stabilize flags
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                haar_faces = HAAR_FACE.detectMultiScale(gray_small, 1.35, 6, minSize=(28, 28))
                haar_count = len(haar_faces)

                # Combine evidence for flags
                if len(candidates) == 0 and haar_count == 0:
                    last_flag = flag_debouncer.step("NO_FACE")
                    tracker = None
                    tracked_reg = None
                    faces_out = []

                elif len(candidates) > 1 or haar_count > 1:
                    last_flag = flag_debouncer.step("MULTIPLE_FACES")
                    # pick primary with center-biased score
                    primary = max(candidates, key=lambda rr: _center_score(rr, W, H)) if candidates else None
                    if primary:
                        reg_full = primary["region"]
                        tracker = _create_tracker()
                        tracker.init(frame, (reg_full["x"], reg_full["y"], reg_full["w"], reg_full["h"]))
                        tracked_reg = reg_full
                    else:
                        tracker = None
                        tracked_reg = None
                    faces_out = []

                else:
                    # Single-face detected -> clear flags explicitly
                    last_flag = flag_debouncer.step("CLEAR")
                    # if DeepFace gave none but Haar saw one, synthesize a region from Haar
                    if not candidates and haar_count == 1:
                        (x, y, w, h) = haar_faces[0]
                        # map from small->full
                        reg_full = _sanitize_and_filter(
                            {"x": int(x/scale), "y": int(y/scale), "w": int(w/scale), "h": int(h/scale)},
                            W, H
                        )
                        if not reg_full:
                            tracker = None
                            tracked_reg = None
                            faces_out = []
                        else:
                            tracker = _create_tracker()
                            tracker.init(frame, (reg_full["x"], reg_full["y"], reg_full["w"], reg_full["h"]))
                            tracked_reg = reg_full
                            last_emotion = last_emotion or "neutral"
                            faces_out = [{"region": reg_full, "emotion": last_emotion}]
                    else:
                        primary = candidates[0] if candidates else None
                        if not primary:
                            tracker = None
                            tracked_reg = None
                            faces_out = []
                        else:
                            reg_full = primary["region"]
                            tracker = _create_tracker()
                            tracker.init(frame, (reg_full["x"], reg_full["y"], reg_full["w"], reg_full["h"]))
                            tracked_reg = reg_full

                            # 3) Refine emotion on the crop only when single-face confirmed
                            need_refresh = (time.time() - last_emo_infer_t) > EMO_STALE_SEC
                            if need_refresh:
                                face_chip = frame[
                                    reg_full["y"]: reg_full["y"] + reg_full["h"],
                                    reg_full["x"]: reg_full["x"] + reg_full["w"]
                                ]
                                try:
                                    emo_res = DeepFace.analyze(
                                        face_chip if face_chip.size else frame,
                                        actions=["emotion"],
                                        enforce_detection=False,
                                        detector_backend="opencv",
                                        align=True,
                                    )
                                    emo_res = emo_res if isinstance(emo_res, list) else [emo_res]
                                    r0 = emo_res[0] if emo_res else {}
                                    probs = r0.get("emotion") if isinstance(r0.get("emotion"), dict) else None
                                    if probs:
                                        emo_ema.update(probs)
                                        label, p = emo_ema.top()
                                        if label and p >= EMO_MIN_PROB:
                                            last_emotion = label
                                            last_emo_infer_t = time.time()
                                except Exception:
                                    pass

                            faces_out = [{"region": reg_full, "emotion": last_emotion}]

            else:
                # TRACK between detections
                faces_out = []
                ok_tr, bbox = (False, None)
                if tracker is not None:
                    ok_tr, bbox = tracker.update(frame)
                if ok_tr:
                    x, y, w, h = map(int, bbox)
                    # sanitize tracked box as well
                    H, W = frame.shape[:2]
                    reg_full = _sanitize_and_filter({"x": x, "y": y, "w": w, "h": h}, W, H)
                    if reg_full:
                        tracked_reg = reg_full
                        faces_out = [{"region": tracked_reg, "emotion": last_emotion}]
                        # Tracking continues; do not update flag counters
                        last_flag = flag_debouncer.step(None)
                    else:
                        tracker = None
                        tracked_reg = None
                        last_flag = flag_debouncer.step("NO_FACE")
                        faces_out = []
                else:
                    tracker = None
                    tracked_reg = None
                    last_flag = flag_debouncer.step("NO_FACE")
                    faces_out = []

            # Smooth text labels for UI
            flag_hist.append(last_flag or "")
            emo_hist.append(last_emotion or "")
            flag_cache = _mode_nonempty(flag_hist)
            emo_cache = _mode_nonempty(emo_hist) or last_emotion
            if faces_out and emo_cache:
                faces_out[0]["emotion"] = emo_cache

            # Draw overlays
            annotated = draw_overlays(frame, faces_out, flag_cache)
            cv2.imshow("Interview Live (q to quit)", annotated)

        except Exception:
            tracker = None
            tracked_reg = None
            last_flag = flag_debouncer.step("NO_FACE")
            annotated = draw_overlays(frame, [], last_flag)
            cv2.imshow("Interview Live (q to quit)", annotated)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

