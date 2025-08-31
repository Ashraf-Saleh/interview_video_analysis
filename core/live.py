"""
Live (real-time) analysis utilities.
Captures webcam frames & mic audio locally, produces rolling snapshots:
- Emotion every LIVE_EMOTION_INTERVAL seconds
- ASR+fluency every LIVE_ASR_INTERVAL seconds over the latest LIVE_WINDOW_SECONDS audio
- Confidence heuristic from current fluency + recent emotion
"""
from __future__ import annotations
import time, threading, collections, io
from typing import Optional, List, Deque, Tuple
import numpy as np
import cv2
import sounddevice as sd
import soundfile as sf
import librosa

from core.config import Settings
# from core.asr import transcribe_audio
from core.fluency import compute_fluency
from core.confidence import infer_confidence
from core.models import LiveSnapshot, LiveFluency, EmotionEntry, SilenceSpan

class LiveAnalyzer:
    def __init__(self, settings: Settings):
        self.s = settings
        self._run = False
        self._video_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        self._last_emotion: Optional[EmotionEntry] = None
        self._audio_ring: Deque[np.ndarray] = collections.deque(maxlen=20)  # ~ LIVE_WINDOW_SECONDS worth of audio chunks
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
        cap = cv2.VideoCapture(self.s.CAMERA_INDEX)
        if not cap.isOpened():
            # Can't open camera; mark NO_FACE snapshots periodically
            while self._run:
                self._last_emotion = EmotionEntry(time=time.time(), flag="NO_FACE")
                time.sleep(self.s.LIVE_EMOTION_INTERVAL)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        next_t = 0.0
        while self._run:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            tnow = time.time()
            if tnow >= next_t:
                # sample this frame for emotion
                try:
                    from deepface import DeepFace
                    result = DeepFace.analyze(
                        frame, actions=["emotion"], enforce_detection=False, detector_backend="opencv"
                    )
                    items = result if isinstance(result, list) else [result]
                    if not items or (len(items) == 1 and items[0].get("face_detected", 1) == 0):
                        self._last_emotion = EmotionEntry(time=tnow, flag="NO_FACE")
                    elif len(items) > 1:
                        self._last_emotion = EmotionEntry(time=tnow, flag="MULTIPLE_FACES")
                    else:
                        emo = items[0].get("dominant_emotion")
                        if emo is None and "emotion" in items[0]:
                            probs = items[0]["emotion"]
                            emo = max(probs, key=probs.get)
                        self._last_emotion = EmotionEntry(time=tnow, emotion=emo)
                except Exception:
                    self._last_emotion = EmotionEntry(time=tnow, flag="NO_FACE")
                next_t = tnow + self.s.LIVE_EMOTION_INTERVAL
            # produce a snapshot if we have audio-derived fluency shortly afterwards
            time.sleep(0.01)
        cap.release()

    def _audio_loop(self):
        from core.asr import transcribe_audio
        # collect audio continuously; do short chunks to ring buffer
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
                    # build rolling window audio
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

                    # temp WAV in memory -> write buffer for Whisper
                    buf = io.BytesIO()
                    sf.write(buf, audio, sr, format="WAV")
                    buf.seek(0)

                    # save to disk (whisper needs file path); use temp file
                    tmp_path = f"temp/live_{int(now)}.wav"
                    sf.write(tmp_path, audio, sr)
                    try:
                        text, lang = transcribe_audio(tmp_path, self.s)
                    finally:
                        try: 
                            import os; os.remove(tmp_path)
                        except Exception:
                            pass

                    # quick silence proxy for fluency: use RMS to approximate within window
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
                                silence_spans.append(SilenceSpan(start=round(s,2), end=round(e,2)))
                            start_idx = None

                    # compute fluency
                    fm = compute_fluency(
                        transcript=text,
                        silence_spans=silence_spans,
                        record_seconds=float(audio.shape[0]) / sr,
                    )

                    # confidence with the latest emotion only (live snapshot)
                    emotions = []
                    if self._last_emotion is not None:
                        emotions = [self._last_emotion]
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
        # --- Live camera overlay (draw face window, emotion, and flags) ---
    # Press 'q' to quit the window.
    from core.config import Settings  # ensure this import is present somewhere in the file

# def run_live_overlay(settings: Settings, camera_index: int | None = None):
#     """
#     Open camera, detect faces/emotions, draw windows + flags in a live window.

#     - Draws 'NO_FACE' when no face is detected
#     - Draws 'MULTIPLE_FACES' when >1 face is detected
#     - Draws a rectangle + emotion label for a single face
#     Sampling period is controlled by settings.LIVE_EMOTION_INTERVAL (seconds).
#     """
#     import time
#     import cv2

#     cam_idx = settings.CAMERA_INDEX if camera_index is None else camera_index
#     cap = cv2.VideoCapture(cam_idx)
#     if not cap.isOpened():
#         raise RuntimeError(f"Could not open camera index {cam_idx}")

#     # Lazy import to avoid pulling TF when not used elsewhere
#     try:
#         from deepface import DeepFace
#     except Exception as e:
#         cap.release()
#         raise RuntimeError(
#             "DeepFace import failed. Ensure your deepface/tensorflow/keras versions are aligned."
#         ) from e

#     faces_cache = []
#     flag_cache = None
#     last_analyze_t = 0.0

#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break

#         t = time.time()
#         if (t - last_analyze_t) >= max(0.1, settings.LIVE_EMOTION_INTERVAL):
#             try:
#                 result = DeepFace.analyze(
#                     frame,
#                     actions=["emotion"],
#                     enforce_detection=False,
#                     detector_backend="opencv",
#                 )
#                 # normalize to list
#                 if isinstance(result, dict):
#                     result = [result]
#                 if len(result) == 0:
#                     flag_cache = "NO_FACE"
#                     faces_cache = []
#                 elif len(result) > 1:
#                     flag_cache = "MULTIPLE_FACES"
#                     faces_cache = []
#                 else:
#                     flag_cache = None
#                     faces_cache = [{
#                         "region": result[0].get("region") or {},
#                         "emotion": (result[0].get("dominant_emotion") or result[0].get("emotion") or "")
#                     }]
#             except Exception:
#                 flag_cache = "NO_FACE"
#                 faces_cache = []
#             last_analyze_t = t

#         # Draw overlays
#         from core.visual import draw_overlays  # local import to avoid cycles
#         annotated = draw_overlays(frame, faces_cache, flag_cache)
#         cv2.imshow("Interview Live", annotated)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()



from core.config import Settings

def run_live_overlay(settings: Settings, camera_index: int | None = None):
    """Open webcam, draw face rectangle + emotion, and show NO_FACE / MULTIPLE_FACES flags."""
    import time, cv2
    from core.visual import draw_overlays  # local to avoid circular import

    cam_idx = settings.CAMERA_INDEX if camera_index is None else camera_index
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_idx}")

    try:
        from deepface import DeepFace  # lazy import (avoid torch)
    except Exception as e:
        cap.release()
        raise RuntimeError("DeepFace import failed; check TF/Keras/DeepFace versions.") from e

    faces_cache = []
    flag_cache = None
    last = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = time.time()
        if (t - last) >= max(0.1, settings.LIVE_EMOTION_INTERVAL):
            try:
                result = DeepFace.analyze(
                    frame, actions=["emotion"],
                    enforce_detection=False, detector_backend="opencv")
                # normalize to list across DeepFace versions
                if isinstance(result, dict):
                    result = [result]
                if len(result) == 0:
                    flag_cache = "NO_FACE"; faces_cache = []
                elif len(result) > 1:
                    flag_cache = "MULTIPLE_FACES"; faces_cache = []
                else:
                    flag_cache = None
                    r0 = result[0]
                    faces_cache = [{
                        "region": r0.get("region") or {},
                        "emotion": (r0.get("dominant_emotion") or r0.get("emotion") or "")
                    }]
            except Exception:
                flag_cache = "NO_FACE"; faces_cache = []
            last = t

        annotated = draw_overlays(frame, faces_cache, flag_cache)
        cv2.imshow("Interview Live (q to quit)", annotated)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()