
# Place this function at the bottom of core/live.py
def run_live_overlay(settings: Settings, camera_index: int | None = None):
    """Open camera, detect faces/emotions, draw windows + flags in a live window.

    Press 'q' to quit the window.
    This function is CPU-friendly by sampling frames per LIVE_EMOTION_INTERVAL.
    """
    import cv2, time
    cam_idx = settings.CAMERA_INDEX if camera_index is None else camera_index
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_idx}")

    try:
        from deepface import DeepFace
    except Exception as e:
        cap.release()
        raise RuntimeError("DeepFace import failed. Install/align deepface/tensorflow.") from e

    last_analyze_t = 0.0
    faces_cache = []
    flag_cache = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = time.time()
        if (t - last_analyze_t) >= max(0.1, settings.LIVE_EMOTION_INTERVAL):
            try:
                result = DeepFace.analyze(frame, actions=['emotion'],
                                          enforce_detection=False,
                                          detector_backend='opencv')
                if isinstance(result, dict):
                    result = [result]
                if len(result) == 0:
                    flag_cache = 'NO_FACE'
                    faces_cache = []
                elif len(result) > 1:
                    flag_cache = 'MULTIPLE_FACES'
                    faces_cache = []
                else:
                    flag_cache = None
                    faces_cache = [{
                        'region': result[0].get('region') or {},
                        'emotion': (result[0].get('dominant_emotion') or result[0].get('emotion') or '')
                    }]
            except Exception:
                flag_cache = 'NO_FACE'
                faces_cache = []
            last_analyze_t = t

        from core.visual import draw_overlays
        annotated = draw_overlays(frame, faces_cache, flag_cache)
        cv2.imshow('Interview Live', annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
