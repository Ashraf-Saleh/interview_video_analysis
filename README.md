# Interview Video Analysis – FastAPI 
Offline-capable **interview video analysis** for:
- 🎤 Speech fluency (WPM, filler words, rate variation) + silence detection
- 🎭 Facial emotion recognition over time from video frames
- 🧭 Confidence inference from multimodal cues

**Folders**
```
api/           # FastAPI app (not app/)
core/          # Core modules: ASR, fluency, emotion, silence, confidence, pipeline
data/          # Place datasets / samples (ignored by git)
output/        # JSON results or exported files
temp/          # runtime scratch (uploads, extracted audio)
scripts/       # CLI and helpers
tests/         # pytest suite
```

## Quick Start
1) Ensure **FFmpeg** is installed and on PATH (Linux: `sudo apt-get install ffmpeg`, macOS: `brew install ffmpeg`, Windows: `choco install ffmpeg`).
2) Create env & install deps:
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
3) Run API:
```bash
uvicorn api.main:app --reload --port 8000
```
4) Endpoints:
- `GET  /health`
- `POST /analyze/video`  (form-data: `file=@video.mp4`, optional `emotion_interval`)
- `POST /analyze/audio`  (form-data: `file=@audio.wav`, optional `min_silence_dur`, `silence_threshold`)

5) CLI:
```bash
python scripts/cli.py --video path/to/video.mp4 --out output/analysis.json
```

## Env Vars (optional)
```
DEVICE=cpu|cuda
WHISPER_MODEL=base           # tiny|base|small|medium|large
AUDIO_SAMPLE_RATE=16000
EMOTION_INTERVAL=5           # seconds
MIN_SILENCE_DUR=0.35         # seconds
SILENCE_THRESHOLD=auto       # numeric to override, else auto
```

## Notes
- Silence detection here uses **energy-based** method offline (librosa RMS). Replaceable with VAD (WebRTC/pyannote).
- DeepFace backend defaults to `opencv` with `enforce_detection=False` for robustness.
- Whisper loads lazily and respects `DEVICE` for CPU/GPU.
