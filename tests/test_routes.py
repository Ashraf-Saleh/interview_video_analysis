
from fastapi.testclient import TestClient
from api.main import app
import api.routes as routes


def test_analyze_audio(sample_audio_path):
    def fake_audio_pipeline(path, settings):
        return {
            "transcript": "hello world",
            "emotion_timeline": [],
            "silence_timeline": [{"start": 0.0, "end": 0.5}],
            "fluency_analysis": {
                "words_per_minute": 120.0,
                "filler_word_count": 1,
                "rate_variation": 0.5,
                "fluency_score": 85.0,
                "accent_signal": "likely fluent"
            },
            "confidence_analysis": {
                "confidence_inferred": True,
                "confidence_score": 3,
                "verdict": "Confident"
            }
        }
    routes.analyze_audio_pipeline = fake_audio_pipeline  # type: ignore

    client = TestClient(app)
    with sample_audio_path.open("rb") as f:
        r = client.post("/analyze/audio", files={"file": (sample_audio_path.name, f, "audio/wav")})
    assert r.status_code == 200
    j = r.json()
    assert "transcript" in j
    assert "confidence_analysis" in j


def test_analyze_video(sample_video_path):
    def fake_video_pipeline(path, settings):
        return {
            "transcript": "video",
            "emotion_timeline": [{"time": 0.0, "emotion": "happy"}],
            "silence_timeline": [],
            "fluency_analysis": {
                "words_per_minute": 100.0,
                "filler_word_count": 0,
                "rate_variation": 0.4,
                "fluency_score": 92.0,
                "accent_signal": "likely fluent"
            },
            "confidence_analysis": {
                "confidence_inferred": True,
                "confidence_score": 4,
                "verdict": "Confident"
            }
        }
    routes.analyze_video_pipeline = fake_video_pipeline  # type: ignore

    client = TestClient(app)
    with sample_video_path.open("rb") as f:
        r = client.post("/analyze/video", files={"file": (sample_video_path.name, f, "video/mp4")})
    assert r.status_code == 200
    j = r.json()
    assert j["transcript"]
    assert "confidence_analysis" in j


def test_live_start():
    client = TestClient(app)
    r = client.post('/live/start')
    # may be "started" or "already_running" if test suite calls twice
    assert r.status_code == 200
    assert r.json()['status'] in ('started', 'already_running')

def test_live_status():
    client = TestClient(app)
    # ensure started
    client.post('/live/start')
    r = client.get('/live/status')
    assert r.status_code == 200
    body = r.json()
    assert 'running' in body
    assert isinstance(body['running'], bool)

def test_live_stop():
    client = TestClient(app)
    # ensure started first
    client.post('/live/start')
    r = client.post('/live/stop')
    assert r.status_code == 200
    assert r.json()['status'] in ('stopped', 'not_running')
