"""
Pydantic data models for API IO.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Deque

class SilenceSpan(BaseModel):
    start: float
    end: float

class FluencyMetrics(BaseModel):
    words_per_minute: float
    filler_word_count: int
    rate_variation: float
    fluency_score: float
    accent_signal: str

class EmotionEntry(BaseModel):
    time: float
    emotion: Optional[str] = None
    flag: Optional[Literal["NO_FACE", "MULTIPLE_FACES"]] = None

class ConfidenceResult(BaseModel):
    confidence_inferred: bool
    confidence_score: int
    verdict: Literal["Confident", "Not Confident"]

class AnalysisResponse(BaseModel):
    transcript: str
    emotion_timeline: List[EmotionEntry] = Field(default_factory=list)
    silence_timeline: List[SilenceSpan] = Field(default_factory=list)
    fluency_analysis: FluencyMetrics
    confidence_analysis: ConfidenceResult



# live model 


class LiveFluency(BaseModel):
    transcript_window: str
    words_per_minute: float
    filler_word_count: int
    rate_variation: float
    fluency_score: float

class LiveSnapshot(BaseModel):
    ts: float
    emotion: Optional[str] = None
    emotion_flag: Optional[Literal["NO_FACE", "MULTIPLE_FACES"]] = None
    fluency: Optional[LiveFluency] = None
    confidence_score: Optional[int] = None
    verdict: Optional[Literal["Confident", "Not Confident"]] = None

class LiveStatus(BaseModel):
    running: bool
    started_at: float | None = None
    last_snapshot: LiveSnapshot | None = None
