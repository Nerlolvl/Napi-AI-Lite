from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config import MAX_MESSAGE_LENGTH

Language = Literal["auto", "ru", "en", "pl"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str = "default"
    language: Language = "auto"
    self_improve: bool = True


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    model: str
    improved: bool
    evaluation_score: float | None = None
    critique: str | None = None
    think: str | None = None
    language: str = "auto"


class VisionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    image_url: str | None = None
    image_data_url: str | None = None
    session_id: str = "default"
    language: Language = "auto"


class FeedbackRequest(BaseModel):
    session_id: str = "default"
    message_id: int
    rating: int = Field(ge=-1, le=1)
    comment: str = Field(default="", max_length=2000)


class FeedbackResponse(BaseModel):
    ok: bool


class KnowledgeAddRequest(BaseModel):
    source: str = Field(min_length=1, max_length=500)
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)


class KnowledgeAddResponse(BaseModel):
    ok: bool
    chunks: int


class KnowledgeIngestRequest(BaseModel):
    directory: str = "./knowledge"


class KnowledgeIngestResponse(BaseModel):
    ok: bool
    directory: str
    chunks: int


class KnowledgeSearchResponse(BaseModel):
    results: list[dict[str, str]]


class HealthResponse(BaseModel):
    status: str
    name: str
    local_model_loaded: bool
    provider_configured: bool
    can_chat: bool
    autonomous_mode: bool
    self_brain_mode: bool
    neural_brain_loaded: bool
    neural_vocab_size: int
    neural_model_path: str
    helper_models: dict[str, bool]
