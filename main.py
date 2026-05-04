from __future__ import annotations

import gc
import logging
from contextlib import asynccontextmanager
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.engine import InferenceEngine
from core.gatekeeper import GateResult, detect_language, gate_check
from core.prompt_builder import (
    FILTER_SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT,
    VISION_SYSTEM_PROMPT,
    build_prompt,
    extract_reflected_rules,
    strip_think_tags,
)
from soft_learning.rule_extractor import process_evaluation
from soft_learning.teacher_api import evaluate_answer, revise_answer
from storage.db_manager import NapiBrain

log = logging.getLogger("napi")

# ── Config ───────────────────────────────────────────────────────────

CONFIG_PATH = __import__("pathlib").Path(__file__).resolve().parent / "config.yaml"


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


CONFIG = _load_config()

DNA_CFG = CONFIG.get("dna", {})
STORAGE_CFG = CONFIG.get("storage", {})
SOFT_CFG = CONFIG.get("soft_learning", {})
REASONING_CFG = CONFIG.get("reasoning", {})
SERVER_CFG = CONFIG.get("server", {})

MAX_MESSAGE_LENGTH = DNA_CFG.get("max_message_length", 4000)
MAX_VISIBLE_TOKENS = DNA_CFG.get("max_visible_tokens", 512)
ENABLE_SOFT_LEARNING = SOFT_CFG.get("enabled", True)
MIN_SCORE_REVISION = SOFT_CFG.get("min_score_for_revision", 8)
ENABLE_REASONING = REASONING_CFG.get("enabled", True)
REASONING_MAX_TOKENS = REASONING_CFG.get("max_tokens", 450)
REASONING_MAX_CONTEXT = REASONING_CFG.get("max_context_chars", 12000)

MAX_HISTORY = STORAGE_CFG.get("max_history_messages", 24)
MAX_NOTES = STORAGE_CFG.get("max_memory_notes", 12)
MAX_CHUNKS = STORAGE_CFG.get("max_knowledge_chunks", 6)
MAX_RULES = STORAGE_CFG.get("max_reflected_rules", 8)
NOTE_MAX_LENGTH = STORAGE_CFG.get("note_max_length", 1000)

PROVIDER_CFG = CONFIG.get("engine", {}).get("provider", {})
CHAT_MODEL = PROVIDER_CFG.get("chat_model", "")
TEACHER_MODEL = PROVIDER_CFG.get("teacher_model", "")
FILTER_MODEL = PROVIDER_CFG.get("filter_model", "")
REASONING_MODEL = PROVIDER_CFG.get("reasoning_model", "")
VISION_MODEL = PROVIDER_CFG.get("vision_model", "")

TEMPS = SOFT_CFG.get("temperatures", {})

DB_PATH = STORAGE_CFG.get("database_path", "./storage/napi_brain.db")

# ── Global objects ─────────────────────────────────────────────────────

brain = NapiBrain(DB_PATH, max_note_length=NOTE_MAX_LENGTH)
engine = InferenceEngine()


# ── Schemas ────────────────────────────────────────────────────────────

Language = __import__("typing").Literal["auto", "ru", "en", "pl"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
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


class HealthResponse(BaseModel):
    status: str
    name: str
    local_model_loaded: bool


# ── App ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.init_local()
    log.info("Napi started. Local model loaded: %s", engine.local.available)
    yield


app = FastAPI(
    title="Napi AI",
    description="Stateless local AI with DNA prompt, THINK tags, reflection diary, and soft learning.",
    version="0.3.0",
    lifespan=lifespan,
)


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        name=SERVER_CFG.get("app_name", "Napi"),
        local_model_loaded=engine.local.available,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # ── Шаг 1: Gatekeeper (префильтрация без нейросети) ──
    gate = gate_check(request.message)
    if not gate.allowed:
        return ChatResponse(
            answer=f"Я не могу помочь с этим запросом. Причина: {gate.reason}",
            session_id=request.session_id,
            model="gatekeeper",
            improved=False,
            language=gate.language,
        )

    language = request.language if request.language != "auto" else gate.language

    # ── Шаг 2: Сборка контекста (Prompt Builder) ──
    notes = brain.recent_notes(request.session_id, MAX_NOTES)
    knowledge_chunks = brain.search_knowledge(gate.cleaned_text, limit=MAX_CHUNKS)
    reflected_rules = extract_reflected_rules(notes)
    reflected_rules += brain.get_reflected_rules(limit=MAX_RULES - len(reflected_rules))

    messages = build_prompt(
        user_message=gate.cleaned_text,
        language=language,
        notes=notes,
        knowledge_chunks=knowledge_chunks,
        reflected_rules=reflected_rules,
    )

    # ── Шаг 2a: Фильтрация через модель (опционально) ──
    if PROVIDER_CFG.get("enabled", True):
        try:
            filter_result = await _model_filter(gate.cleaned_text)
            if filter_result and filter_result.get("memory_note"):
                brain.add_note(request.session_id, filter_result["memory_note"])
                rule_entries = process_evaluation_from_filter(filter_result)
                for entry in rule_entries:
                    brain.add_reflected_rule(entry["rule"], source="filter", category=entry["category"])
        except Exception:
            pass

    # ── Шаг 3: Размышление + Генерация ──
    if ENABLE_REASONING:
        try:
            brief = await engine.chat(
                messages=[
                    {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"System and knowledge context:\n"
                        f"{_messages_to_text(messages)[:REASONING_MAX_CONTEXT]}\n\n"
                        f"User message:\n{gate.cleaned_text}"
                    )},
                ],
                model=REASONING_MODEL,
                temperature=TEMPS.get("reasoning", 0.25),
                max_tokens=REASONING_MAX_TOKENS,
            )
            messages[0]["content"] += f"\n\nВнутренний анализ (не показывай пользователю):\n{brief}"
        except Exception as exc:
            log.warning("Reasoning failed: %s", exc)

    # ── Основная генерация ──
    raw_answer = await engine.chat(
        messages=messages,
        model=CHAT_MODEL,
        temperature=TEMPS.get("chat", 0.55),
        max_tokens=DNA_CFG.get("max_visible_tokens", 512) + 300,
    )
    answer, think_content = strip_think_tags(raw_answer)

    # ── Шаг 4: Stateless drop (сохранение в БД) ──
    brain.add_message(request.session_id, "user", gate.cleaned_text)
    brain.add_message(
        request.session_id, "assistant", answer,
        metadata={"model": CHAT_MODEL, "language": language},
    )

    # ── Шаг 5: Фоновое мягкое обучение ──
    improved = False
    score = None
    critique = None

    if ENABLE_SOFT_LEARNING and request.self_improve:
        try:
            evaluation = await evaluate_answer(
                engine,
                user_message=gate.cleaned_text,
                answer=answer,
                model=TEACHER_MODEL,
            )
            score = evaluation.score
            critique = evaluation.critique

            if evaluation.memory_note:
                brain.add_note(request.session_id, evaluation.memory_note)

            rule_entries = process_evaluation(evaluation)
            for entry in rule_entries:
                brain.add_reflected_rule(entry["rule"], source=entry["source"], category=entry["category"])

            if evaluation.needs_revision and evaluation.score < MIN_SCORE_REVISION:
                revised = await revise_answer(
                    engine,
                    user_message=gate.cleaned_text,
                    original_answer=answer,
                    evaluation=evaluation,
                    model=CHAT_MODEL,
                )
                answer, revised_think = strip_think_tags(revised)
                if revised_think:
                    think_content = revised_think
                improved = True
        except Exception as exc:
            log.warning("Soft learning failed: %s", exc)

    # ── Шаг 4 continued: Очистка памяти ──
    gc.collect()

    return ChatResponse(
        answer=answer,
        session_id=request.session_id,
        model=CHAT_MODEL,
        improved=improved,
        evaluation_score=score,
        critique=critique,
        think=think_content or None,
        language=language,
    )


@app.post("/vision", response_model=ChatResponse)
async def vision(request: VisionRequest) -> ChatResponse:
    gate = gate_check(request.question, is_vision=True)
    if not gate.allowed:
        return ChatResponse(
            answer=f"Я не могу обработать этот запрос. Причина: {gate.reason}",
            session_id=request.session_id,
            model="gatekeeper",
            improved=False,
            language=gate.language,
        )

    if not request.image_url and not request.image_data_url:
        raise HTTPException(status_code=400, detail="Provide image_url or image_data_url")

    image_source = request.image_url or request.image_data_url
    language = request.language if request.language != "auto" else gate.language
    lang_hint = "" if language == "auto" else f"\nReply language code: {language}."

    content = [
        {"type": "text", "text": request.question},
        {"type": "image_url", "image_url": {"url": image_source}},
    ]

    try:
        raw = await engine.chat(
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {"role": "user", "content": content + [{"type": "text", "text": lang_hint}]},
            ],
            model=VISION_MODEL,
            temperature=TEMPS.get("vision", 0.35),
            max_tokens=DNA_CFG.get("max_visible_tokens", 512) + 300,
        )
        answer, think_content = strip_think_tags(raw)

        brain.add_message(request.session_id, "user", f"[vision] {request.question}", {"has_image": True})
        brain.add_message(request.session_id, "assistant", answer, {"model": VISION_MODEL})

        gc.collect()

        return ChatResponse(
            answer=answer,
            session_id=request.session_id,
            model=VISION_MODEL,
            improved=False,
            think=think_content or None,
            language=language,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    brain.add_feedback(request.session_id, request.message_id, request.rating, request.comment)
    return FeedbackResponse(ok=True)


# ── Helpers ────────────────────────────────────────────────────────────

async def _model_filter(text: str) -> dict | None:
    """Use the filter model to classify the request."""
    import json
    from core.prompts import FILTER_SYSTEM_PROMPT

    try:
        raw = await engine.chat(
            messages=[
                {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            model=FILTER_MODEL,
            temperature=TEMPS.get("filter", 0.0),
            max_tokens=500,
        )
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
    except Exception as exc:
        log.warning("Model filter failed: %s", exc)
    return None


def process_evaluation_from_filter(filter_data: dict) -> list[dict[str, str]]:
    """Extract reflected rules from filter output."""
    from soft_learning.rule_extractor import categorize_rule

    entries = []
    note = filter_data.get("memory_note", "")
    if note:
        import re
        rules = re.findall(r"\[REFLECTED_RULE:\s*([^\]]+)\]", note)
        for rule in rules:
            entries.append({"rule": rule, "category": categorize_rule(rule), "source": "filter"})
        if not rules and note.strip():
            entries.append({"rule": note.strip(), "category": categorize_rule(note), "source": "filter"})
    return entries


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        parts.append(f"[{role}]: {content}")
    return "\n".join(parts)


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "main:app",
        host=SERVER_CFG.get("host", "0.0.0.0"),
        port=SERVER_CFG.get("port", 8000),
        workers=SERVER_CFG.get("workers", 1),
    )