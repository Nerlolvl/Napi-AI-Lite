"""Chat, vision, and feedback endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from core.gatekeeper import gate_check
from core.prompt_builder import (
    REASONING_SYSTEM_PROMPT,
    VISION_SYSTEM_PROMPT,
    build_prompt,
    extract_reflected_rules,
    strip_think_tags,
)
from core.self_brain import self_brain_answer
from soft_learning.rule_extractor import process_evaluation
from soft_learning.teacher_api import evaluate_answer, revise_answer

from app.config import (
    CHAT_MODEL,
    ENABLE_REASONING,
    ENABLE_SOFT_LEARNING,
    FILTER_MODEL,
    MAX_CHUNKS,
    MAX_NOTES,
    MAX_RULES,
    MAX_VISIBLE_TOKENS,
    MIN_SCORE_REVISION,
    REASONING_MAX_CONTEXT,
    REASONING_MAX_TOKENS,
    REASONING_MODEL,
    REINFORCE_BONUS,
    TEACHER_MODEL,
    TEMPS,
    VISION_MODEL,
)
from app.schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    VisionRequest,
)
from app.services import (
    build_learning_context,
    helper_model_ready,
    messages_to_text,
    model_filter,
    rules_from_filter,
)
from app.state import brain, count_request, engine, neural_brain

log = logging.getLogger("napi")

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
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

    if not engine.can_chat:
        return await _autonomous_answer(request, gate.cleaned_text, notes, knowledge_chunks, reflected_rules, language)

    if helper_model_ready(FILTER_MODEL):
        try:
            filter_result = await model_filter(gate.cleaned_text)
            if filter_result and filter_result.get("memory_note"):
                brain.add_note(request.session_id, filter_result["memory_note"])
                for entry in rules_from_filter(filter_result):
                    brain.add_reflected_rule(entry["rule"], source="filter", category=entry["category"])
        except Exception:
            log.exception("Filter step failed")

    if ENABLE_REASONING and helper_model_ready(REASONING_MODEL):
        await _prepend_reasoning_brief(messages, gate.cleaned_text)

    raw_answer = await engine.chat(
        messages=messages,
        model=CHAT_MODEL,
        temperature=TEMPS.get("chat", 0.55),
        max_tokens=MAX_VISIBLE_TOKENS + 300,
        prefer_local=engine.local.available and not engine.remote_ready,
    )
    answer, think_content = strip_think_tags(raw_answer)

    learning_context = build_learning_context(knowledge_chunks, reflected_rules)

    brain.add_message(request.session_id, "user", gate.cleaned_text)

    answer, think_content, improved, score, critique = await _maybe_self_improve(
        request, gate.cleaned_text, answer, think_content, learning_context, reflected_rules
    )

    brain.add_message(
        request.session_id,
        "assistant",
        answer,
        metadata={
            "model": CHAT_MODEL,
            "language": language,
            "score": score,
            "improved": improved,
        },
    )

    count_request()

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


@router.post("/vision", response_model=ChatResponse)
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
            max_tokens=MAX_VISIBLE_TOKENS + 300,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer, think_content = strip_think_tags(raw)

    brain.add_message(request.session_id, "user", f"[vision] {request.question}", {"has_image": True})
    brain.add_message(request.session_id, "assistant", answer, {"model": VISION_MODEL})

    return ChatResponse(
        answer=answer,
        session_id=request.session_id,
        model=VISION_MODEL,
        improved=False,
        think=think_content or None,
        language=language,
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    brain.add_feedback(request.session_id, request.message_id, request.rating, request.comment)
    return FeedbackResponse(ok=True)


async def _autonomous_answer(
    request: ChatRequest,
    cleaned_text: str,
    notes: list[dict],
    knowledge_chunks: list[dict],
    reflected_rules: list[str],
    language: str,
) -> ChatResponse:
    neural_context = neural_brain.think(
        cleaned_text,
        knowledge_chunks=knowledge_chunks,
        notes=notes,
    ).as_dict()
    result = self_brain_answer(
        cleaned_text,
        notes=notes,
        knowledge_chunks=knowledge_chunks,
        reflected_rules=reflected_rules,
        neural_context=neural_context,
        language=language,
    )

    if result.learned_note:
        brain.add_note(request.session_id, result.learned_note)
    brain.add_message(request.session_id, "user", cleaned_text)
    brain.add_message(
        request.session_id,
        "assistant",
        result.answer,
        metadata={
            "model": "napi-neural-brain" if neural_brain.available else "napi-self-brain",
            "language": language,
            "neural": neural_context,
        },
    )

    count_request()

    return ChatResponse(
        answer=result.answer,
        session_id=request.session_id,
        model="napi-neural-brain" if neural_brain.available else "napi-self-brain",
        improved=False,
        think=result.think,
        language=language,
    )


async def _prepend_reasoning_brief(messages: list[dict], cleaned_text: str) -> None:
    try:
        brief = await engine.chat(
            messages=[
                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"System and knowledge context:\n"
                    f"{messages_to_text(messages)[:REASONING_MAX_CONTEXT]}\n\n"
                    f"User message:\n{cleaned_text}"
                )},
            ],
            model=REASONING_MODEL,
            temperature=TEMPS.get("reasoning", 0.25),
            max_tokens=REASONING_MAX_TOKENS,
        )
        messages[0]["content"] += f"\n\nВнутренний анализ (не показывай пользователю):\n{brief}"
    except Exception as exc:
        log.warning("Reasoning failed: %s", exc)


async def _maybe_self_improve(
    request: ChatRequest,
    cleaned_text: str,
    answer: str,
    think_content: str | None,
    learning_context: str | None,
    applied_rules: list[str],
) -> tuple[str, str | None, bool, float | None, str | None]:
    if not (ENABLE_SOFT_LEARNING and request.self_improve and helper_model_ready(TEACHER_MODEL)):
        return answer, think_content, False, None, None

    try:
        evaluation = await evaluate_answer(
            engine,
            user_message=cleaned_text,
            answer=answer,
            model=TEACHER_MODEL,
            evaluation_context=learning_context,
        )
        if evaluation.memory_note:
            brain.add_note(request.session_id, evaluation.memory_note)
        for entry in process_evaluation(evaluation):
            brain.add_reflected_rule(entry["rule"], source=entry["source"], category=entry["category"])

        if evaluation.needs_revision and evaluation.score < MIN_SCORE_REVISION:
            revised = await revise_answer(
                engine,
                user_message=cleaned_text,
                original_answer=answer,
                evaluation=evaluation,
                model=CHAT_MODEL,
                revision_context=learning_context,
            )
            answer, revised_think = strip_think_tags(revised)
            if revised_think:
                think_content = revised_think
            return answer, think_content, True, evaluation.score, evaluation.critique

        if evaluation.score >= MIN_SCORE_REVISION:
            reinforced = brain.reinforce_rules(applied_rules, REINFORCE_BONUS)
            if reinforced:
                log.debug("Reinforced %d rules after good answer (score=%.1f)", reinforced, evaluation.score)

        return answer, think_content, False, evaluation.score, evaluation.critique
    except Exception as exc:
        log.warning("Soft learning failed: %s", exc)
        return answer, think_content, False, None, None
