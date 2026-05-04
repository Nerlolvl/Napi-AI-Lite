from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("napi.teacher")

from core.engine import InferenceEngine
from core.prompt_builder import TEACHER_SYSTEM_PROMPT, REVISION_SYSTEM_PROMPT


@dataclass
class Evaluation:
    score: float
    needs_revision: bool
    critique: str
    revision_instructions: str
    memory_note: str


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(text[start : end + 1])


async def evaluate_answer(
    engine: InferenceEngine,
    *,
    user_message: str,
    answer: str,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 500,
) -> Evaluation:
    raw = await engine.chat(
        messages=[
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"User message:\n{user_message}\n\nAssistant answer:\n{answer}",
            },
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    try:
        data = _extract_json(raw)
    except Exception:
        log.warning("Teacher returned non-JSON output")
        return Evaluation(
            score=7.0,
            needs_revision=False,
            critique="Teacher returned non-JSON output.",
            revision_instructions="",
            memory_note="",
        )

    return Evaluation(
        score=float(data.get("score", 7)),
        needs_revision=bool(data.get("needs_revision", False)),
        critique=str(data.get("critique", "")),
        revision_instructions=str(data.get("revision_instructions", "")),
        memory_note=str(data.get("memory_note", "")),
    )


async def revise_answer(
    engine: InferenceEngine,
    *,
    user_message: str,
    original_answer: str,
    evaluation: Evaluation,
    model: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 1400,
) -> str:
    return await engine.chat(
        messages=[
            {"role": "system", "content": REVISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User message:\n{user_message}\n\n"
                    f"Original answer:\n{original_answer}\n\n"
                    f"Teacher critique:\n{evaluation.critique}\n\n"
                    f"Revision instructions:\n{evaluation.revision_instructions}"
                ),
            },
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )