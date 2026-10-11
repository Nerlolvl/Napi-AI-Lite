"""Health endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.config import (
    FILTER_MODEL,
    PROVIDER_CFG,
    REASONING_MODEL,
    SERVER_CFG,
    TEACHER_MODEL,
)
from app.schemas import HealthResponse
from app.state import engine, neural_brain

log = logging.getLogger("napi")

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        name=SERVER_CFG.get("app_name", "Napi"),
        local_model_loaded=engine.local.available,
        provider_configured=engine.remote_ready,
        can_chat=True,
        autonomous_mode=not engine.can_chat and not neural_brain.available,
        self_brain_mode=not engine.can_chat,
        neural_brain_loaded=neural_brain.available,
        neural_vocab_size=neural_brain.vocab_size,
        neural_model_path=str(neural_brain.model_path),
        helper_models={
            "teacher": bool(TEACHER_MODEL and PROVIDER_CFG.get("base_url")),
            "filter": bool(FILTER_MODEL and PROVIDER_CFG.get("base_url")),
            "reasoning": bool(REASONING_MODEL and PROVIDER_CFG.get("base_url")),
        },
    )
