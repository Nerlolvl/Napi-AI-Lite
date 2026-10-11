"""Napi AI Lite — application entry point.

Assembles the FastAPI app from the app/ package and runs uvicorn.
Endpoint logic lives in app/routers/, settings in app/config.py.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import SERVER_CFG
from app.routers import chat, knowledge, system
from app.state import brain, engine, neural_brain

log = logging.getLogger("napi")


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.init_local()
    neural_brain.load()
    log.info("Napi started. Local model loaded: %s", engine.local.available)
    log.info("Neural brain loaded: %s (vocab=%s)", neural_brain.available, neural_brain.vocab_size)
    yield
    await engine.shutdown()
    brain.close()
    log.info("Napi shut down cleanly.")


app = FastAPI(
    title="Napi AI",
    description="Stateless local AI with DNA prompt, THINK tags, reflection diary, and soft learning.",
    version="0.4.0",
    lifespan=lifespan,
)

app.include_router(system.router)
app.include_router(chat.router)
app.include_router(knowledge.router)


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc: RuntimeError) -> JSONResponse:
    detail = str(exc)
    if "Provider base_url not configured" in detail or "No model specified" in detail:
        return JSONResponse(
            status_code=503,
            content={
                "detail": detail,
                "hint": (
                    "Napi server is running, but no model provider is configured. "
                    "Set engine.provider.base_url and model names in config.yaml, then restart the server."
                ),
            },
        )
    if "no primary model configured" in detail:
        return JSONResponse(
            status_code=503,
            content={
                "detail": detail,
                "hint": (
                    "Soft-learning teacher is optional, but Napi still needs one primary model: "
                    "either a local GGUF file in engine.local or a chat provider in engine.provider."
                ),
            },
        )
    return JSONResponse(status_code=500, content={"detail": detail})


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "main:app",
        host=SERVER_CFG.get("host", "0.0.0.0"),
        port=SERVER_CFG.get("port", 8000),
        workers=SERVER_CFG.get("workers", 1),
    )
