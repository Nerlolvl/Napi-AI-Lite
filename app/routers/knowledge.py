"""Knowledge base management endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.schemas import (
    KnowledgeAddRequest,
    KnowledgeAddResponse,
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
    KnowledgeSearchResponse,
)
from app.state import brain

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/add", response_model=KnowledgeAddResponse)
async def add_knowledge(request: KnowledgeAddRequest) -> KnowledgeAddResponse:
    chunks = brain.upsert_document(
        source=request.source,
        title=request.title,
        content=request.content,
    )
    return KnowledgeAddResponse(ok=True, chunks=chunks)


@router.post("/ingest", response_model=KnowledgeIngestResponse)
async def ingest_knowledge(request: KnowledgeIngestRequest) -> KnowledgeIngestResponse:
    directory = Path(request.directory)
    if not directory.exists() or not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Knowledge directory not found: {request.directory}")
    chunks = brain.ingest_directory(str(directory))
    return KnowledgeIngestResponse(ok=True, directory=str(directory), chunks=chunks)


@router.get("/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(q: str, limit: int = 6) -> KnowledgeSearchResponse:
    clean_limit = max(1, min(limit, 20))
    return KnowledgeSearchResponse(results=brain.search_knowledge(q, limit=clean_limit))
