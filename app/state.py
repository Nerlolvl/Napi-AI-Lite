"""Shared process-wide singletons: database, inference engine, neural brain."""

from __future__ import annotations

import gc

from core.engine import InferenceEngine
from core.neural_brain import NeuralBrain
from storage.db_manager import NapiBrain

from app.config import DB_PATH, GC_INTERVAL, NOTE_MAX_LENGTH

brain = NapiBrain(DB_PATH, max_note_length=NOTE_MAX_LENGTH)
engine = InferenceEngine()
neural_brain = NeuralBrain()

_request_count = 0


def count_request() -> None:
    """Count a served request and trigger GC only every GC_INTERVAL requests."""
    global _request_count
    _request_count += 1
    if _request_count % GC_INTERVAL == 0:
        gc.collect()
