"""Shared process-wide singletons: database, inference engine, neural brain."""

from __future__ import annotations

import gc

from core.engine import InferenceEngine
from core.neural_brain import NeuralBrain
from storage.db_manager import NapiBrain

from app.config import DB_PATH, GC_INTERVAL, MAX_RULES_TOTAL, NOTE_MAX_LENGTH, RULE_TTL_DAYS

brain = NapiBrain(
    DB_PATH,
    max_note_length=NOTE_MAX_LENGTH,
    rule_ttl_days=RULE_TTL_DAYS,
    max_rules_total=MAX_RULES_TOTAL,
)
engine = InferenceEngine()
neural_brain = NeuralBrain()

_request_count = 0


def count_request() -> None:
    """Count a served request and trigger GC only every GC_INTERVAL requests."""
    global _request_count
    _request_count += 1
    if _request_count % GC_INTERVAL == 0:
        gc.collect()
