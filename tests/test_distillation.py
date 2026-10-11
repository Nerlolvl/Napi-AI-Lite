"""Tests for the distillation dataset exporter."""

from __future__ import annotations

import json

import pytest

from export_training_data import (
    build_example,
    export_dataset,
    load_pairs,
)
from storage.db_manager import NapiBrain


@pytest.fixture()
def brain(tmp_path):
    return NapiBrain(str(tmp_path / "test.db"))


def _add_pair(brain, question, answer, score, session="s1", user_meta=None):
    brain.add_message(session, "user", question, user_meta)
    brain.add_message(session, "assistant", answer, {"score": score})


def test_only_high_score_answers_are_exported(brain):
    _add_pair(brain, "Good question here?", "A thorough, correct answer.", 9.0, session="a")
    _add_pair(brain, "Bad question here?", "A weak answer that scored low.", 4.0, session="b")

    pairs = load_pairs(brain.all_messages(), min_score=8.0)

    assert pairs == [("Good question here?", "A thorough, correct answer.")]


def test_unscored_answers_are_skipped(brain):
    brain.add_message("s1", "user", "Question without evaluation?")
    brain.add_message("s1", "assistant", "Autonomous answer, no teacher score.")

    assert load_pairs(brain.all_messages()) == []


def test_pairing_uses_the_latest_pending_user(brain):
    brain.add_message("s1", "user", "first question never answered")
    _add_pair(brain, "second question answered", "A thorough, correct answer.", 9.0)

    pairs = load_pairs(brain.all_messages())

    assert pairs == [("second question answered", "A thorough, correct answer.")]


def test_vision_questions_are_skipped(brain):
    _add_pair(brain, "[vision] describe this image", "A thorough, correct answer.", 9.0)

    assert load_pairs(brain.all_messages()) == []


def test_short_answers_are_skipped(brain):
    _add_pair(brain, "A perfectly fine question", "ok", 9.0)

    assert load_pairs(brain.all_messages()) == []


def test_duplicate_questions_are_deduplicated(brain):
    _add_pair(brain, "Repeatable question text", "A thorough, correct answer.", 9.0, session="a")
    _add_pair(brain, "repeatable question text", "Another thorough answer.", 9.0, session="b")

    pairs = load_pairs(brain.all_messages())

    assert len(pairs) == 1
    assert pairs[0][0] == "Repeatable question text"


def test_build_example_shape():
    example = build_example("SYSTEM", "USER", "ASSISTANT")

    assert [m["role"] for m in example["messages"]] == ["system", "user", "assistant"]
    assert example["messages"][0]["content"] == "SYSTEM"


def test_export_dataset_writes_jsonl(brain, tmp_path):
    _add_pair(brain, "Exportable question text", "A thorough, correct answer.", 9.0, session="a")
    _add_pair(brain, "Low score question text", "A weak answer that scored low.", 3.0, session="b")
    output = tmp_path / "dataset.jsonl"

    count = export_dataset(brain.all_messages(), str(output), min_score=8.0)

    assert count == 1
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["messages"][1]["content"] == "Exportable question text"
    assert payload["messages"][2]["content"] == "A thorough, correct answer."
    assert "SYSTEM_PROMPT_NAPI_CORE_V1" in payload["messages"][0]["content"]
