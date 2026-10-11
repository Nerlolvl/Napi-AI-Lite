"""Tests for learning quality: rule validation, reinforcement, learning context."""

from __future__ import annotations

import pytest

from soft_learning.rule_extractor import (
    MAX_RULE_LENGTH,
    MIN_RULE_LENGTH,
    is_usable_rule,
    process_evaluation,
)
from soft_learning.teacher_api import Evaluation
from storage.db_manager import NapiBrain
from app.services import build_learning_context


@pytest.fixture()
def brain(tmp_path):
    return NapiBrain(str(tmp_path / "test.db"))


class TestIsUsableRule:
    def test_accepts_normal_rule(self):
        assert is_usable_rule("Always cite the source table when quoting numbers") is True

    def test_rejects_too_short(self):
        assert is_usable_rule("be nice") is False
        assert is_usable_rule("") is False
        assert len("x" * MIN_RULE_LENGTH) >= MIN_RULE_LENGTH

    def test_rejects_too_long(self):
        assert is_usable_rule("word " * (MAX_RULE_LENGTH // 4)) is False

    def test_rejects_generic_phrases(self):
        assert is_usable_rule("Be better") is False
        assert is_usable_rule("improve") is False

    def test_rejects_single_word(self):
        assert is_usable_rule("conciseness") is False


class TestProcessEvaluationFilters:
    def test_invalid_memory_note_rules_are_dropped(self):
        evaluation = Evaluation(
            score=5.0,
            needs_revision=False,
            critique="Answer was short",
            revision_instructions="",
            memory_note="[REFLECTED_RULE: ok] [REFLECTED_RULE: Always verify API versions before use]",
        )
        entries = process_evaluation(evaluation)
        assert [e["rule"] for e in entries] == ["Always verify API versions before use"]

    def test_generic_note_yields_no_rules(self):
        evaluation = Evaluation(
            score=5.0,
            needs_revision=False,
            critique="Be better.",
            revision_instructions="improve",
            memory_note="",
        )
        assert process_evaluation(evaluation) == []


class TestReinforceRules:
    def test_reinforces_matching_rules_only(self, brain):
        brain.add_reflected_rule("Check language before answering")
        brain.add_reflected_rule("Cite sources when quoting")

        updated = brain.reinforce_rules(
            ["check language before answering", "unknown rule"], bonus=0.1
        )

        assert updated == 1
        stats = {row["rule"]: row["score"] for row in brain.rule_stats()}
        assert stats["Check language before answering"] == pytest.approx(1.1)
        assert stats["Cite sources when quoting"] == pytest.approx(1.0)

    def test_reinforcement_is_capped(self, brain):
        brain.add_reflected_rule("Stay grounded in knowledge")
        for _ in range(20):
            brain.reinforce_rules(["Stay grounded in knowledge"], bonus=0.5)

        assert brain.rule_stats()[0]["score"] == pytest.approx(3.0)

    def test_empty_input_is_noop(self, brain):
        assert brain.reinforce_rules([], bonus=0.1) == 0
        assert brain.reinforce_rules(["   "], bonus=0.1) == 0
        assert brain.rule_stats() == []


class TestBuildLearningContext:
    def test_empty_inputs_return_none(self):
        assert build_learning_context([], []) is None

    def test_contains_knowledge_and_rules(self):
        context = build_learning_context(
            [{"title": "RAG", "content": "Retrieval augmented generation basics."}],
            ["Always define acronyms"],
        )
        assert context is not None
        assert "Knowledge base excerpts" in context
        assert "RAG" in context
        assert "Reflected rules applied" in context
        assert "Always define acronyms" in context
