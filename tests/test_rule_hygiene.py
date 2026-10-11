"""Tests for the reflected-rule hygiene layer: dedup, score, TTL, pruning."""

from __future__ import annotations

import sqlite3
from time import time

import pytest

from storage.db_manager import NapiBrain


@pytest.fixture()
def brain(tmp_path):
    return NapiBrain(str(tmp_path / "test.db"))


@pytest.fixture()
def short_ttl_brain(tmp_path):
    # TTL ~86 seconds so tests can expire rules by backdating timestamps.
    return NapiBrain(str(tmp_path / "short.db"), rule_ttl_days=0.001)


def _backdate(brain: NapiBrain, rule: str, seconds_ago: float, *, used: bool = False) -> None:
    with brain._lock:
        conn = brain._get_conn()
        conn.execute(
            "UPDATE reflection_diary SET created_at = ?, last_used_at = ? "
            "WHERE normalized = ?",
            (time() - seconds_ago, time() - seconds_ago if used else None, rule.lower()),
        )
        conn.commit()


def _backdate_created_only(brain: NapiBrain, rule: str, seconds_ago: float) -> None:
    with brain._lock:
        conn = brain._get_conn()
        conn.execute(
            "UPDATE reflection_diary SET created_at = ? WHERE normalized = ?",
            (time() - seconds_ago, rule.lower()),
        )
        conn.commit()


def test_add_ignores_blank_rules(brain):
    assert brain.add_reflected_rule("   ") is False
    assert brain.rule_stats() == []


def test_duplicate_rules_are_deduplicated_and_reinforced(brain):
    assert brain.add_reflected_rule("Be concise.") is True
    assert brain.add_reflected_rule("be concise. ") is False

    stats = brain.rule_stats()
    assert len(stats) == 1
    assert stats[0]["rule"] == "Be concise."
    assert stats[0]["score"] == pytest.approx(1.25)
    assert stats[0]["last_used_at"] is not None


def test_repeated_reinforcement_is_capped(brain):
    for _ in range(12):
        brain.add_reflected_rule("Stay accurate")
    stats = brain.rule_stats()
    assert len(stats) == 1
    assert stats[0]["score"] == pytest.approx(3.0)


def test_usage_tracking_on_selection(brain):
    brain.add_reflected_rule("Prefer facts over guesses")

    first = brain.get_reflected_rules()
    assert first == ["Prefer facts over guesses"]

    stats = brain.rule_stats()
    assert stats[0]["hits"] == 1
    assert stats[0]["last_used_at"] is not None

    brain.get_reflected_rules()
    assert brain.rule_stats()[0]["hits"] == 2


def test_ttl_excludes_unused_rules(short_ttl_brain):
    short_ttl_brain.add_reflected_rule("old rule")
    short_ttl_brain.add_reflected_rule("fresh rule")

    _backdate(short_ttl_brain, "old rule", seconds_ago=200)

    assert short_ttl_brain.get_reflected_rules() == ["fresh rule"]


def test_used_rules_survive_longer(brain):
    brain.add_reflected_rule("used rule")
    brain.add_reflected_rule("expired rule")

    # "used rule" enters the TTL window and gets picked, which stamps last_used_at.
    _backdate(brain, "used rule", seconds_ago=25 * 86400)
    _backdate(brain, "expired rule", seconds_ago=40 * 86400)
    assert brain.get_reflected_rules() == ["used rule"]

    # Now push created_at beyond the TTL: the rule survives thanks to its
    # fresh last_used_at, while the never-used one is gone.
    _backdate_created_only(brain, "used rule", seconds_ago=40 * 86400)

    assert brain.get_reflected_rules() == ["used rule"]
    assert brain.rule_stats()[0]["hits"] == 2


def test_prune_rules_removes_expired(short_ttl_brain):
    short_ttl_brain.add_reflected_rule("stale rule")
    _backdate(short_ttl_brain, "stale rule", seconds_ago=200)

    removed = short_ttl_brain.prune_rules()

    assert removed == 1
    assert short_ttl_brain.rule_stats() == []


def test_hard_cap_keeps_best_rules(tmp_path):
    brain = NapiBrain(str(tmp_path / "cap.db"), max_rules_total=3)
    for i in range(5):
        brain.add_reflected_rule(f"rule {i}")

    kept = {row["rule"] for row in brain.rule_stats()}
    assert kept == {"rule 2", "rule 3", "rule 4"}


def test_high_score_beats_recency_under_cap(tmp_path):
    brain = NapiBrain(str(tmp_path / "score.db"), max_rules_total=2)
    brain.add_reflected_rule("first")
    brain.add_reflected_rule("second")
    for _ in range(4):
        brain.add_reflected_rule("first")

    kept = {row["rule"] for row in brain.rule_stats()}
    assert kept == {"first", "second"}


def test_migration_from_legacy_schema(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE reflection_diary ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "rule TEXT NOT NULL, "
        "source TEXT NOT NULL DEFAULT 'teacher', "
        "category TEXT NOT NULL DEFAULT 'general', "
        "created_at REAL NOT NULL)"
    )
    now = time()
    conn.execute("INSERT INTO reflection_diary (rule, created_at) VALUES ('Be short', ?)", (now,))
    conn.execute("INSERT INTO reflection_diary (rule, created_at) VALUES ('be short', ?)", (now + 1,))
    conn.commit()
    conn.close()

    brain = NapiBrain(str(db_path))

    stats = brain.rule_stats()
    assert len(stats) == 1
    assert stats[0]["rule"] == "Be short"
    assert stats[0]["hits"] == 0

    assert brain.add_reflected_rule("BE SHORT") is False
