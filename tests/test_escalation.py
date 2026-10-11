"""Tests for backend routing and the escalation decision."""

from __future__ import annotations

from app.services import should_escalate
from core.engine import choose_backend


class TestChooseBackend:
    def test_prefers_local_when_asked(self):
        assert choose_backend(
            local_available=True, remote_ready=True, prefer_local=True, force_remote=False
        ) == "local"

    def test_force_remote_beats_local_preference(self):
        assert choose_backend(
            local_available=True, remote_ready=True, prefer_local=True, force_remote=True
        ) == "provider"

    def test_local_only_uses_local(self):
        assert choose_backend(
            local_available=True, remote_ready=False, prefer_local=False, force_remote=False
        ) == "local"

    def test_remote_only_uses_provider(self):
        assert choose_backend(
            local_available=False, remote_ready=True, prefer_local=True, force_remote=False
        ) == "provider"

    def test_remote_wins_over_idle_local(self):
        assert choose_backend(
            local_available=True, remote_ready=True, prefer_local=False, force_remote=False
        ) == "provider"

    def test_no_model_returns_none(self):
        assert choose_backend(
            local_available=False, remote_ready=False, prefer_local=False, force_remote=False
        ) is None


class TestShouldEscalate:
    @staticmethod
    def _call(**overrides) -> bool:
        base = dict(
            source="local",
            score=5.0,
            escalation_enabled=True,
            remote_ready=True,
            min_score=7.0,
        )
        base.update(overrides)
        return should_escalate(**base)

    def test_escalates_low_scoring_local_answer(self):
        assert self._call() is True

    def test_does_not_escalate_good_local_answer(self):
        assert self._call(score=8.0) is False

    def test_does_not_escalate_provider_answer(self):
        assert self._call(source="provider", score=2.0) is False

    def test_disabled_never_escalates(self):
        assert self._call(escalation_enabled=False) is False

    def test_no_remote_never_escalates(self):
        assert self._call(remote_ready=False) is False

    def test_unscored_answer_never_escalates(self):
        assert self._call(score=None) is False

    def test_boundary_score_is_not_below(self):
        assert self._call(score=7.0) is False
