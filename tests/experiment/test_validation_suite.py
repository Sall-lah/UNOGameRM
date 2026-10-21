"""Tests for pre-pilot validation suite checks (Task 8.4)."""

from __future__ import annotations

from unittest.mock import patch
import pytest

from src.env.barrier import LegalObservation
from src.experiment.validate import (
    ValidationFailure,
    run_all_validation_checks,
    validate_barrier_integrity,
    validate_mcts_forward_model_turn_retention,
)


def test_validation_suite_all_pass():
    """Verify that correctly wired system passes all Phase-1 verification checks."""
    assert run_all_validation_checks() is True


def test_barrier_leak_fails_fast():
    """Verify that detecting opponent hand leakage across the barrier raises ValidationFailure."""
    def leaking_extract_observation(raw_state):
        class LeakingObservation(LegalObservation):
            opp_hand: list[str] = ["green 9"]

        return LeakingObservation(
            round_id=1,
            turn_idx=1,
            agent_hand=("red 1",),
            top_discard="red 5",
            active_colour="red",
            opp_card_count=7,
            legal_actions=("red 1",),
            public_events_since_last_turn=(),
            known_discards_since_reshuffle=(),
        )

    with patch("src.env.barrier.ClientServerBarrier.extract_observation", side_effect=leaking_extract_observation):
        with pytest.raises(ValidationFailure) as exc_info:
            validate_barrier_integrity()
        assert "exposed private opponent hand" in str(exc_info.value)


def test_forward_model_turn_retention_regression_fails_fast():
    """Verify that if forward model passes turn to opponent on retention card, check fails."""
    from src.search.forward_model import SimulatedState

    original_step = SimulatedState.step

    def faulty_step(self, action):
        original_step(self, action)
        # Regress turn retention by giving turn to opponent
        self.current_player = "opponent"

    with patch.object(SimulatedState, "step", faulty_step):
        with pytest.raises(ValidationFailure) as exc_info:
            validate_mcts_forward_model_turn_retention()
        assert "passed turn to opponent" in str(exc_info.value)
