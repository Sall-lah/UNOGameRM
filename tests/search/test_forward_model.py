"""Tests for the 2-player non-stacking forward model (Task 5.1).

Verifies turn retention for Skip/Reverse/+2/+4, non-stacking forced draws,
number card turn transitions, voluntary draws, auto-UNO, and terminal states.
"""

from __future__ import annotations

import pytest

from src.core.cards import (
    COLOURS,
    DRAW_TWO,
    REVERSE,
    SKIP,
    WILD,
    WILD_DRAW_FOUR,
    UNOAction,
    parse_card_string,
)
from src.search.forward_model import (
    SimulatedState,
    create_simulated_state_from_observation,
)
from src.search.mars_mcts import action_from_id


def test_agent_plays_skip_turn_retention(make_simulated_state):
    """Verify acting player playing Skip retains turn control in 2P UNO."""
    state: SimulatedState = make_simulated_state(
        agent_hand=["red skip", "red 5"],
        opp_hand=["green 2", "yellow 7"],
        draw_pile=["blue 1"],
        top_discard="red 3",
        active_colour="red",
        current_player="agent",
    )
    act = action_from_id("red skip")
    state.step(act)

    # Opponent took 0 turns; agent retains control
    assert state.current_player == "agent"
    assert len(state.agent_hand) == 1
    assert state.top_discard.value == SKIP
    assert state.active_colour == "red"
    assert not state.is_terminal()


def test_reverse_behaves_as_skip_in_two_player(make_simulated_state):
    """Verify Reverse acts identically to Skip (turn retention) in 2P UNO."""
    state: SimulatedState = make_simulated_state(
        agent_hand=["red reverse", "blue 5"],
        opp_hand=["green 2", "yellow 7"],
        draw_pile=["blue 1"],
        top_discard="red 3",
        active_colour="red",
        current_player="agent",
    )
    act = action_from_id("red reverse")
    state.step(act)

    assert state.current_player == "agent"
    assert len(state.agent_hand) == 1
    assert state.top_discard.value == REVERSE
    assert not state.is_terminal()


def test_draw_two_forces_penalty_and_retains_turn(make_simulated_state):
    """Verify +2 forces opponent to draw exactly 2 cards with no counter-stacking, retaining turn."""
    state: SimulatedState = make_simulated_state(
        agent_hand=["red + 2", "blue 5"],
        opp_hand=["green 2"],
        draw_pile=["yellow 1", "yellow 2", "yellow 3"],
        top_discard="red 3",
        active_colour="red",
        current_player="agent",
    )
    act = action_from_id("red + 2")
    state.step(act)

    # Opponent drew 2 cards (from 1 to 3); agent retains turn
    assert len(state.opp_hand) == 3
    assert len(state.draw_pile) == 1
    assert state.current_player == "agent"
    assert not state.agent_suffered_a_pen


def test_wild_draw_four_forces_four_cards_and_retains_turn(make_simulated_state):
    """Verify Wild +4 forces opponent to draw 4 cards, sets declared colour, and retains turn."""
    state: SimulatedState = make_simulated_state(
        agent_hand=["WILD + 4 card, set to green", "blue 5"],
        opp_hand=["green 2"],
        draw_pile=["yellow 1", "yellow 2", "yellow 3", "yellow 4", "yellow 5"],
        top_discard="red 3",
        active_colour="red",
        current_player="agent",
    )
    act = action_from_id("WILD + 4 card, set to green")
    state.step(act)

    assert len(state.opp_hand) == 5
    assert len(state.draw_pile) == 1
    assert state.active_colour == "green"
    assert state.current_player == "agent"
    assert not state.agent_suffered_a_pen


def test_opponent_penalty_triggers_agent_suffered_apen(make_simulated_state):
    """Verify when opponent plays +2 or +4, agent suffers A_pen penalty."""
    state: SimulatedState = make_simulated_state(
        agent_hand=["blue 5"],
        opp_hand=["red + 2", "green 1"],
        draw_pile=["yellow 1", "yellow 2"],
        top_discard="red 3",
        active_colour="red",
        current_player="opponent",
    )
    act = action_from_id("red + 2")
    state.step(act)

    assert len(state.agent_hand) == 3
    assert state.current_player == "opponent"  # Opponent retains turn
    assert state.agent_suffered_a_pen is True


def test_number_card_passes_turn(make_simulated_state):
    """Verify normal number card play transitions control to the opponent."""
    state: SimulatedState = make_simulated_state(
        agent_hand=["red 5", "blue 5"],
        opp_hand=["green 2"],
        draw_pile=["yellow 1"],
        top_discard="red 3",
        active_colour="red",
        current_player="agent",
    )
    act = action_from_id("red 5")
    state.step(act)

    assert state.current_player == "opponent"
    assert state.top_discard.value == "5"
    assert state.active_colour == "red"
    assert len(state.agent_hand) == 1


def test_voluntary_draw_passes_turn(make_simulated_state):
    """Verify voluntary draw draws 1 card and passes turn control to the opponent."""
    state: SimulatedState = make_simulated_state(
        agent_hand=["blue 5"],
        opp_hand=["green 2"],
        draw_pile=["yellow 1", "yellow 2"],
        top_discard="red 3",
        active_colour="red",
        current_player="agent",
    )
    act = action_from_id("draw")
    state.step(act)

    assert len(state.agent_hand) == 2
    assert len(state.draw_pile) == 1
    assert state.current_player == "opponent"


def test_terminal_on_empty_hand(make_simulated_state):
    """Verify round terminates immediately when a player empties their hand and sets winner."""
    state: SimulatedState = make_simulated_state(
        agent_hand=["red 5"],
        opp_hand=["green 2"],
        draw_pile=["yellow 1"],
        top_discard="red 3",
        active_colour="red",
        current_player="agent",
    )
    act = action_from_id("red 5")
    state.step(act)

    assert len(state.agent_hand) == 0
    assert state.is_terminal() is True
    assert state.winner == "agent"


def test_auto_uno_on_two_to_one_transition(make_simulated_state):
    """Verify auto-UNO flag is set when acting hand drops from 2 cards to 1 card."""
    state: SimulatedState = make_simulated_state(
        agent_hand=["red 5", "blue 3"],
        opp_hand=["green 2"],
        draw_pile=["yellow 1"],
        top_discard="red 3",
        active_colour="red",
        current_player="agent",
    )
    assert state.agent_uno_flag is False

    act = action_from_id("red 5")
    state.step(act)

    assert len(state.agent_hand) == 1
    assert state.agent_uno_flag is True
