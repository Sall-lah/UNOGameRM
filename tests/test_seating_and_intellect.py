"""Tests for seating configuration, fixed intellect, and starting player selection."""

from src.config import AppConfig
from src.env.uno_engine import ModifiedUNOServerPlatform


def test_seating_and_fixed_intellect():
    """Verify seat 0 is agent, seat 1 is opponent with configured fixed intellect."""
    cfg = AppConfig(opponent={"intellect": 3})
    platform = ModifiedUNOServerPlatform(config=cfg)
    platform.start_new_round(round_id=1, starting_player=0, seed=777)

    assert len(platform.game.players) == 2
    assert platform.game.players[0].name == "AGENT"
    assert platform.game.players[1].name == "OPPONENT"
    assert platform.game.opp.intellect == 3


def test_starting_player_advances_opponent_turns():
    """Verify starting_player=1 runs opponent turns until control reaches agent at seat 0."""
    platform = ModifiedUNOServerPlatform()
    raw_state = platform.start_new_round(round_id=1, starting_player=1, seed=555)

    assert raw_state["current_player_index"] == 0
    assert len(raw_state["agent_hand"]) >= 7  # May have drawn or stayed 7
    # Opening opponent events must be recorded
    assert len(platform.game.recorded_events) > 0
