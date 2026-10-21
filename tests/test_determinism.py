"""Tests verifying determinism, RNG isolation, and non-blocking headless execution."""

import random
from src.env.uno_engine import ModifiedUNOServerPlatform


def test_seed_reproducibility():
    """Verify identical seeds and actions produce identical event streams."""
    seed = 42
    round_id = 1
    starting_player = 0

    platform1 = ModifiedUNOServerPlatform()
    state1 = platform1.start_new_round(round_id, starting_player, seed)

    platform2 = ModifiedUNOServerPlatform()
    state2 = platform2.start_new_round(round_id, starting_player, seed)

    assert state1["agent_hand"] == state2["agent_hand"]
    assert state1["top_card"] == state2["top_card"]
    assert state1["legal_actions"] == state2["legal_actions"]

    # Play first legal action
    action = state1["legal_actions"][0]
    res1 = platform1.dispatch_agent_action_and_observe_horizon(action)
    res2 = platform2.dispatch_agent_action_and_observe_horizon(action)

    assert len(res1["horizon_events"]) == len(res2["horizon_events"])
    for e1, e2 in zip(res1["horizon_events"], res2["horizon_events"]):
        assert e1.event_type == e2.event_type
        assert e1.actor == e2.actor
        assert e1.active == e2.active
        assert e1.card == e2.card
        assert e1.cards_drawn_count == e2.cards_drawn_count


def test_engine_rng_isolated_from_agent_rng():
    """Verify agent RNG consumption does not alter engine reproducibility."""
    seed = 123
    agent_rng = random.Random(999)

    platform = ModifiedUNOServerPlatform()
    state = platform.start_new_round(1, 0, seed)

    # Consume agent RNG
    _ = [agent_rng.random() for _ in range(50)]

    action = state["legal_actions"][0]
    res = platform.dispatch_agent_action_and_observe_horizon(action)

    # Re-run from clean state without agent RNG calls
    platform_clean = ModifiedUNOServerPlatform()
    state_clean = platform_clean.start_new_round(1, 0, seed)
    res_clean = platform_clean.dispatch_agent_action_and_observe_horizon(action)

    assert [e.event_type for e in res["horizon_events"]] == [
        e.event_type for e in res_clean["horizon_events"]
    ]
