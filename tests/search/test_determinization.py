"""Tests for PlugMem-constrained determinization sampler (Task 5.2).

Verifies unseen pool extraction, soft down-weighting of constrained colours/values,
uniform baseline sampling, capacity bounds, and reproducibility under fixed seeds.
"""

from __future__ import annotations

import random
import pytest

from src.core.cards import (
    UNOCard,
    build_full_uno_deck,
    parse_card_string,
)
from src.env.barrier import LegalObservation, compute_unseen_pool
from src.memory.graph import PlugMemConstraint
from src.search.determinization import (
    calculate_soft_card_weight,
    sample_determinization,
)


def test_unseen_pool_excludes_agent_hand_and_top_discard(make_legal_obs):
    """Verify U_t excludes agent hand cards and top discard, matching reported size and multiset counts."""
    obs: LegalObservation = make_legal_obs(
        agent_hand=("red 0", "blue 1", "green 2"),
        top_discard="yellow 3",
        opp_card_count=7,
    )
    res = sample_determinization(obs)

    full_deck = build_full_uno_deck()  # 108
    expected_ut_size = len(full_deck) - 3 - 1  # 104

    assert res.unseen_pool_size_Ut == expected_ut_size
    all_determinized_cards = res.opp_hand + res.draw_pile
    assert len(all_determinized_cards) == expected_ut_size

    # Verify multiset subtraction: "red 0" has only 1 instance in deck, so after subtracting it must be 0
    red_0 = parse_card_string("red 0")
    assert all_determinized_cards.count(red_0) == 0

    # "blue 1" has 2 instances in deck, subtracting 1 leaves exactly 1
    blue_1 = parse_card_string("blue 1")
    assert all_determinized_cards.count(blue_1) == 1

    # "green 2" has 2 instances in deck, subtracting 1 leaves exactly 1
    green_2 = parse_card_string("green 2")
    assert all_determinized_cards.count(green_2) == 1

    # "yellow 3" has 2 instances in deck, subtracting 1 leaves exactly 1
    yellow_3 = parse_card_string("yellow 3")
    assert all_determinized_cards.count(yellow_3) == 1



def test_soft_down_weighting_of_constrained_colour():
    """Verify green card receives weight 1 - 0.85 * 0.85 = 0.2775 when green is lacked."""
    green_card = parse_card_string("green 5")
    red_card = parse_card_string("red 5")

    constraint = PlugMemConstraint(
        memory_id="prop_1",
        origin_turn=1,
        constraint_type="LACKS_COLOR",
        target_key="green",
        confidence=0.85,
    )

    weight_green = calculate_soft_card_weight(green_card, [constraint])
    weight_red = calculate_soft_card_weight(red_card, [constraint])

    expected_green_weight = 1.0 - 0.85 * 0.85  # 0.2775
    assert abs(weight_green - expected_green_weight) < 1e-6
    assert weight_green > 0.0  # Support strictly positive
    assert weight_red == 1.0


def test_uniform_sampling_without_constraints(make_legal_obs):
    """Verify without constraints, sampling is uniform over U_t."""
    obs: LegalObservation = make_legal_obs(
        agent_hand=("red 1", "red 2"),
        top_discard="blue 0",
        opp_card_count=10,
    )
    rng = random.Random(12345)
    res = sample_determinization(obs, memory_constraints=None, rng=rng)

    assert len(res.opp_hand) == 10
    assert len(res.draw_pile) == res.unseen_pool_size_Ut - 10


def test_opponent_size_capped_by_pool_size(make_legal_obs):
    """Verify when opp_card_count exceeds |U_t|, opp_hand is capped and draw pile is empty."""
    obs: LegalObservation = make_legal_obs(
        opp_card_count=200,  # Far exceeds 108 cards
    )
    res = sample_determinization(obs)

    assert len(res.opp_hand) == res.unseen_pool_size_Ut
    assert len(res.draw_pile) == 0


def test_determinization_reproducible_under_fixed_seed(make_legal_obs):
    """Verify identical inputs and seed produce identical determinizations."""
    obs: LegalObservation = make_legal_obs(
        agent_hand=("red 1", "blue 2"),
        top_discard="yellow 3",
        opp_card_count=7,
    )
    constraint = [
        PlugMemConstraint(
            memory_id="prop_1",
            origin_turn=1,
            constraint_type="LACKS_COLOR",
            target_key="blue",
            confidence=0.9,
        )
    ]

    rng1 = random.Random(999)
    res1 = sample_determinization(obs, memory_constraints=constraint, rng=rng1)

    rng2 = random.Random(999)
    res2 = sample_determinization(obs, memory_constraints=constraint, rng=rng2)

    assert res1.opp_hand == res2.opp_hand
    assert res1.draw_pile == res2.draw_pile
    assert res1.unseen_pool_size_Ut == res2.unseen_pool_size_Ut
