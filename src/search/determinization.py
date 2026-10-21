"""PlugMem-constrained determinization sampler over the unseen card pool.

This module samples hidden opponent hands and draw piles from the public observation's
unseen card pool U_t, applying soft down-weighting for opponent lack constraints
derived from PlugMem intra-round knowledge.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Union

from src.core.cards import (
    COLOURS,
    UNOCard,
    _VAL_EXPRESSIONS,
)
from src.env.barrier import LegalObservation, compute_unseen_pool
from src.memory.graph import PlugMemConstraint


@dataclass(frozen=True, slots=True)
class DeterminizationResult:
    """Carries sampled hidden state and unseen pool diagnostics for MCTS rollouts.

    Why: MCTS needs both the concrete sampled state (opp_hand, draw_pile) and the
    grounding metadata (unseen_pool_size_Ut) for two-stage audit logging.
    """

    opp_hand: list[UNOCard]
    draw_pile: list[UNOCard]
    unseen_pool_size_Ut: int


def calculate_soft_card_weight(
    card: UNOCard,
    constraints: Sequence[PlugMemConstraint],
) -> float:
    """Calculates the soft sampling weight omega(c) for a card under PlugMem constraints.

    Why: When PlugMem infers the opponent lacks a colour or value with confidence w_tau,
    omega(c) is reduced to 1 - 0.85 * w_tau. The weight is clamped strictly positive (> 0)
    so that support never collapses, ensuring all hidden states remain probabilistically possible.
    """
    weight = 1.0

    for constraint in constraints:
        conf = float(constraint.confidence)
        if constraint.constraint_type == "LACKS_COLOR":
            if card.colour == constraint.target_key:
                penalty = 1.0 - 0.85 * conf
                weight = min(weight, penalty)
        elif constraint.constraint_type == "LACKS_VALUE":
            card_val_str = _VAL_EXPRESSIONS.get(card.value, card.value)
            if card.value == constraint.target_key or card_val_str == constraint.target_key:
                penalty = 1.0 - 0.85 * conf
                weight = min(weight, penalty)

    # Strictly positive clamp to prevent zero-probability collapse
    return max(0.01, weight)


def sample_determinization(
    obs: LegalObservation,
    memory_constraints: Optional[Union[Iterable[PlugMemConstraint], Any]] = None,
    rng: Optional[Any] = None,
) -> DeterminizationResult:
    """Samples a determinization (Ĥ^opp, D̂^draw) from U_t without replacement.

    Why: In imperfect-information games, open-loop MCTS samples concrete state worlds
    from the public belief distribution to run forward rollouts without leaking server truth.
    """
    if rng is None:
        rng = random.Random()

    # Step 1: Obtain unseen pool U_t from public observation
    unseen_pool = compute_unseen_pool(obs)
    unseen_pool_size_Ut = len(unseen_pool)

    # Parse and normalize memory constraints
    constraints_list: list[PlugMemConstraint] = []
    if memory_constraints is not None:
        if hasattr(memory_constraints, "items") and not isinstance(memory_constraints, dict):
            # If a RetrievalResult was provided, retrieve its constraints if available
            pass
        elif isinstance(memory_constraints, (list, tuple)):
            for c in memory_constraints:
                if isinstance(c, PlugMemConstraint):
                    constraints_list.append(c)

    # Calculate sampling weight for each card in unseen pool
    weights = [calculate_soft_card_weight(card, constraints_list) for card in unseen_pool]

    k = min(obs.opp_card_count, unseen_pool_size_Ut)

    # Degenerate cases: empty pool or full depletion
    if k == 0:
        draw_pile = list(unseen_pool)
        _shuffle_list(rng, draw_pile)
        return DeterminizationResult(
            opp_hand=[],
            draw_pile=draw_pile,
            unseen_pool_size_Ut=unseen_pool_size_Ut,
        )

    if k >= unseen_pool_size_Ut:
        return DeterminizationResult(
            opp_hand=list(unseen_pool),
            draw_pile=[],
            unseen_pool_size_Ut=unseen_pool_size_Ut,
        )

    # Weighted sampling without replacement
    remaining_pool = list(unseen_pool)
    remaining_weights = list(weights)
    opp_hand: list[UNOCard] = []

    for _ in range(k):
        total_w = sum(remaining_weights)
        if total_w <= 0:
            # Uniform fallback if all remaining weights are zero
            pick_idx = _rand_range(rng, len(remaining_pool))
        else:
            r = _uniform_float(rng, 0.0, total_w)
            cum = 0.0
            pick_idx = len(remaining_pool) - 1
            for i, w in enumerate(remaining_weights):
                cum += w
                if r <= cum:
                    pick_idx = i
                    break

        opp_hand.append(remaining_pool.pop(pick_idx))
        remaining_weights.pop(pick_idx)

    draw_pile = remaining_pool
    _shuffle_list(rng, draw_pile)

    return DeterminizationResult(
        opp_hand=opp_hand,
        draw_pile=draw_pile,
        unseen_pool_size_Ut=unseen_pool_size_Ut,
    )


def _uniform_float(rng: Any, low: float, high: float) -> float:
    """Safely extracts a uniform float from either random.Random or numpy.random.Generator."""
    if hasattr(rng, "uniform"):
        return float(rng.uniform(low, high))
    return float(rng.random() * (high - low) + low)


def _rand_range(rng: Any, n: int) -> int:
    """Safely extracts an integer in [0, n-1] from random.Random or numpy.random.Generator."""
    if hasattr(rng, "randrange"):
        return int(rng.randrange(n))
    if hasattr(rng, "integers"):
        return int(rng.integers(0, n))
    return int(rng.random() * n)


def _shuffle_list(rng: Any, items: list[Any]) -> None:
    """Safely in-place shuffles a list using the injected random generator."""
    if hasattr(rng, "shuffle"):
        rng.shuffle(items)
    else:
        # Fisher-Yates shuffle fallback
        for i in range(len(items) - 1, 0, -1):
            j = _rand_range(rng, i + 1)
            items[i], items[j] = items[j], items[i]
