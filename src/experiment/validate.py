"""Phase-1 pre-pilot system validation suite enforcing core research invariants.

Why: Executes doc-08 Phase-1 integrity checks to verify information barrier opacity,
2-player non-stacking turn retention in both engine and MCTS forward model,
A_pen vs A_nop adverse-event separation, and auto-UNO declarations before experimental data collection.
"""

from __future__ import annotations

import sys
from typing import Optional

from card import Card
from src.config import AppConfig, load_config
from src.core.cards import (
    DRAW_TWO,
    REVERSE,
    SKIP,
    WILD_DRAW_FOUR,
    UNOAction,
    UNOCard,
)
from src.env.barrier import ClientServerBarrier, LegalObservation
from src.env.events import HorizonEventType, HorizonServerEvent
from src.env.headless_game import HeadlessGame
from src.env.uno_engine import ModifiedUNOServerPlatform
from src.audit.horizon import HorizonEvaluation, HorizonMonitor
from src.search.forward_model import SimulatedState


class ValidationFailure(RuntimeError):
    """Raised when an experimental system invariant fails verification."""


def validate_barrier_integrity() -> None:
    """Verifies that LegalObservation completely isolates private engine state.

    Why: Guarantees that agent decisions cannot peek at opponent cards, deck ordering,
    or discard pile internals across the information barrier.
    """
    raw_state = {
        "round_id": 1,
        "turn_idx": 1,
        "agent_hand": ["red 1", "blue 2"],
        "top_card": "red 5",
        "active_colour": "red",
        "opp_hand_count": 7,
        "legal_actions": ["red 1", "draw"],
        "events": [],
        "public_discards": ["red 5"],
        # Leaked fields in raw engine state that barrier must strip
        "opp_hand": ["green 9", "yellow skip"],
        "deck_cards": ["blue 7", "red + 2"],
    }

    obs = ClientServerBarrier.extract_observation(raw_state)

    if not isinstance(obs, LegalObservation):
        raise ValidationFailure("extract_observation did not return a LegalObservation instance")

    if hasattr(obs, "opp_hand") or hasattr(obs, "deck_cards") or hasattr(obs, "deck"):
        raise ValidationFailure("LegalObservation exposed private opponent hand or deck attributes")

    if len(obs.agent_hand) != 2 or obs.opp_card_count != 7:
        raise ValidationFailure("Observation corrupted agent hand or opponent card count")


def validate_engine_turn_retention() -> None:
    """Verifies 2-player turn retention in the headless engine platform.

    Why: Playing Skip, Reverse, +2, or Wild +4 under 2-player UNO rules immediately skips
    the opponent's turn and returns control to the agent with zero active opponent turns.
    """
    cfg = AppConfig()
    platform = ModifiedUNOServerPlatform(config=cfg)
    platform.start_new_round(round_id=1, starting_player=0, seed=42)

    # Inject specific cards into agent hand to test turn retention
    game: HeadlessGame = platform.game  # type: ignore
    test_cards = [
        Card("ACT", "red", "⊘"),  # Skip
        Card("ACT", "red", "⇄"),  # Reverse
        Card("ACT", "red", "+ 2"),  # Draw Two
        Card("WIL", "white", "+ 4"),  # Wild Draw Four
    ]

    for card in test_cards:
        game.agent.hand.clear()
        game.opp.hand.clear()
        # Ensure agent has the card and a follow-up card so round doesn't end
        game.agent.hand.extend([card, Card("NUM", "red", "1")])
        game.opp.hand.extend([Card("NUM", "blue", "5"), Card("NUM", "blue", "6")])
        game.deck.discard_pile.clear()
        game.deck.discard_pile.append(Card("NUM", "red", "7"))

        action_id = (
            "WILD + 4 card, set to red"
            if card.value == "+ 4"
            else f"{card.colour} {card.value}"
        )
        if card.value == "⊘":
            action_id = "red skip"
        elif card.value == "⇄":
            action_id = "red reverse"

        res = platform.dispatch_agent_action_and_observe_horizon(action_id)
        events = res["horizon_events"]

        # In 2-player turn retention, opponent gets 0 active turns
        active_opp_events = [e for e in events if e.actor == "OPPONENT" and e.active]
        if len(active_opp_events) > 0:
            raise ValidationFailure(
                f"Engine permitted active opponent action during turn retention for '{action_id}'"
            )


def validate_mcts_forward_model_turn_retention() -> None:
    """Verifies that MCTS lookahead model enforces 2-player turn retention.

    Why: If the forward model mistakenly allows opponent counter-stacking or passes
    control to the opponent on Skip/Reverse/+2/+4, tree search estimates will be fundamentally biased.
    """
    test_actions = [
        UNOAction(action_id="red skip", action_type="PLAY", card=UNOCard(type="ACT", colour="red", value=SKIP)),
        UNOAction(action_id="red reverse", action_type="PLAY", card=UNOCard(type="ACT", colour="red", value=REVERSE)),
        UNOAction(action_id="red + 2", action_type="PLAY", card=UNOCard(type="ACT", colour="red", value=DRAW_TWO)),
        UNOAction(action_id="WILD + 4 card, set to red", action_type="PLAY", card=UNOCard(type="WIL", colour="red", value=WILD_DRAW_FOUR), declared_colour="red"),
    ]

    for act in test_actions:
        state = SimulatedState(
            agent_hand=[act.card, UNOCard(type="NUM", colour="red", value="1")],
            opp_hand=[UNOCard(type="NUM", colour="blue", value="5"), UNOCard(type="NUM", colour="blue", value="6")],
            draw_pile=[UNOCard(type="NUM", colour="green", value="3") for _ in range(20)],
            top_discard=UNOCard(type="NUM", colour="red", value="7"),
            active_colour="red",
            current_player="agent",
        )

        state.step(act)

        # In 2-player UNO, turn retention returns control to agent
        if state.current_player != "agent":
            raise ValidationFailure(
                f"MCTS ForwardModel passed turn to opponent on turn-retention action '{act.action_id}'"
            )


def validate_apen_vs_anop_separation() -> None:
    """Verifies proper separation between forced penalty draws (A_pen) and voluntary draws (A_nop).

    Why: Statistical claim validity for Risk A (y_{t,A}) strictly depends on isolating
    opponent-inflicted penalty draws (+2, +4) from strategic voluntary 1-card draws.
    """
    monitor = HorizonMonitor()

    # 1. Voluntary draw scenario: agent executes voluntary draw
    vol_event = HorizonServerEvent(
        event_index=1,
        actor="AGENT",
        event_type=HorizonEventType.DRAW_VOLUNTARY_ANOP,
        active=True,
        cards_drawn_count=1,
    )
    vol_eval = monitor.process_events(
        events=[vol_event],
        agent_played_retention_card=False,
    )
    if vol_eval.A_nop_voluntary_drawn != 1 or vol_eval.y_t_A_penalty_draw != 0:
        raise ValidationFailure(
            f"Voluntary draw misclassified: A_nop={vol_eval.A_nop_voluntary_drawn}, "
            f"y_tA={vol_eval.y_t_A_penalty_draw}"
        )

    # 2. Penalty draw scenario: agent receives forced penalty draw
    pen_draw_event = HorizonServerEvent(
        event_index=2,
        actor="AGENT",
        event_type=HorizonEventType.DRAW_PENALTY_APEN,
        active=False,
        cards_drawn_count=2,
    )
    pen_eval = monitor.process_events(
        events=[pen_draw_event],
        agent_played_retention_card=False,
    )
    if pen_eval.A_pen_cards_drawn != 2 or pen_eval.y_t_A_penalty_draw != 1:
        raise ValidationFailure(
            f"Penalty draw misclassified: A_pen={pen_eval.A_pen_cards_drawn}, "
            f"y_tA={pen_eval.y_t_A_penalty_draw}"
        )


def validate_auto_uno() -> None:
    """Verifies automatic UNO declaration when hand size reaches exactly 1.

    Why: Upstream UNO rules impose a 2-card draw penalty for failing to declare UNO;
    the agent platform auto-declares UNO to eliminate execution noise.
    """
    cfg = AppConfig()
    platform = ModifiedUNOServerPlatform(config=cfg)
    platform.start_new_round(round_id=1, starting_player=0, seed=42)

    # Set agent hand to 2 cards
    game: HeadlessGame = platform.game  # type: ignore
    game.agent.hand.clear()
    game.agent.hand.extend([Card("NUM", "red", "1"), Card("NUM", "blue", "2")])
    game.deck.discard_pile.clear()
    game.deck.discard_pile.append(Card("NUM", "red", "5"))

    res = platform.dispatch_agent_action_and_observe_horizon("red 1")
    has_uno_event = any(
        e.event_type == HorizonEventType.UNO_DECLARED for e in res.get("horizon_events", [])
    )
    if not (has_uno_event or game.agent.uno):
        raise ValidationFailure("Engine platform failed to auto-declare UNO on hand size transition 2 -> 1")


def run_all_validation_checks(config: Optional[AppConfig] = None) -> bool:
    """Runs all Phase-1 pre-pilot verification checks in sequence, failing fast.

    Why: Serves as the operator-facing gating command before pilot or main experiments.
    """
    checks = [
        ("barrier_integrity", validate_barrier_integrity),
        ("turn_retention_engine", validate_engine_turn_retention),
        ("turn_retention_mcts_forward_model", validate_mcts_forward_model_turn_retention),
        ("apen_vs_anop_separation", validate_apen_vs_anop_separation),
        ("auto_uno", validate_auto_uno),
    ]

    print("=== Running Phase-1 Pre-Pilot System Validation ===")
    for name, check_fn in checks:
        print(f"Checking {name}...", end=" ", flush=True)
        try:
            check_fn()
            print("[PASS]")
        except Exception as exc:
            print(f"[FAIL]: {exc}")
            raise ValidationFailure(f"Phase-1 check '{name}' failed: {exc}") from exc

    print("=== All Phase-1 Verification Checks Passed Successfully ===")
    return True
