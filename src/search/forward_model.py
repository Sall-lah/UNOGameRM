"""Self-contained 2-player non-stacking UNO forward model for MCTS search rollouts.

This module provides a fast, deterministic, agent-side forward simulator that faithfully
reproduces the 2-player rules of UNO.py without ever reading hidden server engine state
or invoking upstream AI opponent methods. In 2-player UNO, action cards (Skip, Reverse,
Draw Two, Wild Draw Four) enforce turn retention where the non-acting player forfeits
their active turn; forced draws do not allow counter-stacking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.core.cards import (
    COLOURS,
    DRAW_TWO,
    REVERSE,
    SKIP,
    WILD,
    WILD_DRAW_FOUR,
    UNOAction,
    UNOCard,
    parse_card_string,
)
from src.env.barrier import LegalObservation


@dataclass(slots=True)
class SimulatedState:
    """Represents an isolated, determinized game state during MCTS lookahead simulation.

    Using frozen UNOCard objects alongside mutable collection copies enables rapid
    branching with minimal memory overhead while guaranteeing zero cross-branch state bleed.
    """

    agent_hand: list[UNOCard]
    opp_hand: list[UNOCard]
    draw_pile: list[UNOCard]
    top_discard: UNOCard
    active_colour: str
    current_player: str = "agent"  # "agent" or "opponent"
    agent_suffered_a_pen: bool = False
    agent_uno_flag: bool = False
    opp_uno_flag: bool = False
    winner: Optional[str] = None  # "agent", "opponent", or None

    def clone(self) -> SimulatedState:
        """Creates an independent copy of this simulated state for tree branch evaluation.

        Why: MCTS requires open-loop rollouts to mutate local state without corrupting
        parallel tree nodes or determinization pools.
        """
        return SimulatedState(
            agent_hand=list(self.agent_hand),
            opp_hand=list(self.opp_hand),
            draw_pile=list(self.draw_pile),
            top_discard=self.top_discard,
            active_colour=self.active_colour,
            current_player=self.current_player,
            agent_suffered_a_pen=self.agent_suffered_a_pen,
            agent_uno_flag=self.agent_uno_flag,
            opp_uno_flag=self.opp_uno_flag,
            winner=self.winner,
        )

    def is_terminal(self) -> bool:
        """Determines if the simulated round has completed.

        Why: A round terminates immediately when a player empties their hand or when
        the determinized draw pile is fully exhausted, cutting off further search depth.
        """
        if self.winner is not None:
            return True
        if len(self.agent_hand) == 0:
            self.winner = "agent"
            return True
        if len(self.opp_hand) == 0:
            self.winner = "opponent"
            return True
        if len(self.draw_pile) == 0:
            return True
        return False

    def get_legal_actions(self) -> list[UNOAction]:
        """Generates all legal actions available to the current acting player.

        Why: Legal action generation strictly mirrors UNO.py rules (Wilds expand into
        four distinct color declaration actions; voluntary draw is always permissible)
        to ensure simulated actions map 1:1 with real game engine options.
        """
        if self.is_terminal():
            return []

        hand = self.agent_hand if self.current_player == "agent" else self.opp_hand
        actions: list[UNOAction] = []
        seen_action_ids: set[str] = set()

        for card in hand:
            if card.is_wild:
                suffix = " + 4" if card.value == WILD_DRAW_FOUR else ""
                for c in COLOURS:
                    action_id = f"WILD{suffix} card, set to {c}"
                    if action_id not in seen_action_ids:
                        seen_action_ids.add(action_id)
                        actions.append(
                            UNOAction(
                                action_id=action_id,
                                action_type="PLAY",
                                card=card,
                                declared_colour=c,
                            )
                        )
            else:
                if card.colour == self.active_colour or card.value == self.top_discard.value:
                    action_id = str(card)
                    if action_id not in seen_action_ids:
                        seen_action_ids.add(action_id)
                        actions.append(
                            UNOAction(
                                action_id=action_id,
                                action_type="PLAY",
                                card=card,
                                declared_colour=None,
                            )
                        )

        # Voluntary draw is always a legal decision in UNO.py
        actions.append(
            UNOAction(
                action_id="draw",
                action_type="DRAW",
                card=None,
                declared_colour=None,
            )
        )
        return actions

    def step(self, action: UNOAction) -> None:
        """Applies an action to the simulated state using 2-player non-stacking rules.

        Why: In 2-player UNO, playing Skip, Reverse, Draw Two, or Wild Draw Four denies
        the opponent an active turn (turn retention). Draw penalties (+2, +4) are forced
        without stacking opportunities. Number cards and regular Wilds pass control.
        """
        if self.is_terminal():
            return

        is_agent_acting = self.current_player == "agent"
        acting_hand = self.agent_hand if is_agent_acting else self.opp_hand
        target_hand = self.opp_hand if is_agent_acting else self.agent_hand

        if action.action_type == "DRAW" or action.action_id == "draw":
            if self.draw_pile:
                drawn = self.draw_pile.pop(0)
                acting_hand.append(drawn)
            # Voluntary draw passes the turn to the opponent in 2P lookahead
            self.current_player = "opponent" if is_agent_acting else "agent"
            return

        # Action is a card play: locate and remove the card from the acting player's hand
        played_card: Optional[UNOCard] = None
        if action.card is not None:
            for idx, c in enumerate(acting_hand):
                if action.card.is_wild and c.is_wild and c.value == action.card.value:
                    played_card = acting_hand.pop(idx)
                    break
                elif c == action.card:
                    played_card = acting_hand.pop(idx)
                    break

        if played_card is None and action.card is not None:
            played_card = action.card

        if played_card is None:
            return

        # Automatic UNO flag upon reducing hand size from 2 to 1 card
        if len(acting_hand) == 1:
            if is_agent_acting:
                self.agent_uno_flag = True
            else:
                self.opp_uno_flag = True

        # Check for immediate round win on empty hand
        if len(acting_hand) == 0:
            self.winner = "agent" if is_agent_acting else "opponent"
            return

        # Update top discard and active color
        if played_card.is_wild:
            self.top_discard = UNOCard(
                type="WIL",
                colour=action.declared_colour or "white",
                value=played_card.value,
            )
            self.active_colour = action.declared_colour or "red"
        else:
            self.top_discard = played_card
            self.active_colour = played_card.colour

        # Apply action card effects and 2-player turn retention mechanics
        if played_card.value == DRAW_TWO:
            draw_count = min(2, len(self.draw_pile))
            for _ in range(draw_count):
                target_hand.append(self.draw_pile.pop(0))
            if not is_agent_acting:
                self.agent_suffered_a_pen = True
            # Turn retention: acting player retains control
        elif played_card.value == WILD_DRAW_FOUR:
            draw_count = min(4, len(self.draw_pile))
            for _ in range(draw_count):
                target_hand.append(self.draw_pile.pop(0))
            if not is_agent_acting:
                self.agent_suffered_a_pen = True
            # Turn retention: acting player retains control
        elif played_card.value in (SKIP, REVERSE):
            # In 2-player UNO, Reverse functions identically to Skip, retaining the turn
            pass
        else:
            # Normal number card or non-+4 Wild passes the turn to the opponent
            self.current_player = "opponent" if is_agent_acting else "agent"


def create_simulated_state_from_observation(
    obs: LegalObservation,
    opp_hand: list[UNOCard],
    draw_pile: list[UNOCard],
) -> SimulatedState:
    """Builds a fresh SimulatedState by pairing a public LegalObservation with determinized cards.

    Why: Ensures root search nodes perfectly reflect the public information state without
    holding leaked references to private engine variables.
    """
    agent_hand = [parse_card_string(c) for c in obs.agent_hand]
    top_discard = parse_card_string(obs.top_discard) if obs.top_discard else parse_card_string("red 0")

    return SimulatedState(
        agent_hand=agent_hand,
        opp_hand=list(opp_hand),
        draw_pile=list(draw_pile),
        top_discard=top_discard,
        active_colour=obs.active_colour,
        current_player="agent",
        agent_suffered_a_pen=False,
        agent_uno_flag=len(agent_hand) == 1,
        opp_uno_flag=len(opp_hand) == 1,
        winner=None,
    )
