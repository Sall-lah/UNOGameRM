"""Modified UNO server platform adapter adhering to the Zone 1 research specification.

This module provides ModifiedUNOServerPlatform, which wraps the headless UNO engine,
translates between high-level UNOAction strings and engine inputs, enforces auto-UNO,
and exposes the exact public observation and dispatch horizon interface.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Union

from card import Card
from src.config import AppConfig, load_config
from src.core.cards import (
    COLOURS,
    DRAW_TWO,
    REVERSE,
    SKIP,
    WILD,
    WILD_DRAW_FOUR,
    UNOAction,
    UNOCard,
)
from src.env.events import HorizonEventType, HorizonServerEvent
from src.env.headless_game import HeadlessGame


class ModifiedUNOServerPlatform:
    """Research adapter wrapping headless UNO.py and observing decision horizons.

    Guarantees that raw states isolate hidden opponent cards and deck ordering,
    while returning atomic telemetry streams covering H_t = [t, t+1).
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or load_config()
        self.game: Optional[HeadlessGame] = None
        self.t: int = 0
        self.round_id: int = 0
        self.opening_events: list[HorizonServerEvent] = []

    def start_new_round(self, round_id: int, starting_player: int, seed: int) -> dict[str, Any]:
        """Initializes a new round and advances to the agent's first decision turn.

        Constructs a fresh 2-player headless game, applies seat configuration,
        and resolves any opening opponent moves if the opponent starts first.
        """
        self.round_id = round_id
        self.t = 1
        self.game = HeadlessGame(
            round_id=round_id,
            starting_player=starting_player,
            seed=seed,
            opponent_intellect=self.config.opponent.intellect,
            agent_d4_response=self.config.agent.d4_response,
        )
        self.opening_events = list(self.game.recorded_events)

        if starting_player == 1 and not self.game.round_over:
            # Advance opponent turns until control transfers to the agent
            self._execute_opponent_turns()

        return self.get_raw_agent_state()

    def get_raw_agent_state(self) -> dict[str, Any]:
        """Builds the raw agent-facing engine state without leaking hidden information.

        Strictly omits opponent hand card identities, remaining draw pile order,
        and internal discard pile cards beyond the top and public discard history.
        """
        if self.game is None:
            raise RuntimeError("Cannot get state before start_new_round is called.")

        top_card = self.game.deck.discard_pile[-1]
        legal_actions = self._build_legal_actions(self.game.agent.hand, top_card)

        return {
            "round_id": self.round_id,
            "t": self.t,
            "current_player_index": 0,
            "agent_hand": [str(c) for c in self.game.agent.hand],
            "agent_hand_cards": [UNOCard.from_uno_py(c) for c in self.game.agent.hand],
            "top_card": str(top_card),
            "discard_pile_top": str(top_card),
            "opp_hand_count": len(self.game.opp.hand),
            "opp_uno": self.game.opp.uno,
            "agent_uno": self.game.agent.uno,
            "legal_actions": legal_actions,
            "public_discards": list(self.game.public_discards),
            "round_over": self.game.round_over,
            "winner": self.game.winner,
        }

    def dispatch_agent_action_and_observe_horizon(
        self,
        locked_action: Union[str, UNOAction],
    ) -> dict[str, Any]:
        """Executes the agent's locked action and collects all events until the next agent turn.

        Handles automatic UNO declarations, applies 2-player turn retention rules,
        drives opponent reactions, and returns the formal horizon observation dict.
        """
        if self.game is None or self.game.round_over:
            raise RuntimeError("Cannot dispatch action on uninitialized or completed round.")

        start_time = time.perf_counter()
        events_start_idx = len(self.game.recorded_events)

        action_str = locked_action.action_id if isinstance(locked_action, UNOAction) else str(locked_action)
        declared_colour = locked_action.declared_colour if isinstance(locked_action, UNOAction) else None

        # 1. Execute Agent turn
        control_returns_to_agent = self._execute_agent_action(action_str, declared_colour)

        # 2. If agent did not retain turn and round is active, drive opponent turns
        if not control_returns_to_agent and not self.game.round_over:
            self._execute_opponent_turns()

        dispatch_latency = time.perf_counter() - start_time
        horizon_events = self.game.recorded_events[events_start_idx:]
        self.t += 1

        return {
            "server_timeout_exceeded": False,
            "dispatch_latency_sec": dispatch_latency,
            "round_over": self.game.round_over,
            "winner": self.game.winner,
            "horizon_events": horizon_events,
            "agent_hand_after_horizon": [str(c) for c in self.game.agent.hand],
            "opp_hand_count_after_horizon": len(self.game.opp.hand),
        }

    def _execute_agent_action(self, action_str: str, explicit_declared_colour: Optional[str]) -> bool:
        """Executes the agent's specified action and returns True if turn is retained."""
        assert self.game is not None
        top_card = self.game.deck.discard_pile[-1]

        if action_str == "draw":
            # Auto-UNO check on draw transition with 1 card remaining
            if len(self.game.agent.hand) == 1:
                self.game.agent.uno = True
                self.game.emit_event(actor="AGENT", event_type=HorizonEventType.UNO_DECLARED, active=False)

            drawn = self.game.deck.draw(1)[0]
            if drawn.is_playable_on(top_card):
                # Automatic play of playable drawn card per UNO.py rules
                self.game.emit_event(
                    actor="AGENT",
                    event_type=HorizonEventType.DRAW_VOLUNTARY_ANOP,
                    active=True,
                    outcome="auto_play",
                    cards_drawn_count=1,
                )
                if drawn.type == "WIL":
                    decl = explicit_declared_colour or "red"
                    drawn.colour = decl
                    self.game.emit_event(
                        actor="AGENT",
                        event_type=HorizonEventType.WILD_COLOUR_DECLARED,
                        active=True,
                        declared_colour=decl,
                    )
                self.game.deck.discard_pile.append(drawn)
                self.game.public_discards.append(str(drawn))
                self.game.emit_event(
                    actor="AGENT",
                    event_type=HorizonEventType.PLAY_CARD,
                    active=True,
                    card=UNOCard.from_uno_py(drawn),
                )
                return self._handle_agent_played_card_effects(drawn)
            else:
                self.game.agent.hand.append(drawn)
                self.game.agent.uno = False
                self.game.emit_event(
                    actor="AGENT",
                    event_type=HorizonEventType.DRAW_VOLUNTARY_ANOP,
                    active=True,
                    outcome="keep",
                    cards_drawn_count=1,
                )
                return False

        # Card play action
        if len(self.game.agent.hand) == 2:
            self.game.agent.uno = True
            self.game.emit_event(actor="AGENT", event_type=HorizonEventType.UNO_DECLARED, active=False)

        card_idx = self._find_matching_hand_card(self.game.agent.hand, action_str)
        if card_idx == -1:
            raise ValueError(f"Action '{action_str}' does not match any card in agent hand.")

        played_card = self.game.agent.hand.pop(card_idx)
        if played_card.type == "WIL":
            decl = explicit_declared_colour or self._extract_colour_from_wild_action(action_str)
            played_card.colour = decl
            self.game.emit_event(
                actor="AGENT",
                event_type=HorizonEventType.WILD_COLOUR_DECLARED,
                active=True,
                declared_colour=decl,
            )

        self.game.deck.discard_pile.append(played_card)
        self.game.public_discards.append(str(played_card))
        self.game.emit_event(
            actor="AGENT",
            event_type=HorizonEventType.PLAY_CARD,
            active=True,
            card=UNOCard.from_uno_py(played_card),
        )

        return self._handle_agent_played_card_effects(played_card)

    def _handle_agent_played_card_effects(self, card: Card) -> bool:
        """Processes 2-player turn-retention or penalty consequences of an agent card."""
        assert self.game is not None
        if len(self.game.agent.hand) == 0:
            self.game.round_over = True
            self.game.winner = "AGENT"
            self.game.emit_event(actor="AGENT", event_type=HorizonEventType.ROUND_WIN, active=True, winner="AGENT")
            return True

        if card.value in (SKIP, REVERSE):
            self.game.emit_event(
                actor="OPPONENT",
                event_type=HorizonEventType.TURN_FORFEIT,
                active=False,
                reason="skip_or_reverse",
            )
            return True

        if card.value == DRAW_TWO:
            self.game.opp.hand.extend(self.game.deck.draw(2))
            self.game.opp.uno = False
            self.game.emit_event(
                actor="OPPONENT",
                event_type=HorizonEventType.DRAW_PENALTY_APEN,
                active=False,
                cards_drawn_count=2,
            )
            return True

        if card.value == WILD_DRAW_FOUR:
            prev_top = self.game.deck.discard_pile[-2]
            choice = self.game.opp.think_challenge(card, self.game.agent)
            if choice == "challenge":
                illegal = any(c.colour == prev_top.colour for c in self.game.agent.hand)
                if illegal:
                    self.game.agent.hand.extend(self.game.deck.draw(4))
                    self.game.agent.uno = False
                    self.game.emit_event(
                        actor="OPPONENT",
                        event_type=HorizonEventType.D4_CHALLENGE_WON,
                        active=False,
                        cards_drawn_count=4,
                    )
                else:
                    self.game.opp.hand.extend(self.game.deck.draw(6))
                    self.game.opp.uno = False
                    self.game.emit_event(
                        actor="OPPONENT",
                        event_type=HorizonEventType.D4_CHALLENGE_LOST,
                        active=False,
                        cards_drawn_count=6,
                    )
            else:
                self.game.opp.hand.extend(self.game.deck.draw(4))
                self.game.opp.uno = False
                self.game.emit_event(
                    actor="OPPONENT",
                    event_type=HorizonEventType.DRAW_PENALTY_APEN,
                    active=False,
                    cards_drawn_count=4,
                )
            return True

        return False

    def _execute_opponent_turns(self) -> None:
        """Executes opponent decisions in a loop until turn passes to agent or round ends."""
        assert self.game is not None
        while not self.game.round_over:
            top_card = self.game.deck.discard_pile[-1]
            choice = self.game.opp.think(top_card, self.game.agent)

            if choice == "draw":
                if len(self.game.opp.hand) == 1 and self.game.opp.uno:
                    self.game.emit_event(actor="OPPONENT", event_type=HorizonEventType.UNO_DECLARED, active=False)

                drawn = self.game.deck.draw(1)[0]
                if drawn.is_playable_on(top_card):
                    self.game.emit_event(
                        actor="OPPONENT",
                        event_type=HorizonEventType.DRAW_VOLUNTARY_ANOP,
                        active=True,
                        outcome="auto_play",
                        cards_drawn_count=1,
                    )
                    if drawn.type == "WIL":
                        is_d4 = drawn.value == WILD_DRAW_FOUR
                        drawn.colour = self.game.opp.think_wild(top_card, self.game.agent, is_d4)
                        self.game.emit_event(
                            actor="OPPONENT",
                            event_type=HorizonEventType.WILD_COLOUR_DECLARED,
                            active=True,
                            declared_colour=drawn.colour,
                        )
                    self.game.deck.discard_pile.append(drawn)
                    self.game.public_discards.append(str(drawn))
                    self.game.emit_event(
                        actor="OPPONENT",
                        event_type=HorizonEventType.PLAY_CARD,
                        active=True,
                        card=UNOCard.from_uno_py(drawn),
                    )
                    retained = self._handle_opponent_played_card_effects(drawn)
                    if not retained:
                        break
                else:
                    self.game.opp.hand.append(drawn)
                    self.game.opp.uno = False
                    self.game.emit_event(
                        actor="OPPONENT",
                        event_type=HorizonEventType.DRAW_VOLUNTARY_ANOP,
                        active=True,
                        outcome="keep",
                        cards_drawn_count=1,
                    )
                    break
            else:
                # 1-based index play
                card_idx = int(choice) - 1
                played = self.game.opp.hand.pop(card_idx)
                if len(self.game.opp.hand) == 1 and self.game.opp.uno:
                    self.game.emit_event(actor="OPPONENT", event_type=HorizonEventType.UNO_DECLARED, active=False)

                if played.type == "WIL":
                    is_d4 = played.value == WILD_DRAW_FOUR
                    played.colour = self.game.opp.think_wild(top_card, self.game.agent, is_d4)
                    self.game.emit_event(
                        actor="OPPONENT",
                        event_type=HorizonEventType.WILD_COLOUR_DECLARED,
                        active=True,
                        declared_colour=played.colour,
                    )

                self.game.deck.discard_pile.append(played)
                self.game.public_discards.append(str(played))
                self.game.emit_event(
                    actor="OPPONENT",
                    event_type=HorizonEventType.PLAY_CARD,
                    active=True,
                    card=UNOCard.from_uno_py(played),
                )
                retained = self._handle_opponent_played_card_effects(played)
                if not retained:
                    break

    def _handle_opponent_played_card_effects(self, card: Card) -> bool:
        """Processes 2-player turn retention or penalties resulting from opponent cards."""
        assert self.game is not None
        if len(self.game.opp.hand) == 0:
            self.game.round_over = True
            self.game.winner = "OPPONENT"
            self.game.emit_event(actor="OPPONENT", event_type=HorizonEventType.ROUND_WIN, active=True, winner="OPPONENT")
            return False

        if card.value in (SKIP, REVERSE):
            self.game.emit_event(
                actor="AGENT",
                event_type=HorizonEventType.TURN_FORFEIT,
                active=False,
                reason="skip_or_reverse",
            )
            return True

        if card.value == DRAW_TWO:
            self.game.agent.hand.extend(self.game.deck.draw(2))
            self.game.agent.uno = False
            self.game.emit_event(
                actor="AGENT",
                event_type=HorizonEventType.DRAW_PENALTY_APEN,
                active=False,
                cards_drawn_count=2,
            )
            return True

        if card.value == WILD_DRAW_FOUR:
            if self.config.agent.d4_response == "accept":
                self.game.agent.hand.extend(self.game.deck.draw(4))
                self.game.agent.uno = False
                self.game.emit_event(
                    actor="AGENT",
                    event_type=HorizonEventType.DRAW_PENALTY_APEN,
                    active=False,
                    cards_drawn_count=4,
                )
            else:
                prev_top = self.game.deck.discard_pile[-2]
                illegal = any(c.colour == prev_top.colour for c in self.game.opp.hand)
                if illegal:
                    self.game.opp.hand.extend(self.game.deck.draw(4))
                    self.game.opp.uno = False
                    self.game.emit_event(
                        actor="AGENT",
                        event_type=HorizonEventType.D4_CHALLENGE_WON,
                        active=False,
                        cards_drawn_count=4,
                    )
                else:
                    self.game.agent.hand.extend(self.game.deck.draw(6))
                    self.game.agent.uno = False
                    self.game.emit_event(
                        actor="AGENT",
                        event_type=HorizonEventType.D4_CHALLENGE_LOST,
                        active=False,
                        cards_drawn_count=6,
                    )
            return True

        return False

    def _build_legal_actions(self, hand: list[Card], top_card: Card) -> list[str]:
        """Constructs unique legal action strings expanding wilds across all four colours."""
        actions: list[str] = []
        for c in hand:
            if c.is_playable_on(top_card):
                if c.type == "WIL":
                    suffix = " + 4" if c.value == WILD_DRAW_FOUR else ""
                    for col in COLOURS:
                        actions.append(f"WILD{suffix} card, set to {col}")
                else:
                    actions.append(str(c))
        actions.append("draw")

        # Deduplicate while preserving insertion order
        seen = set()
        deduped = []
        for a in actions:
            if a not in seen:
                seen.add(a)
                deduped.append(a)
        return deduped

    def _find_matching_hand_card(self, hand: list[Card], action_str: str) -> int:
        """Finds the hand index of the card fulfilling action_str."""
        if action_str.startswith("WILD"):
            is_d4 = " + 4" in action_str
            target_val = WILD_DRAW_FOUR if is_d4 else WILD
            for idx, c in enumerate(hand):
                if c.type == "WIL" and c.value == target_val:
                    return idx
            return -1

        for idx, c in enumerate(hand):
            if str(c) == action_str:
                return idx
        return -1

    def _extract_colour_from_wild_action(self, action_str: str) -> str:
        """Extracts declared color from a serialized wild card string."""
        parts = action_str.split("set to ")
        if len(parts) == 2 and parts[1] in COLOURS:
            return parts[1]
        return "red"
