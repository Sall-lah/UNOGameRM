"""Headless UNO game engine subclassing upstream UNO.py components.

This module strips terminal rendering, interactive inputs, and sleep delays,
providing a deterministic, headless game simulator driven by an isolated random stream.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

from card import Card
from deck import Deck
from player import AIPlayer, Player
from src.core.cards import (
    COLOURS,
    DRAW_TWO,
    REVERSE,
    SKIP,
    WILD,
    WILD_DRAW_FOUR,
    UNOCard,
)
from src.env.events import HorizonEventType, HorizonServerEvent


class HeadlessDeck(Deck):
    """Subclass of UNO.py Deck utilizing an injected RNG and reporting reshuffles.

    Eliminates reliance on module-level random functions so simulation rollouts
    and ground truth games can be precisely repeated from an integer seed.
    """

    def __init__(
        self,
        rng: random.Random,
        on_reshuffle: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.rng = rng
        self.on_reshuffle = on_reshuffle
        self.cards = self.build_deck()
        self.discard_pile: list[Card] = []
        num_shuffles = self.rng.randint(1, 3)
        self.shuffle(num_shuffles)

    def shuffle(self, number_of_shuffles: int = 1) -> None:
        """Shuffles the deck cards using the isolated engine RNG."""
        for _ in range(number_of_shuffles):
            self.rng.shuffle(self.cards)

    def draw(self, count: int = 1) -> list[Card]:
        """Draws cards from the deck, triggering a reshuffle if depleted."""
        drawn_cards: list[Card] = []
        for _ in range(count):
            if not self.cards:
                self.reshuffle_discard_pile()
            if self.cards:
                drawn_cards.append(self.cards.pop())
        for card in drawn_cards:
            if card.type == "WIL":
                card.colour = "white"
        return drawn_cards

    def reshuffle_discard_pile(self) -> None:
        """Recycles discard pile into the draw deck and notifies telemetry listener."""
        recycled_cards = self.discard_pile[:-1].copy()
        recycled_count = len(recycled_cards)
        self.cards = recycled_cards
        self.shuffle(2)
        top_card = self.discard_pile[-1]
        self.discard_pile.clear()
        self.discard_pile.append(top_card)
        if self.on_reshuffle:
            self.on_reshuffle(recycled_count)


class HeadlessAIPlayer(AIPlayer):
    """Subclass of UNO.py AIPlayer with fixed intellect and isolated RNG.

    Preserves upstream decision rules byte-for-byte to maintain opponent policy validity
    while preventing non-deterministic module random mutations.
    """

    def __init__(self, name: str, intellect: int, rng: random.Random) -> None:
        Player.__init__(self, name)
        self.intellect = intellect
        self.rng = rng

    def think(self, top_card: Card, next_player: Player) -> int | str:
        """Faithful port of AIPlayer.think using injected self.rng."""
        playable_card_count = len([card for card in self.hand if card.is_playable_on(top_card)])
        playable_act_or_wil_count = len(
            [card for card in self.hand if card.is_playable_on(top_card) and card.type != "NUM"]
        )

        draw_4_is_safe = True
        for card in self.hand:
            if card.colour == top_card.colour:
                draw_4_is_safe = False

        card_passed = False
        offensive_mode = True if self.rng.randint(0, 2) == 0 else False

        if playable_card_count == 0:
            if len(self.hand) == 1:
                self.uno = self.rng.randint(0, 6) <= self.intellect + 1
            return "draw"

        if (
            offensive_mode
            and self.intellect != 1
            and len(next_player.hand) == 1
            and len(self.hand) != 1
            and playable_act_or_wil_count == 0
        ):
            return "draw"

        self.rng.shuffle(self.hand)

        for i, card in enumerate(self.hand):
            if card.is_playable_on(top_card):
                if len(self.hand) == 1:
                    return 1

                if len(self.hand) == 2:
                    self.uno = self.rng.randint(0, 6) <= self.intellect + 2
                    return i + 1

                if (
                    offensive_mode
                    and len(next_player.hand) <= self.intellect
                    and playable_act_or_wil_count >= 1
                    and card.type == "NUM"
                ):
                    continue

                if (
                    offensive_mode
                    and len(next_player.hand) <= self.intellect
                    and playable_act_or_wil_count >= 1
                ):
                    return i + 1

                if self.rng.randint(0, 1) != self.intellect and playable_card_count == 2:
                    if card.type != "WIL":
                        card_passed = True
                    if card.type == "WIL" and not card_passed:
                        card_passed = True
                        continue

                if playable_card_count <= 2:
                    play_threshold = self.intellect + (8 // len(self.hand))
                    if self.rng.randint(0, play_threshold) != 0:
                        return i + 1

                if card.type == "NUM":
                    if playable_card_count >= 2 * playable_act_or_wil_count and self.intellect == 3:
                        return i + 1
                    if self.rng.randint(0, self.intellect**2 - 4 * self.intellect + 6) <= 1:
                        return i + 1

                if card.type == "ACT":
                    if playable_card_count <= 2 * playable_act_or_wil_count and self.rng.randint(0, 1) == 0:
                        return i + 1
                    if self.rng.randint(0, self.intellect + 1) == 0:
                        return i + 1

                if self.intellect != 1 and card.value == "+ 4" and (not draw_4_is_safe):
                    continue

                if card.type == "WIL":
                    if self.rng.randint(0, self.intellect * 3) <= 1:
                        return i + 1

        return "draw"

    def think_wild(self, top_card: Card, next_player: Optional[Player] = None, is_draw_4: bool = False) -> str:
        """Faithful port of AIPlayer.think_wild using injected self.rng."""
        colours = ["red", "green", "blue", "yellow"]
        self.rng.shuffle(colours)
        colour_count = {colour: 0 for colour in colours}

        for card in self.hand:
            if card.type != "WIL":
                colour_count[card.colour] += 1

        if (next_player is not None) and (not is_draw_4) and self.rng.randint(0, 7 - (self.intellect * 2)) == 0:
            next_colour_count = {colour: 0 for colour in colours}
            for card in next_player.hand:
                if card.type != "WIL":
                    next_colour_count[card.colour] += 1

            chosen_colour = min(next_colour_count, key=next_colour_count.get)  # type: ignore
            if next_colour_count[chosen_colour] == 0 and colour_count[chosen_colour] != 0:
                return chosen_colour
            return self.think_wild(top_card)

        chosen_colour = max(colour_count, key=colour_count.get)  # type: ignore
        if chosen_colour == top_card.colour and (not self.rng.randint(0, 2) > self.intellect):
            colour_count.pop(chosen_colour)
            chosen_colour = max(colour_count, key=colour_count.get)  # type: ignore

        return chosen_colour

    def think_challenge(self, top_card: Card, prev_player: Player) -> str:
        """Faithful port of AIPlayer.think_challenge using injected self.rng."""
        if self.intellect == 3 and self.rng.randint(0, 1) == 0:
            colour_count = 0
            for card in prev_player.hand:
                if card.colour == top_card.colour:
                    colour_count += 1
            return "challenge" if colour_count != 0 else "draw"

        if len(self.hand) <= 2:
            return "challenge" if self.rng.randint(0, 3 - len(self.hand)) != 0 else "draw"

        return "challenge" if self.rng.randint(0, self.intellect**2 - 4 * self.intellect + 5) == 0 else "draw"


class HeadlessGame:
    """Headless 2-player game manager executing turns without UI side effects.

    Maintains round state, emits structured horizon events, and enforces 2-player
    turn-retention rules without blocking terminal operations.
    """

    def __init__(
        self,
        round_id: int,
        starting_player: int,
        seed: int,
        opponent_intellect: int = 2,
        agent_d4_response: str = "accept",
    ) -> None:
        self.round_id = round_id
        self.starting_player = starting_player
        self.seed = seed
        self.opponent_intellect = opponent_intellect
        self.agent_d4_response = agent_d4_response

        self.rng = random.Random(seed)
        self.agent = Player("AGENT")
        self.opp = HeadlessAIPlayer("OPPONENT", intellect=self.opponent_intellect, rng=self.rng)
        self.players = [self.agent, self.opp]

        self.event_counter: int = 0
        self.recorded_events: list[HorizonServerEvent] = []

        self.deck = HeadlessDeck(rng=self.rng, on_reshuffle=self._handle_reshuffle)
        self.public_discards: list[str] = []
        self.round_over: bool = False
        self.winner: Optional[str] = None
        self.first_turn: bool = True

        self._setup_game()

    def _handle_reshuffle(self, count: int) -> None:
        """Emits a public RESHUFFLE event when the discard pile is recycled."""
        self.emit_event(
            actor="OPPONENT",
            event_type=HorizonEventType.RESHUFFLE,
            active=False,
            recycled_count=count,
        )

    def emit_event(
        self,
        actor: str,
        event_type: HorizonEventType,
        active: bool,
        card: Optional[UNOCard] = None,
        declared_colour: Optional[str] = None,
        cards_drawn_count: Optional[int] = None,
        recycled_count: Optional[int] = None,
        outcome: Optional[str] = None,
        winner: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> HorizonServerEvent:
        """Constructs, appends, and returns a typed horizon telemetry event."""
        evt = HorizonServerEvent(
            event_index=self.event_counter,
            actor=actor,
            event_type=event_type,
            active=active,
            card=card,
            declared_colour=declared_colour,
            cards_drawn_count=cards_drawn_count,
            recycled_count=recycled_count,
            outcome=outcome,
            agent_hand_size=len(self.agent.hand),
            opp_hand_size=len(self.opp.hand),
            winner=winner,
            reason=reason,
        )
        self.event_counter += 1
        self.recorded_events.append(evt)
        return evt

    def _setup_game(self) -> None:
        """Deals 7 cards each and flips the starting card according to UNO.py rules."""
        self.agent.hand.extend(self.deck.draw(7))
        self.opp.hand.extend(self.deck.draw(7))

        self.deck.discard_pile.extend(self.deck.draw(1))
        # Upstream rule: if first flipped card is Wild Draw Four, reshuffle into deck
        while self.deck.discard_pile[-1].value == "+ 4":
            self.deck.cards.append(self.deck.discard_pile.pop())
            self.deck.shuffle()
            self.deck.discard_pile.extend(self.deck.draw(1))

        top_card = self.deck.discard_pile[-1]
        self.public_discards.append(str(top_card))

        # Handle initial Wild card colour declaration if top card is a Wild
        if top_card.type == "WIL":
            if self.starting_player == 1:
                top_card.colour = self.opp.think_wild(top_card)
                self.emit_event(
                    actor="OPPONENT",
                    event_type=HorizonEventType.WILD_COLOUR_DECLARED,
                    active=True,
                    declared_colour=top_card.colour,
                )
            else:
                top_card.colour = "red"  # Default initial wild colour for agent
                self.emit_event(
                    actor="AGENT",
                    event_type=HorizonEventType.WILD_COLOUR_DECLARED,
                    active=True,
                    declared_colour="red",
                )
