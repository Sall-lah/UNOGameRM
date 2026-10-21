"""Rule-grounded structuring module for PlugMem knowledge graph extraction.

Implements deterministic rules (Rules 1-5) transforming post-horizon observations
into episodic memories, semantic belief propositions, and procedural prescriptions.
"""

from __future__ import annotations

from typing import Any, Sequence

from src.config import MemoryConfig
from src.core.cards import parse_card_string
from src.env.barrier import LegalObservation
from src.env.events import HorizonEventType, HorizonServerEvent
from src.memory.graph import EpisodicNode, PlugMemGraphStore, PrescriptionNode, PropositionNode


class RuleGroundedExtractor:
    """Extracts verified intra-round beliefs without LLM latency on the decision path."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config

    def extract_and_update(
        self,
        graph: PlugMemGraphStore,
        obs: LegalObservation,
        horizon_events: Sequence[HorizonServerEvent | dict[str, Any]],
        locked_action: str,
        t: int,
    ) -> EpisodicNode:
        """Processes horizon events, ingests episodic experience, and updates propositions."""
        # 1. Parse and classify horizon outcomes
        stratum = self._derive_stratum(horizon_events)
        y_t_A = self._derive_y_A(horizon_events)
        y_t_B = self._derive_y_B(horizon_events)
        a_nop_count = self._count_a_nop_draws(horizon_events)

        raw_event_dicts = tuple(
            e if isinstance(e, dict) else self._event_to_dict(e) for e in horizon_events
        )

        delta_hand = len(obs.agent_hand)  # Hand size delta is factored during procedural scoring

        # 2. Build Episodic Node
        epi_id = f"epi_r{graph.round_id}_t{t}"
        state_summary = f"Round {graph.round_id} Turn {t}: Top '{obs.top_discard}', Colour '{obs.active_colour}', Opponent count {obs.opp_card_count}"
        inferred_subgoal = "retain_turn" if stratum == 1 else "defend_risk" if obs.opp_card_count <= 2 else "advance_hand"

        epi_node = EpisodicNode(
            node_id=epi_id,
            round_id=graph.round_id,
            turn=t,
            state_summary=state_summary,
            inferred_subgoal=inferred_subgoal,
            stratum=stratum,
            y_t_A=y_t_A,
            y_t_B=y_t_B,
            a_nop_count=a_nop_count,
            ordered_events=raw_event_dicts,
            locked_action=locked_action,
        )
        graph.add_episodic(epi_node)

        # 3. Apply Belief Revision (Rule 4a) and Confidence Decay (Rule 4b)
        self._apply_revisions_and_decays(graph, horizon_events, t)

        # 4. Induce New Semantic Beliefs (Rules 1, 2, 3)
        self._induce_semantic_propositions(graph, obs, horizon_events, epi_id, t)

        # 5. Induce Procedural Skill (Rule 5)
        self._induce_procedural_prescription(graph, obs, stratum, y_t_A, y_t_B, delta_hand, epi_id, t)

        return epi_node

    def _apply_revisions_and_decays(
        self,
        graph: PlugMemGraphStore,
        horizon_events: Sequence[HorizonServerEvent | dict[str, Any]],
        t: int,
    ) -> None:
        """Applies contradiction soft-deactivation (4a) and draw decay/pruning (4b)."""
        opp_played_colours: set[str] = set()
        opp_drawn_cards: int = 0

        for e in horizon_events:
            actor = e.get("actor") if isinstance(e, dict) else e.actor
            event_type = e.get("event_type") if isinstance(e, dict) else e.event_type
            if hasattr(event_type, "value"):
                event_type = event_type.value

            if actor == "OPPONENT":
                if event_type == "PLAY_CARD":
                    card_val = e.get("card") if isinstance(e, dict) else e.card
                    if card_val:
                        card_str = str(card_val)
                        try:
                            card_obj = parse_card_string(card_str)
                            opp_played_colours.add(card_obj.colour)
                        except Exception:
                            pass
                elif event_type in ("DRAW_VOLUNTARY_ANOP", "DRAW_PENALTY_APEN"):
                    count = e.get("cards_drawn_count") if isinstance(e, dict) else e.cards_drawn_count
                    opp_drawn_cards += (count or 1)

        # Rule 4a: Contradiction Soft-Deactivation
        if opp_played_colours:
            for prop in graph.proposition_nodes.values():
                if prop.active and prop.category == "OPPONENT_DRAW_CONSTRAINT":
                    lacked_col = prop.metadata.get("lacked_colour")
                    if lacked_col and lacked_col in opp_played_colours:
                        prop.active = False
                        prop.confidence = 0.0
                        prop.deactivation_turn = t
                        prop.deactivation_reason = f"contradicted_by_opp_play_{lacked_col}"

        # Rule 4b: Draw Confidence Decay and Pruning
        if opp_drawn_cards > 0:
            gamma = self.config.draw_decay_gamma
            threshold = self.config.prune_threshold
            decay_factor = gamma ** opp_drawn_cards
            for prop in graph.proposition_nodes.values():
                if prop.active and prop.category == "OPPONENT_DRAW_CONSTRAINT":
                    prop.confidence *= decay_factor
                    if prop.confidence < threshold:
                        prop.active = False
                        prop.deactivation_turn = t
                        prop.deactivation_reason = f"decay_pruned_below_{threshold}"

    def _induce_semantic_propositions(
        self,
        graph: PlugMemGraphStore,
        obs: LegalObservation,
        horizon_events: Sequence[HorizonServerEvent | dict[str, Any]],
        epi_id: str,
        t: int,
    ) -> None:
        """Induces opponent draw constraints (Rule 1), wild declarations (Rule 2), and alerts (Rule 3)."""
        top_card_val = obs.top_discard.split(" ")[-1] if " " in obs.top_discard else ""

        for e in horizon_events:
            actor = e.get("actor") if isinstance(e, dict) else e.actor
            event_type = e.get("event_type") if isinstance(e, dict) else e.event_type
            if hasattr(event_type, "value"):
                event_type = event_type.value

            if actor == "OPPONENT":
                # Rule 1: Non-penalty voluntary draw -> lack constraint
                if event_type == "DRAW_VOLUNTARY_ANOP":
                    prop_id = graph.next_prop_id()
                    statement = f"Opponent was unable to play on colour '{obs.active_colour}' / value '{top_card_val}' at turn {t} and took a voluntary draw."
                    tags = (
                        "constraint:opp_lacks_colour",
                        f"colour:{obs.active_colour}",
                        f"value:{top_card_val}",
                    )
                    prop = PropositionNode(
                        node_id=prop_id,
                        round_id=graph.round_id,
                        origin_turn=t,
                        statement=statement,
                        category="OPPONENT_DRAW_CONSTRAINT",
                        concept_tags=tags,
                        confidence=0.85,
                        active=True,
                        provenance_epi_ids=(epi_id,),
                        metadata={"lacked_colour": obs.active_colour, "lacked_value": top_card_val},
                    )
                    graph.add_proposition(prop)

                # Rule 2: Wild declaration
                elif event_type == "WILD_COLOUR_DECLARED":
                    decl_colour = e.get("declared_colour") if isinstance(e, dict) else e.declared_colour
                    if decl_colour:
                        prop_id = graph.next_prop_id()
                        statement = f"Opponent declared colour '{decl_colour}' at turn {t}."
                        tags = ("constraint:opp_declared_wild", f"colour:{decl_colour}")
                        prop = PropositionNode(
                            node_id=prop_id,
                            round_id=graph.round_id,
                            origin_turn=t,
                            statement=statement,
                            category="WILD_COLOUR_DECLARATION",
                            concept_tags=tags,
                            confidence=0.90,
                            active=True,
                            provenance_epi_ids=(epi_id,),
                            metadata={"declared_colour": decl_colour},
                        )
                        graph.add_proposition(prop)

        # Rule 3: Depletion alert when opponent hand <= 2
        for e in reversed(horizon_events):
            opp_size = e.get("opp_hand_size") if isinstance(e, dict) else e.opp_hand_size
            if opp_size is not None and opp_size <= 2:
                prop_id = graph.next_prop_id()
                statement = f"Opponent hand depleted to {opp_size} cards after turn {t}."
                tags = ("intent:prevent_risk_B", f"hand_size:{opp_size}")
                prop = PropositionNode(
                    node_id=prop_id,
                    round_id=graph.round_id,
                    origin_turn=t,
                    statement=statement,
                    category="DEPLETION_ALERT",
                    concept_tags=tags,
                    confidence=0.95,
                    active=True,
                    provenance_epi_ids=(epi_id,),
                    metadata={"opp_hand_size": opp_size},
                )
                graph.add_proposition(prop)
                break

    def _induce_procedural_prescription(
        self,
        graph: PlugMemGraphStore,
        obs: LegalObservation,
        stratum: int,
        y_t_A: int,
        y_t_B: int,
        delta_hand: int,
        epi_id: str,
        t: int,
    ) -> None:
        """Induces procedural rules scored according to the doc-03 return-score formula (Rule 5)."""
        # Return score rho = clip(6 + 3*I(Stratum=1) - 4*y_A - 5*y_B - 0.5*delta_hand, 1, 10)
        stratum_bonus = 3.0 if stratum == 1 else 0.0
        raw_score = 6.0 + stratum_bonus - (4.0 * y_t_A) - (5.0 * y_t_B) - (0.5 * delta_hand)
        return_score = float(min(10.0, max(1.0, raw_score)))

        presc_id = graph.next_presc_id()
        if stratum == 1:
            intent = "retain_turn"
            prescription = "Play turn-retention action cards (Skip, Reverse, +2, Wild+4) to prevent opponent active decisions."
        elif y_t_A == 1:
            intent = "defend_penalty"
            prescription = "Avoid playing cards that expose agent to opponent +2 or +4 responses."
        else:
            intent = "advance_hand"
            prescription = "Match current active colour or exploit opponent lack constraints."

        tags = (f"stratum:{stratum}", f"intent:{intent}", f"colour:{obs.active_colour}")
        presc = PrescriptionNode(
            node_id=presc_id,
            round_id=graph.round_id,
            origin_turn=t,
            intent=intent,
            prescription=prescription,
            return_score=return_score,
            active=True,
            provenance_epi_ids=(epi_id,),
            tags=tags,
        )
        graph.add_prescription(presc)

    @staticmethod
    def _derive_stratum(events: Sequence[Any]) -> int:
        """Derives horizon Stratum (1 = 0 active opp turns, 2 = 1 active turn, 3 = >=2 active turns)."""
        active_opp_turns = 0
        for e in events:
            actor = e.get("actor") if isinstance(e, dict) else e.actor
            active = e.get("active") if isinstance(e, dict) else e.active
            if actor == "OPPONENT" and active:
                active_opp_turns += 1

        if active_opp_turns == 0:
            return 1
        elif active_opp_turns == 1:
            return 2
        return 3

    @staticmethod
    def _derive_y_A(events: Sequence[Any]) -> int:
        """Derives adverse label y_A (1 if agent suffered penalty draw)."""
        for e in events:
            actor = e.get("actor") if isinstance(e, dict) else e.actor
            event_type = e.get("event_type") if isinstance(e, dict) else e.event_type
            if hasattr(event_type, "value"):
                event_type = event_type.value
            if actor == "AGENT" and event_type == "DRAW_PENALTY_APEN":
                return 1
        return 0

    @staticmethod
    def _derive_y_B(events: Sequence[Any]) -> int:
        """Derives adverse label y_B (1 if opponent won during horizon)."""
        for e in events:
            actor = e.get("actor") if isinstance(e, dict) else e.actor
            event_type = e.get("event_type") if isinstance(e, dict) else e.event_type
            if hasattr(event_type, "value"):
                event_type = event_type.value
            if actor == "OPPONENT" and event_type == "ROUND_WIN":
                return 1
        return 0

    @staticmethod
    def _count_a_nop_draws(events: Sequence[Any]) -> int:
        """Counts voluntary non-penalty draws performed by the opponent."""
        count = 0
        for e in events:
            actor = e.get("actor") if isinstance(e, dict) else e.actor
            event_type = e.get("event_type") if isinstance(e, dict) else e.event_type
            if hasattr(event_type, "value"):
                event_type = event_type.value
            if actor == "OPPONENT" and event_type == "DRAW_VOLUNTARY_ANOP":
                count += 1
        return count

    @staticmethod
    def _event_to_dict(e: HorizonServerEvent) -> dict[str, Any]:
        """Converts HorizonServerEvent into a sanitized dictionary."""
        return {
            "event_index": e.event_index,
            "actor": e.actor,
            "event_type": e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type),
            "active": e.active,
            "card": str(e.card) if e.card else None,
            "declared_colour": e.declared_colour,
            "cards_drawn_count": e.cards_drawn_count,
            "recycled_count": e.recycled_count,
            "outcome": e.outcome,
            "agent_hand_size": e.agent_hand_size,
            "opp_hand_size": e.opp_hand_size,
            "winner": e.winner,
        }
