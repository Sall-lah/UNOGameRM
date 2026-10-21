"""Top-level UNOPlugMem memory manager adapting PlugMem for Zone 2 agent harness.

Maintains intra-round episodic, semantic, and procedural knowledge graphs,
handles per-round lifecycle isolation, seeds baseline priors, and exposes
retrieval and MCTS determinization constraint interfaces.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from src.config import AppConfig, MemoryConfig, load_config
from src.env.barrier import LegalObservation
from src.env.events import HorizonServerEvent
from src.memory.embedding import EmbeddingInterface, LocalFallbackEmbedder
from src.memory.extractor import RuleGroundedExtractor
from src.memory.graph import (
    EpisodicNode,
    PlugMemConstraint,
    PlugMemGraphStore,
    PrescriptionNode,
    RetrievalResult,
)
from src.memory.retrieval import PlugMemRetriever


class UNOPlugMem:
    """Intra-round memory graph management according to the PlugMem framework."""

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embedder: Optional[EmbeddingInterface] = None,
    ) -> None:
        if config is None:
            app_cfg = load_config()
            self.config = app_cfg.memory
        else:
            self.config = config

        self.embedder = embedder or LocalFallbackEmbedder()
        self.graph = PlugMemGraphStore()
        self.extractor = RuleGroundedExtractor(config=self.config)
        self.retriever = PlugMemRetriever(config=self.config, embedder=self.embedder)
        self.current_round: int = 0
        self.current_turn: int = 1

    def reset_for_new_round(self, round_id: int) -> None:
        """Clears all prior-round memory structures and seeds initial priors if enabled."""
        self.current_round = round_id
        self.current_turn = 1
        self.graph.clear(round_id=round_id)

        if self.config.seed_procedural_priors:
            self._seed_procedural_priors(round_id)

    def update(
        self,
        obs: LegalObservation,
        horizon_events: Sequence[HorizonServerEvent | dict[str, Any]],
        locked_action: str,
        t: int,
    ) -> EpisodicNode:
        """Ingests verified public horizon telemetry into G^E, G^S, and G^P."""
        self.current_turn = t + 1
        return self.extractor.extract_and_update(
            graph=self.graph,
            obs=obs,
            horizon_events=horizon_events,
            locked_action=locked_action,
            t=t,
        )

    def retrieve(self, obs: LegalObservation) -> RetrievalResult:
        """Pre-decision retrieval returning distilled context and active constraint mappings."""
        return self.retriever.retrieve(
            obs=obs,
            round_id=self.current_round,
            t=obs.turn_idx,
            graph=self.graph,
        )

    def get_determinization_constraints(self) -> list[PlugMemConstraint]:
        """Extracts active beliefs as PlugMemConstraint objects for MCTS determinization."""
        constraints: list[PlugMemConstraint] = []
        for prop in self.graph.proposition_nodes.values():
            if not prop.active or prop.confidence <= 0.0:
                continue

            if prop.category == "OPPONENT_DRAW_CONSTRAINT":
                lacked_col = prop.metadata.get("lacked_colour")
                if lacked_col:
                    constraints.append(
                        PlugMemConstraint(
                            memory_id=prop.node_id,
                            origin_turn=prop.origin_turn,
                            constraint_type="LACKS_COLOR",
                            target_key=lacked_col,
                            confidence=float(prop.confidence),
                        )
                    )
                lacked_val = prop.metadata.get("lacked_value")
                if lacked_val:
                    constraints.append(
                        PlugMemConstraint(
                            memory_id=prop.node_id,
                            origin_turn=prop.origin_turn,
                            constraint_type="LACKS_VALUE",
                            target_key=lacked_val,
                            confidence=float(prop.confidence),
                        )
                    )
            elif prop.category == "WILD_COLOUR_DECLARATION":
                decl_col = prop.metadata.get("declared_colour")
                if decl_col:
                    constraints.append(
                        PlugMemConstraint(
                            memory_id=prop.node_id,
                            origin_turn=prop.origin_turn,
                            constraint_type="PREFERRED_WILD_COLOR",
                            target_key=decl_col,
                            confidence=float(prop.confidence),
                        )
                    )

        return constraints

    def _seed_procedural_priors(self, round_id: int) -> None:
        """Inserts baseline 2-player UNO procedural skills into G^P at round start."""
        presc_1 = PrescriptionNode(
            node_id=self.graph.next_presc_id(),
            round_id=round_id,
            origin_turn=0,
            intent="retain_turn",
            prescription="Play turn-retention action cards (Skip, Reverse, +2, Wild+4) to prevent opponent active decisions.",
            return_score=9.0,
            active=True,
            provenance_epi_ids=(),
            tags=("stratum:1", "intent:retain_turn"),
        )
        self.graph.add_prescription(presc_1)

        presc_2 = PrescriptionNode(
            node_id=self.graph.next_presc_id(),
            round_id=round_id,
            origin_turn=0,
            intent="exploit_constraint",
            prescription="Exploit identified opponent lack constraints or declared wild colours to force opponent draws.",
            return_score=8.5,
            active=True,
            provenance_epi_ids=(),
            tags=("intent:exploit_constraint",),
        )
        self.graph.add_prescription(presc_2)
