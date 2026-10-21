"""Data models and graph storage for PlugMem intra-round knowledge.

Defines episodic, proposition, and prescription nodes forming the three-tier
knowledge graph (G^E, G^S, G^P), alongside concept and provenance indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class EpisodicNode:
    """Represents a single atomic decision and horizon experience in G^E."""

    node_id: str
    round_id: int
    turn: int
    state_summary: str
    inferred_subgoal: str
    stratum: int
    y_t_A: int
    y_t_B: int
    a_nop_count: int
    ordered_events: tuple[dict[str, Any], ...]
    locked_action: str


@dataclass(slots=True)
class PropositionNode:
    """Represents an atomic semantic belief, constraint, or summary fact in G^S."""

    node_id: str
    round_id: int
    origin_turn: int
    statement: str
    category: str  # "OPPONENT_DRAW_CONSTRAINT", "WILD_COLOUR_DECLARATION", "DEPLETION_ALERT", etc.
    concept_tags: tuple[str, ...]
    confidence: float
    active: bool = True
    provenance_epi_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    deactivation_turn: Optional[int] = None
    deactivation_reason: Optional[str] = None


@dataclass(slots=True)
class PrescriptionNode:
    """Represents a procedural rule, tactic, or skill recommendation in G^P."""

    node_id: str
    round_id: int
    origin_turn: int
    intent: str
    prescription: str
    return_score: float  # [1, 10]
    active: bool = True
    provenance_epi_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlugMemConstraint:
    """Per-constraint object consumable by MCTS determinization samplers."""

    memory_id: str
    origin_turn: int
    constraint_type: str  # "LACKS_COLOR", "LACKS_VALUE", "PREFERRED_WILD_COLOR"
    target_key: str  # e.g. "red" or "skip"
    confidence: float


@dataclass(frozen=True, slots=True)
class RetrievedMemoryItem:
    """Single item retrieved during pre-turn context distillation."""

    node_id: str
    text: str
    relevance_score: float
    origin_turn: int
    node_type: str  # "semantic" or "procedural"
    confidence_or_score: float
    provenance_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Consolidated retrieval output for prompt distillation and audit logging."""

    distilled_context: str
    retrieved_node_ids: tuple[str, ...]
    relevance_scores: tuple[float, ...]
    origin_turns: tuple[int, ...]
    active_opp_lack_colours: dict[str, float]
    active_opp_wild_colours: tuple[tuple[int, str], ...]
    items: tuple[RetrievedMemoryItem, ...]
    token_estimate: int


class PlugMemGraphStore:
    """In-memory storage and indexing for episodic, semantic, and procedural nodes.

    Restricted strictly to intra-round scope: all nodes and indices are cleared
    and counters reset upon starting a new round.
    """

    def __init__(self) -> None:
        self.round_id: int = 0
        self.episodic_nodes: dict[str, EpisodicNode] = {}
        self.proposition_nodes: dict[str, PropositionNode] = {}
        self.prescription_nodes: dict[str, PrescriptionNode] = {}

        self.concept_index: dict[str, set[str]] = {}  # tag -> node_ids
        self.intent_index: dict[str, set[str]] = {}  # intent -> node_ids
        self.provenance_edges: dict[str, set[str]] = {}  # node_id -> episodic_ids

        self._prop_counter: int = 0
        self._presc_counter: int = 0

    def clear(self, round_id: int) -> None:
        """Clears all graph collections and resets counters for the specified round."""
        self.round_id = round_id
        self.episodic_nodes.clear()
        self.proposition_nodes.clear()
        self.prescription_nodes.clear()
        self.concept_index.clear()
        self.intent_index.clear()
        self.provenance_edges.clear()
        self._prop_counter = 0
        self._presc_counter = 0

    def add_episodic(self, node: EpisodicNode) -> None:
        """Stores an episodic experience node."""
        self.episodic_nodes[node.node_id] = node

    def add_proposition(self, prop: PropositionNode) -> None:
        """Stores a semantic proposition and indexes its concept tags."""
        self.proposition_nodes[prop.node_id] = prop
        for tag in prop.concept_tags:
            self.concept_index.setdefault(tag, set()).add(prop.node_id)
        if prop.provenance_epi_ids:
            self.provenance_edges.setdefault(prop.node_id, set()).update(prop.provenance_epi_ids)

    def add_prescription(self, presc: PrescriptionNode) -> None:
        """Stores a procedural prescription and indexes its intent and tags."""
        self.prescription_nodes[presc.node_id] = presc
        self.intent_index.setdefault(presc.intent, set()).add(presc.node_id)
        for tag in presc.tags:
            self.concept_index.setdefault(tag, set()).add(presc.node_id)
        if presc.provenance_epi_ids:
            self.provenance_edges.setdefault(presc.node_id, set()).update(presc.provenance_epi_ids)

    def next_prop_id(self) -> str:
        """Generates the next sequential proposition identifier."""
        pid = f"prop_r{self.round_id}_{self._prop_counter}"
        self._prop_counter += 1
        return pid

    def next_presc_id(self) -> str:
        """Generates the next sequential prescription identifier."""
        mid = f"presc_r{self.round_id}_{self._presc_counter}"
        self._presc_counter += 1
        return mid
