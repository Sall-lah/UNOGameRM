"""Interleaved retrieval and context distillation for PlugMem.

Expands candidates across dense vector and concept tag bipartite channels,
reranks using the multi-component value function V, and formats distilled prompts.
"""

from __future__ import annotations

import math
from typing import Any

from src.config import MemoryConfig
from src.env.barrier import LegalObservation
from src.memory.embedding import EmbeddingInterface, cosine_similarity
from src.memory.graph import (
    PlugMemGraphStore,
    PrescriptionNode,
    PropositionNode,
    RetrievalResult,
    RetrievedMemoryItem,
)


class PlugMemRetriever:
    """Retrieves relevant semantic beliefs and procedural skills for the current turn."""

    def __init__(self, config: MemoryConfig, embedder: EmbeddingInterface) -> None:
        self.config = config
        self.embedder = embedder

    def retrieve(
        self,
        obs: LegalObservation,
        round_id: int,
        t: int,
        graph: PlugMemGraphStore,
    ) -> RetrievalResult:
        """Executes two-channel candidate expansion, value reranking, and context distillation."""
        # 1. Build Query Representation and Tags
        query_text = f"Active colour: {obs.active_colour}. Top card: {obs.top_discard}. Opponent hand count: {obs.opp_card_count}."
        query_vec = self.embedder.encode(query_text)

        query_tags: set[str] = {
            f"colour:{obs.active_colour}",
            f"top:{obs.top_discard}",
        }
        if obs.opp_card_count <= 2:
            query_tags.add("intent:prevent_risk_B")

        # 2. Filter Candidate Nodes (Must be active and origin_turn < t)
        semantic_candidates = [
            p
            for p in graph.proposition_nodes.values()
            if p.active and p.origin_turn < t and p.round_id == round_id
        ]
        procedural_candidates = [
            p
            for p in graph.prescription_nodes.values()
            if p.active and p.origin_turn < t and p.round_id == round_id
        ]

        # 3. Score and Rerank Semantic Candidates
        scored_semantics: list[tuple[float, PropositionNode]] = []
        for prop in semantic_candidates:
            score = self._compute_value_score(
                query_vec=query_vec,
                query_tags=query_tags,
                item_text=prop.statement,
                item_tags=set(prop.concept_tags),
                confidence_or_return=prop.confidence,
                origin_turn=prop.origin_turn,
                t=t,
                is_procedural=False,
            )
            scored_semantics.append((score, prop))

        scored_semantics.sort(key=lambda x: x[0], reverse=True)
        top_semantics = scored_semantics[: self.config.top_k_semantic]

        # 4. Score and Rerank Procedural Candidates
        scored_procedurals: list[tuple[float, PrescriptionNode]] = []
        for presc in procedural_candidates:
            score = self._compute_value_score(
                query_vec=query_vec,
                query_tags=query_tags,
                item_text=f"{presc.intent}: {presc.prescription}",
                item_tags=set(presc.tags),
                confidence_or_return=presc.return_score,
                origin_turn=presc.origin_turn,
                t=t,
                is_procedural=True,
            )
            scored_procedurals.append((score, presc))

        scored_procedurals.sort(key=lambda x: x[0], reverse=True)
        top_procedurals = scored_procedurals[: self.config.top_k_procedural]

        # 5. Extract Structured Constraint Summaries from active propositions
        active_opp_lack_colours: dict[str, float] = {}
        active_opp_wild_colours_list: list[tuple[int, str]] = []

        for _, prop in top_semantics:
            if prop.category == "OPPONENT_DRAW_CONSTRAINT":
                col = prop.metadata.get("lacked_colour")
                if col:
                    active_opp_lack_colours[col] = float(prop.confidence)
            elif prop.category == "WILD_COLOUR_DECLARATION":
                decl_col = prop.metadata.get("declared_colour")
                if decl_col:
                    active_opp_wild_colours_list.append((prop.origin_turn, decl_col))

        # 6. Format Items and Build Distilled Context
        items: list[RetrievedMemoryItem] = []
        retrieved_node_ids: list[str] = []
        relevance_scores: list[float] = []
        origin_turns: list[int] = []

        for score, prop in top_semantics:
            items.append(
                RetrievedMemoryItem(
                    node_id=prop.node_id,
                    text=prop.statement,
                    relevance_score=score,
                    origin_turn=prop.origin_turn,
                    node_type="semantic",
                    confidence_or_score=prop.confidence,
                    provenance_ids=tuple(prop.provenance_epi_ids),
                )
            )
            retrieved_node_ids.append(prop.node_id)
            relevance_scores.append(score)
            origin_turns.append(prop.origin_turn)

        for score, presc in top_procedurals:
            items.append(
                RetrievedMemoryItem(
                    node_id=presc.node_id,
                    text=presc.prescription,
                    relevance_score=score,
                    origin_turn=presc.origin_turn,
                    node_type="procedural",
                    confidence_or_score=presc.return_score,
                    provenance_ids=tuple(presc.provenance_epi_ids),
                )
            )
            retrieved_node_ids.append(presc.node_id)
            relevance_scores.append(score)
            origin_turns.append(presc.origin_turn)

        distilled_context, token_estimate = self._assemble_distilled_context(
            top_semantics=top_semantics,
            top_procedurals=top_procedurals,
            obs=obs,
        )

        return RetrievalResult(
            distilled_context=distilled_context,
            retrieved_node_ids=tuple(retrieved_node_ids),
            relevance_scores=tuple(relevance_scores),
            origin_turns=tuple(origin_turns),
            active_opp_lack_colours=active_opp_lack_colours,
            active_opp_wild_colours=tuple(active_opp_wild_colours_list),
            items=tuple(items),
            token_estimate=token_estimate,
        )

    def _compute_value_score(
        self,
        query_vec: Any,
        query_tags: set[str],
        item_text: str,
        item_tags: set[str],
        confidence_or_return: float,
        origin_turn: int,
        t: int,
        is_procedural: bool,
    ) -> float:
        """Computes reranking value V = w_sim*sim + w_tag*tag + w_conf*conf + w_rec*recency."""
        # Dense channel similarity
        item_vec = self.embedder.encode(item_text)
        sim = max(0.0, cosine_similarity(query_vec, item_vec))

        # Tag channel overlap
        overlap = len(query_tags & item_tags)
        tag_score = float(overlap / max(1, len(query_tags)))

        # Normalized confidence / return score in [0, 1]
        if is_procedural:
            norm_conf = max(0.0, min(1.0, (confidence_or_return - 1.0) / 9.0))
        else:
            norm_conf = max(0.0, min(1.0, confidence_or_return))

        # Recency score
        age = max(0, t - origin_turn)
        recency = math.exp(-self.config.recency_lambda * age)

        v = (
            self.config.w_sim * sim
            + self.config.w_tag * tag_score
            + self.config.w_conf * norm_conf
            + self.config.w_rec * recency
        )
        return float(v)

    @staticmethod
    def _assemble_distilled_context(
        top_semantics: list[tuple[float, PropositionNode]],
        top_procedurals: list[tuple[float, PrescriptionNode]],
        obs: LegalObservation,
    ) -> tuple[str, int]:
        """Formats distilled text block and estimates positive token count."""
        if not top_semantics and not top_procedurals:
            empty_text = (
                "[PlugMem Intra-Round Context: No intra-round opponent constraints or observations recorded yet. "
                "Defaulting to baseline heuristic exploration.]"
            )
            est = max(1, len(empty_text.split()))
            return empty_text, est

        lines: list[str] = ["[PlugMem Intra-Round Distilled Context]"]

        # 1. Verified Opponent Constraints
        lines.append("=== VERIFIED OPPONENT CONSTRAINTS ===")
        constraints = [p for _, p in top_semantics if p.category == "OPPONENT_DRAW_CONSTRAINT"]
        if constraints:
            for p in constraints:
                lines.append(f"- [{p.node_id}] {p.statement} (confidence: {p.confidence:.2f})")
        else:
            lines.append("- No active colour/value lack constraints currently confirmed.")

        # 2. Opponent Threat Signals
        lines.append("=== ACTIVE OPPONENT THREAT SIGNALS ===")
        threats = [p for _, p in top_semantics if p.category != "OPPONENT_DRAW_CONSTRAINT"]
        if threats:
            for p in threats:
                lines.append(f"- [{p.node_id}] {p.statement} (confidence: {p.confidence:.2f})")
        else:
            lines.append(f"- Opponent card count: {obs.opp_card_count}. No severe depletion alerts.")

        # 3. Tactical Procedural Prescriptions
        lines.append("=== TACTICAL PROCEDURAL PRESCRIPTIONS ===")
        if top_procedurals:
            for _, presc in top_procedurals:
                lines.append(
                    f"- [{presc.node_id}] Intent '{presc.intent}': {presc.prescription} (return: {presc.return_score:.1f}/10)"
                )
        else:
            lines.append("- Apply standard legal playability and color alignment.")

        distilled = "\n".join(lines)
        token_estimate = max(1, int(len(distilled.split()) * 1.3))
        return distilled, token_estimate
