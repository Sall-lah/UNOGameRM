"""MARS Budget-Aware Imperfect-Information Monte Carlo Tree Search.

This module implements the lookahead search component (Zone 2) of the budget-aware agent.
It executes prior-guided UCT selection, open-loop re-determinization from public unseen pools,
latency-modulated rollout rewards, most-visited action commitment, and deterministic fallbacks.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

from src.config import MCTSConfig
from src.core.cards import (
    COLOURS,
    DRAW_TWO,
    REVERSE,
    SKIP,
    UNDERLINE_COMBINING,
    WILD,
    WILD_DRAW_FOUR,
    _WILD_STRING_PATTERN,
    UNOAction,
    UNOCard,
    parse_card_string,
)
from src.env.barrier import LegalObservation, compute_unseen_pool
from src.memory.graph import PlugMemConstraint
from src.search.determinization import sample_determinization
from src.search.forward_model import (
    SimulatedState,
    create_simulated_state_from_observation,
)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Consolidated MCTS output matching the search_mcts audit schema and LLM Call #2 inputs.

    Why: Downstream audit loggers and risk evaluation calls require visit counts, action
    values, and rollout diagnostics keyed by canonical action_id strings.
    """

    visit_counts_N: dict[str, int]
    action_values_Q: dict[str, float]
    simulations_N_total: int
    unseen_pool_size_Ut: int
    search_wall_time_ms: float
    fallback_used: bool
    locked_action: UNOAction
    visit_shares: dict[str, float]
    mean_raw_utilities: dict[str, float]
    rollout_apen_rates: dict[str, float]
    rollout_opp_win_rates: dict[str, float]


def action_from_id(action_id: str) -> UNOAction:
    """Reconstructs a canonical UNOAction from its unique string identifier.

    Why: The barrier and audit logs serialize actions as strings; search needs to convert
    these back into typed domain actions without loss of declared colours or card metadata.
    """
    if action_id in ("draw", "challenge"):
        action_type = "DRAW" if action_id == "draw" else "D4_RESPONSE"
        return UNOAction(action_id=action_id, action_type=action_type, card=None, declared_colour=None)

    wild_match = _WILD_STRING_PATTERN.match(action_id)
    if wild_match:
        is_d4 = wild_match.group(1) is not None
        colour = wild_match.group(2)
        val = WILD_DRAW_FOUR if is_d4 else WILD
        card = UNOCard(type="WIL", colour="white", value=val)
        return UNOAction(
            action_id=action_id,
            action_type="PLAY",
            card=card,
            declared_colour=colour,
        )

    card = parse_card_string(action_id)
    return UNOAction(
        action_id=action_id,
        action_type="PLAY",
        card=card,
        declared_colour=None,
    )


def get_card_point_value(card: UNOCard) -> int:
    """Calculates official UNO point value for non-wild card load-shedding heuristics.

    Why: In fallback scenarios, playing high-point cards shedding inventory risk is preferred.
    """
    if card.type == "NUM":
        clean_val = card.value.replace(UNDERLINE_COMBINING, "")
        try:
            return int(clean_val)
        except ValueError:
            return 0
    if card.type == "ACT":
        return 20
    if card.type == "WIL":
        return 50
    return 0


class MARSBudgetAwareMCTS:
    """Implements budget-aware imperfect-information Monte Carlo Tree Search.

    Operates strictly within public observation boundaries using open-loop
    determinization and efficiency-modulated utility backpropagation.
    """

    def __init__(
        self,
        config: Optional[MCTSConfig] = None,
        rng: Optional[Any] = None,
        timer: Optional[Any] = None,
    ) -> None:
        """Initializes MCTS with hyper-parameters and an injected random number generator.

        Why: Strict reproducibility demands that all random operations stem from an
        injected seeded generator rather than global module random state. Injected timer
        enables deterministic testing of latency-modulated reward backpropagation.
        """
        self.config = config or MCTSConfig()
        self.rng = rng if rng is not None else random.Random()
        self.timer = timer if timer is not None else time.perf_counter

    def search(
        self,
        obs: LegalObservation,
        priors: Optional[dict[str, float]] = None,
        memory_constraints: Optional[Union[Sequence[PlugMemConstraint], Any]] = None,
        deadline: Optional[float] = None,
    ) -> SearchResult:
        """Executes budget-bounded MCTS search and returns committed action and search stats.

        Why: Coordinates determinization sampling, prior-guided UCT selection, depth-limited
        rollouts, and tie-broken action commitment within strict wall-clock deadlines.
        """
        start_time = self.timer()
        candidates = list(obs.legal_actions)

        # Degenerate Case: No legal actions available (failsafe to draw)
        if not candidates:
            draw_action = action_from_id("draw")
            return self._build_empty_result(
                locked_action=draw_action,
                candidates=["draw"],
                unseen_pool_size=len(compute_unseen_pool(obs)),
                fallback_used=True,
                elapsed_ms=(self.timer() - start_time) * 1000.0,
            )

        # Immediate Fallback Check: Deadline already exceeded on entry
        if deadline is not None and start_time >= deadline:
            fallback_action = self._deterministic_fallback(obs, candidates)
            return self._build_empty_result(
                locked_action=fallback_action,
                candidates=candidates,
                unseen_pool_size=len(compute_unseen_pool(obs)),
                fallback_used=True,
                elapsed_ms=(self.timer() - start_time) * 1000.0,
            )

        # Optimization: Single legal action commits immediately without burning simulation budget
        if len(candidates) == 1:
            single_action_id = candidates[0]
            locked_action = action_from_id(single_action_id)
            return self._build_empty_result(
                locked_action=locked_action,
                candidates=candidates,
                unseen_pool_size=len(compute_unseen_pool(obs)),
                fallback_used=False,
                elapsed_ms=(self.timer() - start_time) * 1000.0,
            )

        # Normalize prior distribution across candidate actions
        normalized_priors = self._normalize_priors(candidates, priors)

        # Initialize search tracking statistics
        visit_counts: dict[str, int] = {a: 0 for a in candidates}
        action_values_Q: dict[str, float] = {a: 0.0 for a in candidates}
        raw_utilities_sum: dict[str, float] = {a: 0.0 for a in candidates}
        apen_counts: dict[str, int] = {a: 0 for a in candidates}
        opp_win_counts: dict[str, int] = {a: 0 for a in candidates}

        simulations_N_total = 0
        m_min: Optional[float] = None
        m_max: Optional[float] = None
        last_unseen_pool_size = len(compute_unseen_pool(obs))

        # Main simulation loop bounded by K_max and wall-clock deadline
        while simulations_N_total < self.config.max_simulations:
            # Deadline check before starting a new simulation
            if deadline is not None and self.timer() >= deadline:
                break

            # 1. Determinization: Sample concrete hidden world from U_t
            det = sample_determinization(obs, memory_constraints, rng=self.rng)
            last_unseen_pool_size = det.unseen_pool_size_Ut
            root_state = create_simulated_state_from_observation(obs, det.opp_hand, det.draw_pile)

            # 2. Selection: Select root candidate using prior-guided UCT
            selected_action_id = self._select_root_action(
                candidates=candidates,
                visit_counts=visit_counts,
                action_values_Q=action_values_Q,
                priors=normalized_priors,
                simulations_N_total=simulations_N_total,
            )

            # 3. Expansion & Step: Apply root candidate
            sim_state = root_state.clone()
            root_action_obj = action_from_id(selected_action_id)

            t_sim_start = self.timer()
            sim_state.step(root_action_obj)

            # 4. Rollout: Run fast heuristic rollout to max depth or terminal
            self._run_rollout(sim_state, max_depth=self.config.max_rollout_depth)
            t_sim_elapsed = self.timer() - t_sim_start

            # 5. Reward: Calculate raw utility M(v_l)
            m_raw = self._calculate_raw_utility(sim_state)

            if m_min is None or m_raw < m_min:
                m_min = m_raw
            if m_max is None or m_raw > m_max:
                m_max = m_raw

            # Normalization G
            if m_max == m_min:
                norm_g = 0.5
            else:
                norm_g = (m_raw - m_min) / (m_max - m_min)

            # Latency modulation: R = G * [max(t_sim, eps) / L] ^ w
            eps = 1e-5
            latency_ratio = max(t_sim_elapsed, eps) / self.config.target_sim_latency_s
            latency_factor = latency_ratio ** self.config.penalty_weight_w
            reward_r = norm_g * latency_factor

            # 6. Backpropagation: Update running stats for selected root action
            simulations_N_total += 1
            visit_counts[selected_action_id] += 1
            n_a = visit_counts[selected_action_id]
            action_values_Q[selected_action_id] += (reward_r - action_values_Q[selected_action_id]) / n_a

            raw_utilities_sum[selected_action_id] += m_raw
            if sim_state.agent_suffered_a_pen:
                apen_counts[selected_action_id] += 1
            if sim_state.winner == "opponent" or (sim_state.is_terminal() and len(sim_state.opp_hand) == 0):
                opp_win_counts[selected_action_id] += 1

        elapsed_ms = (self.timer() - start_time) * 1000.0

        # Commitment Check: If search could not achieve >= 1 visit per candidate, trigger fallback
        unvisited_exist = any(visit_counts[a] == 0 for a in candidates)
        if simulations_N_total < len(candidates) or unvisited_exist:
            fallback_action = self._deterministic_fallback(obs, candidates)
            return self._build_result(
                locked_action=fallback_action,
                candidates=candidates,
                visit_counts=visit_counts,
                action_values_Q=action_values_Q,
                raw_utilities_sum=raw_utilities_sum,
                apen_counts=apen_counts,
                opp_win_counts=opp_win_counts,
                simulations_N_total=simulations_N_total,
                unseen_pool_size=last_unseen_pool_size,
                fallback_used=True,
                elapsed_ms=elapsed_ms,
            )

        # Most-Visited Action Commitment with deterministic tie-breaks
        locked_action_id = self._commit_most_visited_action(
            candidates=candidates,
            visit_counts=visit_counts,
            action_values_Q=action_values_Q,
            priors=normalized_priors,
        )
        locked_action = action_from_id(locked_action_id)

        return self._build_result(
            locked_action=locked_action,
            candidates=candidates,
            visit_counts=visit_counts,
            action_values_Q=action_values_Q,
            raw_utilities_sum=raw_utilities_sum,
            apen_counts=apen_counts,
            opp_win_counts=opp_win_counts,
            simulations_N_total=simulations_N_total,
            unseen_pool_size=last_unseen_pool_size,
            fallback_used=False,
            elapsed_ms=elapsed_ms,
        )

    def _normalize_priors(
        self,
        candidates: Sequence[str],
        priors: Optional[dict[str, float]],
    ) -> dict[str, float]:
        """Normalizes prior distribution over candidate actions, defaulting missing to uniform.

        Why: Ensures UCT exploration weights are non-zero and sum properly to 1.0.
        """
        uniform_val = 1.0 / len(candidates)
        raw_priors: dict[str, float] = {}

        for a in candidates:
            if priors is not None and a in priors and priors[a] is not None:
                raw_priors[a] = max(0.0, float(priors[a]))
            else:
                raw_priors[a] = uniform_val

        total = sum(raw_priors.values())
        if total > 0.0:
            return {a: raw_priors[a] / total for a in candidates}
        return {a: uniform_val for a in candidates}

    def _select_root_action(
        self,
        candidates: Sequence[str],
        visit_counts: dict[str, int],
        action_values_Q: dict[str, float],
        priors: dict[str, float],
        simulations_N_total: int,
    ) -> str:
        """Selects root action: unvisited first by descending prior, then via UCT.

        Why: Ensures full initial exploration breadth before UCB exploitation begins.
        """
        unvisited = [a for a in candidates if visit_counts[a] == 0]
        if unvisited:
            # Unvisited expanded first in descending order of P_prior; ties broken by lexicographic order
            return min(unvisited, key=lambda a: (-priors[a], a))

        # All candidates visited: maximize UCT formula
        def uct_sort_key(a: str) -> tuple[float, float, float, str]:
            n_a = visit_counts[a]
            q_a = action_values_Q[a]
            p_a = priors[a]
            uct_bonus = self.config.c_uct * p_a * math.sqrt(math.log(simulations_N_total + 1)) / (1.0 + n_a)
            score = q_a + uct_bonus
            # Higher score, higher Q, higher P, earlier lexicographic name
            return (-score, -q_a, -p_a, a)

        return min(candidates, key=uct_sort_key)

    def _run_rollout(self, state: SimulatedState, max_depth: int) -> None:
        """Runs fast heuristic rollout policy reading only simulated state.

        Why: Rollout policy approximates UNO play (prefer plays over draw, non-Wild over Wild)
        without referencing the hidden true state or the upstream AI opponent logic.
        """
        depth = 0
        while not state.is_terminal() and depth < max_depth:
            legal_actions = state.get_legal_actions()
            if not legal_actions:
                break

            # Filter non-draw plays
            plays = [act for act in legal_actions if act.action_id != "draw"]
            if not plays:
                # Must draw
                draw_act = next((act for act in legal_actions if act.action_id == "draw"), None)
                if draw_act:
                    state.step(draw_act)
                break

            # Distinguish non-Wild from Wild plays
            non_wild_plays = [act for act in plays if act.card is not None and not act.card.is_wild]
            if non_wild_plays:
                # Prefer playing non-Wild cards to conserve Wilds; pick highest point value
                chosen = max(non_wild_plays, key=lambda act: (get_card_point_value(act.card), act.action_id))
            else:
                # Only Wilds available: declare most frequent hand colour
                hand = state.agent_hand if state.current_player == "agent" else state.opp_hand
                colour_counts = {c: 0 for c in COLOURS}
                for c in hand:
                    if not c.is_wild and c.colour in colour_counts:
                        colour_counts[c.colour] += 1
                best_colour = min(COLOURS, key=lambda c: (-colour_counts[c], c))

                matching_wild = [act for act in plays if act.declared_colour == best_colour]
                if matching_wild:
                    chosen = matching_wild[0]
                else:
                    chosen = plays[0]

            state.step(chosen)
            depth += 1

    def _calculate_raw_utility(self, state: SimulatedState) -> float:
        """Calculates raw terminal/horizon rollout utility M(v_l).

        Why: Implements doc 04 §5.3 formula: win/loss term, hand differential alpha1,
        held resource conservation alpha2, and penalty aversion alpha3.
        """
        agent_wins = 1.0 if state.winner == "agent" or len(state.agent_hand) == 0 else 0.0
        opp_wins = 1.0 if state.winner == "opponent" or (state.is_terminal() and len(state.opp_hand) == 0) else 0.0

        hand_diff = (len(state.opp_hand) - len(state.agent_hand)) / 15.0
        action_or_wild_count = sum(1 for c in state.agent_hand if c.is_action_or_wild)
        resource_term = action_or_wild_count / 8.0
        pen_term = 1.0 if state.agent_suffered_a_pen else 0.0

        return (
            agent_wins * 1.0
            - opp_wins * 1.0
            + self.config.alpha1 * hand_diff
            + self.config.alpha2 * resource_term
            - self.config.alpha3 * pen_term
        )

    def _commit_most_visited_action(
        self,
        candidates: Sequence[str],
        visit_counts: dict[str, int],
        action_values_Q: dict[str, float],
        priors: dict[str, float],
    ) -> str:
        """Selects action with maximum visits, resolving ties via Q, prior, and lexicographic id.

        Why: Most-visited action commitment is robust to non-stationary normalization shifts in Q.
        """
        def commit_key(a: str) -> tuple[int, float, float, str]:
            # Sort order: max N, max Q, max Prior, min action_id (lexicographic)
            return (-visit_counts[a], -action_values_Q[a], -priors[a], a)

        return min(candidates, key=commit_key)

    def _deterministic_fallback(
        self,
        obs: LegalObservation,
        candidates: Sequence[str],
    ) -> UNOAction:
        """Selects a legal action using deterministic priority rules when search cannot run.

        Priority:
        1. Legal non-Wild action matching active colour, preferring highest point value.
        2. Legal Wild action declaring the most frequent colour in agent's hand.
        3. Voluntary draw.
        """
        # Priority 1: Legal non-Wild card matching active colour
        colour_matching_non_wild: list[tuple[int, str]] = []
        for a in candidates:
            if a == "draw" or a.startswith("WILD"):
                continue
            card = parse_card_string(a)
            if card.colour == obs.active_colour:
                pts = get_card_point_value(card)
                colour_matching_non_wild.append((pts, a))

        if colour_matching_non_wild:
            # Highest point value; ties broken lexicographically
            best_id = min(colour_matching_non_wild, key=lambda item: (-item[0], item[1]))[1]
            return action_from_id(best_id)

        # Priority 2: Legal Wild declaring most frequent hand colour
        colour_counts = {c: 0 for c in COLOURS}
        for card_str in obs.agent_hand:
            card = parse_card_string(card_str)
            if not card.is_wild and card.colour in colour_counts:
                colour_counts[card.colour] += 1
        most_freq_colour = min(COLOURS, key=lambda c: (-colour_counts[c], c))

        matching_wilds = [
            a for a in candidates if a.startswith("WILD") and f"set to {most_freq_colour}" in a
        ]
        if matching_wilds:
            # Prefer regular Wild over +4 if both present to conserve +4
            regular_wild = next((a for a in matching_wilds if "+ 4" not in a), None)
            best_wild = regular_wild or matching_wilds[0]
            return action_from_id(best_wild)

        any_wilds = [a for a in candidates if a.startswith("WILD")]
        if any_wilds:
            any_wilds.sort()
            return action_from_id(any_wilds[0])

        # Priority 3: Voluntary draw action
        if "draw" in candidates:
            return action_from_id("draw")

        # Failsafe: first legal candidate
        return action_from_id(candidates[0])

    def _build_empty_result(
        self,
        locked_action: UNOAction,
        candidates: Sequence[str],
        unseen_pool_size: int,
        fallback_used: bool,
        elapsed_ms: float,
    ) -> SearchResult:
        """Builds a zero-simulation SearchResult for single-action or immediate fallback triggers."""
        visit_counts = {a: 0 for a in candidates}
        action_values = {a: 0.0 for a in candidates}
        shares = {a: (1.0 if a == locked_action.action_id else 0.0) for a in candidates}

        return SearchResult(
            visit_counts_N=visit_counts,
            action_values_Q=action_values,
            simulations_N_total=0,
            unseen_pool_size_Ut=unseen_pool_size,
            search_wall_time_ms=elapsed_ms,
            fallback_used=fallback_used,
            locked_action=locked_action,
            visit_shares=shares,
            mean_raw_utilities={a: 0.0 for a in candidates},
            rollout_apen_rates={a: 0.0 for a in candidates},
            rollout_opp_win_rates={a: 0.0 for a in candidates},
        )

    def _build_result(
        self,
        locked_action: UNOAction,
        candidates: Sequence[str],
        visit_counts: dict[str, int],
        action_values_Q: dict[str, float],
        raw_utilities_sum: dict[str, float],
        apen_counts: dict[str, int],
        opp_win_counts: dict[str, int],
        simulations_N_total: int,
        unseen_pool_size: int,
        fallback_used: bool,
        elapsed_ms: float,
    ) -> SearchResult:
        """Assembles fully populated SearchResult with per-candidate diagnostic stats."""
        n_total = max(1, simulations_N_total)
        shares = {a: visit_counts[a] / n_total for a in candidates}
        mean_utilities = {
            a: (raw_utilities_sum[a] / visit_counts[a]) if visit_counts[a] > 0 else 0.0
            for a in candidates
        }
        apen_rates = {
            a: (apen_counts[a] / visit_counts[a]) if visit_counts[a] > 0 else 0.0
            for a in candidates
        }
        opp_win_rates = {
            a: (opp_win_counts[a] / visit_counts[a]) if visit_counts[a] > 0 else 0.0
            for a in candidates
        }

        return SearchResult(
            visit_counts_N=visit_counts,
            action_values_Q=action_values_Q,
            simulations_N_total=simulations_N_total,
            unseen_pool_size_Ut=unseen_pool_size,
            search_wall_time_ms=elapsed_ms,
            fallback_used=fallback_used,
            locked_action=locked_action,
            visit_shares=shares,
            mean_raw_utilities=mean_utilities,
            rollout_apen_rates=apen_rates,
            rollout_opp_win_rates=opp_win_rates,
        )
