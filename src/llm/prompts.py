"""Prompt templates for LLM Call #1 (Action Pruning & Prior) and Call #2 (Risk Elicitation).

Renders prompt strings adhering verbatim to docs/05 specifications, formatting public
observations, PlugMem distilled context, and MCTS statistics using canonical UNO.py encodings.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence, Union

from src.env.barrier import LegalObservation


CALL1_SYSTEM_PROMPT: str = (
    "You are an expert 2-player UNO strategist operating under 2-player non-stacking rules:\n"
    "- Skip, Reverse, Draw Two (+2), and Wild Draw 4 (+4) immediately skip the opponent's turn and return the turn to you.\n"
    "- Penalty cards (+2, +4) cannot be stacked."
)

CALL1_USER_TEMPLATE: str = (
    "=== CURRENT LEGAL OBSERVATION (o_t) ===\n"
    "Round ID: {round_id} | Decision Index (t): {turn_idx}\n"
    "Your Hand ({agent_hand_count} cards): {agent_hand}\n"
    "Top Discard Card: {top_discard} | Active Color: {active_colour}\n"
    "Opponent Hand Size: {opp_card_count} cards\n"
    "Legal Actions Available (A_t): {legal_actions}\n\n"
    "=== INTRA-ROUND MEMORY FROM PLUGMEM (Exp_t) ===\n"
    "{plugmem_distilled_context}\n\n"
    "=== TASK ===\n"
    "1. Evaluate the legal actions in A_t using your hand composition, opponent hand size, and PlugMem constraints.\n"
    "2. Prune clearly dominated actions (keep at least 1 and at most {max_candidates} actions; if |A_t| <= 2, keep all).\n"
    "3. Assign a prior probability distribution P_prior over the kept actions that sums to 1.0.\n\n"
    "Return STRICT JSON ONLY matching the schema below."
)

CALL2_SYSTEM_PROMPT: str = (
    "You are an analytical risk-calibration module for a 2-player non-stacking UNO agent.\n"
    "The agent has ALREADY LOCKED its action a_t for turn t, and this action is about to be sent to the server."
)

CALL2_USER_TEMPLATE: str = (
    "=== DEFINITIONS OF HORIZON H_t AND RISKS ===\n"
    "- Observation Horizon H_t = [t, t+1): The interval starting immediately after locked action a_t is executed on the server until right before your next decision opportunity at t+1, or the end of the round, whichever comes first.\n"
    "- Note on 2-Player Turn Retention: If your locked action a_t is a Skip (\"<colour> skip\"), Reverse (\"<colour> reverse\"), Draw Two (\"<colour> + 2\"), or Wild Draw 4 (\"WILD + 4 card, set to <colour>\"), or if a_t plays your final card, the opponent gets 0 active turns in H_t (Stratum 1).\n"
    "- Risk A (y_{{t,A}} / A_pen): Probability (0.00 to 1.00) that you suffer a FORCED PENALTY DRAW (from an opponent playing +2 or Wild +4) during H_t. (Voluntary 1-card draws A_nop do NOT count as Risk A).\n"
    "- Risk B (y_{{t,B}}): Probability (0.00 to 1.00) that the opponent empties their entire hand and WINS THE ROUND during H_t.\n\n"
    "=== INPUT STATE & LOCKED DECISION ===\n"
    "Your Current Hand ({agent_hand_count} cards): {agent_hand}\n"
    "Top Discard Card: {top_discard} | Active Color: {active_colour}\n"
    "Opponent Hand Size BEFORE a_t: {opp_card_count} cards\n"
    "PlugMem Opponent Constraints (Exp_t): {plugmem_distilled_context}\n"
    "MCTS Search Statistics (N visits, Q value per candidate): {mcts_stats_summary}\n"
    "LOCKED ACTION (a_t): {locked_action}\n\n"
    "=== TASK ===\n"
    "Estimate the conditional probabilities p_{{t,A}} and p_{{t,B}} in [0.0, 1.0] given that locked action a_t is executed, and provide a concise rationale (1-3 sentences) referencing the observation (hand/opponent card count/action type), PlugMem constraints, and MCTS statistics.\n\n"
    "Return STRICT JSON ONLY:\n"
    "{{\n"
    '  "p_t_A": <float between 0.0 and 1.0>,\n'
    '  "p_t_B": <float between 0.0 and 1.0>,\n'
    '  "expected_stratum": <int 1, 2, or 3>,\n'
    '  "short_rationale": "<1-3 concise sentences grounding p_t_A and p_t_B in o_t, Exp_t, and MCTS (N, Q)>"\n'
    "}}"
)


def format_mcts_stats_summary(mcts_stats: Any) -> str:
    """Formats MCTS visit counts and Q values into a clean string for LLM Call #2.

    Why: The prompt requires candidate-level search statistics so Call #2 can ground
    its probability estimates in empirical lookahead simulations.
    """
    if isinstance(mcts_stats, str):
        return mcts_stats

    # Check for SearchResult dataclass
    if hasattr(mcts_stats, "visit_counts_N") and hasattr(mcts_stats, "action_values_Q"):
        entries = []
        for a, n in mcts_stats.visit_counts_N.items():
            q = mcts_stats.action_values_Q.get(a, 0.0)
            entries.append(f"{a}: N={n}, Q={q:.3f}")
        return ", ".join(entries) if entries else "None"

    # Check for dict
    if isinstance(mcts_stats, Mapping):
        entries = []
        for a, val in mcts_stats.items():
            if isinstance(val, tuple) and len(val) == 2:
                entries.append(f"{a}: N={val[0]}, Q={val[1]:.3f}")
            else:
                entries.append(f"{a}: {val}")
        return ", ".join(entries) if entries else "None"

    return str(mcts_stats)


def render_call1_prompts(
    obs: LegalObservation,
    distilled_context: str,
    max_candidates: int = 4,
) -> tuple[str, str]:
    """Renders (system_prompt, user_prompt) for LLM Call #1 (Action Pruning & Prior).

    Why: Populates the prompt strictly from public LegalObservation and PlugMem context
    without leaking private server state.
    """
    hand_repr = list(obs.agent_hand)
    legal_repr = list(obs.legal_actions)
    context_repr = distilled_context if distilled_context.strip() else "(No previous history)"

    user_prompt = CALL1_USER_TEMPLATE.format(
        round_id=obs.round_id,
        turn_idx=obs.turn_idx,
        agent_hand_count=len(obs.agent_hand),
        agent_hand=json.dumps(hand_repr),
        top_discard=obs.top_discard,
        active_colour=obs.active_colour,
        opp_card_count=obs.opp_card_count,
        legal_actions=json.dumps(legal_repr),
        plugmem_distilled_context=context_repr,
        max_candidates=max_candidates,
    )
    return CALL1_SYSTEM_PROMPT, user_prompt


def render_call2_prompts(
    obs: LegalObservation,
    distilled_context: str,
    locked_action: str,
    mcts_stats: Any,
) -> tuple[str, str]:
    """Renders (system_prompt, user_prompt) for LLM Call #2 (Risk Elicitation & Rationale).

    Why: Call #2 is executed after action commitment to estimate conditional adverse risks
    p_{t,A} and p_{t,B} given that locked_action is about to be played.
    """
    hand_repr = list(obs.agent_hand)
    context_repr = distilled_context if distilled_context.strip() else "(No previous history)"
    stats_summary = format_mcts_stats_summary(mcts_stats)

    user_prompt = CALL2_USER_TEMPLATE.format(
        agent_hand_count=len(obs.agent_hand),
        agent_hand=json.dumps(hand_repr),
        top_discard=obs.top_discard,
        active_colour=obs.active_colour,
        opp_card_count=obs.opp_card_count,
        plugmem_distilled_context=context_repr,
        mcts_stats_summary=stats_summary,
        locked_action=locked_action,
    )
    return CALL2_SYSTEM_PROMPT, user_prompt
