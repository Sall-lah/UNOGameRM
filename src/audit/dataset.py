"""DecisionRecord schema (d_{r,t}) and JSONL dataset persistence for Zone 3.

Defines the complete eight-group Pydantic v2 clustered decision record model matching
docs/06 §3, enforcing British 'colour' keys, UNO.py string card representations,
and atomic line-by-line JSONL serialization for pilot and main experiments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class IdentityGroup(BaseModel):
    """Decision round, step indexing, starting player, and engine configuration version."""

    model_config = ConfigDict(extra="forbid")

    round_id: int
    turn_idx: int
    starting_player: str
    config_version: str = "v1.0-frozen"
    first_turn_wild_colour: bool = False


class ObservationOtGroup(BaseModel):
    """Sanitized public observation exported to the agent at decision step t."""

    model_config = ConfigDict(extra="forbid")

    agent_hand: list[str]
    agent_hand_count: int
    action_wild_count: int
    top_discard: str
    active_colour: str
    opp_card_count: int
    legal_actions: list[str]


class MemoryPlugMemGroup(BaseModel):
    """PlugMem intra-round retrieved context, belief constraints, and token metrics."""

    model_config = ConfigDict(extra="forbid")

    retrieved_node_ids: list[str]
    opponent_draw_constraints: list[dict[str, Any]]
    opponent_wild_declarations: list[dict[str, Any]]
    relevance_scores: list[float]
    distilled_context_tokens: int

    @field_validator("opponent_draw_constraints", "opponent_wild_declarations", mode="before")
    @classmethod
    def normalise_british_colour_keys(cls, v: Any) -> Any:
        """Normalises American 'color' keys to canonical British 'colour'."""
        if not isinstance(v, list):
            return v
        normalised_list = []
        for item in v:
            if isinstance(item, dict):
                clean_item = dict(item)
                if "color" in clean_item and "colour" not in clean_item:
                    clean_item["colour"] = clean_item.pop("color")
                normalised_list.append(clean_item)
            else:
                normalised_list.append(item)
        return normalised_list


class SearchMCTSGroup(BaseModel):
    """MARS Budget-Aware MCTS lookahead statistics and candidate distributions."""

    model_config = ConfigDict(extra="forbid")

    pruned_candidates: list[str]
    p_prior: dict[str, float]
    simulations_N_total: int
    visit_counts_N: dict[str, int]
    action_values_Q: dict[str, float]
    unseen_pool_size_Ut: int
    search_wall_time_ms: float


class Stage1FreezeLogGroup(BaseModel):
    """Tamper-evident pre-execution prediction frozen before server dispatch."""

    model_config = ConfigDict(extra="forbid")

    locked_action_at: str
    p_t_A: float
    p_t_B: float
    short_rationale: str
    freeze_timestamp_ns: int
    stage1_sha256: str


class Stage2PostHorizonGroup(BaseModel):
    """Observed outcome labels, turn progression, and strata derived post-horizon."""

    model_config = ConfigDict(extra="forbid")

    dispatched_command: str
    auto_uno_declared: bool
    dispatch_timestamp_ns: int
    eval_timestamp_ns: int
    y_t_A_penalty_draw: int
    A_pen_cards_drawn: int
    A_nop_voluntary_drawn: int
    y_t_B_opp_win: int
    opp_turns_in_horizon: int
    stratum_horizon: int
    round_status: str
    challenge_penalty: int = 0


class GameResourcesGroup(BaseModel):
    """Card economy trajectory and inventory dynamics tracked across horizons."""

    model_config = ConfigDict(extra="forbid")

    delta_hand_size_next: int
    agent_hand_count_next: int
    used_action_or_wild_at_t: int
    action_wild_count_next: int
    cumulative_cards_drawn_round: int
    act_opportunities_so_far: int


class SystemReliabilityGroup(BaseModel):
    """Turn-level latency, budget overruns, and component fallback flags."""

    model_config = ConfigDict(extra="forbid")

    turn_total_time_ms: float
    budget_limit_ms: float
    timeout_exceeded: bool
    mcts_fallback_used: bool
    llm_call1_invalid_json: bool
    llm_call2_invalid_json: bool
    dispatch_error: bool


class DecisionRecord(BaseModel):
    """Complete clustered decision record d_{r,t} comprising all eight data groups.

    Why: Serves as the immutable analytical contract for evaluation (RQ1-RQ3),
    statistical verification, and audit trace validation.
    """

    model_config = ConfigDict(extra="forbid")

    identity: IdentityGroup
    observation_ot: ObservationOtGroup
    memory_plugmem: MemoryPlugMemGroup
    search_mcts: SearchMCTSGroup
    stage1_freeze_log: Stage1FreezeLogGroup
    stage2_post_horizon: Stage2PostHorizonGroup
    game_resources: GameResourcesGroup
    system_reliability: SystemReliabilityGroup


class DatasetWriter:
    """Manages atomic JSONL persistence of DecisionRecords routed by experiment split."""

    def __init__(
        self,
        output_root: str = "data",
        dataset_split: Literal["pilot", "main", "main_experiment"] = "pilot",
        target_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initializes dataset directory and target JSONL filepath.

        Why: Segregates pilot calibration runs from main experiment data to prevent
        cross-contamination of evaluation samples.
        """
        self.output_root = output_root
        if target_dir is not None:
            self.target_dir = Path(target_dir)
        else:
            subfolder = "main_experiment" if dataset_split in ("main", "main_experiment") else "pilot"
            self.target_dir = Path(output_root) / subfolder
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.target_dir / "dataset_drt.jsonl"

    def append(self, record: DecisionRecord) -> None:
        """Appends one validated DecisionRecord as a single JSON line with fsync.

        Why: Line-by-line fsync ensures that if a run crashes or is killed, all preceding
        decision records remain durably intact and parseable on disk.
        """
        json_line = record.model_dump_json() + "\n"
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json_line)
            f.flush()
            os.fsync(f.fileno())

    def read_all(self) -> list[DecisionRecord]:
        """Reads and validates all DecisionRecords from the current JSONL file."""
        if not self.file_path.exists():
            return []

        records: list[DecisionRecord] = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    data = json.loads(stripped)
                    records.append(DecisionRecord.model_validate(data))
        return records
