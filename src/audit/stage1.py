"""Stage 1 Pre-Execution Freeze Record implementation.

Serializes the agent's pre-execution state, priors, lookahead statistics, and risk
predictions into canonical JSON, computes a SHA-256 integrity hash, stamps monotonic
nanosecond timestamps, and durably persists (fsync) the record strictly before action dispatch.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union


@dataclass(frozen=True, slots=True)
class Stage1FreezeResult:
    """Contains frozen bytes, cryptographic digest, and monotonic timestamp for Stage 1.

    Why: Provides unforgeable proof that the model's adverse risk estimates were
    committed and persisted to disk prior to any server action execution.
    """

    canonical_bytes: bytes
    stage1_sha256: str
    freeze_timestamp_ns: int
    file_path: Path


def serialize_stage1_payload(
    round_id: int,
    turn_idx: int,
    observation_ot: dict[str, Any],
    memory_plugmem: dict[str, Any],
    p_prior: dict[str, float],
    visit_counts_N: dict[str, int],
    action_values_Q: dict[str, float],
    locked_action_at: str,
    p_t_A: float,
    p_t_B: float,
    short_rationale: str,
) -> bytes:
    """Produces canonical JSON bytes for Stage 1 freeze verification.

    Why: Exact key sorting, compact delimiters (',', ':'), and UTF-8 encoding ensure
    deterministic, byte-identical serialization across different runtime environments.
    """
    payload = {
        "round_id": round_id,
        "turn_idx": turn_idx,
        "observation_ot": observation_ot,
        "memory_plugmem": memory_plugmem,
        "p_prior": p_prior,
        "visit_counts_N": visit_counts_N,
        "action_values_Q": action_values_Q,
        "locked_action_at": locked_action_at,
        "p_t_A": round(p_t_A, 6),
        "p_t_B": round(p_t_B, 6),
        "short_rationale": short_rationale,
    }
    canonical_str = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return canonical_str.encode("utf-8")


class Stage1Writer:
    """Handles durable disk persistence and hashing for Stage 1 freeze records."""

    def __init__(self, freeze_dir: Union[str, Path] = "data/stage1_freeze") -> None:
        """Configures freeze target directory and ensures parent paths exist."""
        self.freeze_dir = Path(freeze_dir)
        self.freeze_dir.mkdir(parents=True, exist_ok=True)

    def freeze(
        self,
        round_id: int,
        turn_idx: int,
        observation_ot: dict[str, Any],
        memory_plugmem: dict[str, Any],
        p_prior: dict[str, float],
        visit_counts_N: dict[str, int],
        action_values_Q: dict[str, float],
        locked_action_at: str,
        p_t_A: float,
        p_t_B: float,
        short_rationale: str,
        custom_file_path: Optional[Union[str, Path]] = None,
    ) -> Stage1FreezeResult:
        """Serializes, hashes, and fsyncs the Stage 1 record to disk.

        Why: Fsync guarantees that bytes are written to persistent physical storage
        before control returns to the harness, preventing speculative dispatch.
        """
        canonical_bytes = serialize_stage1_payload(
            round_id=round_id,
            turn_idx=turn_idx,
            observation_ot=observation_ot,
            memory_plugmem=memory_plugmem,
            p_prior=p_prior,
            visit_counts_N=visit_counts_N,
            action_values_Q=action_values_Q,
            locked_action_at=locked_action_at,
            p_t_A=p_t_A,
            p_t_B=p_t_B,
            short_rationale=short_rationale,
        )

        stage1_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
        freeze_timestamp_ns = time.monotonic_ns()

        if custom_file_path is not None:
            target_path = Path(custom_file_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            target_path = self.freeze_dir / f"freeze_r{round_id}_t{turn_idx}.json"

        # Durable write: open, write, flush, fsync
        with open(target_path, "wb") as f:
            f.write(canonical_bytes)
            f.flush()
            os.fsync(f.fileno())

        return Stage1FreezeResult(
            canonical_bytes=canonical_bytes,
            stage1_sha256=stage1_sha256,
            freeze_timestamp_ns=freeze_timestamp_ns,
            file_path=target_path,
        )
