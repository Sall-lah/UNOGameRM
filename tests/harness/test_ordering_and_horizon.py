"""Tests for freeze-dispatch ordering, horizon evaluation, and barrier invariant (Tasks 7.4, 7.5, 7.6).

Verifies freeze fsync precedes server dispatch, monotonic timestamp invariant checks,
horizon event label compilation, memory update invocation, and barrier encapsulation.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.config import AppConfig
from src.env.barrier import LegalObservation
from src.env.events import HorizonEventType, HorizonServerEvent
from src.harness.agent_harness import AgentHarness
from src.audit.stage2 import TimestampInvariantError


def test_stage1_freeze_precedes_dispatch_and_timestamp_ordering_holds(
    tmp_path: Path,
    sample_legal_obs,
    fake_llm,
):
    """Verify Stage 1 freeze is persisted before dispatch, and t_freeze < t_dispatch <= t_eval."""
    cfg = AppConfig.from_yaml(f"""
    audit:
      output_root: {str(tmp_path)}
      strict_invariant: true
    """)

    dispatch_called_at = []
    freeze_file_existed = []

    mock_platform = MagicMock()
    def stub_dispatch(command):
        # Check that freeze files exist in output dir
        freeze_files = list((tmp_path / "stage1_freeze").glob("*.json"))
        freeze_file_existed.append(len(freeze_files) > 0)
        dispatch_called_at.append(time.monotonic_ns())
        return {
            "horizon_events": (),
            "round_over": False,
            "next_state": {"agent_hand": ["blue skip"], "top_card": "red 5"},
        }
    mock_platform.dispatch_agent_action_and_observe_horizon.side_effect = stub_dispatch

    harness = AgentHarness(
        config=cfg,
        server_platform=mock_platform,
        llm_client=fake_llm,
    )

    obs = sample_legal_obs()
    rec = harness.on_agent_turn(obs)

    assert freeze_file_existed == [True]
    t_freeze = rec.stage1_freeze_log.freeze_timestamp_ns
    t_dispatch = rec.stage2_post_horizon.dispatch_timestamp_ns
    t_eval = rec.stage2_post_horizon.eval_timestamp_ns

    # Invariant: freeze < dispatch <= eval
    assert t_freeze < t_dispatch <= t_eval


def test_ordering_violation_raises_and_emits_no_record(tmp_path: Path, sample_legal_obs, fake_llm):
    """Verify when dispatch timestamp is before freeze timestamp, an assertion error is raised."""
    cfg = AppConfig.from_yaml(f"""
    audit:
      output_root: {str(tmp_path)}
      strict_invariant: true
    """)

    mock_platform = MagicMock()
    mock_platform.dispatch_agent_action_and_observe_horizon.return_value = {
        "horizon_events": (),
        "round_over": False,
        "next_state": {"agent_hand": ["blue skip"], "top_card": "red 5"},
    }

    harness = AgentHarness(
        config=cfg,
        server_platform=mock_platform,
        llm_client=fake_llm,
    )

    # Corrupt stage1 writer to emit a future timestamp violating invariant
    original_freeze = harness.stage1_writer.freeze
    def corrupted_freeze(*args, **kwargs):
        res = original_freeze(*args, **kwargs)
        # Return a timestamp in the far future
        future_ns = time.monotonic_ns() + 10_000_000_000
        from src.audit.stage1 import Stage1FreezeResult
        return Stage1FreezeResult(
            canonical_bytes=res.canonical_bytes,
            stage1_sha256=res.stage1_sha256,
            freeze_timestamp_ns=future_ns,
            file_path=res.file_path,
        )
    harness.stage1_writer.freeze = corrupted_freeze

    obs = sample_legal_obs()
    with pytest.raises(TimestampInvariantError):
        harness.on_agent_turn(obs)

    # No record should be written to dataset file
    dataset_records = harness.dataset_writer.read_all()
    assert len(dataset_records) == 0


def test_dispatch_horizon_evaluation_and_memory_update(tmp_path: Path, sample_legal_obs, fake_llm):
    """Verify horizon events are processed for y_t_A, y_t_B, stratum, and PlugMem is updated."""
    cfg = AppConfig.from_yaml(f"""
    audit:
      output_root: {str(tmp_path)}
      strict_invariant: true
    """)

    penalty_event = HorizonServerEvent(
        event_index=1,
        actor="AGENT",
        event_type=HorizonEventType.DRAW_PENALTY_APEN,
        active=False,
        cards_drawn_count=2,
    )

    mock_platform = MagicMock()
    mock_platform.dispatch_agent_action_and_observe_horizon.return_value = {
        "horizon_events": (penalty_event,),
        "round_over": False,
        "next_state": {"agent_hand": ["blue skip", "yellow 1", "yellow 2"], "top_card": "red 5"},
    }

    mock_plugmem = MagicMock()
    ret_mock = MagicMock()
    ret_mock.distilled_context = "context"
    ret_mock.retrieved_node_ids = []
    ret_mock.active_opp_lack_colours = {}
    ret_mock.active_opp_wild_colours = ()
    ret_mock.relevance_scores = []
    ret_mock.token_estimate = 0
    mock_plugmem.retrieve.return_value = ret_mock
    mock_plugmem.get_determinization_constraints.return_value = []

    harness = AgentHarness(
        config=cfg,
        server_platform=mock_platform,
        plugmem=mock_plugmem,
        llm_client=fake_llm,
    )

    obs = sample_legal_obs()
    rec = harness.on_agent_turn(obs)

    # Horizon labels evaluated
    assert rec.stage2_post_horizon.y_t_A_penalty_draw == 1
    assert rec.stage2_post_horizon.A_pen_cards_drawn == 2
    # PlugMem update called with the observation and horizon events
    mock_plugmem.update.assert_called_once()
    # Emitted to dataset
    dataset_records = harness.dataset_writer.read_all()
    assert len(dataset_records) == 1


def test_barrier_only_observation_invariant(sample_legal_obs):
    """Verify harness operates strictly on LegalObservation without referencing Game or Deck."""
    obs = sample_legal_obs()
    # Ensure observation conforms to LegalObservation
    assert isinstance(obs, LegalObservation)

    # Verify no private engine attributes exist on LegalObservation
    assert not hasattr(obs, "players")
    assert not hasattr(obs, "deck")
    assert not hasattr(obs, "cards")
    assert not hasattr(obs, "cpu_hand")
