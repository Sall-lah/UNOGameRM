"""Two-stage audit logging package for tamper-evident agent prediction verification.

Exports Stage1Writer, HorizonMonitor, Stage2Evaluator, DecisionRecord, and DatasetWriter.
"""

from __future__ import annotations

from src.audit.dataset import (
    DatasetWriter,
    DecisionRecord,
    GameResourcesGroup,
    IdentityGroup,
    MemoryPlugMemGroup,
    ObservationOtGroup,
    SearchMCTSGroup,
    Stage1FreezeLogGroup,
    Stage2PostHorizonGroup,
    SystemReliabilityGroup,
)
from src.audit.horizon import (
    HorizonEvaluation,
    HorizonMonitor,
    classify_stratum,
)
from src.audit.stage1 import (
    Stage1FreezeResult,
    Stage1Writer,
    serialize_stage1_payload,
)
from src.audit.stage2 import (
    Stage2Evaluator,
    TamperError,
    TimestampInvariantError,
)

__all__ = [
    "Stage1Writer",
    "Stage1FreezeResult",
    "serialize_stage1_payload",
    "HorizonMonitor",
    "HorizonEvaluation",
    "classify_stratum",
    "Stage2Evaluator",
    "TimestampInvariantError",
    "TamperError",
    "DecisionRecord",
    "DatasetWriter",
    "IdentityGroup",
    "ObservationOtGroup",
    "MemoryPlugMemGroup",
    "SearchMCTSGroup",
    "Stage1FreezeLogGroup",
    "Stage2PostHorizonGroup",
    "GameResourcesGroup",
    "SystemReliabilityGroup",
]
