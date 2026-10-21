"""Analysis configuration model and default hyperparameter values.

Why: Re-exports AnalysisConfig from src.config to maintain clean subpackage namespaces
and prevent circular dependency cycles.
"""

from __future__ import annotations

from src.config import AnalysisConfig

__all__ = ["AnalysisConfig"]
