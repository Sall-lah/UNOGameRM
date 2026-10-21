"""Pluggable embedding interface and deterministic local fallback for PlugMem.

This module provides a fast, offline vector representation without requiring
external neural model dependencies, using a stable hash-based token projection.
"""

from __future__ import annotations

import math
import zlib
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class EmbeddingInterface(Protocol):
    """Protocol for text embedding backends."""

    def encode(self, text: str) -> np.ndarray:
        """Encodes an input string into a normalized 1D numpy array."""
        ...


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes cosine similarity between two 1D vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


class LocalFallbackEmbedder:
    """Deterministic, process-stable local feature hasher for fast offline similarity.

    Uses CRC32 hashing across n-grams to construct a fixed-dimension vector,
    guaranteeing identical embeddings across runs and processes without network access.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        """Projects token n-grams into a fixed-dimension unit vector."""
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = text.lower().replace(":", " ").replace("/", " ").split()
        if not tokens:
            return vec

        for token in tokens:
            # Stable crc32 hash rather than Python's salted hash()
            h = zlib.crc32(token.encode("utf-8"))
            idx = h % self.dim
            sign = 1.0 if (h >> 16) & 1 else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0.0:
            vec /= norm
        return vec
