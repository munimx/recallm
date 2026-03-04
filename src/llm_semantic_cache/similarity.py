"""cosine_similarity — pure math, no side effects."""
from __future__ import annotations

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a float in [-1.0, 1.0]. Returns 0.0 if either vector is zero.
    Raises ValueError if vectors have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vectors must have the same length, got {len(a)} and {len(b)}"
        )
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))
