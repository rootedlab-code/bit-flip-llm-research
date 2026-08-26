"""Weighted statistics: the outcomes are 65,536, the weights are how often each occurs."""

from __future__ import annotations

import numpy as np


def weighted_quantile(values: np.ndarray, counts: np.ndarray, quantile: float) -> float:
    """Quantile of the distribution where `values[i]` occurs `counts[i]` times.

    Follows the `inverted_cdf` convention: the value returned is one that was
    actually observed. Raises `ValueError` on a quantile outside [0, 1] or an
    empty distribution.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile {quantile} outside [0, 1]")
    if values.size == 0 or counts.sum() == 0:
        raise ValueError("empty distribution")

    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(counts[order].astype(np.float64))
    position = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(ordered_values[min(position, ordered_values.size - 1)])
