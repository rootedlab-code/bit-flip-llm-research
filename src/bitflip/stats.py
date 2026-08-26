"""Statistiche pesate: gli esiti sono 65.536, i pesi sono quante volte ricorrono."""

from __future__ import annotations

import numpy as np


def weighted_quantile(values: np.ndarray, counts: np.ndarray, quantile: float) -> float:
    """Quantile della distribuzione in cui `values[i]` compare `counts[i]` volte.

    Solleva `ValueError` su quantile fuori da [0, 1] o su distribuzione vuota.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile {quantile} fuori da [0, 1]")
    if values.size == 0 or counts.sum() == 0:
        raise ValueError("distribuzione vuota")

    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(counts[order].astype(np.float64))
    position = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(ordered_values[min(position, ordered_values.size - 1)])
