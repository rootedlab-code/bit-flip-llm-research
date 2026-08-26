"""Flip outcomes, per bit position, computed over the entire pattern space.

The 16-bit pattern space has 65,536 elements: where everything can be enumerated,
estimating is a lazy choice. Every fraction produced here is an exact count, weighted
by the histogram of the observed population -- model weights in E1, block scales in E3.
"""

from __future__ import annotations

import numpy as np

from bitflip.codec import FloatFormat, field_at, flip_bit, to_float32
from bitflip.stats import weighted_quantile

CODE_SPACE = 1 << 16
CATASTROPHIC_RATIO = 2.0**16
AMPLIFYING_RATIO = 2.0
ALL_CODES = np.arange(CODE_SPACE, dtype=np.uint16)


def values_of(fmt: FloatFormat) -> np.ndarray:
    with np.errstate(invalid="ignore", over="ignore"):
        return to_float32(ALL_CODES, fmt).astype(np.float64)


def flip_outcomes(fmt: FloatFormat, position: int) -> tuple[np.ndarray, ...]:
    """For every pattern: |delta|, the ratio |after/before|, and whether it stays
    finite."""
    before = values_of(fmt)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        after = to_float32(flip_bit(ALL_CODES, position, fmt), fmt).astype(np.float64)
        delta = np.abs(after - before)
        ratio = np.where(before != 0, np.abs(after / before), np.inf)
    return delta, ratio, np.isfinite(after)


def bit_rows(counts: np.ndarray, fmt: FloatFormat) -> list[dict[str, object]]:
    """One row per bit position, weighted by the observed population."""
    if counts.shape != (CODE_SPACE,):
        raise ValueError(f"histogram of {counts.shape}, expected ({CODE_SPACE},)")
    total = float(counts.sum())
    if total == 0:
        raise ValueError("empty population")

    with np.errstate(invalid="ignore"):
        usable = (counts > 0) & np.isfinite(values_of(fmt))
    weights = counts[usable].astype(np.float64)
    rows = []

    for position in range(fmt.total_bits):
        delta, ratio, finite = flip_outcomes(fmt, position)
        catastrophic = ~finite | (ratio >= CATASTROPHIC_RATIO)
        finite_delta = delta[usable & finite]

        rows.append(
            {
                "bit": position,
                "field": field_at(position, fmt),
                "zero_bit_fraction": float(
                    counts[((ALL_CODES >> position) & 1) == 0].sum() / total
                ),
                "median_delta": weighted_quantile(delta[usable], weights, 0.5),
                "p99_delta": weighted_quantile(delta[usable], weights, 0.99),
                "max_finite_delta": (
                    float(finite_delta.max()) if finite_delta.size else 0.0
                ),
                "amplified_fraction": float(
                    counts[usable & (ratio >= AMPLIFYING_RATIO)].sum() / total
                ),
                "non_finite_fraction": float(counts[usable & ~finite].sum() / total),
                "catastrophic_fraction": float(
                    counts[usable & catastrophic].sum() / total
                ),
            }
        )
    return rows


def catastrophic_bit_fraction(rows: list[dict[str, object]], fmt: FloatFormat) -> float:
    """The fraction of the population's bits whose flip is catastrophic."""
    return sum(float(row["catastrophic_fraction"]) for row in rows) / fmt.total_bits


def format_table(rows: list[dict[str, object]]) -> str:
    header = (
        f"{'bit':>3} {'field':<9} {'bit=0':>8} {'|d| median':>13} "
        f"{'|d| p99':>11} {'|d| max':>11} {'>=x2':>9} {'catastr.':>10}"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row['bit']:>3} {row['field']:<9} {row['zero_bit_fraction']:>7.2%} "
            f"{row['median_delta']:>13.3e} {row['p99_delta']:>11.3e} "
            f"{row['max_finite_delta']:>11.3e} {row['amplified_fraction']:>8.2%} "
            f"{row['catastrophic_fraction']:>10.4%}"
        )
    return "\n".join(lines)
