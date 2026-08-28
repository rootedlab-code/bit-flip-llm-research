"""Flip outcomes, per bit position, computed over the entire pattern space.

The 16-bit pattern space has 65,536 elements: where everything can be enumerated,
estimating is a lazy choice. Every fraction produced here is an exact count, weighted
by the histogram of the observed population -- model weights in E1, block scales in E3.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from bitflip.codec import FloatFormat, decompose, field_at, flip_bit, to_float32
from bitflip.stats import weighted_quantile

CODE_SPACE = 1 << 16
CATASTROPHIC_RATIO = 2.0**16
AMPLIFYING_RATIO = 2.0
# The mirror images of the two above. A flip that divides a weight by 2**16 removes
# it as surely as one that multiplies it explodes it, and until now only one
# direction had a name.
COLLAPSE_RATIO = 2.0**-16
ATTENUATING_RATIO = 0.5
ALL_CODES = np.arange(CODE_SPACE, dtype=np.uint16)


class BitRow(TypedDict):
    """One row of the per-bit table, which is also one row of a published CSV.

    Written as a schema rather than a bare mapping because these rows are read back
    by other experiments and by the reader of `results/e1-bit-hierarchy-*.csv`. A
    dictionary of `object` cannot say that `catastrophic_fraction` is a number, and
    every consumer was converting it by hand to find out.
    """

    bit: int
    field: str
    zero_bit_fraction: float
    median_delta: float
    p99_delta: float
    max_finite_delta: float
    amplified_fraction: float
    non_finite_fraction: float
    catastrophic_fraction: float


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


def bit_rows(counts: np.ndarray, fmt: FloatFormat) -> list[BitRow]:
    """One row per bit position, weighted by the observed population."""
    if counts.shape != (CODE_SPACE,):
        raise ValueError(f"histogram of {counts.shape}, expected ({CODE_SPACE},)")
    total = float(counts.sum())
    if total == 0:
        raise ValueError("empty population")

    with np.errstate(invalid="ignore"):
        usable = (counts > 0) & np.isfinite(values_of(fmt))
    weights = counts[usable].astype(np.float64)
    rows: list[BitRow] = []

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


class SpectrumRow(TypedDict):
    """One class of flip outcome, and how much of the bit space it accounts for."""

    outcome: str
    bit_share: float
    positions: str


# Ordered: each pattern is classified by the first condition it satisfies, so the
# classes partition the space rather than overlapping. `non_finite` precedes the ratio
# tests because a NaN has no meaningful ratio, and `sign_inversion` precedes the
# moderate bands because negating a weight leaves |after/before| at exactly 1.
SPECTRUM_OUTCOMES = (
    "non_finite",
    "catastrophic_amplification",
    "collapse",
    "sign_inversion",
    "moderate_amplification",
    "moderate_attenuation",
    "negligible",
)


def perturbation_spectrum(counts: np.ndarray, fmt: FloatFormat) -> list[SpectrumRow]:
    """Every outcome a flip can have, partitioned, not only the catastrophic one.

    `bit_rows` answers "how much of this population's bit space is catastrophic", and
    everything else is left as a residue with no name. That hides a second channel of
    damage: an exponent bit flipped *downward* divides the weight instead of
    multiplying it, which removes the weight rather than exploding it. Both destroy a
    weight, and only one of them was being counted.

    The classification is by **outcome**, not by bit position. Which bit does what is a
    consequence of the format's geometry, and deriving it rather than assuming it is
    the whole point of the experiment -- the `positions` column then shows the
    correspondence instead of presupposing it.

    Note what this does *not* do: `CATASTROPHIC_RATIO` is untouched, and
    `non_finite` + `catastrophic_amplification` reproduce
    `catastrophic_bit_fraction` exactly. No published figure moves.
    """
    if counts.shape != (CODE_SPACE,):
        raise ValueError(f"histogram of {counts.shape}, expected ({CODE_SPACE},)")
    total = float(counts.sum())
    if total == 0:
        raise ValueError("empty population")

    before = values_of(fmt)
    with np.errstate(invalid="ignore"):
        usable = (counts > 0) & np.isfinite(before)

    shares = dict.fromkeys(SPECTRUM_OUTCOMES, 0.0)
    seen: dict[str, list[int]] = {name: [] for name in SPECTRUM_OUTCOMES}

    for position in range(fmt.total_bits):
        delta, ratio, finite = flip_outcomes(fmt, position)
        with np.errstate(invalid="ignore", over="ignore"):
            after = to_float32(flip_bit(ALL_CODES, position, fmt), fmt).astype(np.float64)
        conditions = (
            ("non_finite", ~finite),
            ("catastrophic_amplification", ratio >= CATASTROPHIC_RATIO),
            ("collapse", ratio <= COLLAPSE_RATIO),
            ("sign_inversion", after == -before),
            ("moderate_amplification", ratio >= AMPLIFYING_RATIO),
            ("moderate_attenuation", ratio <= ATTENUATING_RATIO),
            ("negligible", np.ones(CODE_SPACE, dtype=bool)),
        )

        claimed = np.zeros(CODE_SPACE, dtype=bool)
        for name, condition in conditions:
            selected = usable & condition & ~claimed
            claimed |= selected
            weight = float(counts[selected].sum())
            if weight:
                # Accumulated per position and divided once at the end, matching the
                # association in `catastrophic_bit_fraction` exactly. Dividing inside
                # the loop rounds differently and puts the two figures one ULP apart,
                # which is the whole identity this function must not break.
                shares[name] += weight / total
                seen[name].append(position)

    return [
        SpectrumRow(
            outcome=name,
            bit_share=shares[name] / fmt.total_bits,
            positions=" ".join(str(p) for p in seen[name]),
        )
        for name in SPECTRUM_OUTCOMES
    ]


def catastrophic_bit_fraction(rows: list[BitRow], fmt: FloatFormat) -> float:
    """The fraction of the population's bits whose flip is catastrophic."""
    return sum(row["catastrophic_fraction"] for row in rows) / fmt.total_bits


def exponent_profile(counts: np.ndarray, fmt: FloatFormat) -> dict[str, float]:
    """How small the population's values are, and what that implies for the top bit.

    The top exponent bit is the universal attack surface only because it is
    predictably zero, and it is predictably zero only because almost every weight
    has |w| < 1. These two figures are what makes that argument checkable.
    """
    _, exponents, _ = decompose(ALL_CODES, fmt)
    total = float(counts.sum())
    return {
        "fraction_below_one": float(counts[exponents < fmt.bias].sum() / total),
        "median_exponent": weighted_quantile(
            exponents.astype(np.float64), counts.astype(np.float64), 0.5
        ),
    }


def population_summary(
    counts: np.ndarray, rows: list[BitRow], fmt: FloatFormat
) -> dict[str, object]:
    """The figures quoted in prose, in persistable form.

    No number this project publishes may live on screen only. Identity fields are
    the caller's to prepend: which model was measured is not something the
    arithmetic knows, and the same summary is produced locally and on a hosted run.
    """
    total = int(counts.sum())
    fraction = catastrophic_bit_fraction(rows, fmt)
    return {
        "weights": total,
        "total_bits": total * fmt.total_bits,
        **exponent_profile(counts, fmt),
        "exponent_bias": fmt.bias,
        "catastrophic_bits": round(fraction * total * fmt.total_bits),
        "catastrophic_bit_fraction": fraction,
        "one_bit_in": 1 / fraction,
    }


def format_table(rows: list[BitRow]) -> str:
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
