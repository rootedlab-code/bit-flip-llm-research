"""The published per-bit figures: exact counts over a population, not estimates."""

from __future__ import annotations

import numpy as np
import pytest

from bitflip.codec import BF16, from_float32
from bitflip.fragility import (
    CODE_SPACE,
    bit_rows,
    catastrophic_bit_fraction,
    exponent_profile,
    population_summary,
)

TOP_EXPONENT_BIT = 14


def population(values: list[float], repeats: list[int] | None = None) -> np.ndarray:
    """A histogram over the 16-bit code space, holding exactly these values."""
    codes = from_float32(np.array(values, dtype=np.float32), BF16)
    counts = np.zeros(CODE_SPACE, dtype=np.uint64)
    for code, times in zip(codes, repeats or [1] * len(values), strict=True):
        counts[int(code)] += times
    return counts


def test_exponent_profile_counts_the_weights_below_one():
    counts = population([0.5, 1.0], repeats=[3, 1])

    profile = exponent_profile(counts, BF16)

    assert profile["fraction_below_one"] == pytest.approx(0.75)
    assert profile["median_exponent"] == pytest.approx(BF16.bias - 1)


def test_exponent_profile_reports_no_weight_below_one_when_none_is():
    counts = population([1.0, 2.0, 4.0])

    assert exponent_profile(counts, BF16)["fraction_below_one"] == 0.0


def test_population_summary_keeps_its_bit_count_consistent_with_its_fraction():
    counts = population([0.02, -1.5, 3.25, 0.125])
    rows = bit_rows(counts, BF16)

    totals = population_summary(counts, rows, BF16)

    assert totals["weights"] == 4
    assert totals["total_bits"] == 4 * BF16.total_bits
    assert totals["catastrophic_bits"] == round(
        totals["catastrophic_bit_fraction"] * totals["total_bits"]
    )
    assert totals["one_bit_in"] == pytest.approx(1 / totals["catastrophic_bit_fraction"])


def test_population_summary_leaves_the_identity_fields_to_its_caller():
    counts = population([0.5])

    assert "model" not in population_summary(counts, bit_rows(counts, BF16), BF16)


def test_a_population_of_small_weights_loses_one_bit_in_sixteen():
    """The result E1 rests on: with |w| < 1 the top exponent bit is zero, and it is
    the only bit of the sixteen whose flip is catastrophic."""
    counts = population([0.02, 0.5, 0.125, -0.75, -0.001])
    rows = bit_rows(counts, BF16)

    fraction = catastrophic_bit_fraction(rows, BF16)

    assert rows[TOP_EXPONENT_BIT]["zero_bit_fraction"] == 1.0
    assert rows[TOP_EXPONENT_BIT]["catastrophic_fraction"] == pytest.approx(1.0)
    assert fraction == pytest.approx(1 / BF16.total_bits)
