"""The published per-bit figures: exact counts over a population, not estimates."""

from __future__ import annotations

import numpy as np
import pytest

from bitflip.codec import BF16, from_float32
from bitflip.fragility import (
    CODE_SPACE,
    SPECTRUM_OUTCOMES,
    bit_rows,
    catastrophic_bit_fraction,
    exponent_profile,
    perturbation_spectrum,
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


# --- the perturbation spectrum ---------------------------------------------------
#
# `bit_rows` names one outcome and leaves the rest as a residue. The spectrum names
# them all, so that the channel which removes a weight is counted rather than implied.


def spectrum_of(values: list[float]) -> dict[str, float]:
    rows = perturbation_spectrum(population(values), BF16)
    return {row["outcome"]: row["bit_share"] for row in rows}


def test_perturbation_spectrum_partitions_the_whole_bit_space():
    shares = spectrum_of([0.02, 0.5, 0.125, -0.75, -0.001])

    assert set(shares) == set(SPECTRUM_OUTCOMES)
    assert sum(shares.values()) == 1.0


def test_perturbation_spectrum_reproduces_the_published_catastrophic_fraction():
    """The classification is additive: it renames nothing the criterion already
    decided, so the two catastrophic classes cover exactly the positions the criterion
    marks, and agree numerically.

    Agreement is asserted to a tolerance rather than to the bit, deliberately. It is
    the same quantity summed in a different order -- one accumulator per class here,
    one combined there -- and float addition is not associative, so the two can sit
    one ULP apart. Demanding exact equality would be testing an accident of ordering.
    What must hold is that no pattern moves between the classes, and that is the
    set identity asserted first.
    """
    counts = population([0.02, 0.5, 0.125, -0.75, -0.001, 3.5])
    rows = perturbation_spectrum(counts, BF16)
    shares = {row["outcome"]: row["bit_share"] for row in rows}
    catastrophic_positions = {
        position
        for row in rows
        if row["outcome"] in ("non_finite", "catastrophic_amplification")
        for position in row["positions"].split()
    }

    published_rows = bit_rows(counts, BF16)
    published = catastrophic_bit_fraction(published_rows, BF16)

    assert catastrophic_positions == {
        str(row["bit"]) for row in published_rows if row["catastrophic_fraction"] > 0
    }
    assert shares["non_finite"] + shares["catastrophic_amplification"] == pytest.approx(
        published, rel=1e-15
    )


def test_perturbation_spectrum_counts_the_collapse_channel_the_criterion_omits():
    """The result P3 exists for: on weights below one the exponent bits that divide
    outnumber the one that multiplies, and only the second was ever counted."""
    shares = spectrum_of([0.02, 0.5, 0.125, -0.75, -0.001])

    catastrophic = shares["non_finite"] + shares["catastrophic_amplification"]
    assert shares["collapse"] > catastrophic


def test_perturbation_spectrum_derives_the_sign_bit_instead_of_assuming_it():
    """Classification is by outcome, not by position: that bit 15 is the sign bit is
    a conclusion the table reaches, not an input it is given."""
    rows = {
        row["outcome"]: row for row in perturbation_spectrum(population([0.02]), BF16)
    }

    assert rows["sign_inversion"]["positions"] == "15"
    assert rows["sign_inversion"]["bit_share"] == pytest.approx(1 / BF16.total_bits)


def test_perturbation_spectrum_rejects_an_empty_population():
    with pytest.raises(ValueError, match="empty population"):
        perturbation_spectrum(np.zeros(CODE_SPACE, dtype=np.uint64), BF16)
