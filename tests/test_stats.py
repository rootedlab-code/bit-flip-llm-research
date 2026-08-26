"""Contratto del quantile pesato."""

from __future__ import annotations

import numpy as np
import pytest

from bitflip.stats import weighted_quantile

VALUES = np.array([10.0, 20.0, 30.0])
COUNTS = np.array([1, 8, 1], dtype=np.uint64)


def test_median_follows_the_mass_not_the_values():
    assert weighted_quantile(VALUES, COUNTS, 0.5) == 20.0


def test_quantile_zero_returns_the_smallest_represented_value():
    assert weighted_quantile(VALUES, COUNTS, 0.0) == 10.0


def test_quantile_one_returns_the_largest_represented_value():
    assert weighted_quantile(VALUES, COUNTS, 1.0) == 30.0


def test_values_with_zero_count_are_ignored():
    counts = np.array([0, 1, 0], dtype=np.uint64)

    assert weighted_quantile(VALUES, counts, 0.5) == 20.0


def test_unsorted_input_is_handled():
    values = np.array([30.0, 10.0, 20.0])
    counts = np.array([1, 1, 8], dtype=np.uint64)

    assert weighted_quantile(values, counts, 0.5) == 20.0


def test_matches_numpy_on_an_expanded_distribution():
    """La convenzione e quella di `inverted_cdf`: si restituisce un valore osservato."""
    values = np.arange(5, dtype=np.float64)
    counts = np.array([3, 1, 4, 1, 5], dtype=np.uint64)
    expanded = np.repeat(values, counts.astype(np.int64))

    for quantile in (0.0, 0.25, 0.5, 0.75, 1.0):
        expected = np.quantile(expanded, quantile, method="inverted_cdf")
        assert weighted_quantile(values, counts, quantile) == expected


def test_quantile_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="fuori da"):
        weighted_quantile(VALUES, COUNTS, 1.5)


def test_empty_distribution_is_rejected():
    with pytest.raises(ValueError, match="vuota"):
        weighted_quantile(VALUES, np.zeros(3, dtype=np.uint64), 0.5)
