"""The contract of damage classification."""

from __future__ import annotations

import math

import pytest

from bitflip.damage import (
    DESTROYED,
    INTACT,
    NUMERIC_COLLAPSE,
    PARTIAL,
    UNIFORM_COLLAPSE,
    damage_class,
    is_uniform,
)

VOCABULARY = 151_936
BASELINE = 16.106150421005136


def test_an_untouched_model_is_intact():
    assert damage_class(BASELINE, 1.0, VOCABULARY) == INTACT


def test_a_barely_moved_model_is_still_intact():
    assert damage_class(16.0984, 0.9938, VOCABULARY) == INTACT


def test_the_one_partial_case_observed_in_e2_is_partial():
    assert damage_class(10_524.0466, 0.1364, VOCABULARY) == PARTIAL


def test_perplexity_equal_to_the_vocabulary_is_a_uniform_collapse():
    assert damage_class(151_935.9607, 0.0, VOCABULARY) == UNIFORM_COLLAPSE


def test_not_a_number_is_a_numeric_collapse():
    assert damage_class(math.nan, 0.0, VOCABULARY) == NUMERIC_COLLAPSE


def test_infinity_is_a_numeric_collapse():
    assert damage_class(math.inf, 0.0, VOCABULARY) == NUMERIC_COLLAPSE


def test_a_numeric_collapse_is_decided_before_agreement_is_consulted():
    """NaN never equals anything, so agreement there measures the arithmetic,
    not the model."""
    assert damage_class(math.nan, 1.0, VOCABULARY) == NUMERIC_COLLAPSE


def test_a_ruined_model_that_is_not_uniform_is_merely_destroyed():
    assert damage_class(5_000.0, 0.0, VOCABULARY) == DESTROYED


def test_uniformity_is_judged_within_a_declared_tolerance():
    assert is_uniform(VOCABULARY * 1.005, VOCABULARY)
    assert not is_uniform(VOCABULARY * 1.5, VOCABULARY)


def test_uniformity_is_never_claimed_for_a_non_number():
    assert not is_uniform(math.nan, VOCABULARY)


def test_an_agreement_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        damage_class(BASELINE, 1.5, VOCABULARY)
