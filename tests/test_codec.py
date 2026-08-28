"""The codec contract: the law of exponent bits, verified rather than cited."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bitflip.codec import (
    BF16,
    EXPONENT,
    FORMATS,
    FP16,
    MANTISSA,
    SIGN,
    compose,
    decompose,
    exponent_multiplier,
    exponent_shift,
    field_at,
    flip_bit,
    from_float32,
    to_float32,
)

FORMAT_LIST = list(FORMATS.values())
ALL_CODES = np.arange(1 << 16, dtype=np.uint16)


@pytest.mark.parametrize("fmt", FORMAT_LIST, ids=lambda f: f.name)
def test_roundtrip_is_exact_for_every_pattern(fmt):
    restored = from_float32(to_float32(ALL_CODES, fmt), fmt)

    assert np.array_equal(restored, ALL_CODES)


@pytest.mark.parametrize("fmt", FORMAT_LIST, ids=lambda f: f.name)
def test_compose_inverts_decompose_for_every_pattern(fmt):
    assert np.array_equal(compose(*decompose(ALL_CODES, fmt), fmt), ALL_CODES)


@pytest.mark.parametrize("fmt", FORMAT_LIST, ids=lambda f: f.name)
def test_flipping_the_same_bit_twice_restores_the_pattern(fmt):
    for position in range(fmt.total_bits):
        assert np.array_equal(
            flip_bit(flip_bit(ALL_CODES, position, fmt), position, fmt), ALL_CODES
        )


@pytest.mark.parametrize("fmt", FORMAT_LIST, ids=lambda f: f.name)
def test_scalar_input_yields_scalar_output(fmt):
    assert from_float32(np.float32(0.02), fmt).shape == ()
    assert to_float32(np.uint16(0x3CA4), fmt).shape == ()
    assert from_float32(np.zeros(3, np.float32), fmt).shape == (3,)


def test_flip_rejects_a_position_outside_the_word():
    with pytest.raises(ValueError, match="outside the 16 bits"):
        flip_bit(np.uint16(0), 16, BF16)


@pytest.mark.parametrize("fmt", FORMAT_LIST, ids=lambda f: f.name)
def test_every_position_belongs_to_exactly_one_field(fmt):
    fields = [field_at(position, fmt) for position in range(fmt.total_bits)]

    assert fields.count(SIGN) == 1
    assert fields.count(EXPONENT) == fmt.exponent_bits
    assert fields.count(MANTISSA) == fmt.mantissa_bits


@pytest.mark.parametrize("fmt", FORMAT_LIST, ids=lambda f: f.name)
def test_exponent_shift_doubles_with_each_position(fmt):
    shifts = [exponent_shift(position, fmt) for position in fmt.exponent_positions]

    assert shifts == [2**index for index in range(fmt.exponent_bits)]


def test_exponent_shift_refuses_a_mantissa_bit():
    with pytest.raises(ValueError, match="is not an exponent bit"):
        exponent_shift(0, BF16)


@pytest.mark.parametrize("fmt", FORMAT_LIST, ids=lambda f: f.name)
def test_exponent_flip_multiplies_the_value_by_two_to_the_shift(fmt):
    """The project's law: an exponent bit multiplies by 2**(2**index).

    Verified over *every* normal exponent for which the flip does not overflow, not
    over one hand-picked value.
    """
    max_exponent = (1 << fmt.exponent_bits) - 1

    for position in fmt.exponent_positions:
        shift = exponent_shift(position, fmt)
        bit_index = position - fmt.mantissa_bits
        candidates = np.array(
            [
                exponent
                for exponent in range(1, max_exponent)
                if not (exponent >> bit_index) & 1
                and 1 <= exponent + shift < max_exponent
            ],
            dtype=np.uint16,
        )
        assert candidates.size, f"nessun esponente utile per la posizione {position}"

        zeros = np.zeros_like(candidates)
        codes = compose(zeros, candidates, zeros, fmt)
        before = to_float32(codes, fmt).astype(np.float64)
        after = to_float32(flip_bit(codes, position, fmt), fmt).astype(np.float64)

        # The comparison must happen in float64: for bf16 the top bit's factor is
        # 2**128, which exceeds the float32 maximum and would become inf in the
        # comparison itself.
        assert np.array_equal(after, before * exponent_multiplier(position, fmt))


def test_the_top_exponent_bit_costs_2_to_the_128_in_bf16_and_2_to_the_16_in_fp16():
    """The asymmetry that makes bf16 the more fragile format under attack."""
    with np.errstate(over="ignore"):  # the overflow is exactly what is asserted
        assert np.float32(exponent_multiplier(14, BF16)) == np.inf
    assert np.isfinite(np.float32(exponent_multiplier(14, FP16)))
    assert exponent_multiplier(14, BF16) == 2.0**128
    assert exponent_multiplier(14, FP16) == 2.0**16
    assert exponent_multiplier(14, BF16) / exponent_multiplier(14, FP16) == 2.0**112


def test_the_severity_gap_is_thirty_three_orders_of_magnitude_not_twelve():
    """The technical note quoted twelve for a while. 2**112 is 5.2e33, and a boundary
    stated wrong by twenty orders is a boundary a reader stops trusting."""
    gap = exponent_multiplier(14, BF16) / exponent_multiplier(14, FP16)

    assert math.log10(gap) == pytest.approx(33.7, abs=0.05)
    assert math.log10(gap) > 12


@pytest.mark.parametrize("fmt", FORMAT_LIST, ids=lambda f: f.name)
def test_mantissa_top_bit_adds_exactly_half_to_the_significand(fmt):
    one = from_float32(np.float32(1.0), fmt)
    top_mantissa_bit = fmt.mantissa_bits - 1

    assert to_float32(flip_bit(one, top_mantissa_bit, fmt), fmt) == np.float32(1.5)


@pytest.mark.parametrize("fmt", FORMAT_LIST, ids=lambda f: f.name)
def test_mantissa_flips_never_move_a_weight_more_than_a_factor_of_two(fmt):
    """The counterpart: no mantissa bit can do comparable damage."""
    weights = to_float32(
        from_float32(np.linspace(0.001, 0.5, 400, dtype=np.float32), fmt), fmt
    )
    codes = from_float32(weights, fmt)

    for position in fmt.mantissa_positions:
        ratios = to_float32(flip_bit(codes, position, fmt), fmt) / weights

        assert np.all(ratios > 0.5)
        assert np.all(ratios < 2.0)


@pytest.mark.parametrize("fmt", FORMAT_LIST, ids=lambda f: f.name)
def test_sign_flip_negates_the_value(fmt):
    weights = np.array([0.02, 1.0, 3.5], dtype=np.float32)
    codes = from_float32(weights, fmt)

    flipped = to_float32(flip_bit(codes, fmt.sign_position, fmt), fmt)

    assert np.array_equal(flipped, -to_float32(codes, fmt))


def test_rounding_to_nearest_even_beats_truncation_on_a_real_weight():
    source_bits = np.float32(0.02).view(np.uint32)

    assert int(from_float32(np.float32(0.02), BF16)) == 0x3CA4
    assert int(source_bits >> 16) == 0x3CA3


@pytest.mark.parametrize("fmt", FORMAT_LIST, ids=lambda f: f.name)
def test_not_a_number_survives_the_roundtrip(fmt):
    quiet_nan = compose(
        np.uint16(0),
        np.uint16((1 << fmt.exponent_bits) - 1),
        np.uint16(1 << (fmt.mantissa_bits - 1)),
        fmt,
    )

    assert np.isnan(to_float32(quiet_nan, fmt))
    assert int(from_float32(to_float32(quiet_nan, fmt), fmt)) == int(quiet_nan)
