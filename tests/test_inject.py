"""The injection contract: reproducible, uniform, and always reversible."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from bitflip.codec import BF16, from_float32, to_float32
from bitflip.inject import (
    COLLAPSE_BIT,
    TOP_EXPONENT_BIT,
    Flip,
    apply_flips,
    collapse_flips,
    flipped_model,
    largest_magnitude_flips,
    model_codes,
    parameter_codes,
    random_flips,
)

SIZES = {"a": 100, "b": 300}


def test_the_same_seed_draws_the_same_faults():
    assert random_flips(SIZES, 50, seed=7) == random_flips(SIZES, 50, seed=7)


def test_different_seeds_draw_different_faults():
    assert random_flips(SIZES, 50, seed=7) != random_flips(SIZES, 50, seed=8)


def test_faults_land_inside_the_tensors_they_name():
    for flip in random_flips(SIZES, 500, seed=1):
        assert 0 <= flip.index < SIZES[flip.tensor]
        assert 0 <= flip.bit < BF16.total_bits


def test_tensors_are_hit_in_proportion_to_their_size():
    """A tensor three times larger offers three times the targets,
    not three times the importance."""
    counts = Counter(flip.tensor for flip in random_flips(SIZES, 40_000, seed=3))

    assert counts["b"] / counts["a"] == pytest.approx(3.0, rel=0.05)


def test_the_bit_choice_can_be_restricted():
    flips = random_flips(SIZES, 100, seed=2, bits=[TOP_EXPONENT_BIT])

    assert {flip.bit for flip in flips} == {TOP_EXPONENT_BIT}


def test_zero_faults_is_a_valid_request():
    assert random_flips(SIZES, 0, seed=1) == []


def test_a_negative_count_is_rejected():
    with pytest.raises(ValueError, match="negative fault count"):
        random_flips(SIZES, -1, seed=1)


def test_an_empty_model_is_rejected():
    with pytest.raises(ValueError, match="no tensor"):
        random_flips({}, 1, seed=1)


def test_targeted_faults_pick_the_largest_weights_and_the_top_exponent_bit():
    codes = {"w": from_float32(np.array([0.01, 0.5, 0.02, 0.9], np.float32), BF16)}

    flips = largest_magnitude_flips(codes, count=2)

    assert [flip.index for flip in flips] == [3, 1]
    assert {flip.bit for flip in flips} == {TOP_EXPONENT_BIT}


def test_targeted_faults_rank_across_tensors():
    codes = {
        "small": from_float32(np.array([0.01, 0.02], np.float32), BF16),
        "large": from_float32(np.array([0.9, 0.03], np.float32), BF16),
    }

    assert largest_magnitude_flips(codes, count=1) == [Flip("large", 0, TOP_EXPONENT_BIT)]


def test_applying_a_fault_leaves_the_source_untouched():
    codes = from_float32(np.array([0.02, 0.02], np.float32), BF16)
    original = codes.copy()

    apply_flips(codes, [Flip("w", 0, TOP_EXPONENT_BIT)])

    assert np.array_equal(codes, original)


def test_a_top_exponent_flip_multiplies_the_weight_by_two_to_the_128():
    codes = from_float32(np.array([2.0**-64], np.float32), BF16)

    flipped = apply_flips(codes, [Flip("w", 0, TOP_EXPONENT_BIT)])

    assert float(to_float32(flipped, BF16)[0]) == 2.0**64


def build_tiny_model():
    torch.manual_seed(0)
    return torch.nn.Linear(4, 3, bias=False).to(torch.bfloat16)


def test_the_model_is_restored_exactly_after_the_context():
    model = build_tiny_model()
    before = model.weight.detach().clone()

    with flipped_model(model, [Flip("weight", 0, TOP_EXPONENT_BIT)]):
        assert not torch.equal(model.weight, before)

    assert torch.equal(model.weight, before)


def test_the_model_is_restored_even_when_the_body_fails():
    model = build_tiny_model()
    before = model.weight.detach().clone()

    with pytest.raises(RuntimeError), flipped_model(model, [Flip("weight", 1, 14)]):
        raise RuntimeError("guasto nel corpo")

    assert torch.equal(model.weight, before)


def test_repeated_faults_on_one_weight_are_undone_in_reverse_order():
    model = build_tiny_model()
    before = model.weight.detach().clone()
    flips = [Flip("weight", 2, 14), Flip("weight", 2, 13), Flip("weight", 2, 14)]

    with flipped_model(model, flips):
        pass

    assert torch.equal(model.weight, before)


def test_the_torch_flip_matches_the_numpy_flip():
    model = build_tiny_model()
    codes = model.weight.detach().view(-1).view(torch.int16).numpy().view(np.uint16)
    expected = apply_flips(codes, [Flip("weight", 3, TOP_EXPONENT_BIT)])

    with flipped_model(model, [Flip("weight", 3, TOP_EXPONENT_BIT)]):
        observed = (
            model.weight.detach().view(-1).view(torch.int16).numpy().view(np.uint16)
        )

        assert np.array_equal(observed, expected)


def test_a_bfloat16_weight_promoted_to_float32_keeps_its_pattern_in_the_high_bits():
    """The premise that makes float32 compute exact on GPUs without native bf16."""
    weights = torch.tensor([0.02, -1.5, 3.25], dtype=torch.bfloat16)

    promoted = weights.to(torch.float32)
    high_bits = (promoted.view(torch.int32).numpy().astype(np.uint32) >> 16).astype(
        np.uint16
    )

    assert np.array_equal(high_bits, weights.view(torch.int16).numpy().view(np.uint16))


def test_the_float32_path_flips_the_same_bit_as_the_bfloat16_path():
    reference = torch.nn.Linear(4, 3, bias=False).to(torch.bfloat16)
    torch.nn.init.constant_(reference.weight, 0.02)
    promoted = torch.nn.Linear(4, 3, bias=False).to(torch.float32)
    with torch.no_grad():
        promoted.weight.copy_(reference.weight.to(torch.float32))

    flip = [Flip("weight", 5, TOP_EXPONENT_BIT)]
    with flipped_model(reference, flip), flipped_model(promoted, flip):
        assert (
            promoted.weight.view(-1)[5].item()
            == reference.weight.view(-1)[5].float().item()
        )


def test_the_float32_model_is_restored_exactly():
    model = torch.nn.Linear(4, 3, bias=False).to(torch.float32)
    before = model.weight.detach().clone()

    with flipped_model(model, [Flip("weight", 0, TOP_EXPONENT_BIT)]):
        assert not torch.equal(model.weight, before)

    assert torch.equal(model.weight, before)


def test_an_unsupported_dtype_is_refused():
    model = torch.nn.Linear(4, 3, bias=False).to(torch.float64)

    with (
        pytest.raises(TypeError, match="not supported"),
        flipped_model(model, [Flip("weight", 0, 14)]),
    ):
        pass


def test_targeted_faults_skip_weights_whose_bit_is_already_set():
    """The bug this test exists for: the largest weights are the wrong target.

    A weight of 200 has |w| >= 2, so bit 14 is already 1 and flipping it divides by
    2**128 -- no damage at all. The policy must pass it over for the largest weight
    that the flip actually amplifies.
    """
    codes = {"w": from_float32(np.array([200.0, 0.9, 0.02], np.float32), BF16)}

    assert largest_magnitude_flips(codes, count=1) == [Flip("w", 1, TOP_EXPONENT_BIT)]


def test_targeted_faults_ignore_a_tensor_with_no_amplifying_candidate():
    codes = {
        "all_large": from_float32(np.array([200.0, 300.0], np.float32), BF16),
        "usable": from_float32(np.array([0.5], np.float32), BF16),
    }

    assert largest_magnitude_flips(codes, count=2) == [
        Flip("usable", 0, TOP_EXPONENT_BIT)
    ]


def test_the_chosen_target_amplifies_without_overflowing():
    """1.9 x 2**128 exceeds the bfloat16 maximum and would be NaN, not damage."""
    codes = from_float32(np.array([200.0, 1.9, 0.9], np.float32), BF16)
    flip = largest_magnitude_flips({"w": codes}, count=1)[0]

    before = abs(float(to_float32(codes[flip.index], BF16)))
    after = abs(float(to_float32(apply_flips(codes, [flip])[flip.index], BF16)))

    assert flip.index == 2
    assert after / before == 2.0**128


def test_overflow_can_be_allowed_explicitly():
    codes = from_float32(np.array([200.0, 1.9, 0.9], np.float32), BF16)

    flip = largest_magnitude_flips({"w": codes}, count=1, require_finite=False)[0]

    assert flip.index == 1
    assert not np.isfinite(to_float32(apply_flips(codes, [flip])[flip.index], BF16))


def test_parameter_codes_match_the_bfloat16_patterns():
    weights = torch.tensor([0.02, -1.5, 3.25, 0.0], dtype=torch.bfloat16)
    expected = weights.view(torch.int16).numpy().view(np.uint16)

    assert np.array_equal(parameter_codes(weights), expected)


def test_parameter_codes_read_the_same_patterns_out_of_a_float32_copy():
    """The promoted model must yield the identical codes, or targeting would drift."""
    weights = torch.tensor([0.02, -1.5, 3.25, 0.0], dtype=torch.bfloat16)

    assert np.array_equal(
        parameter_codes(weights.to(torch.float32)), parameter_codes(weights)
    )


def test_parameter_codes_flatten_a_matrix():
    weights = torch.zeros(3, 4, dtype=torch.bfloat16)

    assert parameter_codes(weights).shape == (12,)


def test_model_codes_cover_every_named_parameter():
    model = build_tiny_model()

    codes = model_codes(model)

    assert set(codes) == {"weight"}
    assert codes["weight"].shape == (12,)


def test_a_dtype_without_a_bfloat16_pattern_is_refused():
    with pytest.raises(TypeError, match="carries no bf16 pattern"):
        parameter_codes(torch.zeros(4, dtype=torch.float64))


# --- the collapse policy ---------------------------------------------------------
#
# The amplifying policy is unusable for E5: E2 measured a single chosen flip taking
# perplexity to NaN, because a weight near 3e38 overflows the activations even though
# the weight itself is representable. These faults divide instead, so nothing grows.


def test_collapse_faults_pick_the_largest_weights_and_the_collapse_bit():
    codes = {"w": from_float32(np.array([0.01, 0.5, 0.02, 0.9], np.float32), BF16)}

    flips = collapse_flips(codes, count=2)

    assert [flip.index for flip in flips] == [3, 1]
    assert {flip.bit for flip in flips} == {COLLAPSE_BIT}


def test_a_collapse_fault_divides_the_weight_instead_of_multiplying_it():
    """The property that makes this policy usable where the amplifying one is not:
    no value grows, so no activation downstream can overflow."""
    codes = from_float32(np.array([0.02], np.float32), BF16)

    after = to_float32(apply_flips(codes, [Flip("w", 0, COLLAPSE_BIT)]), BF16)

    assert after[0] == pytest.approx(0.02 / 2**64, rel=1e-2)
    assert np.isfinite(after[0])


def test_collapse_faults_skip_weights_whose_bit_is_already_zero():
    """A weight small enough that the bit is 0 would be *multiplied* by the same flip.
    Those are the ones this policy must not choose, and they are the mirror image of
    the trap the amplifying policy documents."""
    tiny = np.array([2.0**-80], np.float32)
    codes = {
        "w": from_float32(np.concatenate([tiny, np.array([0.02], np.float32)]), BF16)
    }

    flips = collapse_flips(codes, count=2)

    assert [flip.index for flip in flips] == [1]


def test_collapse_faults_rank_across_tensors():
    codes = {
        "small": from_float32(np.array([0.01, 0.02], np.float32), BF16),
        "large": from_float32(np.array([0.9, 0.03], np.float32), BF16),
    }

    assert collapse_flips(codes, count=1) == [Flip("large", 0, COLLAPSE_BIT)]


def test_asking_for_more_collapse_faults_than_there_are_weights_returns_what_exists():
    codes = {"w": from_float32(np.array([0.01, 0.02], np.float32), BF16)}

    assert len(collapse_flips(codes, count=10)) == 2


def test_collapse_faults_ignore_weights_that_are_not_finite():
    codes = {"w": np.array([0x7FC0, 0x3C00], dtype=np.uint16)}  # NaN, then 1.0

    assert [flip.index for flip in collapse_flips(codes, count=2)] == [1]
