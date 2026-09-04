"""Fault injection into weights: in memory, and always reversible.

Constitution Principles I and III: no modified weight touches the disk, and the choice
of bits is reproducible from a seed. Selection is pure numpy arithmetic, so it is
verified locally without models; applying it to a real model is a thin adapter at the
bottom of this file.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import DTypeLike

from bitflip.codec import BF16, FloatFormat

if TYPE_CHECKING:
    import torch

TOP_EXPONENT_BIT = 14
# The strongest exponent bit whose downward flip removes a weight without any value
# growing: 1 -> 0 here divides by 2**64. Bits 11 and 12 do the same at 2**16 and
# 2**32. E1's spectrum names this the collapse channel; it is 18.74% of the bit
# space against the catastrophic 6.26%.
COLLAPSE_BIT = 13


@dataclass(frozen=True)
class Flip:
    """A fault: which tensor, which weight, which bit."""

    tensor: str
    index: int
    bit: int


def random_flips(
    sizes: Mapping[str, int],
    count: int,
    seed: int,
    bits: Sequence[int] | None = None,
) -> list[Flip]:
    """Faults uniform over every bit of every weight -- the cosmic-ray model.

    Uniform means proportional to tensor size: a large tensor is hit more often
    because it offers more targets, not because it matters more.
    """
    if count < 0:
        raise ValueError(f"negative fault count: {count}")
    names = list(sizes)
    if not names:
        raise ValueError("no tensor to inject into")
    positions = list(bits) if bits is not None else list(range(BF16.total_bits))

    boundaries = np.cumsum([sizes[name] for name in names])
    generator = np.random.default_rng(seed)
    drawn = generator.integers(0, int(boundaries[-1]), size=count)
    chosen_tensor = np.searchsorted(boundaries, drawn, side="right")
    starts = np.concatenate(([0], boundaries[:-1]))
    chosen_bits = generator.choice(positions, size=count)

    return [
        Flip(names[tensor], int(offset - starts[tensor]), int(bit))
        for tensor, offset, bit in zip(chosen_tensor, drawn, chosen_bits, strict=True)
    ]


def largest_magnitude_flips(
    codes: Mapping[str, np.ndarray],
    count: int,
    fmt: FloatFormat = BF16,
    bit: int = TOP_EXPONENT_BIT,
    require_finite: bool = True,
) -> list[Flip]:
    """Chosen faults: the largest weight the flip amplifies while staying finite.

    Two traps sit between the naive reading of E1 and a flip that actually does damage.

    The first: "bit 14 is zero in 100% of weights, so hit the largest weight" is wrong.
    The weights of largest magnitude are precisely those with |w| >= 2, which is exactly
    the condition for bit 14 to already be **set**; flipping it there divides by 2**128
    rather than multiplying, and the attack quietly does nothing.

    The second: among the weights the flip does amplify, the largest ones overflow. A
    weight in [1, 2) multiplied by 2**128 exceeds the bfloat16 maximum and becomes NaN,
    which destroys the model outright instead of corrupting it. When `require_finite`
    is set, those are excluded too, and what is chosen is the largest weight the flip
    can amplify while leaving a representable number behind.
    """
    from bitflip.codec import flip_bit, to_float32

    ranked: list[tuple[float, str, int]] = []
    for name, tensor_codes in codes.items():
        # A flipped pattern can land on a signalling NaN -- exponent all ones, quiet bit
        # clear -- and promoting one to float64 raises the invalid flag by design. The
        # result is a quiet NaN that `isfinite` excludes, so the flag carries nothing.
        with np.errstate(invalid="ignore"):
            values = to_float32(tensor_codes, fmt).astype(np.float64)
            after = to_float32(flip_bit(tensor_codes, bit, fmt), fmt).astype(np.float64)
        usable = (((tensor_codes >> np.uint16(bit)) & np.uint16(1)) == 0) & np.isfinite(
            values
        )
        if require_finite:
            usable &= np.isfinite(after)
        positions = np.flatnonzero(usable)
        if positions.size == 0:
            continue
        magnitudes = np.abs(values[positions])
        take = min(count, magnitudes.size)
        top = np.argpartition(magnitudes, -take)[-take:]
        ranked.extend(
            (float(magnitudes[index]), name, int(positions[index])) for index in top
        )

    ranked.sort(key=lambda item: -item[0])
    return [Flip(name, index, bit) for _, name, index in ranked[:count]]


def collapse_flips(
    codes: Mapping[str, np.ndarray],
    count: int,
    fmt: FloatFormat = BF16,
    bit: int = COLLAPSE_BIT,
) -> list[Flip]:
    """Chosen faults in the other direction: remove the largest weight, do not explode it.

    `largest_magnitude_flips` amplifies, and E2 measured what that costs: a single chosen
    flip takes perplexity to NaN and top-1 agreement to zero. Even with `require_finite`
    the surviving weight is around 3e38, and it overflows the activations downstream. A
    model that has become NaN has not lost its alignment quietly; it has stopped being a
    model, and E5 is asking what happens *underneath* that.

    This policy flips an exponent bit that is already 1, downward, so the weight is
    divided rather than multiplied -- by 2**64 at the default bit. On a typical weight of
    0.02 that leaves 1e-21, which is removal, not perturbation. Nothing can overflow,
    because no value grows: the failure mode that makes the amplifying policy unusable
    for this question cannot arise here.

    Weights are ranked by magnitude, exactly as in the amplifying policy. Selection is
    therefore a property of the weights and not of any alignment measurement -- choosing
    targets by what the oracle says about them would tune the attack against the
    hypothesis it is meant to test.

    `require_finite` has no counterpart here and is deliberately absent rather than
    accepted and ignored: dividing a finite number by 2**64 cannot leave anything that
    is not representable.
    """
    from bitflip.codec import to_float32

    ranked: list[tuple[float, str, int]] = []
    for name, tensor_codes in codes.items():
        # See `largest_magnitude_flips`: a stored pattern can itself be a signalling NaN.
        with np.errstate(invalid="ignore"):
            values = to_float32(tensor_codes, fmt).astype(np.float64)
        # Eligible where the bit is already set, which is where the flip goes downward.
        # In bf16 that is almost everywhere: E1 measures bit 13 as 1 in 99.998% of
        # weights, so the policy is not short of targets.
        usable = (((tensor_codes >> np.uint16(bit)) & np.uint16(1)) == 1) & np.isfinite(
            values
        )
        positions = np.flatnonzero(usable)
        if positions.size == 0:
            continue
        magnitudes = np.abs(values[positions])
        take = min(count, magnitudes.size)
        top = np.argpartition(magnitudes, -take)[-take:]
        ranked.extend(
            (float(magnitudes[index]), name, int(positions[index])) for index in top
        )

    ranked.sort(key=lambda item: -item[0])
    return [Flip(name, index, bit) for _, name, index in ranked[:count]]


def apply_flips(codes: np.ndarray, flips: Sequence[Flip]) -> np.ndarray:
    """Apply the faults to an array of patterns, returning a modified copy."""
    modified = np.array(codes, dtype=np.uint16, copy=True)
    for flip in flips:
        modified[flip.index] ^= np.uint16(1 << flip.bit)
    return modified


def _integer_view(parameter: torch.Tensor) -> tuple[torch.Tensor, int, DTypeLike]:
    """An integer view of a parameter, with the bit shift its dtype imposes.

    A bf16 weight promoted to float32 keeps its pattern **exactly** in the top 16 bits,
    because the bf16 -> float32 conversion zero-fills the low bits. Bit `b` of the
    stored bf16 is therefore bit `b + 16` of the float32 holding it. This is what
    allows computing in float32 on GPUs without native bf16 -- such as Kaggle's T4s --
    while staying faithful to the fault that happens in DRAM.
    """
    import torch

    flat = parameter.view(-1)
    if flat.dtype == torch.bfloat16:
        return flat.view(torch.int16), 0, np.int16
    if flat.dtype == torch.float32:
        return flat.view(torch.int32), 16, np.int32
    raise TypeError(f"dtype {flat.dtype} is not supported for injection")


def parameter_codes(parameter: torch.Tensor) -> np.ndarray:
    """The stored bf16 pattern of every weight in a torch parameter.

    numpy has no bfloat16, so the tensor cannot simply be converted. It does not need
    to be: a bf16 weight promoted to float32 keeps its pattern in the top 16 bits, so
    the codes are a shift away from the integer view -- the same identity that lets the
    injection work on a float32 model in the first place.
    """
    import torch

    flat = parameter.detach().reshape(-1).cpu()
    if flat.dtype == torch.float32:
        raw = np.asarray(flat.view(torch.int32).numpy()).astype(np.uint32)
        return (raw >> 16).astype(np.uint16)
    if flat.dtype == torch.bfloat16:
        return np.asarray(flat.view(torch.int16).numpy()).view(np.uint16)
    raise TypeError(f"dtype {flat.dtype} carries no bf16 pattern")


def model_codes(model: torch.nn.Module) -> dict[str, np.ndarray]:
    """The bf16 patterns of every named parameter, ready for target selection."""
    return {name: parameter_codes(p) for name, p in model.named_parameters()}


@contextmanager
def flipped_model(
    model: torch.nn.Module, flips: Sequence[Flip]
) -> Iterator[dict[str, int]]:
    """Apply the faults to a torch model and **always** undo them on exit.

    The original values are kept in memory and restored even if the body fails:
    without that guarantee a later measurement would inherit the previous one's damage,
    and nobody would notice.
    """
    import torch

    parameters = dict(model.named_parameters())
    originals: list[tuple[str, int, int]] = []

    try:
        with torch.no_grad():
            for flip in flips:
                view, shift, dtype = _integer_view(parameters[flip.tensor])
                originals.append((flip.tensor, flip.index, int(view[flip.index])))
                mask = int(np.array(1 << (flip.bit + shift)).astype(dtype))
                view[flip.index] ^= mask
        yield {"applied": len(flips)}
    finally:
        with torch.no_grad():
            for name, index, value in reversed(originals):
                view, _, _ = _integer_view(parameters[name])
                view[index] = value
