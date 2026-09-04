"""The correspondence between a 16-bit pattern and the value it represents.

The project turns on one fact: the bits of a weight are not worth the same. This
module makes that computable -- no conversion goes through a third-party library, so
every figure we publish traces back to integer operations verifiable here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

SIGN = "sign"
EXPONENT = "exponent"
MANTISSA = "mantissa"


@dataclass(frozen=True)
class FloatFormat:
    """The geometry of a 16-bit floating-point format."""

    name: str
    exponent_bits: int
    mantissa_bits: int

    @property
    def total_bits(self) -> int:
        return 1 + self.exponent_bits + self.mantissa_bits

    @property
    def sign_position(self) -> int:
        return self.total_bits - 1

    @property
    def exponent_positions(self) -> range:
        return range(self.mantissa_bits, self.sign_position)

    @property
    def mantissa_positions(self) -> range:
        return range(self.mantissa_bits)

    @property
    def bias(self) -> int:
        return (1 << (self.exponent_bits - 1)) - 1


BF16 = FloatFormat(name="bf16", exponent_bits=8, mantissa_bits=7)
FP16 = FloatFormat(name="fp16", exponent_bits=5, mantissa_bits=10)

FORMATS = {fmt.name: fmt for fmt in (BF16, FP16)}


def _contiguous(values: ArrayLike, dtype: DTypeLike) -> np.ndarray:
    # ascontiguousarray promotes scalars to 1-d arrays; here the shape must be
    # preserved, because a caller passing one weight expects one value back.
    array = np.asarray(values, dtype=dtype)
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def _as_codes(values: ArrayLike) -> np.ndarray:
    return _contiguous(values, np.uint16)


def _bf16_to_float32(codes: np.ndarray) -> np.ndarray:
    return (codes.astype(np.uint32) << 16).view(np.float32)


def _float32_to_bf16(values: np.ndarray) -> np.ndarray:
    bits = values.view(np.uint32)
    # Round to nearest even: without this correction the conversion would truncate,
    # introducing a systematic bias towards zero across our samples.
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) >> 16
    truncated = bits >> np.uint32(16)
    return np.where(np.isnan(values), truncated, rounded).astype(np.uint16)


def _fp16_to_float32(codes: np.ndarray) -> np.ndarray:
    return codes.view(np.float16).astype(np.float32)


def _float32_to_fp16(values: np.ndarray) -> np.ndarray:
    return _contiguous(values.astype(np.float16), np.float16).view(np.uint16)


_CODECS = {
    BF16.name: (_bf16_to_float32, _float32_to_bf16),
    FP16.name: (_fp16_to_float32, _float32_to_fp16),
}


def to_float32(codes: ArrayLike, fmt: FloatFormat) -> np.ndarray:
    """The float32 values corresponding to the given bit patterns."""
    return _CODECS[fmt.name][0](_as_codes(codes))


def from_float32(values: ArrayLike, fmt: FloatFormat) -> np.ndarray:
    """The bit patterns corresponding to the given values."""
    return _CODECS[fmt.name][1](_contiguous(values, np.float32))


def flip_bit(codes: ArrayLike, position: int, fmt: FloatFormat) -> np.ndarray:
    """Flip the bit at `position` (0 is the least significant)."""
    if not 0 <= position < fmt.total_bits:
        raise ValueError(
            f"position {position} outside the {fmt.total_bits} bits of {fmt.name}"
        )
    return _as_codes(codes) ^ np.uint16(1 << position)


def field_at(position: int, fmt: FloatFormat) -> str:
    """The IEEE-754 field a position belongs to: sign, exponent or mantissa."""
    if position == fmt.sign_position:
        return SIGN
    if position in fmt.exponent_positions:
        return EXPONENT
    if position in fmt.mantissa_positions:
        return MANTISSA
    raise ValueError(
        f"position {position} outside the {fmt.total_bits} bits of {fmt.name}"
    )


def exponent_shift(position: int, fmt: FloatFormat) -> int:
    """How much the biased exponent changes when that bit is flipped."""
    if field_at(position, fmt) != EXPONENT:
        raise ValueError(f"position {position} of {fmt.name} is not an exponent bit")
    return 1 << (position - fmt.mantissa_bits)


def exponent_multiplier(position: int, fmt: FloatFormat) -> float:
    """The factor the value is multiplied by (0->1) or divided by (1->0).

    This is the law that makes the project interesting: an exponent bit at position p
    multiplies the weight by 2**(2**(p - mantissa_bits)). For bf16 the top bit is
    worth 2**128; for fp16, with three fewer exponent bits, it is worth 2**16.
    """
    return float(2.0 ** exponent_shift(position, fmt))


def compose(
    sign: ArrayLike, exponent: ArrayLike, mantissa: ArrayLike, fmt: FloatFormat
) -> np.ndarray:
    """Rebuild a pattern from its three fields. Inverse of `decompose`."""
    sign = _contiguous(sign, np.uint16)
    exponent = _contiguous(exponent, np.uint16)
    mantissa = _contiguous(mantissa, np.uint16)
    pattern = (
        (sign << np.uint16(fmt.sign_position))
        | (exponent << np.uint16(fmt.mantissa_bits))
        | mantissa
    )
    return np.asarray(pattern, dtype=np.uint16)


def decompose(
    codes: ArrayLike, fmt: FloatFormat
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split patterns into (sign, biased exponent, mantissa)."""
    codes = _as_codes(codes)
    sign = (codes >> fmt.sign_position) & np.uint16(1)
    exponent = (codes >> fmt.mantissa_bits) & np.uint16((1 << fmt.exponent_bits) - 1)
    mantissa = codes & np.uint16((1 << fmt.mantissa_bits) - 1)
    return sign, exponent, mantissa
