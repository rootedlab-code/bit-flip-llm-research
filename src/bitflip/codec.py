"""Corrispondenza tra il pattern di 16 bit e il valore che rappresenta.

Il progetto ruota attorno a un fatto: i bit di un peso non valgono uguale. Questo
modulo lo rende calcolabile — nessuna conversione passa da una libreria di terze
parti, cosi ogni cifra pubblicata risale a operazioni intere verificabili qui.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIGN = "sign"
EXPONENT = "exponent"
MANTISSA = "mantissa"


@dataclass(frozen=True)
class FloatFormat:
    """Geometria di un formato in virgola mobile a 16 bit."""

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
        return 2 ** (self.exponent_bits - 1) - 1


BF16 = FloatFormat(name="bf16", exponent_bits=8, mantissa_bits=7)
FP16 = FloatFormat(name="fp16", exponent_bits=5, mantissa_bits=10)

FORMATS = {fmt.name: fmt for fmt in (BF16, FP16)}


def _contiguous(values, dtype) -> np.ndarray:
    # ascontiguousarray promuove gli scalari a array 1-d: qui la forma va conservata,
    # perche il chiamante che passa un peso singolo si aspetta un valore singolo.
    array = np.asarray(values, dtype=dtype)
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def _as_codes(values) -> np.ndarray:
    return _contiguous(values, np.uint16)


def _bf16_to_float32(codes: np.ndarray) -> np.ndarray:
    return (codes.astype(np.uint32) << 16).view(np.float32)


def _float32_to_bf16(values: np.ndarray) -> np.ndarray:
    bits = values.view(np.uint32)
    # Arrotondamento al pari piu vicino: senza questa correzione la conversione
    # troncherebbe, introducendo un bias sistematico verso lo zero nei nostri campioni.
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


def to_float32(codes, fmt: FloatFormat) -> np.ndarray:
    """Valori float32 corrispondenti ai pattern di bit dati."""
    return _CODECS[fmt.name][0](_as_codes(codes))


def from_float32(values, fmt: FloatFormat) -> np.ndarray:
    """Pattern di bit corrispondenti ai valori dati."""
    return _CODECS[fmt.name][1](_contiguous(values, np.float32))


def flip_bit(codes, position: int, fmt: FloatFormat) -> np.ndarray:
    """Ribalta il bit in posizione `position` (0 = meno significativo)."""
    if not 0 <= position < fmt.total_bits:
        raise ValueError(
            f"posizione {position} fuori dai {fmt.total_bits} bit di {fmt.name}"
        )
    return _as_codes(codes) ^ np.uint16(1 << position)


def field_at(position: int, fmt: FloatFormat) -> str:
    """Campo IEEE-754 a cui appartiene la posizione: segno, esponente o mantissa."""
    if position == fmt.sign_position:
        return SIGN
    if position in fmt.exponent_positions:
        return EXPONENT
    if position in fmt.mantissa_positions:
        return MANTISSA
    raise ValueError(f"posizione {position} fuori dai {fmt.total_bits} bit di {fmt.name}")


def exponent_shift(position: int, fmt: FloatFormat) -> int:
    """Di quanto cambia l'esponente polarizzato ribaltando quel bit."""
    if field_at(position, fmt) != EXPONENT:
        raise ValueError(
            f"la posizione {position} di {fmt.name} non e un bit di esponente"
        )
    return 2 ** (position - fmt.mantissa_bits)


def exponent_multiplier(position: int, fmt: FloatFormat) -> float:
    """Fattore per cui il valore viene moltiplicato (bit 0->1) o diviso (1->0).

    E la legge che rende il progetto interessante: un bit di esponente in posizione p
    moltiplica il peso per 2**(2**(p - mantissa_bits)). Per bf16 il bit alto vale
    2**128; per fp16, con tre bit di esponente in meno, vale 2**16.
    """
    return float(2.0 ** exponent_shift(position, fmt))


def compose(sign, exponent, mantissa, fmt: FloatFormat) -> np.ndarray:
    """Ricompone un pattern dai suoi tre campi. Inversa di `decompose`."""
    sign = _contiguous(sign, np.uint16)
    exponent = _contiguous(exponent, np.uint16)
    mantissa = _contiguous(mantissa, np.uint16)
    return (
        (sign << np.uint16(fmt.sign_position))
        | (exponent << np.uint16(fmt.mantissa_bits))
        | mantissa
    ).astype(np.uint16)


def decompose(codes, fmt: FloatFormat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scompone i pattern in (segno, esponente polarizzato, mantissa)."""
    codes = _as_codes(codes)
    sign = (codes >> fmt.sign_position) & np.uint16(1)
    exponent = (codes >> fmt.mantissa_bits) & np.uint16((1 << fmt.exponent_bits) - 1)
    mantissa = codes & np.uint16((1 << fmt.mantissa_bits) - 1)
    return sign, exponent, mantissa
