"""Esiti di un flip, per posizione di bit, calcolati sull'intero spazio dei pattern.

Lo spazio dei pattern a 16 bit ha 65.536 elementi: dove si puo enumerare tutto,
stimare e una scelta pigra. Ogni frazione prodotta qui e un conteggio esatto, pesato
dall'istogramma della popolazione osservata — pesi di un modello in E1, scale di
blocco in E3.
"""

from __future__ import annotations

import numpy as np

from bitflip.codec import FloatFormat, field_at, flip_bit, to_float32
from bitflip.stats import weighted_quantile

CODE_SPACE = 1 << 16
CATASTROPHIC_RATIO = 2.0**16
AMPLIFYING_RATIO = 2.0
ALL_CODES = np.arange(CODE_SPACE, dtype=np.uint16)


def values_of(fmt: FloatFormat) -> np.ndarray:
    with np.errstate(invalid="ignore", over="ignore"):
        return to_float32(ALL_CODES, fmt).astype(np.float64)


def flip_outcomes(fmt: FloatFormat, position: int) -> tuple[np.ndarray, ...]:
    """Per ogni pattern: |Δ|, rapporto |dopo/prima|, e se l'esito resta finito."""
    before = values_of(fmt)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        after = to_float32(flip_bit(ALL_CODES, position, fmt), fmt).astype(np.float64)
        delta = np.abs(after - before)
        ratio = np.where(before != 0, np.abs(after / before), np.inf)
    return delta, ratio, np.isfinite(after)


def bit_rows(counts: np.ndarray, fmt: FloatFormat) -> list[dict[str, object]]:
    """Una riga per posizione di bit, pesata sulla popolazione osservata."""
    if counts.shape != (CODE_SPACE,):
        raise ValueError(f"istogramma di {counts.shape}, atteso ({CODE_SPACE},)")
    total = float(counts.sum())
    if total == 0:
        raise ValueError("popolazione vuota")

    with np.errstate(invalid="ignore"):
        usable = (counts > 0) & np.isfinite(values_of(fmt))
    weights = counts[usable].astype(np.float64)
    rows = []

    for position in range(fmt.total_bits):
        delta, ratio, finite = flip_outcomes(fmt, position)
        catastrophic = ~finite | (ratio >= CATASTROPHIC_RATIO)
        finite_delta = delta[usable & finite]

        rows.append(
            {
                "bit": position,
                "campo": field_at(position, fmt),
                "frazione_bit_a_zero": float(
                    counts[((ALL_CODES >> position) & 1) == 0].sum() / total
                ),
                "delta_mediano": weighted_quantile(delta[usable], weights, 0.5),
                "delta_p99": weighted_quantile(delta[usable], weights, 0.99),
                "delta_massimo_finito": (
                    float(finite_delta.max()) if finite_delta.size else 0.0
                ),
                "frazione_amplificati": float(
                    counts[usable & (ratio >= AMPLIFYING_RATIO)].sum() / total
                ),
                "frazione_non_finiti": float(counts[usable & ~finite].sum() / total),
                "frazione_catastrofici": float(
                    counts[usable & catastrophic].sum() / total
                ),
            }
        )
    return rows


def catastrophic_bit_fraction(rows: list[dict[str, object]], fmt: FloatFormat) -> float:
    """Frazione dei bit della popolazione il cui flip e catastrofico."""
    return sum(float(row["frazione_catastrofici"]) for row in rows) / fmt.total_bits


def format_table(rows: list[dict[str, object]]) -> str:
    header = (
        f"{'bit':>3} {'campo':<9} {'bit=0':>8} {'|Δ| mediano':>13} "
        f"{'|Δ| p99':>11} {'|Δ| max':>11} {'≥×2':>9} {'catastr.':>10}"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row['bit']:>3} {row['campo']:<9} {row['frazione_bit_a_zero']:>7.2%} "
            f"{row['delta_mediano']:>13.3e} {row['delta_p99']:>11.3e} "
            f"{row['delta_massimo_finito']:>11.3e} {row['frazione_amplificati']:>8.2%} "
            f"{row['frazione_catastrofici']:>10.4%}"
        )
    return "\n".join(lines)
