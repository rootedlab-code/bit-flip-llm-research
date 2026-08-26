"""E1 — gerarchia di fragilita dei 16 bit di un peso, misurata sui pesi veri.

Ogni statistica e **esatta**, non campionata: l'istogramma dei 65.536 pattern riassume
senza perdita l'intero modello, e da li ogni esito di flip si calcola per costruzione.

Uso:  uv run python experiments/e1_bit_hierarchy.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from bitflip.codec import BF16, decompose, field_at, flip_bit, to_float32
from bitflip.fetch import ABLITERATED, BASE, PROJECT_ROOT, Artifact
from bitflip.guard import immutable
from bitflip.stats import weighted_quantile
from bitflip.weights import CODE_SPACE, SafetensorsFile, code_histogram

RESULTS_DIR = PROJECT_ROOT / "results"
CATASTROPHIC_RATIO = 2.0**16
AMPLIFYING_RATIO = 2.0
ALL_CODES = np.arange(CODE_SPACE, dtype=np.uint16)


def flip_outcomes(position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per ogni pattern possibile: |Δw|, rapporto |w'/w|, e se l'esito resta finito."""
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        before = to_float32(ALL_CODES, BF16).astype(np.float64)
        after = to_float32(flip_bit(ALL_CODES, position, BF16), BF16).astype(np.float64)
        delta = np.abs(after - before)
        ratio = np.where(before != 0, np.abs(after / before), np.inf)
    return delta, ratio, np.isfinite(after)


def analyse(counts: np.ndarray) -> list[dict[str, object]]:
    with np.errstate(invalid="ignore"):
        usable = (counts > 0) & np.isfinite(
            to_float32(ALL_CODES, BF16).astype(np.float64)
        )
    total = float(counts.sum())
    weights = counts[usable].astype(np.float64)
    rows = []

    for position in range(BF16.total_bits):
        delta, ratio, finite = flip_outcomes(position)
        catastrophic = ~finite | (ratio >= CATASTROPHIC_RATIO)
        finite_delta = delta[usable & finite]
        zero_here = ((ALL_CODES >> position) & 1) == 0

        rows.append(
            {
                "bit": position,
                "campo": field_at(position, BF16),
                "frazione_bit_a_zero": float(counts[zero_here].sum() / total),
                "delta_mediano": weighted_quantile(delta[usable], weights, 0.5),
                "delta_p99": weighted_quantile(delta[usable], weights, 0.99),
                "delta_massimo_finito": float(finite_delta.max())
                if finite_delta.size
                else 0.0,
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


def exponent_profile(counts: np.ndarray) -> dict[str, float]:
    """Quanto sono piccoli i pesi, e cosa implica per il bit alto dell'esponente."""
    _, exponents, _ = decompose(ALL_CODES, BF16)
    total = float(counts.sum())
    return {
        "frazione_sotto_uno": float(counts[exponents < BF16.bias].sum() / total),
        "esponente_mediano": weighted_quantile(
            exponents.astype(np.float64), counts.astype(np.float64), 0.5
        ),
    }


def report(name: str, artifact: Artifact) -> list[dict[str, object]]:
    with immutable([artifact.primary_path]):
        file = SafetensorsFile(artifact.primary_path)
        counts = code_histogram(file, BF16)

    total = int(counts.sum())
    if total != file.parameter_count:
        raise RuntimeError(f"istogramma {total} != parametri {file.parameter_count}")

    rows = analyse(counts)
    profile = exponent_profile(counts)

    print(f"\n=== {name}: {total:,} pesi bf16 in {len(file)} tensori ===")
    header = (
        f"{'bit':>3} {'campo':<9} {'bit=0':>8} {'|Δw| mediano':>13} "
        f"{'|Δw| p99':>11} {'|Δw| max':>11} {'≥×2':>9} {'catastr.':>10}"
    )
    print(header)
    for row in rows:
        print(
            f"{row['bit']:>3} {row['campo']:<9} {row['frazione_bit_a_zero']:>7.2%} "
            f"{row['delta_mediano']:>13.3e} {row['delta_p99']:>11.3e} "
            f"{row['delta_massimo_finito']:>11.3e} {row['frazione_amplificati']:>8.2%} "
            f"{row['frazione_catastrofici']:>10.4%}"
        )

    catastrophic_bits = sum(row["frazione_catastrofici"] for row in rows) * total
    total_bits = total * BF16.total_bits
    print(
        f"pesi con |w| < 1: {profile['frazione_sotto_uno']:.4%} · "
        f"esponente mediano {profile['esponente_mediano']:.0f} (bias {BF16.bias})"
    )
    print(
        f"bit catastrofici: {catastrophic_bits:,.0f} su {total_bits:,} = "
        f"{catastrophic_bits / total_bits:.4%}, uno ogni "
        f"{total_bits / catastrophic_bits:.2f}"
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    for name, artifact in (("base", BASE), ("abliterato", ABLITERATED)):
        if not artifact.primary_path.exists():
            print(f"manca {artifact.primary_path}", file=sys.stderr)
            return 1
        write_csv(
            RESULTS_DIR / f"e1-bit-hierarchy-{artifact.key}.csv", report(name, artifact)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
