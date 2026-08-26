"""E1 — gerarchia di fragilita dei 16 bit di un peso, misurata sui pesi veri.

Ogni statistica e **esatta**, non campionata: l'istogramma dei 65.536 pattern riassume
senza perdita l'intero modello, e da li ogni esito di flip si calcola per costruzione.
L'analisi per posizione di bit vive in `bitflip.fragility`, condivisa con E3.

Uso:  uv run python experiments/e1_bit_hierarchy.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from bitflip.codec import BF16, decompose
from bitflip.fetch import ABLITERATED, BASE, PROJECT_ROOT, Artifact
from bitflip.fragility import ALL_CODES, bit_rows, catastrophic_bit_fraction, format_table
from bitflip.guard import immutable
from bitflip.stats import weighted_quantile
from bitflip.weights import SafetensorsFile, code_histogram

RESULTS_DIR = PROJECT_ROOT / "results"


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


def summary(
    name: str, artifact: Artifact, counts: np.ndarray, rows: list[dict[str, object]]
) -> dict[str, object]:
    """Le cifre citate a testo, in forma persistibile: nessun numero solo a schermo."""
    total = int(counts.sum())
    profile = exponent_profile(counts)
    fraction = catastrophic_bit_fraction(rows, BF16)
    return {
        "modello": name,
        "artefatto": artifact.key,
        "revisione": artifact.revision,
        "pesi": total,
        "bit_totali": total * BF16.total_bits,
        "frazione_sotto_uno": profile["frazione_sotto_uno"],
        "esponente_mediano": profile["esponente_mediano"],
        "bias_esponente": BF16.bias,
        "bit_catastrofici": round(fraction * total * BF16.total_bits),
        "frazione_bit_catastrofici": fraction,
        "un_bit_ogni": 1 / fraction,
    }


def report(name: str, artifact: Artifact) -> tuple[list[dict[str, object]], dict]:
    with immutable([artifact.primary_path]):
        file = SafetensorsFile(artifact.primary_path)
        counts = code_histogram(file, BF16)

    total = int(counts.sum())
    if total != file.parameter_count:
        raise RuntimeError(f"istogramma {total} != parametri {file.parameter_count}")

    rows = bit_rows(counts, BF16)
    profile = exponent_profile(counts)

    print(f"\n=== {name}: {total:,} pesi bf16 in {len(file)} tensori ===")
    print(format_table(rows))
    print(
        f"pesi con |w| < 1: {profile['frazione_sotto_uno']:.4%} · "
        f"esponente mediano {profile['esponente_mediano']:.0f} (bias {BF16.bias})"
    )
    fraction = catastrophic_bit_fraction(rows, BF16)
    print(
        f"bit catastrofici: {fraction * total * BF16.total_bits:,.0f} su "
        f"{total * BF16.total_bits:,} = {fraction:.4%}, uno ogni {1 / fraction:.2f}"
    )
    return rows, summary(name, artifact, counts, rows)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    summaries = []
    for name, artifact in (("base", BASE), ("abliterato", ABLITERATED)):
        if not artifact.primary_path.exists():
            print(f"manca {artifact.primary_path}", file=sys.stderr)
            return 1
        rows, totals = report(name, artifact)
        write_csv(RESULTS_DIR / f"e1-bit-hierarchy-{artifact.key}.csv", rows)
        summaries.append(totals)
    write_csv(RESULTS_DIR / "e1-summary.csv", summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
