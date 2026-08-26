"""E3 — la superficie critica di un modello quantizzato.

La quantizzazione protegge? La domanda non si decide contando i bit ma separandoli per
**funzione**: i quanti sono interi, e un flip li sposta di un gradino limitato; le
scale sono fp16, e un flip nel loro esponente alto moltiplica per 65.536 *tutti* i
pesi del blocco che governano. Qui si contano le due popolazioni e si misura, con lo
stesso metodo esatto di E1, quanto vale un guasto casuale nei due formati.

Uso:  uv run python experiments/e3_gguf_surface.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np

from bitflip.codec import BF16, FP16
from bitflip.fetch import BASE, PROJECT_ROOT, QUANTIZED
from bitflip.fragility import (
    CODE_SPACE,
    bit_rows,
    catastrophic_bit_fraction,
    format_table,
)
from bitflip.gguf import SCALE_FP16, GGUFFile
from bitflip.guard import immutable
from bitflip.weights import SafetensorsFile, code_histogram

RESULTS_DIR = PROJECT_ROOT / "results"


def scale_histograms(file: GGUFFile) -> dict[int, np.ndarray]:
    """Istogrammi dei pattern delle scale fp16, separati per dimensione del blocco."""
    histograms: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros(CODE_SPACE, dtype=np.uint64)
    )
    for tensor in file.tensors:
        codes = file.field_codes(tensor, SCALE_FP16)
        if codes.size:
            histograms[tensor.layout.elements] += np.bincount(
                codes, minlength=CODE_SPACE
            ).astype(np.uint64)
    return dict(histograms)


def catastrophic_bits(rows: list[dict[str, object]], population: int) -> float:
    """Quanti bit, nella popolazione data, sono catastrofici se ribaltati."""
    return sum(float(row["frazione_catastrofici"]) for row in rows) * population


def main() -> int:
    if not QUANTIZED.primary_path.exists() or not BASE.primary_path.exists():
        print(
            "artefatti mancanti: eseguire `uv run python -m bitflip.fetch`",
            file=sys.stderr,
        )
        return 1
    RESULTS_DIR.mkdir(exist_ok=True)

    with immutable([QUANTIZED.primary_path, BASE.primary_path]):
        gguf = GGUFFile(QUANTIZED.primary_path)
        census = gguf.bit_census()
        histograms = scale_histograms(gguf)
        safetensors = SafetensorsFile(BASE.primary_path)
        bf16_counts = code_histogram(safetensors, BF16)

    gguf_bits = sum(census.values())
    print(f"=== {QUANTIZED.primary_path.name} ===")
    print(
        f"{len(gguf)} tensori · {gguf.parameter_count:,} pesi · {gguf_bits:,} bit di dati"
    )
    for kind, bits in sorted(census.items(), key=lambda item: -item[1]):
        print(f"  {kind:<14} {bits:>15,} bit  {bits / gguf_bits:>9.5%}")

    all_rows = []
    catastrophic_scale_bits = 0.0
    damaged_weights = 0.0
    for elements, counts in sorted(histograms.items()):
        scales = int(counts.sum())
        rows = bit_rows(counts, FP16)
        bits = catastrophic_bits(rows, scales)
        catastrophic_scale_bits += bits
        damaged_weights += bits * elements

        print(f"\n--- scale fp16 di blocchi da {elements} pesi: {scales:,} scale ---")
        print(format_table(rows))
        print(
            f"bit catastrofici in questa popolazione: {bits:,.0f} "
            f"({catastrophic_bit_fraction(rows, FP16):.4%} dei loro bit), "
            f"raggio {elements} pesi ciascuno"
        )
        for row in rows:
            all_rows.append({"blocco_elementi": elements, **row})

    bf16_rows = bit_rows(bf16_counts, BF16)
    bf16_total_bits = safetensors.parameter_count * BF16.total_bits
    bf16_catastrophic = catastrophic_bits(bf16_rows, safetensors.parameter_count)

    print("\n=== confronto: quanto costa un guasto in un bit a caso ===")
    print(
        f"{'formato':<22} {'bit totali':>16} {'catastrofici':>15} "
        f"{'quota':>9} {'raggio':>8} {'pesi persi':>12}"
    )
    for name, total_bits, catastrophic, radius in (
        ("bf16 safetensors", bf16_total_bits, bf16_catastrophic, 1.0),
        (
            "gguf q4_k_m",
            gguf_bits,
            catastrophic_scale_bits,
            damaged_weights / catastrophic_scale_bits,
        ),
    ):
        expected = catastrophic * radius / total_bits
        print(
            f"{name:<22} {total_bits:>16,} {catastrophic:>15,.0f} "
            f"{catastrophic / total_bits:>8.4%} {radius:>8.1f} {expected:>12.6f}"
        )

    bf16_expected = bf16_catastrophic / bf16_total_bits
    gguf_expected = damaged_weights / gguf_bits
    print(
        f"\nrapporto pesi persi per flip casuale, gguf / bf16: "
        f"{gguf_expected / bf16_expected:.3f}"
    )

    with (RESULTS_DIR / "e3-gguf-scale-fragility.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (RESULTS_DIR / "e3-gguf-bit-census.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ruolo", "bit", "quota"])
        for kind, bits in sorted(census.items(), key=lambda item: -item[1]):
            writer.writerow([kind, bits, bits / gguf_bits])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
