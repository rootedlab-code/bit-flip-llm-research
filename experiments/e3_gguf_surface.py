"""E3 -- the critical surface of a quantized model.

Does quantization protect? The question is not settled by counting bits but by
separating them by **function**: quants are integers, and a flip moves them by a
bounded step; scales are fp16, and a flip in their top exponent bit multiplies *all*
the weights of the block they govern by 65,536. Here the two populations are counted
and, with the same exact method as E1, the cost of a random fault is measured in both
formats.

Usage:  uv run python experiments/e3_gguf_surface.py
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
    """Histograms of the fp16 scale patterns, split by block size."""
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
    """How many bits, in the given population, are catastrophic when flipped."""
    return sum(float(row["catastrophic_fraction"]) for row in rows) * population


def main() -> int:
    if not QUANTIZED.primary_path.exists() or not BASE.primary_path.exists():
        print(
            "missing artifacts: run `uv run python -m bitflip.fetch`",
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
        f"{len(gguf)} tensors · {gguf.parameter_count:,} weights · "
        f"{gguf_bits:,} data bits"
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

        print(f"\n--- fp16 scales of {elements}-weight blocks: {scales:,} ---")
        print(format_table(rows))
        print(
            f"catastrophic bits in this population: {bits:,.0f} "
            f"({catastrophic_bit_fraction(rows, FP16):.4%} of their bits), "
            f"blast radius {elements} weights each"
        )
        for row in rows:
            all_rows.append({"block_elements": elements, **row})

    bf16_rows = bit_rows(bf16_counts, BF16)
    bf16_total_bits = safetensors.parameter_count * BF16.total_bits
    bf16_catastrophic = catastrophic_bits(bf16_rows, safetensors.parameter_count)

    comparison = [
        {
            "format": "bf16 safetensors",
            "artifact": BASE.key,
            "weights": safetensors.parameter_count,
            "total_bits": bf16_total_bits,
            "catastrophic_bits": round(bf16_catastrophic),
            "catastrophic_bit_share": bf16_catastrophic / bf16_total_bits,
            "mean_blast_radius": 1.0,
            "weights_lost_per_random_flip": bf16_catastrophic / bf16_total_bits,
        },
        {
            "format": "gguf q4_k_m",
            "artifact": QUANTIZED.key,
            "weights": gguf.parameter_count,
            "total_bits": gguf_bits,
            "catastrophic_bits": round(catastrophic_scale_bits),
            "catastrophic_bit_share": catastrophic_scale_bits / gguf_bits,
            "mean_blast_radius": damaged_weights / catastrophic_scale_bits,
            "weights_lost_per_random_flip": damaged_weights / gguf_bits,
        },
    ]

    print("\n=== comparison: what one fault in a random bit costs ===")
    print(
        f"{'format':<22} {'total bits':>16} {'catastrophic':>15} "
        f"{'share':>9} {'radius':>8} {'weights lost':>12}"
    )
    for row in comparison:
        print(
            f"{row['format']:<22} {row['total_bits']:>16,} "
            f"{row['catastrophic_bits']:>15,} {row['catastrophic_bit_share']:>8.4%} "
            f"{row['mean_blast_radius']:>8.1f} "
            f"{row['weights_lost_per_random_flip']:>12.6f}"
        )

    bf16_expected = bf16_catastrophic / bf16_total_bits
    gguf_expected = damaged_weights / gguf_bits
    print(
        f"\nratio of weights lost per random flip, gguf / bf16: "
        f"{gguf_expected / bf16_expected:.3f}"
    )

    with (RESULTS_DIR / "e3-gguf-scale-fragility.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (RESULTS_DIR / "e3-comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    with (RESULTS_DIR / "e3-gguf-bit-census.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["role", "bits", "share"])
        for kind, bits in sorted(census.items(), key=lambda item: -item[1]):
            writer.writerow([kind, bits, bits / gguf_bits])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
