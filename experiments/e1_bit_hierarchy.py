"""E1 -- the fragility hierarchy of a weight's 16 bits, measured on real weights.

Every statistic is **exact**, not sampled: the histogram of the 65,536 patterns
summarises the whole model without loss, and from it every flip outcome follows by
construction. The per-bit-position analysis lives in `bitflip.fragility`, shared with E3.

Usage:  uv run python experiments/e1_bit_hierarchy.py
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
    """How small the weights are, and what that implies for the top exponent bit."""
    _, exponents, _ = decompose(ALL_CODES, BF16)
    total = float(counts.sum())
    return {
        "fraction_below_one": float(counts[exponents < BF16.bias].sum() / total),
        "median_exponent": weighted_quantile(
            exponents.astype(np.float64), counts.astype(np.float64), 0.5
        ),
    }


def summary(
    name: str, artifact: Artifact, counts: np.ndarray, rows: list[dict[str, object]]
) -> dict[str, object]:
    """The figures quoted in prose, in persistable form.

    No number this project publishes may live on screen only.
    """
    total = int(counts.sum())
    profile = exponent_profile(counts)
    fraction = catastrophic_bit_fraction(rows, BF16)
    return {
        "model": name,
        "artifact": artifact.key,
        "revision": artifact.revision,
        "weights": total,
        "total_bits": total * BF16.total_bits,
        "fraction_below_one": profile["fraction_below_one"],
        "median_exponent": profile["median_exponent"],
        "exponent_bias": BF16.bias,
        "catastrophic_bits": round(fraction * total * BF16.total_bits),
        "catastrophic_bit_fraction": fraction,
        "one_bit_in": 1 / fraction,
    }


def report(name: str, artifact: Artifact) -> tuple[list[dict[str, object]], dict]:
    with immutable([artifact.primary_path]):
        file = SafetensorsFile(artifact.primary_path)
        counts = code_histogram(file, BF16)

    total = int(counts.sum())
    if total != file.parameter_count:
        raise RuntimeError(f"histogram {total} != parameters {file.parameter_count}")

    rows = bit_rows(counts, BF16)
    profile = exponent_profile(counts)

    print(f"\n=== {name}: {total:,} bf16 weights across {len(file)} tensors ===")
    print(format_table(rows))
    print(
        f"weights with |w| < 1: {profile['fraction_below_one']:.4%} · "
        f"median exponent {profile['median_exponent']:.0f} (bias {BF16.bias})"
    )
    fraction = catastrophic_bit_fraction(rows, BF16)
    print(
        f"catastrophic bits: {fraction * total * BF16.total_bits:,.0f} of "
        f"{total * BF16.total_bits:,} = {fraction:.4%}, one in {1 / fraction:.2f}"
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
    for name, artifact in (("base", BASE), ("abliterated", ABLITERATED)):
        if not artifact.primary_path.exists():
            print(f"missing {artifact.primary_path}", file=sys.stderr)
            return 1
        rows, totals = report(name, artifact)
        write_csv(RESULTS_DIR / f"e1-bit-hierarchy-{artifact.key}.csv", rows)
        summaries.append(totals)
    write_csv(RESULTS_DIR / "e1-summary.csv", summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
