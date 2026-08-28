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

from bitflip.codec import BF16
from bitflip.fetch import ABLITERATED, BASE, PROJECT_ROOT, Artifact
from bitflip.fragility import (
    bit_rows,
    catastrophic_bit_fraction,
    exponent_profile,
    format_table,
    population_summary,
)
from bitflip.guard import immutable
from bitflip.weights import code_histogram, open_weights

RESULTS_DIR = PROJECT_ROOT / "results"


def summary(
    name: str, artifact: Artifact, counts: np.ndarray, rows: list[dict[str, object]]
) -> dict[str, object]:
    """A summary row: which model was measured, then what the arithmetic found."""
    identity = {"model": name, "artifact": artifact.key, "revision": artifact.revision}
    return identity | population_summary(counts, rows, BF16)


def report(name: str, artifact: Artifact) -> tuple[list[dict[str, object]], dict]:
    weights = open_weights(artifact.local_dir)
    with immutable(weights.paths):
        counts = code_histogram(weights, BF16)

    total = int(counts.sum())
    if total != weights.parameter_count:
        raise RuntimeError(f"histogram {total} != parameters {weights.parameter_count}")

    rows = bit_rows(counts, BF16)
    profile = exponent_profile(counts, BF16)

    print(f"\n=== {name}: {total:,} bf16 weights across {len(weights)} tensors ===")
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
