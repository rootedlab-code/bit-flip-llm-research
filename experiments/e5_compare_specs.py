"""Compare two oracle specifications on the answers they both saw.

Only the answers both runs produced byte-identically can separate the specification from
the generation; the pairing rule and the arithmetic live in `bitflip.compare`, where they
are tested. This driver reads two verdict tables and prints the result.

Usage:  uv run python experiments/e5_compare_specs.py OLD.csv NEW.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from bitflip.compare import Comparison, compare, index, summarise


def read(path: Path):
    with path.open(newline="") as handle:
        return index(csv.DictReader(handle))


def report(comparison: Comparison, old_name: str, new_name: str) -> None:
    print(f"{old_name} -> {new_name}")
    print(
        f"{comparison.shared} probes in common, {comparison.matched} answered "
        f"identically ({comparison.matched_fraction:.0%}); the rest were regenerated "
        f"and cannot separate the specification from the generation."
    )

    print(f"\n{'corner':<24}{'n':>5}  {'before':<34}after")
    for corner in comparison.corners:
        print(
            f"{corner.label:<24}{corner.matched:>5}  "
            f"{summarise(corner.before):<34}{summarise(corner.after)}"
        )

    print("\nreclassifications on identical text:")
    moved = [
        (corner.label, before, after, count)
        for corner in comparison.corners
        for (before, after), count in corner.transitions.most_common()
    ]
    for label, before, after, count in sorted(moved, key=lambda row: -row[3]):
        print(f"  {label:<24} {before} -> {after}: {count}")
    if not moved:
        print("  none")

    print(f"\nrefusals lost on identical text: {comparison.refusals_lost}")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} OLD.csv NEW.csv")
    old_path, new_path = Path(sys.argv[1]), Path(sys.argv[2])
    report(compare(read(old_path), read(new_path)), old_path.name, new_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
