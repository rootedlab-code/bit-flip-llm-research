"""Compare two oracle specifications on the answers they both saw.

Verdict shares from two runs are not a comparison of two specifications. Generation is
re-run each time, and it does not reproduce exactly across configurations: greedy
decoding takes an argmax, which is discontinuous, so a logit difference in the last bits
flips a near-tie and the answer diverges from there. Changing the batch size changes the
left-padding width applied to each prompt, which is enough.

What the verdict tables carry is a truncated SHA-256 of every answer. Answers whose
digests agree are the same text, so on that subset the only thing that changed is the
specification -- which is the comparison one actually wanted. The subset is smaller than
the run and it is reported, because a matched comparison over an unstated fraction is
worth no more than the unmatched one.

    python experiments/e5_compare_specs.py OLD.csv NEW.csv
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

VERDICTS = ("refusal", "compliance", "degenerate", "indeterminate")


def load(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Verdict rows keyed by the condition and probe they belong to."""
    with path.open(newline="") as handle:
        return {(row["condition"], row["probe"]): row for row in csv.DictReader(handle)}


def matched_corners(
    old: dict[tuple[str, str], dict[str, str]],
    new: dict[tuple[str, str], dict[str, str]],
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """The probes each corner answered identically in both runs."""
    corners: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for key in old.keys() & new.keys():
        if old[key]["answer_sha256"] == new[key]["answer_sha256"]:
            corners[(old[key]["condition"], old[key]["kind"])].append(key)
    return corners


def shares(rows: list[dict[str, str]]) -> Counter:
    return Counter(row["verdict"] for row in rows)


def summarise(counts: Counter) -> str:
    """The verdict counts of one corner, in the specification's own order."""
    return " ".join(f"{name[:6]}={counts[name]}" for name in VERDICTS if counts[name])


def main() -> int:
    old_path, new_path = Path(sys.argv[1]), Path(sys.argv[2])
    old, new = load(old_path), load(new_path)
    shared = old.keys() & new.keys()
    if not shared:
        raise SystemExit(f"{old_path} and {new_path} have no probe in common")

    corners = matched_corners(old, new)
    total_matched = sum(len(keys) for keys in corners.values())
    print(f"{old_path.name} -> {new_path.name}")
    print(
        f"{len(shared)} probes in common, {total_matched} answered identically "
        f"({total_matched / len(shared):.0%}); the rest were regenerated and cannot "
        f"separate the specification from the generation."
    )

    print(f"\n{'corner':<24}{'n':>5}  {'before':<34}after")
    lost_refusals = 0
    for corner in sorted(corners):
        keys = corners[corner]
        before, after = shares([old[k] for k in keys]), shares([new[k] for k in keys])
        lost_refusals += max(0, before["refusal"] - after["refusal"])
        label = f"{corner[0]}/{corner[1]}"
        print(f"{label:<24}{len(keys):>5}  {summarise(before):<34}{summarise(after)}")

    moves = Counter(
        (old[k]["condition"], old[k]["kind"], old[k]["verdict"], new[k]["verdict"])
        for keys in corners.values()
        for k in keys
        if old[k]["verdict"] != new[k]["verdict"]
    )
    print("\nreclassifications on identical text:")
    for (condition, kind, before_class, after_class), count in moves.most_common():
        corner = f"{condition}/{kind}"
        print(f"  {corner:<24} {before_class} -> {after_class}: {count}")
    if not moves:
        print("  none")

    print(f"\nrefusals lost on identical text: {lost_refusals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
