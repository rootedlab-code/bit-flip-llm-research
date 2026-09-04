"""Score an E5b arm from its checkpoint, exactly as the notebook does after its run.

The notebook and this driver call the same function on the same file, so a run whose
last cell failed scores as the run that finished would have. The verdict table is
optional: without it the paired test is not attached, and the scores say nothing about
it rather than inventing it.

Usage:  uv run python experiments/e5b_score.py COUNTS.csv OUT.csv [VERDICTS.csv]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from bitflip.alignment import E5Spec
from bitflip.scoring import compliance_by_probe, conditions_from_rows, score_run, table


def main() -> int:
    if len(sys.argv) not in (3, 4):
        raise SystemExit(f"usage: {sys.argv[0]} COUNTS.csv OUT.csv [VERDICTS.csv]")
    counts_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    spec = E5Spec.load()

    with counts_path.open(newline="") as handle:
        conditions = conditions_from_rows(csv.DictReader(handle))
    compliance = None
    if len(sys.argv) == 4:
        with Path(sys.argv[3]).open(newline="") as handle:
            compliance = compliance_by_probe(csv.DictReader(handle))

    scores = score_run(conditions, spec, compliance)
    print(f"e5_spec.yaml v{spec.version} sha256 {spec.digest}\n")
    print(table(scores, spec.primary_rule))

    rows = [entry.as_row() for entry in scores]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} scored conditions -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
