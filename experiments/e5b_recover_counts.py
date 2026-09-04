"""Rebuild an E5b arm's checkpoint table from the public log of its Kaggle run.

The chosen arm ran before the notebook wrote anything to disk. Its verdict shares
survive in the log Kaggle keeps for a public notebook, and `bitflip.recover` turns them
back into the rows the notebook would have checkpointed, refusing any share that does
not identify exactly one count. The per-probe table is not recoverable: what this
produces is enough to score the arm, not to pair it.

Usage:  uv run python experiments/e5b_recover_counts.py KERNEL.log OUT.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from bitflip.recover import recover_rows, stdout_lines
from bitflip.scoring import COUNTS_COLUMNS


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} KERNEL.log OUT.csv")
    log_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    rows = recover_rows(stdout_lines(json.loads(log_path.read_text())))
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COUNTS_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    conditions = {row["condition"] for row in rows}
    print(f"{len(rows)} rows over {len(conditions)} conditions -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
