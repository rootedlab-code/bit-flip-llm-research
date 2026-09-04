"""Rebuild the checkpoint table of an E5b arm from the public log of its Kaggle run.

The chosen arm ran before the notebook checkpointed anything. It completed every
condition, printed each one's verdict shares and perplexity, and died in its summary;
the per-probe table went with the kernel. The shares did not: Kaggle keeps the log of a
public notebook. And at 300 harmful and 100 benign probes a share printed to a tenth of
a percent identifies its count uniquely, because consecutive counts sit a third of a
percent apart. This turns those printed lines back into the rows the notebook would have
written, and refuses any line whose share does not round-trip from exactly one count.

What it cannot recover is declared in the output rather than papered over: the
perplexity carries the six decimals the log printed and no more, and the top-1 columns
are empty.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from bitflip.scoring import BENIGN, COUNTS_COLUMNS, HARMFUL

ARM_LINE = re.compile(r"^arm: (\w+) ·")
PROBES_LINE = re.compile(r"^probes\s+harmful (\d+) · benign (\d+)$")
CONDITION_LINE = re.compile(r"^=== (\S+) ===$")
SHARES_LINE = re.compile(
    r"^\s+(harmful|benign)\s+refusa\s+([\d.]+)%\s+compli\s+([\d.]+)%"
    r"\s+degene\s+([\d.]+)%\s+indete\s+([\d.]+)%$"
)
# The agreement printed after the perplexity arrived with the top-1 checkpoint; the
# logs this recovers predate it, and a later log must still parse.
PERPLEXITY_LINE = re.compile(r"^\s+perplexity ([\d.]+)(?: ·.*)?$")
DOSED_NAME = re.compile(r"^\w+-d(\d+)-s(\d+)$")

VERDICT_ORDER = ("refusal", "compliance", "degenerate", "indeterminate")


class RecoveryError(ValueError):
    """The log does not identify what the run measured."""


def stdout_lines(entries: Iterable[Mapping[str, object]]) -> list[str]:
    """The notebook's own output, in order, out of the log Kaggle serves."""
    text = "".join(
        str(entry["data"]) for entry in entries if entry["stream_name"] == "stdout"
    )
    return text.splitlines()


def count_from_share(printed: str, total: int) -> int:
    """The one count of `total` that prints as this share, to one decimal.

    Every candidate is tried rather than the share being multiplied back, so that a
    printed value two counts could round to is refused instead of resolved by
    arithmetic that happens to land on one of them.
    """
    matches = [
        count for count in range(total + 1) if f"{count / total:.1%}" == f"{printed}%"
    ]
    if len(matches) != 1:
        raise RecoveryError(
            f"{printed}% of {total} identifies {len(matches)} counts, not one"
        )
    return matches[0]


def recover_rows(lines: Iterable[str]) -> list[dict[str, object]]:
    """Checkpoint rows, in the order the run measured them.

    Raises rather than skipping when a condition is missing a half or its perplexity:
    a run that died between two prints leaves a condition that must be visible as
    incomplete, not silently absent.
    """
    arm: str | None = None
    totals: dict[str, int] = {}
    conditions: list[str] = []
    shares: dict[tuple[str, str], tuple[str, ...]] = {}
    perplexities: dict[str, str] = {}
    current: str | None = None

    for line in lines:
        if found := ARM_LINE.match(line):
            arm = found.group(1)
        elif found := PROBES_LINE.match(line):
            totals = {HARMFUL: int(found.group(1)), BENIGN: int(found.group(2))}
        elif found := CONDITION_LINE.match(line):
            current = found.group(1)
            conditions.append(current)
        elif (found := SHARES_LINE.match(line)) and current:
            shares[(current, found.group(1))] = found.groups()[1:]
        elif (found := PERPLEXITY_LINE.match(line)) and current:
            perplexities[current] = found.group(1)

    if arm is None or not totals:
        raise RecoveryError("the log does not name the arm and the probe counts")

    rows: list[dict[str, object]] = []
    for name in conditions:
        if name not in perplexities:
            raise RecoveryError(f"{name}: no perplexity was printed")
        for kind in (HARMFUL, BENIGN):
            if (name, kind) not in shares:
                raise RecoveryError(f"{name}: no {kind} shares were printed")
            counts = {
                verdict: count_from_share(printed, totals[kind])
                for verdict, printed in zip(
                    VERDICT_ORDER, shares[name, kind], strict=True
                )
            }
            if sum(counts.values()) != totals[kind]:
                raise RecoveryError(f"{name}/{kind}: counts do not sum to {totals[kind]}")
            dose, seed = _dose_and_seed(name)
            values: dict[str, object] = {
                "condition": name,
                "arm": arm,
                "dose": dose,
                "seed": seed,
                "kind": kind,
                **counts,
                "perplexity": perplexities[name],
                "top1_agreement": "",
                "top1_positions": "",
            }
            rows.append({column: values[column] for column in COUNTS_COLUMNS})
    return rows


def _dose_and_seed(name: str) -> tuple[int, object]:
    if found := DOSED_NAME.match(name):
        return int(found.group(1)), int(found.group(2))
    return 0, ""
