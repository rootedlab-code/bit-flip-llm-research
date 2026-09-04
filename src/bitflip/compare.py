"""Compare two classification runs on the answers they both produced.

Verdict shares from two runs are not a comparison of two specifications. Generation is
re-run each time and does not reproduce across configurations: greedy decoding takes an
argmax, which is discontinuous, so a logit difference in the last bits flips a near-tie
and the answer diverges from there. Changing the batch size changes the left-padding
width applied to each prompt, which is enough -- measured at 341 identical answers out
of 600, in `docs/environment-notes.md`.

What the verdict tables carry is a SHA-256 of every answer. Answers whose digests agree
are the same text, so on that subset the only thing that changed is the specification,
which is the comparison one actually wanted. The matched fraction is part of the result
rather than an implementation detail: a paired comparison over an unstated fraction is
worth no more than the unpaired one.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from bitflip.oracle import COMPLIANCE, DEGENERATE, INDETERMINATE, REFUSAL

VERDICT_ORDER = (REFUSAL, COMPLIANCE, DEGENERATE, INDETERMINATE)
REQUIRED_COLUMNS = frozenset({"condition", "probe", "kind", "verdict", "answer_sha256"})

Row = Mapping[str, str]
ProbeKey = tuple[str, str]
VerdictTable = dict[ProbeKey, Row]
Transition = tuple[str, str]


class CompareError(ValueError):
    """The verdict tables cannot be compared."""


@dataclass(frozen=True)
class CornerComparison:
    """One condition/set corner, over the probes both runs answered identically."""

    condition: str
    kind: str
    matched: int
    before: Counter[str]
    after: Counter[str]
    transitions: Counter[Transition]

    @property
    def label(self) -> str:
        return f"{self.condition}/{self.kind}"

    @property
    def refusals_lost(self) -> int:
        """Refusals the new specification no longer awards on unchanged text.

        Counted from the transitions rather than from the difference of the two totals:
        a net figure would report zero for a corner that lost three refusals and gained
        three others, which is not the same event at all.
        """
        return sum(
            count
            for (before, after), count in self.transitions.items()
            if before == REFUSAL and after != REFUSAL
        )


@dataclass(frozen=True)
class Comparison:
    """The paired comparison, with the fraction it was able to pair on."""

    shared: int
    corners: tuple[CornerComparison, ...]

    @property
    def matched(self) -> int:
        return sum(corner.matched for corner in self.corners)

    @property
    def matched_fraction(self) -> float:
        return self.matched / self.shared if self.shared else 0.0

    @property
    def refusals_lost(self) -> int:
        return sum(corner.refusals_lost for corner in self.corners)


def index(rows: Iterable[Row]) -> VerdictTable:
    """Verdict rows keyed by the condition and probe they belong to.

    Raises `CompareError` on a table missing a column the comparison needs, or holding
    the same probe twice, rather than failing later with an error that names one row and
    not the defect.
    """
    table: VerdictTable = {}
    for row in rows:
        missing = REQUIRED_COLUMNS - set(row)
        if missing:
            raise CompareError(f"verdict table lacks {sorted(missing)}")
        key = (row["condition"], row["probe"])
        if key in table:
            raise CompareError(f"duplicate probe {key} in one table")
        table[key] = row
    return table


def matched_probes(old: VerdictTable, new: VerdictTable) -> list[ProbeKey]:
    """The probes both runs answered with byte-identical text.

    Identity is decided on the digest alone. Two runs that answered the same probe with
    the same text agree on everything a classifier can see, so any difference in verdict
    between them is the specification and nothing else.
    """
    return [
        key
        for key in old.keys() & new.keys()
        if old[key]["answer_sha256"] == new[key]["answer_sha256"]
    ]


def compare(old: VerdictTable, new: VerdictTable) -> Comparison:
    """The paired comparison of two runs, corner by corner.

    Raises `CompareError` when the two tables share no probe: comparing two unrelated
    runs is a mistake worth stopping on, not an empty table worth printing.
    """
    shared = old.keys() & new.keys()
    if not shared:
        raise CompareError("the two verdict tables have no probe in common")

    grouped: dict[tuple[str, str], list[ProbeKey]] = defaultdict(list)
    for key in matched_probes(old, new):
        grouped[(old[key]["condition"], old[key]["kind"])].append(key)

    corners = tuple(
        CornerComparison(
            condition=condition,
            kind=kind,
            matched=len(keys),
            before=Counter(old[key]["verdict"] for key in keys),
            after=Counter(new[key]["verdict"] for key in keys),
            transitions=Counter(
                (old[key]["verdict"], new[key]["verdict"])
                for key in keys
                if old[key]["verdict"] != new[key]["verdict"]
            ),
        )
        for (condition, kind), keys in sorted(grouped.items())
    )
    return Comparison(shared=len(shared), corners=corners)


def summarise(counts: Counter[str]) -> str:
    """The verdict counts of one corner, in the specification's own order."""
    return " ".join(
        f"{name[:6]}={counts[name]}" for name in VERDICT_ORDER if counts[name]
    )
