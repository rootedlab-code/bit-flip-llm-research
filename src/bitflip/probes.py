"""Probe sets: the requests a model is asked, and what counts as answering them.

Two sets, and both are mandatory. The harmful set measures whether refusal survives a
fault. The benign set -- where complying is the correct behaviour -- is what separates
"stopped refusing" from "stopped understanding": a model reduced to noise refuses
nothing either, and without the control that would read as de-alignment.

The prompts themselves come from a public benchmark, cited rather than written here.
The keywords are derived from each prompt by the registered rule, so nothing about the
classification is authored per item.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from bitflip.oracle import OracleSpec, keywords_for, required_hits

HARMFUL = "harmful"
BENIGN = "benign"


@dataclass(frozen=True)
class Probe:
    """One request, with the keywords derived from it and the bar they set."""

    identifier: str
    kind: str
    prompt: str
    keywords: tuple[str, ...]
    required: int


@dataclass(frozen=True)
class ProbeSet:
    """The probes that could be used, and an honest count of those that could not."""

    kind: str
    probes: tuple[Probe, ...]
    discarded: int

    def __len__(self) -> int:
        return len(self.probes)

    @property
    def coverage(self) -> float:
        total = len(self.probes) + self.discarded
        return len(self.probes) / total if total else 0.0


def build_probe_set(
    prompts: Iterable[str], kind: str, spec: OracleSpec, limit: int | None = None
) -> ProbeSet:
    """Derive keywords for each prompt, dropping those that yield none.

    A request from which no content word can be derived cannot be judged: there is
    nothing for an answer to be about. Those are discarded and **counted**, because a
    probe set that silently shrinks is a probe set whose coverage is unknown.
    """
    probes: list[Probe] = []
    discarded = 0
    for index, prompt in enumerate(prompts):
        keywords = keywords_for(prompt, spec)
        if not keywords:
            discarded += 1
            continue
        probes.append(
            Probe(
                identifier=f"{kind}-{index:04d}",
                kind=kind,
                prompt=prompt.strip(),
                keywords=keywords,
                required=required_hits(keywords, spec),
            )
        )
        if limit is not None and len(probes) == limit:
            break
    return ProbeSet(kind=kind, probes=tuple(probes), discarded=discarded)
