"""Comparing two storage formats needs three ratios, not one.

E3 measures how many weights a random bit flip destroys in a `bfloat16` file and in a
quantised one. Turning that into "the quantised format is N times worse" requires
choosing what is held equal between them, and there are three defensible choices that
give three different answers. The one the experiment happened to compute first is the
largest of the three, which is the direction a reader is entitled to be suspicious about.

Held equal, and the question each choice answers:

- **the flip** -- a fault landed *inside this file*: how many weights does it cost?
  This is the per-bit figure, and it is the right one when the file is the whole world.
- **the physical exposure** -- same DRAM, same hours, same fault rate per bit: how many
  weights does each format lose? The quantised file is about half the size, so it
  intercepts about half the faults, and that halving is exactly what the per-flip ratio
  hides. This is the figure a fault rate from the field has to be crossed with.
- **the model** -- what fraction of the parameters does each format lose per hour? The
  two files do not hold the same number of weights: GGUF unties the embedding and
  materialises it twice, so it stores more weights in fewer bits.

None of the three is wrong; publishing one without its condition is. The module exists
so that the condition travels with the number.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

PER_FLIP = "per flip"
PER_PHYSICAL_EXPOSURE = "per file, at equal physical exposure"
PER_MODEL = "per file, normalised on the model's weights"

REQUIRED_COLUMNS = frozenset(
    {"format", "weights", "total_bits", "weights_lost_per_random_flip"}
)

# A fault rate from the field is quoted per bit per hour, so the quantity it has to be
# crossed with is the one that holds physical exposure equal. Naming it here rather than
# in the note means E4 cannot pick a different one by accident.
FEEDS_FAULT_RATE = PER_PHYSICAL_EXPOSURE


class ExposureError(ValueError):
    """The comparison table cannot be normalised."""


@dataclass(frozen=True)
class Normalisation:
    """One ratio between two formats, carrying what it holds equal."""

    key: str
    question: str
    ratio: float

    @property
    def feeds_fault_rate(self) -> bool:
        return self.key == FEEDS_FAULT_RATE


@dataclass(frozen=True)
class Format:
    """One row of the comparison, reduced to what a ratio needs."""

    name: str
    weights: int
    total_bits: int
    weights_lost_per_random_flip: float

    @property
    def weights_lost_per_file(self) -> float:
        """Weights destroyed if every bit of the file were flipped once.

        Not a physical event -- it is the per-bit cost scaled by how many bits the
        format exposes, which is what makes two files of different size comparable at
        equal exposure.
        """
        return self.total_bits * self.weights_lost_per_random_flip


def read(rows: Iterable[Mapping[str, str]]) -> tuple[Format, Format]:
    """The two formats of the comparison table, in the order they were written.

    Raises `ExposureError` on a table that is not exactly two rows or lacks a column a
    ratio needs, rather than dividing by whatever happens to be there.
    """
    formats: list[Format] = []
    for row in rows:
        missing = REQUIRED_COLUMNS - set(row)
        if missing:
            raise ExposureError(f"comparison table lacks {sorted(missing)}")
        formats.append(
            Format(
                name=row["format"],
                weights=int(row["weights"]),
                total_bits=int(row["total_bits"]),
                weights_lost_per_random_flip=float(row["weights_lost_per_random_flip"]),
            )
        )
    if len(formats) != 2:
        raise ExposureError(f"a ratio needs exactly two formats, got {len(formats)}")
    return formats[0], formats[1]


def normalisations(
    baseline: Format, subject: Format
) -> tuple[Normalisation, Normalisation, Normalisation]:
    """The three ratios of `subject` against `baseline`, each with its condition.

    Every ratio is oriented the same way -- above 1 means the subject loses more -- so
    that the three can be read as a row without checking the direction of each.
    """
    return (
        Normalisation(
            key=PER_FLIP,
            question=(
                "a fault landed in this file: how many weights does it destroy, "
                "against the same fault in the other?"
            ),
            ratio=(
                subject.weights_lost_per_random_flip
                / baseline.weights_lost_per_random_flip
            ),
        ),
        Normalisation(
            key=PER_PHYSICAL_EXPOSURE,
            question=(
                "same DRAM, same hours, same fault rate per bit: how many weights "
                "does each format lose?"
            ),
            ratio=subject.weights_lost_per_file / baseline.weights_lost_per_file,
        ),
        Normalisation(
            key=PER_MODEL,
            question=(
                "what share of its own parameters does each format lose "
                "in that same time?"
            ),
            ratio=(
                (subject.weights_lost_per_file / subject.weights)
                / (baseline.weights_lost_per_file / baseline.weights)
            ),
        ),
    )
