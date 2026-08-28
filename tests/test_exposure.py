"""The three normalisations, and the one the fault-rate bridge is required to use.

Most of these run on synthetic rows, so that the arithmetic is checked against cases
whose answer is known by construction rather than against the artifact it produced. The
last one goes the other way and pins the published figures: it is the reason the module
exists, and a change that silently moved 1.3792 would otherwise reach the paper.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from bitflip.exposure import (
    FEEDS_FAULT_RATE,
    PER_FLIP,
    PER_MODEL,
    PER_PHYSICAL_EXPOSURE,
    ExposureError,
    normalisations,
    read,
)

COMPARISON = Path(__file__).resolve().parents[1] / "results" / "e3-comparison.csv"


def row(name, weights, total_bits, lost_per_flip):
    return {
        "format": name,
        "weights": str(weights),
        "total_bits": str(total_bits),
        "weights_lost_per_random_flip": str(lost_per_flip),
    }


def test_a_table_missing_a_column_is_refused():
    with pytest.raises(ExposureError, match="lacks"):
        read([{"format": "bf16", "weights": "10"}])


def test_a_ratio_needs_exactly_two_formats():
    with pytest.raises(ExposureError, match="exactly two"):
        read([row("only one", 10, 160, 0.5)])


def test_the_per_flip_ratio_ignores_how_big_the_files_are():
    """The figure that answers 'a fault landed in this file': file size cannot enter it,
    because the fault is already known to have landed."""
    baseline, subject = read([row("big", 100, 1600, 0.1), row("small", 100, 16, 0.2)])

    per_flip, _, _ = normalisations(baseline, subject)

    assert per_flip.key == PER_FLIP
    assert per_flip.ratio == pytest.approx(2.0)


def test_halving_the_file_halves_its_exposure():
    """The whole reason three ratios exist. Both formats lose the same per flip, but one
    file is half the size, so at equal physical exposure it loses half as much -- which
    the per-flip ratio reports as parity."""
    baseline, subject = read([row("big", 100, 1600, 0.1), row("half", 100, 800, 0.1)])

    per_flip, per_exposure, _ = normalisations(baseline, subject)

    assert per_flip.ratio == pytest.approx(1.0)
    assert per_exposure.ratio == pytest.approx(0.5)


def test_the_per_model_ratio_divides_by_each_format_s_own_weights():
    """Two files can hold different numbers of weights -- GGUF unties the embedding and
    stores it twice -- so 'what fraction of the model is lost' is a third question."""
    baseline, subject = read([row("bf16", 100, 1600, 0.1), row("gguf", 200, 1600, 0.1)])

    _, per_exposure, per_model = normalisations(baseline, subject)

    assert per_exposure.ratio == pytest.approx(1.0)
    assert per_model.ratio == pytest.approx(0.5)


def test_every_ratio_points_the_same_way():
    """Above one means the subject loses more, in all three. A row of ratios that each
    needed its direction checked would be read wrong."""
    baseline, subject = read(
        [row("baseline", 100, 1600, 0.1), row("worse", 100, 1600, 0.3)]
    )

    assert all(norm.ratio > 1 for norm in normalisations(baseline, subject))


def test_only_the_physical_exposure_ratio_feeds_the_fault_rate():
    """A field fault rate is quoted per bit per hour, so crossing it with the per-flip
    figure would overstate the bridge by the file-size ratio."""
    baseline, subject = read([row("bf16", 100, 1600, 0.1), row("gguf", 100, 800, 0.2)])

    feeding = [n.key for n in normalisations(baseline, subject) if n.feeds_fault_rate]

    assert feeding == [PER_PHYSICAL_EXPOSURE]
    assert FEEDS_FAULT_RATE == PER_PHYSICAL_EXPOSURE


def test_the_published_comparison_still_gives_the_published_ratios():
    """The proof of this step: 2.807 was published alone, and the two figures that put
    it in proportion are 1.379 and 1.081. If any of the three moves, it moves here
    first."""
    with COMPARISON.open(newline="") as handle:
        baseline, subject = read(csv.DictReader(handle))

    by_key = {norm.key: norm.ratio for norm in normalisations(baseline, subject)}

    assert by_key[PER_FLIP] == pytest.approx(2.8071, abs=5e-5)
    assert by_key[PER_PHYSICAL_EXPOSURE] == pytest.approx(1.3792, abs=5e-5)
    assert by_key[PER_MODEL] == pytest.approx(1.0812, abs=5e-5)
