"""The paired-comparison contract: only identical answers may be compared.

The rows here are inert stand-ins carrying the five columns the comparison reads. What
they exercise is the pairing rule and the arithmetic on top of it, not any property of a
model -- which is the point, since the whole purpose of this module is to keep the model
out of a comparison between two specifications.
"""

from __future__ import annotations

from collections import Counter

import pytest

from bitflip.compare import (
    CompareError,
    compare,
    index,
    matched_probes,
    summarise,
)
from bitflip.oracle import COMPLIANCE, INDETERMINATE, REFUSAL


def row(probe, verdict, digest, condition="base", kind="harmful"):
    return {
        "condition": condition,
        "probe": probe,
        "kind": kind,
        "verdict": verdict,
        "answer_sha256": digest,
    }


def test_rows_are_keyed_by_condition_and_probe():
    table = index([row("p1", REFUSAL, "aa"), row("p1", COMPLIANCE, "bb", "abliterated")])

    assert set(table) == {("base", "p1"), ("abliterated", "p1")}


def test_a_table_missing_a_column_is_refused():
    incomplete = [{"condition": "base", "probe": "p1", "verdict": REFUSAL}]

    with pytest.raises(CompareError, match="lacks"):
        index(incomplete)


def test_the_same_probe_twice_in_one_table_is_refused():
    with pytest.raises(CompareError, match="duplicate"):
        index([row("p1", REFUSAL, "aa"), row("p1", COMPLIANCE, "bb")])


def test_only_identical_answers_are_paired():
    """The rule the whole module exists for: a regenerated answer cannot tell the
    specification apart from the generation, so it is excluded rather than counted."""
    old = index([row("p1", REFUSAL, "aa"), row("p2", INDETERMINATE, "bb")])
    new = index([row("p1", REFUSAL, "aa"), row("p2", COMPLIANCE, "REGENERATED")])

    assert matched_probes(old, new) == [("base", "p1")]


def test_the_matched_fraction_is_part_of_the_result():
    """A paired comparison over an unstated fraction is worth no more than an unpaired
    one, so the fraction is carried, not left to the caller to work out."""
    old = index([row("p1", REFUSAL, "aa"), row("p2", REFUSAL, "bb")])
    new = index([row("p1", REFUSAL, "aa"), row("p2", REFUSAL, "CHANGED")])

    result = compare(old, new)

    assert result.shared == 2
    assert result.matched == 1
    assert result.matched_fraction == 0.5


def test_probes_only_one_run_saw_are_not_shared():
    old = index([row("p1", REFUSAL, "aa"), row("p2", REFUSAL, "bb")])
    new = index([row("p1", REFUSAL, "aa")])

    assert compare(old, new).shared == 1


def test_two_unrelated_runs_are_refused_rather_than_summarised():
    old = index([row("p1", REFUSAL, "aa")])
    new = index([row("p9", REFUSAL, "aa", "abliterated")])

    with pytest.raises(CompareError, match="no probe in common"):
        compare(old, new)


def test_corners_are_kept_apart():
    old = index(
        [
            row("p1", REFUSAL, "aa"),
            row("p2", COMPLIANCE, "bb", kind="benign"),
        ]
    )
    new = index(
        [
            row("p1", REFUSAL, "aa"),
            row("p2", COMPLIANCE, "bb", kind="benign"),
        ]
    )

    labels = [corner.label for corner in compare(old, new).corners]

    assert labels == ["base/benign", "base/harmful"]


def test_a_reclassification_is_recorded_as_a_transition():
    old = index([row("p1", INDETERMINATE, "aa")])
    new = index([row("p1", COMPLIANCE, "aa")])

    (corner,) = compare(old, new).corners

    assert corner.transitions == {(INDETERMINATE, COMPLIANCE): 1}
    assert corner.before[INDETERMINATE] == 1
    assert corner.after[COMPLIANCE] == 1


def test_an_unchanged_verdict_is_not_a_transition():
    old = index([row("p1", REFUSAL, "aa")])
    new = index([row("p1", REFUSAL, "aa")])

    (corner,) = compare(old, new).corners

    assert corner.transitions == {}
    assert corner.matched == 1


def test_a_lost_refusal_is_counted():
    """The quantity a specification change has to be judged on: the protected class can
    only lose members when the text is held constant."""
    old = index([row("p1", REFUSAL, "aa"), row("p2", REFUSAL, "bb")])
    new = index([row("p1", REFUSAL, "aa"), row("p2", INDETERMINATE, "bb")])

    result = compare(old, new)

    assert result.refusals_lost == 1


def test_refusals_lost_is_counted_from_transitions_and_not_from_totals():
    """A net figure would report zero for a corner that traded three refusals for three
    others. That is not the same event, and the difference is the whole finding."""
    old = index([row("p1", REFUSAL, "aa"), row("p2", COMPLIANCE, "bb")])
    new = index([row("p1", COMPLIANCE, "aa"), row("p2", REFUSAL, "bb")])

    result = compare(old, new)

    assert result.corners[0].before[REFUSAL] == result.corners[0].after[REFUSAL]
    assert result.refusals_lost == 1


def test_refusals_lost_is_never_negative():
    old = index([row("p1", INDETERMINATE, "aa")])
    new = index([row("p1", REFUSAL, "aa")])

    assert compare(old, new).refusals_lost == 0


def test_an_empty_pairing_is_a_result_and_not_an_error():
    """Two runs that share probes but agree on no answer are comparable and the answer
    is 'nothing can be said' -- which the matched fraction states."""
    old = index([row("p1", REFUSAL, "aa")])
    new = index([row("p1", COMPLIANCE, "CHANGED")])

    result = compare(old, new)

    assert result.corners == ()
    assert result.matched == 0
    assert result.matched_fraction == 0.0
    assert result.refusals_lost == 0


def test_counts_are_summarised_in_the_specifications_own_order():
    text = summarise(Counter({COMPLIANCE: 2, REFUSAL: 1}))

    assert text == "refusa=1 compli=2"


def test_a_verdict_with_no_members_is_left_out_of_the_summary():
    assert summarise(Counter({REFUSAL: 3})) == "refusa=3"
