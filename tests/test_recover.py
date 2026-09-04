"""Rebuilding a checkpoint from a run's printed output, and refusing to guess."""

from __future__ import annotations

import pytest

from bitflip.recover import (
    RecoveryError,
    count_from_share,
    recover_rows,
    stdout_lines,
)
from bitflip.scoring import COUNTS_COLUMNS

HEADER = [
    "arm: chosen · attestation only: False",
    "probes      harmful 300 · benign 100",
]


def condition_lines(name: str, harmful: str, benign: str, perplexity: str) -> list[str]:
    return [
        f"=== {name} ===",
        f"  harmful  {harmful}",
        f"  benign   {benign}",
        f"  perplexity {perplexity}",
    ]


ALIGNED = "refusa  83.7%  compli   1.3%  degene   0.0%  indete  15.0%"
WORKING = "refusa   0.0%  compli  72.0%  degene   0.0%  indete  28.0%"


def test_a_share_printed_to_a_tenth_identifies_one_count_of_three_hundred():
    assert count_from_share("83.7", 300) == 251
    assert count_from_share("1.3", 300) == 4
    assert count_from_share("0.0", 300) == 0


def test_a_share_no_count_prints_as_is_refused():
    """The parser tries every count rather than multiplying the share back, so a value
    the run could not have printed is an error and not a rounded guess."""
    with pytest.raises(RecoveryError, match="identifies 0 counts"):
        count_from_share("83.5", 300)


def test_a_share_two_counts_print_as_is_refused():
    """At n = 1000 consecutive counts are a tenth of a percent apart, so a printed
    tenth no longer identifies one of them; the parser must say so."""
    with pytest.raises(RecoveryError, match="not one"):
        count_from_share("10.0", 2000)


def test_the_printed_lines_become_the_rows_the_notebook_would_have_written():
    lines = (
        HEADER
        + condition_lines("base", ALIGNED, WORKING, "9.248813")
        + condition_lines("chosen-d5-s0", ALIGNED, WORKING, "9.250966")
    )

    rows = recover_rows(lines)

    assert [tuple(row) for row in rows] == [COUNTS_COLUMNS] * 4
    assert rows[0]["condition"] == "base"
    assert (rows[0]["dose"], rows[0]["seed"]) == (0, "")
    assert (rows[0]["refusal"], rows[0]["compliance"]) == (251, 4)
    assert rows[1]["kind"] == "benign"
    assert (rows[1]["compliance"], rows[1]["indeterminate"]) == (72, 28)
    assert (rows[2]["dose"], rows[2]["seed"]) == (5, 0)
    assert rows[2]["perplexity"] == "9.250966"
    assert rows[2]["top1_agreement"] == ""


def test_a_condition_that_died_before_its_perplexity_is_refused():
    lines = HEADER + condition_lines("base", ALIGNED, WORKING, "9.248813")[:-1]

    with pytest.raises(RecoveryError, match="base: no perplexity"):
        recover_rows(lines)


def test_a_condition_missing_its_benign_shares_is_refused():
    lines = HEADER + ["=== base ===", f"  harmful  {ALIGNED}", "  perplexity 9.248813"]

    with pytest.raises(RecoveryError, match="no benign shares"):
        recover_rows(lines)


def test_a_log_that_never_names_the_arm_is_refused():
    with pytest.raises(RecoveryError, match="name the arm"):
        recover_rows(condition_lines("base", ALIGNED, WORKING, "9.248813"))


def test_only_stdout_is_read_and_split_where_the_run_split_it():
    entries = [
        {"stream_name": "stdout", "time": 1.0, "data": "=== base ===\n  harm"},
        {"stream_name": "stderr", "time": 1.5, "data": "Traceback\n"},
        {"stream_name": "stdout", "time": 2.0, "data": "ful  x\n"},
    ]

    assert stdout_lines(entries) == ["=== base ===", "  harmful  x"]
