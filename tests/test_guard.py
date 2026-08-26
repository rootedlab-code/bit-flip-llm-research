"""The contract of the host guards."""

from __future__ import annotations

import pytest

from bitflip.guard import (
    FileIntegrityError,
    InsufficientDiskSpaceError,
    free_gib,
    immutable,
    is_readonly,
    make_readonly,
    require_free_space,
    sha256_file,
)

PROBE_TEXT = b"sonda-integrita-bit-flip"
PROBE_DIGEST = "c362452ca9364bbcc0137d43d5576c7005874bef21ca427a7b16250ebfdda03b"


@pytest.fixture
def probe_file(tmp_path):
    path = tmp_path / "probe.bin"
    path.write_bytes(PROBE_TEXT)
    return path


def test_sha256_file_matches_reference_digest(probe_file):
    assert sha256_file(probe_file) == PROBE_DIGEST


def test_free_gib_reports_positive_space(tmp_path):
    assert free_gib(tmp_path) > 0


def test_require_free_space_returns_available_when_above_threshold(tmp_path):
    available = require_free_space(tmp_path, min_free_gib=0.0)

    assert available == pytest.approx(free_gib(tmp_path), rel=1e-6)


def test_require_free_space_raises_when_below_threshold(tmp_path):
    with pytest.raises(InsufficientDiskSpaceError, match="threshold"):
        require_free_space(tmp_path, min_free_gib=1e9)


def test_make_readonly_forbids_writing(probe_file):
    make_readonly(probe_file)

    assert is_readonly(probe_file)
    with pytest.raises(PermissionError):
        probe_file.open("ab")


def test_immutable_accepts_unchanged_file(probe_file):
    with immutable([probe_file]) as recorded:
        assert recorded[str(probe_file)] == PROBE_DIGEST


def test_immutable_raises_when_content_changes(probe_file):
    with pytest.raises(FileIntegrityError, match="changed"), immutable([probe_file]):
        probe_file.write_bytes(PROBE_TEXT + b"!")


def test_immutable_flips_a_single_bit_and_notices(probe_file):
    original = probe_file.read_bytes()
    corrupted = bytes([original[0] ^ 0b0000_0001]) + original[1:]

    with pytest.raises(FileIntegrityError), immutable([probe_file]):
        probe_file.write_bytes(corrupted)


def test_immutable_verifies_even_when_body_fails(probe_file):
    with pytest.raises(FileIntegrityError) as caught, immutable([probe_file]):
        probe_file.write_bytes(b"corrotto")
        raise ValueError("guasto nel corpo")

    assert isinstance(caught.value.__context__, ValueError)
