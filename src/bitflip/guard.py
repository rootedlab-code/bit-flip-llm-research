"""Guards that keep this project from harming the machine hosting it.

Constitution, Principle I: the flips studied here are arithmetic in memory, never
physical events. What can actually damage the host is prosaic -- a filled disk, an
overwritten model file -- and that is what these guards defend against.
"""

from __future__ import annotations

import hashlib
import shutil
import stat
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

BYTES_PER_GIB = 1024**3
DEFAULT_MIN_FREE_GIB = 10.0

WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


class HostSafetyError(RuntimeError):
    """The safety envelope was breached: the operation must stop."""


class InsufficientDiskSpaceError(HostSafetyError):
    """Free space is below the required threshold."""


class FileIntegrityError(HostSafetyError):
    """A file declared immutable has changed."""


def free_gib(path: Path | str) -> float:
    """Free space, in GiB, on the filesystem holding `path`."""
    return shutil.disk_usage(Path(path)).free / BYTES_PER_GIB


def require_free_space(
    path: Path | str, min_free_gib: float = DEFAULT_MIN_FREE_GIB
) -> float:
    """Check free space before an operation that writes.

    Returns the available GiB; raises `InsufficientDiskSpaceError` below threshold.
    """
    available = free_gib(path)
    if available < min_free_gib:
        raise InsufficientDiskSpaceError(
            f"{available:.1f} GiB free on {path}, threshold {min_free_gib:.1f} GiB"
        )
    return available


def sha256_file(path: Path | str) -> str:
    """Hexadecimal SHA-256 digest of the file contents."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def make_readonly(path: Path | str) -> None:
    """Strip write permissions from the file, following any symlinks."""
    target = Path(path).resolve()
    target.chmod(target.stat().st_mode & ~WRITE_BITS)


def is_readonly(path: Path | str) -> bool:
    return not Path(path).resolve().stat().st_mode & WRITE_BITS


@contextmanager
def immutable(paths: Iterable[Path | str]) -> Iterator[dict[str, str]]:
    """A context demanding that the named files do not change by a single bit.

    Digests are recorded on entry and re-verified on exit, raising
    `FileIntegrityError` at the first difference.
    """
    recorded = {str(Path(p).resolve()): sha256_file(p) for p in paths}
    try:
        yield recorded
    finally:
        # Verification runs even when the body failed: a file corrupted during an
        # error is graver than the error itself, and Python preserves the original
        # exception in the chain anyway.
        for filename, expected in recorded.items():
            actual = sha256_file(filename)
            if actual != expected:
                raise FileIntegrityError(
                    f"{filename} changed: expected {expected[:16]}…, found {actual[:16]}…"
                )
