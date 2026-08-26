"""Reading weights at the bit level, without going through floats.

The safetensors format is simple enough to read directly: eight bytes of length, a
JSON header, one contiguous buffer. Doing it by hand has three advantages that matter
here -- read-only mmap access (Principle I), no conversion that would falsify the bit
patterns, and arithmetic that must **close exactly** on the file size, which makes the
parser its own test.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np

from bitflip.codec import BF16, FP16, FloatFormat
from bitflip.fragility import CODE_SPACE

HEADER_LENGTH_BYTES = 8

ITEM_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}

DTYPE_FORMATS = {"BF16": BF16, "F16": FP16}


class SafetensorsError(ValueError):
    """The file does not match the format it declares."""


@dataclass(frozen=True)
class TensorEntry:
    """A tensor in the buffer: where it lives and what shape it has."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def count(self) -> int:
        return prod(self.shape) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def format(self) -> FloatFormat | None:
        return DTYPE_FORMATS.get(self.dtype)


def _parse_header(path: Path) -> tuple[dict[str, TensorEntry], int, dict]:
    with path.open("rb") as handle:
        raw_length = handle.read(HEADER_LENGTH_BYTES)
        if len(raw_length) < HEADER_LENGTH_BYTES:
            raise SafetensorsError(f"{path}: file truncated before the header")
        header_length = int.from_bytes(raw_length, "little")
        header = json.loads(handle.read(header_length))

    metadata = header.pop("__metadata__", {})
    entries = {}
    for name, spec in header.items():
        start, end = spec["data_offsets"]
        entries[name] = TensorEntry(
            name=name,
            dtype=spec["dtype"],
            shape=tuple(spec["shape"]),
            start=start,
            end=end,
        )
    return entries, HEADER_LENGTH_BYTES + header_length, metadata


def _validate(entries: dict[str, TensorEntry], data_start: int, file_size: int) -> None:
    """The parser's own test: the arithmetic must close on the file's bytes."""
    for entry in entries.values():
        item_size = ITEM_SIZES.get(entry.dtype)
        if item_size is None:
            raise SafetensorsError(f"{entry.name}: unknown dtype {entry.dtype}")
        if entry.nbytes != entry.count * item_size:
            raise SafetensorsError(
                f"{entry.name}: {entry.nbytes} bytes for {entry.count} "
                f"elements of {item_size}"
            )

    ordered = sorted(entries.values(), key=lambda entry: entry.start)
    cursor = 0
    for entry in ordered:
        if entry.start != cursor:
            raise SafetensorsError(
                f"{entry.name}: starts at {entry.start}, the buffer is at {cursor}"
            )
        cursor = entry.end

    if data_start + cursor != file_size:
        raise SafetensorsError(
            f"arithmetic does not close: header {data_start} + data {cursor} "
            f"!= {file_size} bytes of file"
        )


class SafetensorsFile:
    """Read-only access to the bit patterns of a safetensors file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.tensors, self._data_start, self.metadata = _parse_header(self.path)
        _validate(self.tensors, self._data_start, self.path.stat().st_size)
        self._raw: np.memmap | None = None

    @property
    def raw(self) -> np.memmap:
        """The whole file, mapped read-only, created on first access."""
        if self._raw is None:
            self._raw = np.memmap(self.path, dtype=np.uint8, mode="r")
        return self._raw

    def __len__(self) -> int:
        return len(self.tensors)

    @property
    def parameter_count(self) -> int:
        return sum(entry.count for entry in self.tensors.values())

    def entries_of_format(self, fmt: FloatFormat) -> list[TensorEntry]:
        return [entry for entry in self.tensors.values() if entry.format is fmt]

    def codes(self, name: str) -> np.ndarray:
        """The tensor's bit patterns, as read-only uint16, without copying.

        The returned view keeps alive the mapping behind it: there is no close to
        remember, and no path in which an array outlives its own buffer.
        """
        entry = self.tensors[name]
        if entry.format is None:
            raise SafetensorsError(f"{name}: dtype {entry.dtype} is not 16-bit")
        offset = self._data_start + entry.start
        if offset % 2:
            raise SafetensorsError(
                f"{name}: odd offset {offset}, cannot be mapped as uint16"
            )
        return self.raw[offset : offset + entry.nbytes].view(np.uint16)

    def iter_codes(self, fmt: FloatFormat) -> Iterator[tuple[TensorEntry, np.ndarray]]:
        for entry in self.entries_of_format(fmt):
            yield entry, self.codes(entry.name)


HISTOGRAM_CHUNK = 1 << 24


def code_histogram(
    file: SafetensorsFile, fmt: FloatFormat, chunk: int = HISTOGRAM_CHUNK
) -> np.ndarray:
    """An exact count of every 16-bit pattern present in the format's tensors.

    For the study of flips this histogram **is** the model: the outcome of a flip
    depends on the pattern, not on which weight carries it. Half a billion parameters
    therefore collapse into 65,536 cells without losing a single useful piece of
    information, and every per-bit-position statistic becomes exact rather than
    sampled.
    """
    counts = np.zeros(CODE_SPACE, dtype=np.uint64)
    for _, codes in file.iter_codes(fmt):
        for start in range(0, codes.size, chunk):
            block = codes[start : start + chunk]
            counts += np.bincount(block, minlength=CODE_SPACE).astype(np.uint64)
    return counts
