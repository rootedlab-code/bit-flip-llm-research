"""The anatomy of a GGUF file: where the bits are, and which ones matter.

E3 asks whether quantization protects. The answer lies not in the number of bits but
in their **function**: in a Q4_K block, 128 bytes carry the actual quants and 4 bytes
carry the fp16 scales on which all 256 weights of the block depend. A flip in the
former moves one weight by one step; a flip in the latter moves 256 at once. This
module separates the two populations and counts them.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

SCALE_FP16 = "scale_fp16"
SCALE_INT = "int_scale"
QUANT = "quant"
FLOAT = "float"


class GGUFError(ValueError):
    """The file does not match the GGUF format it declares."""


class ValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


SCALAR_FORMATS = {
    ValueType.UINT8: "<B",
    ValueType.INT8: "<b",
    ValueType.UINT16: "<H",
    ValueType.INT16: "<h",
    ValueType.UINT32: "<I",
    ValueType.INT32: "<i",
    ValueType.FLOAT32: "<f",
    ValueType.BOOL: "<?",
    ValueType.UINT64: "<Q",
    ValueType.INT64: "<q",
    ValueType.FLOAT64: "<d",
}


@dataclass(frozen=True)
class Field:
    """A slice of a block, with the role it plays."""

    name: str
    offset: int
    nbytes: int
    kind: str


@dataclass(frozen=True)
class BlockLayout:
    """How the bits are laid out inside a quantized block."""

    type_name: str
    elements: int
    nbytes: int
    fields: tuple[Field, ...]

    def bytes_of_kind(self, kind: str) -> int:
        return sum(field.nbytes for field in self.fields if field.kind == kind)


def _scalar_block(type_name: str, nbytes: int) -> BlockLayout:
    return BlockLayout(
        type_name=type_name,
        elements=1,
        nbytes=nbytes,
        fields=(Field(type_name.lower(), 0, nbytes, FLOAT),),
    )


BLOCK_LAYOUTS = {
    0: _scalar_block("F32", 4),
    1: _scalar_block("F16", 2),
    8: BlockLayout(
        "Q8_0",
        32,
        34,
        (Field("d", 0, 2, SCALE_FP16), Field("qs", 2, 32, QUANT)),
    ),
    2: BlockLayout(
        "Q4_0",
        32,
        18,
        (Field("d", 0, 2, SCALE_FP16), Field("qs", 2, 16, QUANT)),
    ),
    3: BlockLayout(
        "Q4_1",
        32,
        20,
        (
            Field("d", 0, 2, SCALE_FP16),
            Field("m", 2, 2, SCALE_FP16),
            Field("qs", 4, 16, QUANT),
        ),
    ),
    6: BlockLayout(
        "Q5_0",
        32,
        22,
        (
            Field("d", 0, 2, SCALE_FP16),
            Field("qh", 2, 4, QUANT),
            Field("qs", 6, 16, QUANT),
        ),
    ),
    7: BlockLayout(
        "Q5_1",
        32,
        24,
        (
            Field("d", 0, 2, SCALE_FP16),
            Field("m", 2, 2, SCALE_FP16),
            Field("qh", 4, 4, QUANT),
            Field("qs", 8, 16, QUANT),
        ),
    ),
    10: BlockLayout(
        "Q2_K",
        256,
        84,
        (
            Field("scales", 0, 16, SCALE_INT),
            Field("qs", 16, 64, QUANT),
            Field("d", 80, 2, SCALE_FP16),
            Field("dmin", 82, 2, SCALE_FP16),
        ),
    ),
    11: BlockLayout(
        "Q3_K",
        256,
        110,
        (
            Field("hmask", 0, 32, QUANT),
            Field("qs", 32, 64, QUANT),
            Field("scales", 96, 12, SCALE_INT),
            Field("d", 108, 2, SCALE_FP16),
        ),
    ),
    12: BlockLayout(
        "Q4_K",
        256,
        144,
        (
            Field("d", 0, 2, SCALE_FP16),
            Field("dmin", 2, 2, SCALE_FP16),
            Field("scales", 4, 12, SCALE_INT),
            Field("qs", 16, 128, QUANT),
        ),
    ),
    13: BlockLayout(
        "Q5_K",
        256,
        176,
        (
            Field("d", 0, 2, SCALE_FP16),
            Field("dmin", 2, 2, SCALE_FP16),
            Field("scales", 4, 12, SCALE_INT),
            Field("qh", 16, 32, QUANT),
            Field("qs", 48, 128, QUANT),
        ),
    ),
    14: BlockLayout(
        "Q6_K",
        256,
        210,
        (
            Field("ql", 0, 128, QUANT),
            Field("qh", 128, 64, QUANT),
            Field("scales", 192, 16, SCALE_INT),
            Field("d", 208, 2, SCALE_FP16),
        ),
    ),
    30: _scalar_block("BF16", 2),
}


@dataclass(frozen=True)
class GGUFTensor:
    """A tensor in the file: name, shape, type, position in the data."""

    name: str
    shape: tuple[int, ...]
    type_id: int
    offset: int

    @property
    def elements(self) -> int:
        count = 1
        for dimension in self.shape:
            count *= dimension
        return count

    @property
    def layout(self) -> BlockLayout:
        layout = BLOCK_LAYOUTS.get(self.type_id)
        if layout is None:
            raise GGUFError(f"{self.name}: ggml type {self.type_id} is not described")
        return layout

    @property
    def blocks(self) -> int:
        layout = self.layout
        if self.elements % layout.elements:
            raise GGUFError(
                f"{self.name}: {self.elements} elements do not divide "
                f"into blocks of {layout.elements}"
            )
        return self.elements // layout.elements

    @property
    def nbytes(self) -> int:
        return self.blocks * self.layout.nbytes


class _Reader:
    """Sequential reading of GGUF primitive types."""

    def __init__(self, data) -> None:
        self.data = data
        self.cursor = 0

    def take(self, count: int) -> bytes:
        if self.cursor + count > len(self.data):
            raise GGUFError("file truncated while reading the header")
        chunk = self.data[self.cursor : self.cursor + count]
        self.cursor += count
        return chunk

    def scalar(self, value_type: ValueType):
        fmt = SCALAR_FORMATS[value_type]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.take(self.scalar(ValueType.UINT64)).decode("utf-8", "replace")

    def value(self, value_type: ValueType):
        if value_type == ValueType.STRING:
            return self.string()
        if value_type == ValueType.ARRAY:
            item_type = ValueType(self.scalar(ValueType.UINT32))
            length = self.scalar(ValueType.UINT64)
            return [self.value(item_type) for _ in range(length)]
        return self.scalar(value_type)


class GGUFFile:
    """The structure of a GGUF file, read without loading any tensor data."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        with (
            self.path.open("rb") as handle,
            mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
        ):
            self._parse(_Reader(mapped))
        self._validate()

    def _parse(self, reader: _Reader) -> None:
        """Read the header, the metadata and the tensor table.

        A GGUF header has no predictable size: it also holds the tokenizer vocabulary,
        which for this model exceeds four megabytes. The file is mapped rather than
        guessed at.
        """
        if reader.take(4) != GGUF_MAGIC:
            raise GGUFError(f"{self.path}: GGUF magic missing")
        self.version = reader.scalar(ValueType.UINT32)
        tensor_count = reader.scalar(ValueType.UINT64)
        metadata_count = reader.scalar(ValueType.UINT64)

        self.metadata = {}
        for _ in range(metadata_count):
            key = reader.string()
            self.metadata[key] = reader.value(ValueType(reader.scalar(ValueType.UINT32)))

        self.tensors = []
        for _ in range(tensor_count):
            name = reader.string()
            dimensions = reader.scalar(ValueType.UINT32)
            shape = tuple(reader.scalar(ValueType.UINT64) for _ in range(dimensions))
            self.tensors.append(
                GGUFTensor(
                    name=name,
                    shape=shape,
                    type_id=reader.scalar(ValueType.UINT32),
                    offset=reader.scalar(ValueType.UINT64),
                )
            )

        self.alignment = int(self.metadata.get("general.alignment", DEFAULT_ALIGNMENT))
        self.data_start = reader.cursor + (-reader.cursor % self.alignment)

    def _validate(self) -> None:
        """The parser's own test: the arithmetic must close on the file's bytes."""
        cursor = 0
        for tensor in sorted(self.tensors, key=lambda item: item.offset):
            if tensor.offset != cursor:
                raise GGUFError(
                    f"{tensor.name}: starts at {tensor.offset}, data are at {cursor}"
                )
            cursor += tensor.nbytes
            cursor += -cursor % self.alignment

        file_size = self.path.stat().st_size
        expected = self.data_start + cursor
        if expected != file_size:
            raise GGUFError(
                f"arithmetic does not close: header {self.data_start} + data "
                f"{cursor} = {expected} != {file_size} bytes of file"
            )

    @property
    def raw(self) -> np.memmap:
        """The whole file, mapped read-only, created on first access."""
        if getattr(self, "_raw", None) is None:
            self._raw = np.memmap(self.path, dtype=np.uint8, mode="r")
        return self._raw

    def field_codes(self, tensor: GGUFTensor, kind: str = SCALE_FP16) -> np.ndarray:
        """The 16-bit patterns of every field of the given role, for that tensor.

        Scale fields are not contiguous: they sit at the start of each block. The
        tensor's byte range is reshaped into (blocks x bytes_per_block) and the column
        is taken.

        The output is ordered **by field, not by block**: all the `d` first, then all
        the `dmin`. It makes no difference to a histogram, but the contract must be
        stated.
        """
        layout = tensor.layout
        start = self.data_start + tensor.offset
        blocks = self.raw[start : start + tensor.nbytes].reshape(
            tensor.blocks, layout.nbytes
        )
        columns = [
            np.ascontiguousarray(
                blocks[:, field.offset : field.offset + field.nbytes]
            ).view(np.uint16)
            for field in layout.fields
            if field.kind == kind
        ]
        if not columns:
            return np.zeros(0, dtype=np.uint16)
        return np.concatenate([column.ravel() for column in columns])

    def __len__(self) -> int:
        return len(self.tensors)

    @property
    def parameter_count(self) -> int:
        return sum(tensor.elements for tensor in self.tensors)

    def bit_census(self) -> dict[str, int]:
        """How many data bits belong to each role."""
        census: dict[str, int] = {}
        for tensor in self.tensors:
            layout = tensor.layout
            for field in layout.fields:
                census[field.kind] = (
                    census.get(field.kind, 0) + field.nbytes * 8 * tensor.blocks
                )
        return census

    def scale_fields(self) -> int:
        """The number of fp16 scale fields present in the file."""
        return sum(
            tensor.blocks
            * sum(1 for field in tensor.layout.fields if field.kind == SCALE_FP16)
            for tensor in self.tensors
        )
