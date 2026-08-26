"""The GGUF parser contract: here too the test is that the arithmetic closes."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from bitflip.fetch import QUANTIZED
from bitflip.gguf import (
    BLOCK_LAYOUTS,
    GGUF_MAGIC,
    QUANT,
    SCALE_FP16,
    GGUFError,
    GGUFFile,
    ValueType,
)

Q4_K_TYPE = 12
ALIGNMENT = 32


def encode_string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def build_gguf(path, *, blocks=2, type_id=Q4_K_TYPE, offset_shift=0, truncate=0):
    """Write a minimal GGUF: one quantized tensor and one metadata entry."""
    layout = BLOCK_LAYOUTS[type_id]
    header = bytearray(GGUF_MAGIC)
    header += struct.pack("<IQQ", 3, 1, 1)
    header += encode_string("general.alignment")
    header += struct.pack("<I", int(ValueType.UINT32)) + struct.pack("<I", ALIGNMENT)
    header += encode_string("blk.0.weight")
    header += struct.pack("<I", 1) + struct.pack("<Q", blocks * layout.elements)
    header += struct.pack("<I", type_id) + struct.pack("<Q", offset_shift)

    padding = -len(header) % ALIGNMENT
    payload = bytes(range(256)) * ((blocks * layout.nbytes) // 256 + 1)
    data = payload[: blocks * layout.nbytes]
    body = bytes(header) + b"\x00" * padding + b"\x00" * offset_shift + data
    path.write_bytes(body[: len(body) - truncate] if truncate else body)
    return path


@pytest.fixture
def synthetic(tmp_path):
    return build_gguf(tmp_path / "tiny.gguf")


def test_header_and_metadata_are_parsed(synthetic):
    file = GGUFFile(synthetic)

    assert file.version == 3
    assert len(file) == 1
    assert file.alignment == ALIGNMENT
    assert file.parameter_count == 512


def test_arithmetic_closes_on_the_file_size(synthetic):
    file = GGUFFile(synthetic)

    assert file.data_start + file.tensors[0].nbytes == synthetic.stat().st_size


def test_a_truncated_file_is_rejected(tmp_path):
    path = build_gguf(tmp_path / "short.gguf", truncate=8)

    with pytest.raises(GGUFError, match="arithmetic does not close"):
        GGUFFile(path)


def test_a_tensor_that_does_not_start_where_the_data_do_is_rejected(tmp_path):
    path = build_gguf(tmp_path / "shifted.gguf", offset_shift=ALIGNMENT)

    with pytest.raises(GGUFError, match="data are at"):
        GGUFFile(path)


def test_a_file_without_the_magic_is_rejected(tmp_path):
    path = tmp_path / "notgguf.bin"
    path.write_bytes(b"XXXX" + b"\x00" * 64)

    with pytest.raises(GGUFError, match="GGUF magic missing"):
        GGUFFile(path)


@pytest.mark.parametrize("type_id", sorted(BLOCK_LAYOUTS))
def test_every_layout_declares_consistent_field_coverage(type_id):
    layout = BLOCK_LAYOUTS[type_id]

    assert sum(field.nbytes for field in layout.fields) == layout.nbytes
    cursor = 0
    for field in layout.fields:
        assert field.offset == cursor
        cursor += field.nbytes


def test_q4_k_superblock_is_two_scales_twelve_bytes_and_one_hundred_twenty_eight_quants():
    layout = BLOCK_LAYOUTS[Q4_K_TYPE]

    assert (layout.elements, layout.nbytes) == (256, 144)
    assert layout.bytes_of_kind(SCALE_FP16) == 4
    assert layout.bytes_of_kind(QUANT) == 128


def test_bit_census_accounts_for_every_data_bit(synthetic):
    file = GGUFFile(synthetic)

    assert sum(file.bit_census().values()) == file.tensors[0].nbytes * 8


def test_field_codes_returns_two_scales_per_block(synthetic):
    file = GGUFFile(synthetic)

    codes = file.field_codes(file.tensors[0], SCALE_FP16)

    assert codes.shape == (4,)
    assert codes.dtype == np.uint16


def test_field_codes_groups_by_field_not_by_block(synthetic):
    file = GGUFFile(synthetic)
    tensor = file.tensors[0]
    block_bytes = tensor.layout.nbytes
    data = file.raw[file.data_start :]

    def word(offset):
        return int(np.frombuffer(data[offset : offset + 2].tobytes(), dtype=np.uint16)[0])

    codes = file.field_codes(tensor, SCALE_FP16)

    # all the `d` of both blocks first, then the `dmin`: the order is by field
    assert list(codes) == [
        word(0),
        word(block_bytes),
        word(2),
        word(block_bytes + 2),
    ]


@pytest.mark.skipif(not QUANTIZED.primary_path.exists(), reason="gguf non scaricato")
def test_the_real_file_closes_and_every_scale_field_is_counted():
    file = GGUFFile(QUANTIZED.primary_path)

    scales_from_layout = sum(
        tensor.blocks
        * sum(1 for field in tensor.layout.fields if field.kind == SCALE_FP16)
        for tensor in file.tensors
    )
    census_scale_bits = file.bit_census()[SCALE_FP16]

    assert file.scale_fields() == scales_from_layout
    assert census_scale_bits == scales_from_layout * 16
