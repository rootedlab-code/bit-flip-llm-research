"""Contratto del lettore safetensors: il parser si convalida chiudendo l'aritmetica."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bitflip.codec import BF16, from_float32, to_float32
from bitflip.fetch import BASE
from bitflip.weights import SafetensorsError, SafetensorsFile, code_histogram

QWEN_05B_PARAMETERS = 494_032_768


HEADER_ALIGNMENT = 8


def encode_header(header: dict) -> bytes:
    """Intestazione JSON riempita di spazi: la specifica allinea i dati a 8 byte."""
    encoded = json.dumps(header).encode()
    padding = -(len(encoded)) % HEADER_ALIGNMENT
    return encoded + b" " * padding


def build_file(path, tensors, shift_tensor=None, truncate=0):
    """Scrive un safetensors sintetico: intestazione JSON piu buffer contiguo."""
    header = {}
    buffer = bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        start = len(buffer)
        buffer.extend(payload.tobytes())
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(buffer)],
        }
    if shift_tensor:
        # Sposta in avanti l'intervallo senza cambiarne la lunghezza: crea un buco
        # nel buffer, che e esattamente cio che il controllo di contiguita deve vedere.
        header[shift_tensor]["data_offsets"] = [
            offset + 2 for offset in header[shift_tensor]["data_offsets"]
        ]
        buffer.extend(b"\x00\x00")

    encoded = encode_header(header)
    body = len(encoded).to_bytes(8, "little") + encoded + bytes(buffer)
    path.write_bytes(body[: len(body) - truncate] if truncate else body)
    return path


@pytest.fixture
def weights():
    return np.array([0.02, -1.5, 3.25, 0.0], dtype=np.float32)


@pytest.fixture
def synthetic(tmp_path, weights):
    codes = from_float32(weights, BF16)
    return build_file(
        tmp_path / "model.safetensors",
        {
            "block.weight": ("BF16", (2, 2), codes),
            "block.bias": ("F32", (2,), np.array([1.0, 2.0], dtype=np.float32)),
        },
    )


def test_header_is_parsed_into_entries(synthetic):
    file = SafetensorsFile(synthetic)

    assert len(file) == 2
    assert file.tensors["block.weight"].shape == (2, 2)
    assert file.tensors["block.weight"].dtype == "BF16"
    assert file.parameter_count == 6


def test_codes_expose_the_stored_bit_patterns(synthetic, weights):
    file = SafetensorsFile(synthetic)

    restored = to_float32(file.codes("block.weight"), BF16)

    assert np.array_equal(restored, to_float32(from_float32(weights, BF16), BF16))


def test_codes_are_not_writable(synthetic):
    assert not SafetensorsFile(synthetic).codes("block.weight").flags.writeable


def test_view_outlives_the_file_object(synthetic):
    file = SafetensorsFile(synthetic)
    codes = file.codes("block.weight")
    del file

    assert codes[0] == codes[0]


def test_codes_refuse_a_dtype_that_is_not_sixteen_bit(synthetic):
    with pytest.raises(SafetensorsError, match="non e a 16 bit"):
        SafetensorsFile(synthetic).codes("block.bias")


def test_entries_of_format_selects_only_bf16(synthetic):
    names = [entry.name for entry in SafetensorsFile(synthetic).entries_of_format(BF16)]

    assert names == ["block.weight"]


def test_a_gap_between_tensors_is_rejected(tmp_path, weights):
    path = build_file(
        tmp_path / "gap.safetensors",
        {
            "a": ("BF16", (4,), from_float32(weights, BF16)),
            "b": ("BF16", (4,), from_float32(weights, BF16)),
        },
        shift_tensor="b",
    )

    with pytest.raises(SafetensorsError, match="il buffer e a"):
        SafetensorsFile(path)


def test_a_truncated_file_fails_the_closing_arithmetic(tmp_path, weights):
    path = build_file(
        tmp_path / "short.safetensors",
        {"a": ("BF16", (4,), from_float32(weights, BF16))},
        truncate=2,
    )

    with pytest.raises(SafetensorsError, match="l'aritmetica non chiude"):
        SafetensorsFile(path)


def test_a_declared_shape_that_contradicts_the_byte_range_is_rejected(tmp_path):
    payload = np.zeros(4, dtype=np.uint16)
    path = tmp_path / "mismatch.safetensors"
    header = encode_header(
        {"a": {"dtype": "BF16", "shape": [8], "data_offsets": [0, len(payload) * 2]}}
    )
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload.tobytes())

    with pytest.raises(SafetensorsError, match="byte per 8 elementi"):
        SafetensorsFile(path)


@pytest.mark.skipif(not BASE.primary_path.exists(), reason="modello non scaricato")
def test_the_real_model_closes_exactly_and_is_entirely_bf16():
    file = SafetensorsFile(BASE.primary_path)

    assert file.parameter_count == QWEN_05B_PARAMETERS
    assert {entry.dtype for entry in file.tensors.values()} == {"BF16"}
    assert len(file.entries_of_format(BF16)) == len(file)


def test_code_histogram_counts_every_pattern(tmp_path):
    codes = np.array([0x3CA4, 0x3CA4, 0x0000, 0xFFFF], dtype=np.uint16)
    path = build_file(tmp_path / "hist.safetensors", {"w": ("BF16", (4,), codes)})

    counts = code_histogram(SafetensorsFile(path), BF16)

    assert counts.sum() == 4
    assert counts[0x3CA4] == 2
    assert counts[0x0000] == 1
    assert counts[0xFFFF] == 1


def test_code_histogram_is_chunk_invariant(tmp_path):
    codes = np.arange(1000, dtype=np.uint16)
    path = build_file(tmp_path / "chunk.safetensors", {"w": ("BF16", (1000,), codes)})
    file = SafetensorsFile(path)

    assert np.array_equal(code_histogram(file, BF16, chunk=7), code_histogram(file, BF16))
