"""The safetensors reader contract: the parser validates itself by closing the sums."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bitflip.codec import BF16, from_float32, to_float32
from bitflip.fetch import BASE
from bitflip.weights import (
    SHARD_INDEX_NAME,
    SafetensorsError,
    SafetensorsFile,
    ShardedWeights,
    ShardIndex,
    code_histogram,
    dtype_census,
    open_weights,
)

QWEN_05B_PARAMETERS = 494_032_768


HEADER_ALIGNMENT = 8


def encode_header(header: dict) -> bytes:
    """JSON header padded with spaces: the spec aligns the data to 8 bytes."""
    encoded = json.dumps(header).encode()
    padding = -(len(encoded)) % HEADER_ALIGNMENT
    return encoded + b" " * padding


def build_file(path, tensors, shift_tensor=None, truncate=0):
    """Write a synthetic safetensors file: JSON header plus contiguous buffer."""
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
        # Shift the range forward without changing its length: this creates a hole
        # in the buffer, which is exactly what the contiguity check must catch.
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
    with pytest.raises(SafetensorsError, match="is not 16-bit"):
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

    with pytest.raises(SafetensorsError, match="the buffer is at"):
        SafetensorsFile(path)


def test_a_truncated_file_fails_the_closing_arithmetic(tmp_path, weights):
    path = build_file(
        tmp_path / "short.safetensors",
        {"a": ("BF16", (4,), from_float32(weights, BF16))},
        truncate=2,
    )

    with pytest.raises(SafetensorsError, match="arithmetic does not close"):
        SafetensorsFile(path)


def test_a_declared_shape_that_contradicts_the_byte_range_is_rejected(tmp_path):
    payload = np.zeros(4, dtype=np.uint16)
    path = tmp_path / "mismatch.safetensors"
    header = encode_header(
        {"a": {"dtype": "BF16", "shape": [8], "data_offsets": [0, len(payload) * 2]}}
    )
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload.tobytes())

    with pytest.raises(SafetensorsError, match="bytes for 8 elements"):
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


# --- sharded models -------------------------------------------------------------
#
# Above a few gigabytes safetensors splits a model across files. The histogram that
# E1 publishes is a whole-population count, so the only failure that would corrupt it
# silently is a shard that goes missing: the run would simply measure fewer weights.

SHARD_NAMES = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


def build_shards(directory, groups, total_size=None, promise=()):
    """Write one safetensors file per group, plus the index that joins them."""
    weight_map = {}
    stored = 0
    for shard_name, tensors in zip(SHARD_NAMES, groups, strict=True):
        build_file(directory / shard_name, tensors)
        for name, (_, _, payload) in tensors.items():
            weight_map[name] = shard_name
            stored += payload.nbytes
    for name in promise:
        weight_map[name] = SHARD_NAMES[0]

    index = {
        "metadata": {"total_size": stored if total_size is None else total_size},
        "weight_map": weight_map,
    }
    (directory / SHARD_INDEX_NAME).write_text(json.dumps(index))
    return directory


@pytest.fixture
def split_codes(weights):
    return from_float32(np.concatenate([weights, weights * 3.5]), BF16)


@pytest.fixture
def split_model(tmp_path, split_codes):
    directory = tmp_path / "split"
    directory.mkdir()
    half = split_codes.size // 2
    return build_shards(
        directory,
        [
            {"a.weight": ("BF16", (half,), split_codes[:half])},
            {"b.weight": ("BF16", (half,), split_codes[half:])},
        ],
    )


def test_open_weights_reads_an_unsplit_model_as_one_file(synthetic):
    source = open_weights(synthetic.parent)

    assert isinstance(source, SafetensorsFile)
    assert source.paths == [synthetic]


def test_open_weights_joins_the_shards_of_a_split_model(split_model, split_codes):
    source = open_weights(split_model)

    assert isinstance(source, ShardedWeights)
    assert len(source) == 2
    assert source.parameter_count == split_codes.size
    assert [path.name for path in source.paths] == list(SHARD_NAMES)


def test_sharded_histogram_equals_the_histogram_of_the_unsplit_weights(
    tmp_path, split_model, split_codes
):
    whole = build_file(
        tmp_path / "model.safetensors",
        {"a.weight": ("BF16", (split_codes.size,), split_codes)},
    )

    split = code_histogram(open_weights(split_model), BF16)
    unsplit = code_histogram(SafetensorsFile(whole), BF16)

    assert np.array_equal(split, unsplit)
    assert int(split.sum()) == split_codes.size


def test_sharded_weights_reject_an_index_promising_an_absent_tensor(
    tmp_path, split_codes
):
    directory = tmp_path / "incomplete"
    directory.mkdir()
    half = split_codes.size // 2
    build_shards(
        directory,
        [
            {"a.weight": ("BF16", (half,), split_codes[:half])},
            {"b.weight": ("BF16", (half,), split_codes[half:])},
        ],
        promise=("c.weight",),
    )

    with pytest.raises(SafetensorsError, match="1 missing"):
        open_weights(directory)


def test_sharded_weights_reject_a_payload_the_index_does_not_declare(
    tmp_path, split_codes
):
    directory = tmp_path / "mismatched"
    directory.mkdir()
    half = split_codes.size // 2
    build_shards(
        directory,
        [
            {"a.weight": ("BF16", (half,), split_codes[:half])},
            {"b.weight": ("BF16", (half,), split_codes[half:])},
        ],
        total_size=1,
    )

    with pytest.raises(SafetensorsError, match="index declares 1"):
        open_weights(directory)


def test_shard_index_rejects_a_document_without_a_weight_map(tmp_path):
    path = tmp_path / SHARD_INDEX_NAME
    path.write_text(json.dumps({"metadata": {"total_size": 8}}))

    with pytest.raises(SafetensorsError, match="index missing"):
        ShardIndex.load(path)


def test_shard_index_names_each_shard_once_in_first_seen_order(tmp_path):
    path = tmp_path / SHARD_INDEX_NAME
    path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {"a": "two.st", "b": "one.st", "c": "two.st"},
            }
        )
    )

    assert [shard.name for shard in ShardIndex.load(path).shards] == ["two.st", "one.st"]


def test_open_weights_rejects_a_directory_holding_no_weights(tmp_path):
    with pytest.raises(SafetensorsError, match="neither"):
        open_weights(tmp_path)


def test_dtype_census_accounts_for_every_stored_tensor(synthetic):
    census = dtype_census(SafetensorsFile(synthetic))

    assert census["BF16"] == {"tensors": 1, "parameters": 4, "bytes": 8}
    assert census["F32"] == {"tensors": 1, "parameters": 2, "bytes": 8}
    assert sum(row["parameters"] for row in census.values()) == 6


def test_dtype_census_sums_across_shards(split_model, split_codes):
    census = dtype_census(open_weights(split_model))

    assert census == {
        "BF16": {
            "tensors": 2,
            "parameters": split_codes.size,
            "bytes": split_codes.nbytes,
        }
    }
