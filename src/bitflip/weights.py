"""Lettura dei pesi al livello dei bit, senza passare per i float.

Il formato safetensors e semplice abbastanza da leggerlo direttamente: otto byte di
lunghezza, un'intestazione JSON, un buffer contiguo. Farlo a mano ha tre vantaggi che
qui contano — accesso in mmap di sola lettura (Principio I), nessuna conversione che
falsifichi i pattern di bit, e un'aritmetica che deve **chiudere esattamente** sulla
dimensione del file, il che rende il parser il proprio collaudo.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np

from bitflip.codec import BF16, FP16, FloatFormat

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
    """Il file non rispetta il formato dichiarato."""


@dataclass(frozen=True)
class TensorEntry:
    """Un tensore nel buffer: dove sta e che forma ha."""

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
            raise SafetensorsError(f"{path}: file troncato prima dell'intestazione")
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
    for entry in entries.values():
        item_size = ITEM_SIZES.get(entry.dtype)
        if item_size is None:
            raise SafetensorsError(f"{entry.name}: dtype sconosciuto {entry.dtype}")
        if entry.nbytes != entry.count * item_size:
            raise SafetensorsError(
                f"{entry.name}: {entry.nbytes} byte per {entry.count} "
                f"elementi da {item_size}"
            )

    ordered = sorted(entries.values(), key=lambda entry: entry.start)
    cursor = 0
    for entry in ordered:
        if entry.start != cursor:
            raise SafetensorsError(
                f"{entry.name}: inizia a {entry.start}, il buffer e a {cursor}"
            )
        cursor = entry.end

    if data_start + cursor != file_size:
        raise SafetensorsError(
            f"l'aritmetica non chiude: intestazione {data_start} + dati {cursor} "
            f"!= {file_size} byte di file"
        )


class SafetensorsFile:
    """Accesso in sola lettura ai pattern di bit di un file safetensors."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.tensors, self._data_start, self.metadata = _parse_header(self.path)
        _validate(self.tensors, self._data_start, self.path.stat().st_size)
        self._raw: np.memmap | None = None

    @property
    def raw(self) -> np.memmap:
        """Il file intero, mappato in sola lettura, creato al primo accesso."""
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
        """Pattern di bit del tensore, come uint16 di sola lettura, senza copia.

        La vista restituita tiene in vita la mappa che la sostiene: non c'e una
        chiusura da ricordare, e nessun percorso in cui un array sopravviva al
        proprio buffer.
        """
        entry = self.tensors[name]
        if entry.format is None:
            raise SafetensorsError(f"{name}: dtype {entry.dtype} non e a 16 bit")
        offset = self._data_start + entry.start
        if offset % 2:
            raise SafetensorsError(
                f"{name}: offset dispari {offset}, non mappabile a uint16"
            )
        return self.raw[offset : offset + entry.nbytes].view(np.uint16)

    def iter_codes(self, fmt: FloatFormat) -> Iterator[tuple[TensorEntry, np.ndarray]]:
        for entry in self.entries_of_format(fmt):
            yield entry, self.codes(entry.name)


CODE_SPACE = 1 << 16
HISTOGRAM_CHUNK = 1 << 24


def code_histogram(
    file: SafetensorsFile, fmt: FloatFormat, chunk: int = HISTOGRAM_CHUNK
) -> np.ndarray:
    """Conteggio esatto di ogni pattern di 16 bit presente nei tensori del formato.

    Per lo studio dei flip questo istogramma **e** il modello: l'esito di un flip
    dipende dal pattern, non da quale peso lo porti. Mezzo miliardo di parametri si
    riducono cosi a 65.536 celle senza perdere una sola informazione utile, e ogni
    statistica per posizione di bit diventa esatta invece che campionata.
    """
    counts = np.zeros(CODE_SPACE, dtype=np.uint64)
    for _, codes in file.iter_codes(fmt):
        for start in range(0, codes.size, chunk):
            block = codes[start : start + chunk]
            counts += np.bincount(block, minlength=CODE_SPACE).astype(np.uint64)
    return counts
