"""Anatomia di un file GGUF: dove stanno i bit, e quali contano.

La domanda di E3 e se la quantizzazione protegga. La risposta non sta nel numero di
bit ma nella loro **funzione**: in un blocco Q4_K, 128 byte portano i quanti veri e
4 byte portano le scale in fp16 da cui dipendono tutti e 256 i pesi del blocco. Un
flip nei primi sposta un peso di un gradino; un flip nei secondi ne sposta 256 insieme.
Questo modulo separa le due popolazioni e le conta.
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

SCALE_FP16 = "scala_fp16"
SCALE_INT = "scala_intera"
QUANT = "quanti"
FLOAT = "float"


class GGUFError(ValueError):
    """Il file non rispetta il formato GGUF dichiarato."""


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
    """Una porzione di blocco, con il ruolo che ricopre."""

    name: str
    offset: int
    nbytes: int
    kind: str


@dataclass(frozen=True)
class BlockLayout:
    """Come sono disposti i bit dentro un blocco quantizzato."""

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
    """Un tensore nel file: nome, forma, tipo, posizione nei dati."""

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
            raise GGUFError(f"{self.name}: tipo ggml {self.type_id} non descritto")
        return layout

    @property
    def blocks(self) -> int:
        layout = self.layout
        if self.elements % layout.elements:
            raise GGUFError(
                f"{self.name}: {self.elements} elementi non divisibili "
                f"in blocchi da {layout.elements}"
            )
        return self.elements // layout.elements

    @property
    def nbytes(self) -> int:
        return self.blocks * self.layout.nbytes


class _Reader:
    """Lettura sequenziale dei tipi primitivi GGUF."""

    def __init__(self, data) -> None:
        self.data = data
        self.cursor = 0

    def take(self, count: int) -> bytes:
        if self.cursor + count > len(self.data):
            raise GGUFError("file troncato durante la lettura dell'intestazione")
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
    """Struttura di un file GGUF, letta senza caricare i dati dei tensori."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        with (
            self.path.open("rb") as handle,
            mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
        ):
            self._parse(_Reader(mapped))
        self._validate()

    def _parse(self, reader: _Reader) -> None:
        """Legge intestazione, metadati e tabella dei tensori.

        L'intestazione di un GGUF non ha dimensione prevedibile: qui dentro c'e anche
        il vocabolario del tokenizer, che per questo modello supera i quattro megabyte.
        Si mappa il file invece di indovinare quanto leggerne.
        """
        if reader.take(4) != GGUF_MAGIC:
            raise GGUFError(f"{self.path}: magic GGUF assente")
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
        """Il collaudo del parser: l'aritmetica deve chiudere sui byte del file."""
        cursor = 0
        for tensor in sorted(self.tensors, key=lambda item: item.offset):
            if tensor.offset != cursor:
                raise GGUFError(
                    f"{tensor.name}: inizia a {tensor.offset}, i dati sono a {cursor}"
                )
            cursor += tensor.nbytes
            cursor += -cursor % self.alignment

        file_size = self.path.stat().st_size
        expected = self.data_start + cursor
        if expected != file_size:
            raise GGUFError(
                f"l'aritmetica non chiude: intestazione {self.data_start} + dati "
                f"{cursor} = {expected} != {file_size} byte di file"
            )

    @property
    def raw(self) -> np.memmap:
        """Il file intero, mappato in sola lettura, creato al primo accesso."""
        if getattr(self, "_raw", None) is None:
            self._raw = np.memmap(self.path, dtype=np.uint8, mode="r")
        return self._raw

    def field_codes(self, tensor: GGUFTensor, kind: str = SCALE_FP16) -> np.ndarray:
        """Pattern a 16 bit di tutti i campi del ruolo indicato, per quel tensore.

        I campi scala non sono contigui: stanno all'inizio di ogni blocco. Si rimodella
        l'intervallo del tensore in (blocchi × byte_per_blocco) e si prende la colonna.

        L'ordine dell'uscita e **per campo, non per blocco**: prima tutte le `d`, poi
        tutte le `dmin`. Per un istogramma e indifferente, ma il contratto va detto.
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
        """Quanti bit di dati appartengono a ciascun ruolo."""
        census: dict[str, int] = {}
        for tensor in self.tensors:
            layout = tensor.layout
            for field in layout.fields:
                census[field.kind] = (
                    census.get(field.kind, 0) + field.nbytes * 8 * tensor.blocks
                )
        return census

    def scale_fields(self) -> int:
        """Numero di campi scala in fp16 presenti nel file."""
        return sum(
            tensor.blocks
            * sum(1 for field in tensor.layout.fields if field.kind == SCALE_FP16)
            for tensor in self.tensors
        )
