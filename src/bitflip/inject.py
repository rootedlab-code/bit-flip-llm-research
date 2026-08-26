"""Iniezione di guasti nei pesi, in memoria e sempre reversibile.

Principio I e III della costituzione: nessun peso modificato tocca il disco, e la
scelta dei bit e riproducibile da un seme. La selezione e aritmetica pura su numpy,
quindi si verifica in locale senza modelli; l'applicazione a un modello vero e un
adattatore sottile che vive in fondo al file.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from bitflip.codec import BF16, FloatFormat

TOP_EXPONENT_BIT = 14


@dataclass(frozen=True)
class Flip:
    """Un guasto: quale tensore, quale peso, quale bit."""

    tensor: str
    index: int
    bit: int


def random_flips(
    sizes: Mapping[str, int],
    count: int,
    seed: int,
    bits: Sequence[int] | None = None,
) -> list[Flip]:
    """Guasti uniformi su tutti i bit di tutti i pesi — il modello del raggio cosmico.

    Uniforme significa proporzionale alla dimensione dei tensori: un tensore grande
    e colpito piu spesso perche offre piu bersagli, non perche conti di piu.
    """
    if count < 0:
        raise ValueError(f"numero di guasti negativo: {count}")
    names = list(sizes)
    if not names:
        raise ValueError("nessun tensore in cui iniettare")
    positions = list(bits) if bits is not None else list(range(BF16.total_bits))

    boundaries = np.cumsum([sizes[name] for name in names])
    generator = np.random.default_rng(seed)
    drawn = generator.integers(0, int(boundaries[-1]), size=count)
    chosen_tensor = np.searchsorted(boundaries, drawn, side="right")
    starts = np.concatenate(([0], boundaries[:-1]))
    chosen_bits = generator.choice(positions, size=count)

    return [
        Flip(names[tensor], int(offset - starts[tensor]), int(bit))
        for tensor, offset, bit in zip(chosen_tensor, drawn, chosen_bits, strict=True)
    ]


def largest_magnitude_flips(
    codes: Mapping[str, np.ndarray],
    count: int,
    fmt: FloatFormat = BF16,
    bit: int = TOP_EXPONENT_BIT,
) -> list[Flip]:
    """Guasti scelti: il bit alto dell'esponente dei pesi di modulo maggiore.

    E la politica che la letteratura trova efficace, ed E1 spiega perche funziona senza
    conoscere il modello: quel bit vale zero nel 100% dei pesi, quindi il flip amplifica
    sempre. Qui si aggiunge l'unica informazione che serve — dove stanno i pesi grandi.
    """
    from bitflip.codec import to_float32

    ranked: list[tuple[float, str, int]] = []
    for name, tensor_codes in codes.items():
        values = np.abs(to_float32(tensor_codes, fmt).astype(np.float64))
        take = min(count, values.size)
        top = np.argpartition(values, -take)[-take:]
        ranked.extend((float(values[index]), name, int(index)) for index in top)

    ranked.sort(key=lambda item: -item[0])
    return [Flip(name, index, bit) for _, name, index in ranked[:count]]


def apply_flips(codes: np.ndarray, flips: Sequence[Flip]) -> np.ndarray:
    """Applica i guasti a un array di pattern, restituendo una copia modificata."""
    modified = np.array(codes, dtype=np.uint16, copy=True)
    for flip in flips:
        modified[flip.index] ^= np.uint16(1 << flip.bit)
    return modified


def _integer_view(parameter):
    """Vista intera di un parametro, con lo scorrimento di bit che il dtype impone.

    Un peso bf16 promosso a float32 conserva il proprio pattern **esattamente** nei
    16 bit alti, perche la conversione bf16 → float32 riempie di zeri i bit bassi. Il
    bit `b` del bf16 memorizzato e quindi il bit `b + 16` del float32 che lo ospita.
    E cio che permette di calcolare in float32 su GPU senza bf16 nativo — come le T4
    di Kaggle — restando fedeli al guasto che avviene in DRAM.
    """
    import torch

    flat = parameter.view(-1)
    if flat.dtype == torch.bfloat16:
        return flat.view(torch.int16), 0, np.int16
    if flat.dtype == torch.float32:
        return flat.view(torch.int32), 16, np.int32
    raise TypeError(f"dtype {flat.dtype} non supportato per l'iniezione")


@contextmanager
def flipped_model(model, flips: Sequence[Flip]) -> Iterator[dict[str, int]]:
    """Applica i guasti a un modello torch e li **annulla sempre** all'uscita.

    I valori originali si conservano in memoria e si ripristinano anche se il corpo
    fallisce: senza questa garanzia una misura successiva erediterebbe il guasto della
    precedente, e nessuno se ne accorgerebbe.
    """
    import torch

    parameters = dict(model.named_parameters())
    originals: list[tuple[str, int, int]] = []

    try:
        with torch.no_grad():
            for flip in flips:
                view, shift, dtype = _integer_view(parameters[flip.tensor])
                originals.append((flip.tensor, flip.index, int(view[flip.index])))
                mask = int(np.array(1 << (flip.bit + shift)).astype(dtype))
                view[flip.index] ^= mask
        yield {"applicati": len(flips)}
    finally:
        with torch.no_grad():
            for name, index, value in reversed(originals):
                view, _, _ = _integer_view(parameters[name])
                view[index] = value
