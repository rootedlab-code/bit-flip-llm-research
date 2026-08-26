"""Misure deterministiche di qualita del modello.

Il modello non viene costruito qui: si riceve una funzione che, dati dei token,
restituisce i logit. Cosi l'aritmetica della perplexity si verifica in locale con
funzioni finte in millisecondi, e su Kaggle la stessa funzione riceve un modello vero
senza che il codice cambi.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator

import numpy as np
import torch

ScoreFunction = Callable[[torch.Tensor], torch.Tensor]

DEFAULT_WINDOW = 1024
DEFAULT_STRIDE = 512

GREEDY_GENERATION = {
    "do_sample": False,
    "num_beams": 1,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}


def set_determinism(seed: int = 0) -> None:
    """Azzera ogni sorgente di casualita che possa entrare in una misura."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sliding_windows(
    length: int, window: int = DEFAULT_WINDOW, stride: int = DEFAULT_STRIDE
) -> Iterator[tuple[int, int, int]]:
    """Finestre (inizio, fine, primo indice da valutare) che coprono la sequenza.

    Ogni token viene valutato **una volta sola**, ma con il contesto piu ampio che la
    finestra gli possa dare: e la differenza tra una perplexity onesta e una gonfiata
    dal doppio conteggio. Il token in posizione 0 non ha predecessore e non si valuta.
    """
    if window <= stride:
        raise ValueError(f"finestra {window} non maggiore del passo {stride}")
    if length < 2:
        return

    scored_up_to = 1
    for start in range(0, length, stride):
        end = min(start + window, length)
        if end <= scored_up_to:
            continue
        yield start, end, scored_up_to - start
        scored_up_to = end
        if end == length:
            return


def perplexity(
    score: ScoreFunction,
    token_ids: torch.Tensor,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
) -> float:
    """Perplexity a finestra scorrevole: exp della log-verosimiglianza media per token.

    `score` riceve i token di una finestra e restituisce i logit di forma
    (lunghezza, vocabolario). Solleva `ValueError` su sequenze troppo corte.
    """
    token_ids = token_ids.flatten()
    if token_ids.numel() < 2:
        raise ValueError("servono almeno due token per valutare una predizione")

    total_nll = 0.0
    total_tokens = 0
    for start, end, first in sliding_windows(token_ids.numel(), window, stride):
        chunk = token_ids[start:end]
        logits = score(chunk).reshape(chunk.numel(), -1)
        targets = chunk[first:]
        predictions = logits[first - 1 : -1]
        total_nll += float(
            torch.nn.functional.cross_entropy(
                predictions.float(), targets, reduction="sum"
            )
        )
        total_tokens += targets.numel()

    return math.exp(total_nll / total_tokens)
