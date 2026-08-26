"""Deterministic measurements of model quality.

The model is not built here: a function is received that, given tokens, returns
logits. The perplexity arithmetic is therefore verified locally with stub functions in
milliseconds, and on Kaggle the same function receives a real model without a line of
code changing.
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
    """Zero out every source of randomness that could enter a measurement."""
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
    """Windows (start, end, first index to score) covering the sequence.

    Every token is scored **exactly once**, but with the widest context the window can
    give it: that is the difference between an honest perplexity and one inflated by
    double counting. The token at position 0 has no predecessor and is never scored.
    """
    if window <= stride:
        raise ValueError(f"window {window} is not greater than stride {stride}")
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
    """Sliding-window perplexity: exp of the mean per-token negative log-likelihood.

    `score` receives the tokens of one window and returns logits of shape
    (length, vocabulary). Raises `ValueError` on sequences that are too short.
    """
    token_ids = token_ids.flatten()
    if token_ids.numel() < 2:
        raise ValueError("at least two tokens are needed to score a prediction")

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
