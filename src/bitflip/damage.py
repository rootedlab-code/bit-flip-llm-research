"""Classifying what a fault actually did to a model.

Two distinct ways of dying showed up in E2 and a single number cannot tell them apart.
Random flips leave the model numerically alive but ignorant: its output goes uniform
and perplexity pins to the vocabulary size. Chosen flips overflow the arithmetic and
NaN propagates, so perplexity is not a number at all. Reporting either as "damage" of
some magnitude loses the distinction that matters -- one of those models answers, the
other one crashes.

Thresholds live here as constants rather than in the caller: they are pre-registered
criteria under Principle IV, and a threshold chosen after seeing the data is not a
criterion.
"""

from __future__ import annotations

import math

INTACT = "intact"
PARTIAL = "partial"
UNIFORM_COLLAPSE = "uniform_collapse"
NUMERIC_COLLAPSE = "numeric_collapse"
DESTROYED = "destroyed"

INTACT_AGREEMENT = 0.99
DESTROYED_AGREEMENT = 0.01
UNIFORM_TOLERANCE = 0.01


def is_uniform(
    perplexity: float, vocabulary: int, tolerance: float = UNIFORM_TOLERANCE
) -> bool:
    """Whether the output is indistinguishable from a uniform distribution.

    A model that has lost every distinction assigns the same probability to every
    token, and the perplexity of a uniform distribution over V symbols is exactly V.
    """
    if not math.isfinite(perplexity) or vocabulary <= 0:
        return False
    return abs(perplexity - vocabulary) / vocabulary <= tolerance


def damage_class(perplexity: float, agreement: float, vocabulary: int) -> str:
    """Which of the four observed outcomes a measurement belongs to.

    Ordered so that the numeric outcome is decided first: when perplexity is not a
    number, agreement carries no information about the model, only about the fact that
    NaN never equals anything.
    """
    if not 0.0 <= agreement <= 1.0:
        raise ValueError(f"agreement {agreement} outside [0, 1]")
    if not math.isfinite(perplexity):
        return NUMERIC_COLLAPSE
    if agreement >= INTACT_AGREEMENT:
        return INTACT
    if agreement > DESTROYED_AGREEMENT:
        return PARTIAL
    return UNIFORM_COLLAPSE if is_uniform(perplexity, vocabulary) else DESTROYED
