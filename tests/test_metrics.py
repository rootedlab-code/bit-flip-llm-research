"""Contratto delle misure: l'aritmetica si verifica senza eseguire un modello."""

from __future__ import annotations

import math

import pytest
import torch

from bitflip.metrics import perplexity, set_determinism, sliding_windows

VOCABULARY = 50


def uniform_scorer(chunk: torch.Tensor) -> torch.Tensor:
    """Modello che non sa nulla: stessa probabilita a ogni token."""
    return torch.zeros(chunk.numel(), VOCABULARY)


def oracle_scorer(chunk: torch.Tensor) -> torch.Tensor:
    """Modello che conosce il token successivo e gli da quasi tutta la massa."""
    logits = torch.zeros(chunk.numel(), VOCABULARY)
    for position in range(chunk.numel() - 1):
        logits[position, chunk[position + 1]] = 30.0
    return logits


def test_windows_cover_every_predictable_token_exactly_once():
    length = 3000
    scored = []
    for start, end, first in sliding_windows(length, window=1024, stride=512):
        scored.extend(range(start + first, end))

    assert scored == list(range(1, length))


def test_windows_use_the_widest_available_context():
    """Dalla seconda finestra in poi ogni token valutato ha mezza finestra di contesto.

    L'ultima e piu corta perche la sequenza finisce, non perche il contesto manchi.
    """
    windows = list(sliding_windows(3000, window=1024, stride=512))

    for start, end, first in windows[1:]:
        assert first == 512
        assert end - start == min(1024, 3000 - start)
    full = [end - start for start, end, _ in windows[:-1]]
    assert full == [1024] * (len(windows) - 1)


def test_a_short_sequence_yields_a_single_window():
    assert list(sliding_windows(10, window=1024, stride=512)) == [(0, 10, 1)]


def test_a_stride_at_least_as_long_as_the_window_is_rejected():
    with pytest.raises(ValueError, match="non maggiore del passo"):
        list(sliding_windows(100, window=512, stride=512))


def test_uniform_model_has_perplexity_equal_to_the_vocabulary_size():
    """Se ogni token e equiprobabile, la sorpresa per token e ln(V), quindi PPL = V."""
    tokens = torch.randint(0, VOCABULARY, (2000,))

    assert perplexity(uniform_scorer, tokens, window=256, stride=128) == pytest.approx(
        VOCABULARY, rel=1e-6
    )


def test_a_model_that_knows_the_next_token_approaches_perplexity_one():
    tokens = torch.randint(0, VOCABULARY, (500,))

    assert perplexity(oracle_scorer, tokens, window=256, stride=128) < 1.001


def test_perplexity_is_reproducible_to_the_last_digit():
    tokens = torch.randint(0, VOCABULARY, (1500,))

    first = perplexity(uniform_scorer, tokens, window=256, stride=128)
    second = perplexity(uniform_scorer, tokens, window=256, stride=128)

    assert first == second


def test_windowing_does_not_change_the_measure_of_a_context_free_model():
    """Un modello che ignora il contesto deve dare la stessa cifra a ogni finestratura."""
    tokens = torch.randint(0, VOCABULARY, (2000,))

    wide = perplexity(uniform_scorer, tokens, window=1024, stride=512)
    narrow = perplexity(uniform_scorer, tokens, window=128, stride=64)

    # La tolleranza e quella del softmax in float32 (~1e-7 relativo), non una comodita:
    # sotto quella soglia si misurerebbe l'errore di arrotondamento, non il modello.
    assert wide == pytest.approx(narrow, rel=1e-6)


def test_a_sequence_too_short_to_predict_is_rejected():
    with pytest.raises(ValueError, match="almeno due token"):
        perplexity(uniform_scorer, torch.tensor([7]))


def test_known_nll_gives_the_expected_perplexity():
    """Controllo numerico chiuso: due token, distribuzione nota, PPL calcolata a mano."""
    tokens = torch.tensor([0, 1])

    def biased(chunk):
        logits = torch.full((chunk.numel(), 2), 0.0)
        logits[0, 1] = math.log(3.0)  # p(token 1) = 3/4 dopo il softmax su [0, ln 3]
        return logits

    assert perplexity(biased, tokens, window=4, stride=2) == pytest.approx(4.0 / 3.0)


def test_set_determinism_is_idempotent_on_the_drawn_values():
    set_determinism(1234)
    first = torch.randn(5)
    set_determinism(1234)

    assert torch.equal(first, torch.randn(5))
