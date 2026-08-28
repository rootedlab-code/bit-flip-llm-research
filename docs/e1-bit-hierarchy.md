# E1 — The fragility hierarchy of the bits

Subjects: `Qwen/Qwen2.5-0.5B-Instruct` @`7ae5576` and
`huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3` @`3dee99d`.
494,032,768 bf16 weights across 290 tensors, all of them BF16.
Raw data: `results/e1-bit-hierarchy-{base,abliterated}.csv`, `results/e1-summary.csv`.

## Method

The statistics are **exact, not sampled**. The outcome of a flip depends only on the
16-bit pattern, not on which weight carries it: the histogram of the 65,536 patterns
therefore summarises the entire model without loss, and every fraction below is a
count rather than an estimate. Coverage checks itself — the histogram total must equal
the parameter count declared by the header, and it does.

The model files were read inside the `immutable` context: their SHA-256 was
re-verified at the end of the experiment and is unchanged.

## The result

| bit | field | is 0 in | median \|Δw\| | max \|Δw\| | catastrophic |
|---|---|---|---|---|---|
| 0-6 | mantissa | ~50-58% | 6.1e-05 … 3.9e-03 | ≤ 64 | **0%** |
| 7-10 | low exponent | 36-66% | 9.3e-03 … 1.7e-01 | ≤ 5.5e+04 | 0% |
| 11 | exponent | 0.15% | 1.1e-02 | 1.4e+07 | 0.1508% |
| 12-13 | exponent | 0.00% | 1.1e-02 | 3.9e+21 | 0.0020% |
| **14** | **top exponent** | **99.9980%** | **3.9e+36** | 3.4e+38 | **99.998%** |
| 15 | sign | 50.02% | 2.3e-02 | 428 | 0% |

## Why bit 14 is the universal attack surface

It is not only that its multiplier is the largest (2¹²⁸, proven in
`tests/test_codec.py`). It is that its **value is predictable**: 99.9926% of weights
have |w| < 1, the median exponent is 120 against a bias of 127, and bit 14 is zero in
**99.9980%** of the weights of both models.

From which follows the fact that makes the attacks in the literature practical:

> to amplify a weight, an attacker does not need to know **which** weight is being
> hit. It is enough to hit bit 14, and in all but one weight in 49,797 the flip
> will be 0→1.

**The exception is worth stating rather than rounding away.** 9,921 weights of the
494,032,768 already carry a 1 in bit 14, and flipping theirs divides by 2¹²⁸ instead of
multiplying. Earlier revisions of this page reported the zero fraction as 100.00%, which
is what two decimals render it as, and wrote "any weight at all" underneath — a table
rounding that had turned into a stronger claim than the count supports. It does not
change the conclusion, and one in fifty thousand is not a defence anyone can build on,
but the fraction is 99.9980% and the sentence now says so.

The counterpart is just as sharp: bits 11-13 are 1 almost everywhere, so flipping them
**divides** rather than multiplies — they are harmless by construction, not by luck.
Every weight has one bit that amplifies it above all others, and it is always the same
one.

## The figure E4 will need

**6.2595%** of the file's bits are catastrophic (494,787,536 of 7,904,524,288), that
is **one in every 15.98**. A fault landing at random in this file therefore has roughly
a one-in-sixteen chance of being destructive — and that is the fraction which, crossed
with a FIT rate, will give the mean time before natural damage.

## Ablation does not change the fragility

The abliterated model has the same profile, figure for figure to four decimal places:
the differences in zero-bit fraction stay below 0.003% at every position, and the
catastrophic bits go from 494,787,536 to 494,787,360 — **176 bits of difference out of
7.9 billion**. Removing alignment neither hardens nor weakens the model in the face of
a fault: it moves the behaviour, not the geometry of the weights.

## Boundaries

This is the *arithmetic* fragility of an isolated weight. It says nothing yet about how
much the **model** degrades: a weight driven to 6.8e+36 in a rarely used tensor may not
change a comma of the output. That is E2's question.
