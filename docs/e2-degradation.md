# E2 — How much does the model actually degrade?

Subject: `Qwen/Qwen2.5-0.5B-Instruct` @`7ae5576`, bfloat16 patterns with float32
arithmetic. Corpus: WikiText-2 test split, first 8,192 tokens, sliding window 1024 /
stride 512. Raw data: `results/e2-degradation.csv`. Notebook: `kaggle/e2_degradation.py`.

E1 and E3 measured a **surface**: which bits are dangerous, and how many. Neither says
what happens to the model. This does.

## The control that makes the rest admissible

```
baseline perplexity : 16.105662448130897
second run          : 16.105662448130897   identical
zero flips          : 16.105662448130897   identical
```

The notebook asserts both before measuring anything and stops if either fails. Without
them, every number below would be noise with a decimal point.

## What a fault does

| faults | policy | outcome |
|---|---|---|
| 1 | random | 5/5 **intact** (agreement 0.9938 – 1.0000) |
| 10 | random | 2/5 intact · 1 **partial** · 2 uniform collapse |
| 100 | random | 0/5 intact · 5 uniform collapse |
| 1,000 | random | 0/5 intact · 5 uniform collapse |
| 10,000 | random | 0/5 intact · 5 uniform collapse |
| **1** | **chosen** | **numeric collapse** |

Degradation is not gradual. Of 29 measured configurations, 7 are intact, **1 is
partial**, and 21 are destroyed. The model either survives a fault untouched or stops
being a model; the middle is one case wide.

## Two different ways of dying

Random and chosen faults do not produce the same corpse.

- **Random → uniform collapse.** Perplexity settles at exactly 151,935.96 ≈ 151,936,
  the vocabulary size. That is the perplexity of a uniform distribution: the model is
  numerically alive and still answers, but every token is equally likely to it.
- **Chosen → numeric collapse.** Perplexity is NaN. The weight itself stays finite by
  construction — the injection policy requires it — but multiplying activations by
  3 × 10³⁸ overflows anyway, and NaN propagates through the graph.

The distinction matters beyond bookkeeping: a model that went uniform *responds*, and
one that went NaN does not. Neither is a stealthy failure, which is what E5 must look
for underneath both.

## The leverage of choosing the address

E1 measured, statically, that **6.259549%** of a bfloat16 model's bits are catastrophic.
E2 observed that a chosen bit is catastrophic in 4 configurations out of 4. So:

```
leverage per flip = 1 / 0.06259549 = 15.98x
```

which is exactly the "one catastrophic bit in every 15.98" that E1 produced by counting
patterns in a file — and that is agreement, **not independent confirmation**. 15.98 is the
reciprocal of E1's own counted fraction, so the two cannot disagree: the same number stands
on both sides of the comparison. An earlier version of this page called it "two methods, no
shared assumption, same figure", which was wrong in the middle clause.

What E2 contributes is the **numerator** — that a chosen flip is catastrophic at all — and
it contributes it on four configurations. Four successes in four trials put a one-sided 95%
lower bound of **0.47** on that certainty, so the leverage this measurement supports is **at
least 7.6x**, reaching 15.98x only if a chosen flip never misses. E2 is consistent with E1's
static prediction, with a numerator estimated on four configurations.

The static prediction also anticipates the dose–response curve without being fitted to
it. P(at least one catastrophic bit in *n* random flips) = 1 − (1 − 0.0626)ⁿ:

| random flips | predicted | observed |
|---|---|---|
| 1 | 6.26% | 0/5 |
| 10 | 47.61% | 3/5 |
| 100 | 99.84% | 5/5 |
| 1,000 | 100.00% | 5/5 |

And a different question, which the leverage does not answer: how many unchosen faults it
takes before destruction becomes likely, at a chosen level of likely.

| to destroy with probability | random flips needed |
|---|---|
| 50% | 11 |
| 90% | 36 |
| 99% | 72 |
| 99.9% | 107 |

**Two quantities live in those two tables, and merging them is how this page went wrong.**
The leverage of choosing is **15.98x per flip**, and it is the first table's business. The
second answers something else: how many unchosen faults are needed to reach a given
*probability* of destruction — **11 at even odds, 107 at 99.9%**. An earlier version
collapsed the two into "roughly a hundred cosmic rays", which took the last row of the
second table and reported it as the leverage. It is neither: the band is 11-107 and it
depends entirely on the confidence being asked for.

## Two traps found while measuring, both worth stating

The first run reported that chosen faults did no damage at all. It was right, and the
policy was wrong: it selected the largest-magnitude weights, which are exactly the
weights with |w| ≥ 2 — the condition for bit 14 to be **already set**. Flipping it
there divides by 2¹²⁸ rather than multiplying. In this model, all 1,000 largest weights
are in that class, so the attack was measuring nothing. The naive reading of E1 —
"bit 14 is zero everywhere, so hit the biggest weight" — is a trap.

Behind it sits a second: among the weights the flip does amplify, those in [1, 2)
overflow bfloat16 and become NaN. The policy now selects the largest weight the flip
amplifies while staying finite.

## Perplexity saturates; agreement does not

Every destroyed configuration reports the same perplexity, because the vocabulary size
is the ceiling. Top-1 agreement with the undamaged model was added for that reason, and
immediately earned it: the single partial case reads 653× on perplexity, which says
little, and 0.1364 on agreement, which says the model still makes one prediction in
seven correctly.

## Determinism holds within a session, not across them

Two sessions ran the identical configuration on different hardware:

```
session A : 16.106150421005136
session B : 16.105662448130897
difference: 3.030e-05 relative
```

BLAS kernels and summation order differ between machines. This does not affect any
conclusion — every comparison is against a baseline measured in the same session — but
it bounds what "deterministic" can be claimed to mean, and it is the reason the claim
in the README is scoped.

One detail is more than bookkeeping: **top-1 agreement was identical across the two
sessions in all 30 configurations, while 9 of 30 perplexities were not.** `argmax` is
invariant to perturbations that do not reorder the logits; a sum of thousands of
logarithms is not. The metric added to avoid saturation turns out to be the more robust
of the two.

That number is also the first honest floor for E5: a fault that moves perplexity by
less than 3 × 10⁻⁵ is indistinguishable from changing machines, which is an operational
definition of invisible to anyone monitoring quality. The one partial case moves it by
653×, six orders of magnitude above that floor. If a stealthy window exists, it lies
between them.
