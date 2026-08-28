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

The counterpart is just as sharp, and it is not that the other exponent bits are safe:
bits 11-13 are 1 almost everywhere, so flipping them **divides** rather than multiplies.
Every weight has one bit that amplifies it above all others, and it is always the same
one. What flipping the other three does is measured in the next section, and it is not
nothing — an earlier version of this page called them "harmless by construction", which
was a claim about the criterion rather than about the weights.

## What the catastrophic fraction does not count

6.2595% answers one question: how much of the bit space, flipped, makes a weight
explode. Anything that does not explode a weight is left as a residue with no name,
and this page used to read that residue as safety. Partitioning the whole space
instead of thresholding it shows what was in there.

Measured on the base model, exact counts, `results/e1-perturbation-spectrum.csv`:

| outcome | share of the bit space | bits |
|---|---|---|
| negligible — mantissa perturbation | 43.75000% | 0-6 |
| **collapse — divides by ≥ 2¹⁶** | **18.74045%** | 11-14 |
| moderate amplification, ×2 … ×2⁸ | 13.31981% | 7-10 |
| moderate attenuation | 11.68019% | 7-10 |
| sign inversion | 6.25000% | 15 |
| catastrophic amplification | 6.25921% | 11-14 |
| non-finite | 0.00034% | 14 |
| **total** | **100.00000%** | |

The classes are assigned by **outcome, not by position** — which bit does what is a
conclusion of the table rather than an input to it, and the `bits` column is where that
correspondence shows up. The last two rows are the published 6.2595% split into its two
halves; `CATASTROPHIC_RATIO` is untouched and every figure above this section is
reproduced unchanged.

**The channel that had no name is three times the size of the one that did.** Flipping
bit 11 divides a weight by 2¹⁶, bit 12 by 2³², bit 13 by 2⁶⁴. On a typical weight of
0.02 the first leaves 3·10⁻⁷: the weight is not perturbed, it is gone. That is 18.74045%
of the bit space against 6.25955%, a ratio of **2.994**. And it is not a property of one
model — across the three measured in `docs/e1-scale.md` it holds at 18.74045%, 18.74118%
and 18.71612%, from 0.5B to 7.6B.

**What this does not license.** In the **random** arm it is noise. Removing weights at
random, at a rate of one in a hundred million, is unstructured pruning at a sparsity the
literature finds does nothing, and offering it as a mechanism would be offering noise
with a large percentage attached. The channel is only interesting in the **chosen** arm,
where it is pruning at the granularity of one bit, of a weight the attacker selected,
without access to the file.

**Why it is worth naming even so.** A catastrophic flip produces NaN, and NaN is the
loudest thing a model can do — the output collapses and any operator notices within one
request. Silent de-alignment, which is the thing E5 exists to detect, therefore cannot
live inside the 6.2595% **by construction**. If it exists it is in one of the other
classes, and collapse is the only one both large enough and destructive enough to be a
candidate. That is not evidence that it happens. It is an argument about where to look,
and the first thing this table is good for.

**A third channel, and the control it implies.** Bit 15 inverts the sign: 6.25000% of
the space, median |Δw| 0.0227, zero catastrophic. A safety-relevant weight with its sign
inverted is neither attenuated nor removed — it is *anti*-aligned. If collapse is the
hypothesis, sign inversion is a condition that can disconfirm it rather than agree with
it, which is the same shape as E5's benign controls.

**The criterion is one-sided, and that is now stated rather than omitted.**
`catastrophic = non-finite OR ratio ≥ 2¹⁶` asks only whether a magnitude exploded. It
has nowhere to put "the weight is gone", which is why the collapse channel went
unnamed. The same one-sidedness has a second consequence, measured in
`docs/e1-scale.md`: a weight of exactly zero has no ratio, the code substitutes
infinity, and 77 such weights in the 7.6B are counted catastrophic at all sixteen
positions. Both follow from the same predicate. Changing it would move figures already
published in several places, so it has not been changed — the spectrum is additive, and
what it adds is a name for what the predicate could not see.

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
