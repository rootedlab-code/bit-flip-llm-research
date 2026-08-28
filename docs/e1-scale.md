# E1 at scale — the fragility belongs to the format, not to the model

Subjects, each pinned by revision: `Qwen/Qwen2.5-0.5B-Instruct` @`7ae5576`,
`Qwen/Qwen3-4B-Instruct-2507` @`cdbee75f`, `Qwen/Qwen2.5-7B-Instruct` @`a09a3545`.
12.1 billion bf16 weights read in one pass, on a Kaggle CPU session.
Raw data: `results/e1-scale-summary.csv`, `results/e1-bit-hierarchy-{qwen3-4b,qwen25-7b}.csv`,
with the shard digests and the environment in `results/e1-scale-manifest.json`.
Notebook: `kaggle/e1/`.

## The question this answers

E1 measured 6.2595% of catastrophic bits on half a billion parameters. That figure is
what a fault rate gets crossed with, so the quantitative argument rests on it — and it
had been established on one model, of a size nobody deploys. Two different things could
have made it not travel: the **size** of the model, or the habits of one **family**.

The design separates them. Qwen2.5 at 0.5B and at 7.6B varies size at constant family;
Qwen3-4B varies family and generation at a size between them.

## The result

| model | stored parameters | bf16 | \|w\| < 1 | catastrophic bits | fraction | one bit in |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 494,032,768 | 100% | 99.9926% | 494,787,536 | **6.2595%** | 15.976 |
| Qwen3-4B | 4,022,468,096 | 100% | 99.9984% | 4,028,147,088 | **6.2588%** | 15.977 |
| Qwen2.5-7B | 7,615,616,512 | 100% | 99.9989% | 7,656,896,751 | **6.2839%** | 15.914 |

**It is the format.** Across a 15× range of size and two model generations the fraction
moves by 0.0251 percentage points — two orders of magnitude less than the quantity
itself. Every model sits just above 1/16 = 6.25%, which is the floor the geometry
imposes: of sixteen bits exactly one is the top exponent bit, and for weights below 1 it
is the only one whose flip is catastrophic. The excess over that floor is +0.0095,
+0.0088 and +0.0339 points.

That floor is not a construct. A population measured elsewhere in this project lands
**exactly** on it: the fp16 block scales of the `q4_k_m` file in E3 have a catastrophic
fraction of 0.0625 at 32-element blocks and at 256-element blocks alike, because their
top exponent bit is zero in 100.00% of cases — exactly, not to two decimals — and nothing
below it contributes anything. The bf16 weights measured here miss that floor in both
directions at once, and the two terms are separable: the weights that already carry a one
in bit 14 subtract from it, and the near-zero weights whose bits 11-13 are zero add to it.
The second term is the larger, which is why all three models sit above 6.25% rather than
below.

All three models are **100% BF16**, so the denominator is the whole model in each case
and the fraction needs no qualification.

## Where the 7B's extra hundredth comes from

The largest model is not the least fragile, which is worth explaining rather than
reporting. The catastrophic fraction is a sum over four bits:

| | bit 11 | bit 12 | bit 13 | bit 14 | sum / 16 |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.1508% | 0.0020% | 0.0020% | 99.9980% | 6.2595% |
| Qwen3-4B | 0.1407% | 0.0005% | 0.0005% | 99.9995% | 6.2588% |
| Qwen2.5-7B | **0.3201%** | **0.1111%** | **0.1111%** | 99.9998% | **6.2839%** |

Bits 11-13 are catastrophic only where they are **zero**, and they are zero only in
weights whose exponent is tiny — the ones sitting very close to nothing. The 7B has more
of those, and they open a second, small catastrophic surface underneath the first.

Two effects therefore run in opposite directions as models grow, and both are visible in
the table above:

- Weights get smaller on average — 99.9926% → 99.9989% below 1 — so bit 14 is zero in
  more of them, and the primary surface gets **more** uniform.
- More weights crowd toward zero, so bits 11-13 are zero in more of them, and the
  secondary surface **grows**.

The second effect is the larger one here, which is why the 7B ends up slightly above the
0.5B. It is not monotone in size: the 4B sits marginally *below* the 0.5B. Nothing in
this should be read as a trend with three points — what the three points establish is the
**bound**, that the fraction does not leave the third decimal place.

## The universal bit gets more universal, and never becomes universal

| | bit 14 is zero in | flipping it is catastrophic in |
|---|---|---|
| Qwen2.5-0.5B | 99.9980% | 99.9980% |
| Qwen3-4B | 99.9995% | 99.9995% |
| Qwen2.5-7B | 99.9998% | 99.9998% |

The attack claim — that a fault does not need to know *which* weight it lands on, only
that bit 14 is a zero to flip upward — holds more tightly on the larger models, not less.
It still never reaches 100%: see `docs/e1-bit-hierarchy.md`, where the weights that
already carry a one there are counted rather than rounded away.

## The anchor

The 0.5B was measured again in this run rather than carried over, through the same reader
as the two models nobody had read, so that the old row would have to come back if the new
ones were to be believed. It did, and the notebook printed the comparison itself:

```
✓ weights: published 494032768, measured 494032768
✓ catastrophic_bits: published 494787536, measured 494787536
✓ catastrophic_bit_fraction: published 0.06259548556908678, measured 0.06259548556908678
ANCHOR REPRODUCED
```

Its per-bit table came back **byte-identical** to the committed one, SHA-256
`94edde4a4b740f5574bf117951624fdaf4146b07bde21a1511798d68f170edb5` — on a different
machine, under Python 3.12.13 against 3.13.5 and numpy 2.0.2 against 2.5.2. No copy of it
is shipped: two identical files in a repository are indistinguishable from one file
copied, so the digest is the evidence and the duplicate would not have been.

That reproduction is of a different kind from E5's. Nothing here is generated, so there
is no decoding, no batching and no accelerator arithmetic to depend on — which is exactly
why this half of the programme reproduces where the other half needs a manifest to say
what it held fixed.

## Notes on the artifacts

**Stored parameters, not advertised ones.** `Qwen3-4B-Instruct-2507` ties its output
projection to its embedding, so that tensor exists once on disk and the model stores
4.02B parameters. That is the right denominator: a fault strikes what is in memory.

**The shard index is not a checksum.** `model.safetensors.index.json` carries a
`total_size`, and on the 4B it declares 655,360 bytes more than its three shards hold
(the 7B's closes exactly). It is written by whichever library saved the model, not by the
format. What is checked exactly is the format's own arithmetic, per shard, and the join:
the tensors found across the shards must be exactly the ones the index names — 398 of 398
and 339 of 339 here. The gap is recorded in the manifest rather than allowed to stop a
run.

**77 weights of the 7.6B are exactly zero, and the criterion mishandles them.** The
catastrophic test is `non-finite OR |after/before| >= 2**16`, and a weight of exactly zero
has no ratio — the code substitutes infinity, so those weights count as catastrophic at
**every** bit position. That is defensible at bit 14, where the flip turns 0 into 2.0. It
is wrong at bit 0, where the flip turns 0 into 9.18e-41: an infinite ratio, and a change
of no consequence whatever.

The error is bounded and was measured rather than estimated. Those 77 weights of
7,615,616,512 contribute 1.011e-08 to the fraction — **0.00000101 percentage points**
against an excess over the floor of 0.0339, about one part in thirty-three thousand. The
attribution to bits 11-13 above is unaffected, and the 0.5B and the 4B contain no exactly
zero weights at all, so their figures cannot carry this at all.

It is left in rather than fixed: the criterion produces every published figure in E1 and
E3, and changing it for an effect in the eighth significant digit would be a worse trade
than recording it. **What would change that judgement is a sparse model.** Pruning
produces exact zeros deliberately and in quantity, and on such a model this false positive
stops being negligible and starts being the measurement. Anyone extending this to a pruned
or sparse checkpoint has to fix the criterion first.

## Boundaries

**Three Qwen models are still one vendor.** The size axis is now covered and the
generation axis is covered once, but a result that holds across Qwen2.5 and Qwen3 has not
been shown to hold across Llama, Mistral or Gemma. The mechanism argues that it should —
the fraction is a consequence of bf16's geometry and of weights being small, neither of
which is a Qwen property — but that argument is not a measurement, and the cheapest way
to close the gap is one non-Qwen model of any size.

**This is arithmetic fragility, not damage.** As in E1, it says how many bits are
catastrophic *when flipped*, not how much the model degrades when one is. That is E2's
question, and E2 is measured on 0.5B.

**Quantised formats are not covered here.** E3 measured that surface separately, and its
conclusion is the opposite in sign: quantisation concentrates fragility rather than
diluting it.
