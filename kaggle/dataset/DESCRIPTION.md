# bit-flip results

Raw results from the research project *bit-flip: from the cosmic ray to the attacker who
picks the bit*. The dataset contains **only** the CSVs produced by the experiments in the
repository, plus the manifests describing what was measured and under which configuration:
no model weights, no generated text, no attack artefacts.

- Code, method and technical notes:
  **https://github.com/rootedlab-code/bit-flip-llm-research**
- The notebooks that produced the runnable half:
  [oracle validation](https://www.kaggle.com/code/seb001010/bit-flip-e5-oracle-validation)
  and [E2 degradation](https://www.kaggle.com/code/seb001010/bit-flip-e2-degradation)

Author: `rootedlab-code`.

## The question

A cosmic ray hitting a DRAM cell and a Rowhammer attack produce the same physical event: a
bit changes state. Exactly one thing differs — who picks the address. **What is that
worth, in numbers?**

## What is in it

| File | Rows | Contents |
|---|---|---|
| `e1-bit-hierarchy-base.csv` | 16 | E1 — one row per bit position of a `bfloat16` weight, `Qwen/Qwen2.5-0.5B-Instruct` |
| `e1-bit-hierarchy-abliterated.csv` | 16 | E1 — the same profile on the abliterated positive control |
| `e1-summary.csv` | 2 | E1 — one row per model: catastrophic bit share and the exponent statistics behind it |
| `e2-degradation.csv` | 30 | E2 — perplexity, top-1 agreement and damage class for 30 configurations of random and chosen faults |
| `e3-gguf-bit-census.csv` | 4 | E3 — bits of the `q4_k_m` GGUF file grouped by function |
| `e3-gguf-scale-fragility.csv` | 32 | E3 — flip outcomes for the `fp16` block scales, one row per (block size, bit position) |
| `e3-comparison.csv` | 2 | E3 — `bf16` against `q4_k_m`: catastrophic share, blast radius, weights lost per random fault |
| `e5-oracle-validation.csv` | 6 | Oracle validation — verdict shares at six corners: three models against two probe sets |
| `e5-verdicts.csv` | 600 | Oracle validation — one row per answer: verdict, length and a truncated SHA-256 |
| `e5-run-manifest.json` | — | the generation configuration that produced the two files above |
| `models-manifest.json` | 3 entries | repository, revision, byte count and SHA-256 of every artefact measured |

**E4** (from field fault rates to time-before-a-critical-flip) and **E5 proper** (the
de-alignment fraction under increasing flips) have not been run. Nothing in this dataset
stands in for them. No file here contains estimates, interpolations, or figures taken from
other work.

## Where the numbers come from

E1 and E3 are produced by `experiments/e1_bit_hierarchy.py` and
`experiments/e3_gguf_surface.py`, and neither has any source of randomness: regenerating
the CSVs must return byte-identical files. E2 and the oracle validation are produced by
the two Kaggle notebooks linked above.

The method for the static half, in one sentence: the outcome of flipping a 16-bit
floating-point value depends **only on its 16-bit pattern**, not on which weight or scale
carries it; so the exact histogram of the 65,536 patterns present in a file summarises the
whole population without loss, and every reported fraction is a **count**, not a sample
estimate.

Coverage is self-verifying. For E1, the histogram total must equal the parameter count
declared by the safetensors header, and it does (494,032,768 weights in 290 tensors, all
BF16). For E3, the GGUF parser must close its arithmetic exactly on the file size —
5,947,744 bytes of header plus data equals 491,400,032 bytes — and it does.

## Data dictionary — `e1-bit-hierarchy-*.csv`

The `bfloat16` format has 1 sign bit (position 15), 8 exponent bits (7–14) and 7 mantissa
bits (0–6). Position 0 is the least significant bit.

| Column | Type | Definition |
|---|---|---|
| `bit` | integer 0–15 | position of the flipped bit |
| `field` | `sign` / `exponent` / `mantissa` | IEEE-754 field the position belongs to |
| `zero_bit_fraction` | float ∈ [0,1] | fraction of the population in which that bit is 0. Tells you which way the flip will go for a randomly chosen value: close to 1 means "almost always 0→1" |
| `median_delta` | float ≥ 0 | median of \|Δw\| = \|w′ − w\| over the real distribution, weighted by how often each pattern occurs |
| `p99_delta` | float ≥ 0 | 99th percentile of the same distribution |
| `max_finite_delta` | float ≥ 0 | largest \|Δw\| among outcomes that stay finite (infinite outcomes are excluded from this column only) |
| `amplified_fraction` | float ∈ [0,1] | fraction of the population for which \|w′/w\| ≥ 2 |
| `non_finite_fraction` | float ∈ [0,1] | fraction for which the flip produces a non-finite value (Inf or NaN) |
| `catastrophic_fraction` | float ∈ [0,1] | fraction for which \|w′/w\| ≥ 2¹⁶ **or** the outcome is non-finite |

## Data dictionary — `e1-summary.csv`

| Column | Definition |
|---|---|
| `model`, `artifact`, `revision` | which bytes were measured; the revision is the exact commit |
| `weights`, `total_bits` | population size, and 16 × that |
| `fraction_below_one` | share of weights with \|w\| < 1 — the reason the top exponent bit is almost always 0 |
| `median_exponent`, `exponent_bias` | the stored exponent field's median, and the format's bias |
| `catastrophic_bits`, `catastrophic_bit_fraction` | how many bits of the file are catastrophic when flipped, and their share |
| `one_bit_in` | the same share as "one bit in N", which is the form the figure is usually quoted in |

## Data dictionary — `e2-degradation.csv`

| Column | Definition |
|---|---|
| `policy` | `baseline` (no fault), `random` (addresses drawn uniformly), `targeted` (addresses chosen) |
| `flips` | number of bits flipped in that configuration |
| `seed` | RNG seed for `random`, `-1` where the configuration has no randomness |
| `perplexity` | on the WikiText-2 test split, so the absolute value is comparable with the literature |
| `ratio_to_baseline` | perplexity divided by the undamaged model's. **Not** used for the leverage figure: chosen faults overflow the arithmetic and produce NaN |
| `top1_agreement` | fraction of positions where the damaged model predicts the same token as the undamaged one. Does not saturate, and is the metric to trust |
| `damage_class` | the categorical outcome, defined for every configuration including the NaN ones |

Perplexity saturates at the vocabulary size — a destroyed model outputs a uniform
distribution and every degree of destruction then looks identical. That is why agreement
and a damage class are carried alongside it.

## Data dictionary — `e3-gguf-scale-fragility.csv`

Same columns as `e1-bit-hierarchy-*.csv`, applied to a different population: the `fp16`
block scales of the quantised file rather than the model weights. One extra leading column:

| Column | Type | Definition |
|---|---|---|
| `block_elements` | 32 or 256 | number of weights the scale governs. 32 for the legacy quantisations (Q5_0, Q8_0), 256 for the K-quants (Q4_K, Q6_K) |

The `fp16` format has 1 sign bit (position 15), 5 exponent bits (10–14) and 10 mantissa
bits (0–9), so the field boundaries in `field` differ from the `bfloat16` files above. The
top exponent bit is again position 14, but its multiplier is 2¹⁶ rather than 2¹²⁸.

## Data dictionary — `e3-gguf-bit-census.csv` and `e3-comparison.csv`

| Column | Definition |
|---|---|
| `role` | function of the bits: `quanti` (the quantised values themselves), `scala_fp16` (per-block scales in fp16), `scala_intera` (integer sub-block scales inside K-quant super-blocks), `float` (unquantised tensors: norms and biases) |
| `bits`, `share` | count of bits in that role, and its share of the file's data bits |
| `mean_blast_radius` | weights affected by one catastrophic flip, averaged over the file: 1 in `bf16`, more in a quantised format where a scale governs a block |
| `weights_lost_per_random_flip` | catastrophic share × blast radius — the quantity the two formats are compared on |

## Data dictionary — `e5-oracle-validation.csv` and `e5-verdicts.csv`

The classifier assigns one of four verdicts to a full generated answer — never to a prefix,
and never using a second language model: `refusal`, `compliance`, `degenerate`,
`indeterminate`. Its criteria are frozen in a versioned file with a digest **before** the
run that uses them.

`e5-oracle-validation.csv` holds one row per corner (`condition` × `kind`) with the share
of each verdict. `e5-verdicts.csv` holds one row per answer:

| Column | Definition |
|---|---|
| `condition` | `base` (the aligned model), `abliterated` (a build of the same model with refusal ablated — the positive control), `brick` (the base with one NaN-inducing flip) |
| `probe` | index within its set, e.g. `harmful-0000`. **The prompt text is not included** |
| `kind` | `harmful` (AdvBench) or `benign` (Alpaca) |
| `keywords` | how many keywords were derived from the prompt by the registered rule — never authored per item |
| `required` | how many of them an answer had to contain to count as on-topic |
| `verdict` | the class assigned |
| `answer_chars` | length of the generated answer in characters |
| `answer_sha256` | first 16 hex characters of the SHA-256 of the answer — enough to check whether two runs produced the same text, not enough to recover it |

**The indeterminate share is published rather than hidden**, and it is the honest statement
of how much this instrument does not know: 28% on the aligned model's benign answers. Any
de-alignment figure built on this oracle has that as its resolution floor.

## Conventions and thresholds

Declared because they are choices, not facts:

- **amplifying** = ratio ≥ 2;
- **catastrophic** = ratio ≥ 2¹⁶ or non-finite outcome;
- values whose original is non-finite are excluded from all statistics (none exist in the
  artefacts measured);
- where the original value is zero the ratio is defined as infinite, so the flip counts as
  catastrophic.

Changing these thresholds changes the last three columns of the E1/E3 files and nothing
else: the first six are independent of the convention.

## How to read it — the warning that matters

The E1 and E3 columns describe the **arithmetic fragility of an isolated value**: what
happens to the number. They say nothing about how much the **model** degrades. A weight
driven to a huge value inside a rarely used tensor may not change a comma of the output —
which is exactly why `e2-degradation.csv` exists.

This applies with particular force to the E3 comparison. The headline figure derived from
those files — that the quantised format loses 2.807 times more weights per random fault
than the `bfloat16` one — counts *corrupted weights*, not *lost quality*, and it treats a
2¹²⁸ multiplier and a 2¹⁶ multiplier as equivalent because both cross the catastrophic
threshold. Anyone using this dataset to estimate behavioural impact is extrapolating beyond
what the data measures.

## How you can help

Two gaps in this research are the kind a community can close and one author cannot.

**1. Does a run reproduce on a different accelerator?** Two sessions produced
byte-identical answers to all 600 probes — and both drew two Tesla T4s. Since the cause of
divergence is the order of floating-point reductions, the expectation is that a different
card does *not* reproduce, but expectation is not measurement. *Copy & Edit* the
[oracle-validation notebook](https://www.kaggle.com/code/seb001010/bit-flip-e5-oracle-validation),
run it, and post the digest block its last cell prints. **A mismatch is the more
interesting result** — please post it either way.

**2. Has the classifier ever agreed with a human?** No Cohen's κ has been measured. The
protocol for contributing labels without any generated text changing hands is in
[CONTRIBUTING.md](https://github.com/rootedlab-code/bit-flip-llm-research/blob/main/CONTRIBUTING.md).

## Reproduction

```sh
uv sync
uv run pytest
uv sync --extra download
uv run python -m bitflip.fetch          # ~2.4 GB, pinned revisions, frozen read-only
uv run python experiments/e1_bit_hierarchy.py
uv run python experiments/e3_gguf_surface.py
```

The versioned CSVs must be identical to the regenerated ones. The two notebooks import the
same `bitflip` package from the public repository, so a notebook run and a local run
execute the same code.

## What this dataset does not contain, and will not

- no model weights, original or modified;
- no text generated by a model under test, in any form;
- no reusable attack procedure: no optimal bit addresses for a specific model, no DRAM
  hammering parameters.

The project measures the *payload* — what the flipped bit does — not the *delivery vector*.

## Licence

MIT. The model weights remain subject to their upstream licences and are not
redistributed: they are downloaded at the revisions pinned in the repository.
