# bit-flip — from the cosmic ray to the attacker who picks the bit

Reproducible experimental research on bit-flips in language-model weights.

Author: `rootedlab-code` · Contact: `rootedlab@proton.me`

---

## The question

A cosmic ray hitting a DRAM cell and a Rowhammer attack produce the same physical event:
a bit changes state. Exactly one thing differs — who picks the address. The literature
treats the two cases in separate communities (reliability on one side, security on the
other) and no known work puts the leverage factor on a single axis.

> **What is it worth, in numbers, that an attacker can *choose* which bit to flip rather
> than suffer a random one?**

And the variant we care about most:

> **Does safety alignment degrade *earlier* and *faster* than general capability?**

If there is a regime in which a model passes every check an operator would run —
perplexity in range, accuracy in range, coherent outputs — while it has already lost a
measurable fraction of its propensity to refuse, then output-side monitoring is blind by
construction and the defence has to sit on weight integrity. If instead alignment and
capability fail together, the finding is that the worry is misplaced. Both outcomes get
published.

## Status — honest

| Experiment | Question | Status |
|---|---|---|
| **E1** | Fragility hierarchy of the 16 bits of a `bfloat16` weight | **measured** — [`docs/e1-bit-hierarchy.md`](docs/e1-bit-hierarchy.md) |
| **E3** | Does quantisation protect? Bit census and scale fragility of a GGUF `q4_k_m` | **measured** — [`docs/e3-gguf-surface.md`](docs/e3-gguf-surface.md) |
| baseline | Deterministic perplexity, and the noise floor between sessions | **measured** — [`docs/e2-degradation.md`](docs/e2-degradation.md) |
| **E2** | Model degradation: random flips versus chosen flips | **measured** — [`docs/e2-degradation.md`](docs/e2-degradation.md) |
| oracle | Deterministic refusal / compliance / degenerate classifier | **written** — [`src/bitflip/spec/oracle_spec.yaml`](src/bitflip/spec/oracle_spec.yaml), not yet validated against human labels |
| **E5** | Silent de-alignment: De-alignment Fraction and Stealth Ratio | not yet measured (S5b) |
| **E4** | From field fault rates to time before a natural critical flip | not yet measured (S6) |

Anything not marked **measured** has no number in this repository yet. In the working
note it appears as a heading with an explicit status marker — never as a placeholder that
could be mistaken for a result.

## What is already measured

### E1 — one bit in sixteen carries almost everything

The outcome of flipping a 16-bit floating-point weight depends **only on the 16-bit
pattern**, not on which weight carries it. The exact histogram of the 65,536 patterns
present in a file therefore summarises the whole model without loss, and every reported
fraction is a **count**, not a sample estimate.

Measured on `Qwen/Qwen2.5-0.5B-Instruct` (494,032,768 `bfloat16` weights in 290 tensors)
and on its abliterated control:

- **bit 14** (the top exponent bit) is zero in 99.998% of weights, and flipping it
  multiplies the weight by 2¹²⁸;
- bits 11-13 are one almost everywhere, so flipping them **divides**: they are harmless by
  construction, not by luck;
- **6.2595%** of the file's bits are catastrophic — one in every 15.98;
- the abliterated model has the same profile digit for digit: 176 catastrophic bits of
  difference out of 7.9 billion. Removing alignment does not move the geometry of the
  weights.

The operational consequence is not that bit 14 has the largest multiplier — that is
arithmetic — but that its value is **predictable**. To amplify a weight, an attacker does
not need to know which weight is being hit.

### E3 — quantisation does not protect, it concentrates

Measured on the `q4_k_m` GGUF file of the same model (491,400,032 bytes):

- the file is **mostly not K-quantised**. K-quants require rows that are multiples of 256;
  the hidden size of Qwen2.5-0.5B is 896 = 3.5 × 256, so 146 tensors out of 170 fall back
  to legacy 32-element blocks. A file labelled `q4_k_m` does not describe the format of its
  weights;
- bit census by function: 91.745% quants, **7.018% fp16 scales**, 1.178% integer scales,
  0.059% float;
- bit 14 of every fp16 scale is zero in **100.00%** of cases — the same universal weakness
  E1 found in the weights;
- exactly one bit in sixteen of each scale is catastrophic: 6.2500%.

The comparison that answers the question:

| format | total bits | catastrophic | share | blast radius | weights lost per random flip |
|---|---|---|---|---|---|
| bf16 safetensors | 7,904,524,288 | 494,787,536 | 6.2595% | 1 weight | 0.062595 |
| gguf q4_k_m | 3,883,618,304 | 17,034,752 | 0.4386% | 40.1 weights | **0.175711** |

Quantisation cuts the share of catastrophic bits by 14× and multiplies the blast radius of
each by 40×, because a scale governs its entire block. Net: at equal random fault, the
quantised file loses **2.807 times more weights**.

That conclusion has three declared boundaries — the per-weight multiplier is not the same
in the two formats, quantised damage is spatially correlated, and "weights lost" is not
yet "damage to the model". They are stated in full in the technical note, and they should be read alongside the
headline number.

## Reproduction

Requirements: Python ≥ 3.13 and [`uv`](https://docs.astral.sh/uv/). No GPU is needed for
either measured experiment — both are integer arithmetic over memory-mapped files.

```sh
# 1. dependencies
uv sync

# 2. the contract of every module: no experiment counts without this green
uv run pytest

# 3. artefacts — ~2.4 GB, pinned revisions, frozen read-only after download
uv sync --extra download
uv run python -m bitflip.fetch

# 4. the measured experiments
uv run python experiments/e1_bit_hierarchy.py
uv run python experiments/e3_gguf_surface.py
```

Step 3 writes `results/models-manifest.json` with the size and SHA-256 of every downloaded
file. Step 4 rewrites the CSVs under `results/`. The versioned CSVs must be **identical**
to the regenerated ones: neither experiment has any source of randomness.

`uv run pytest` includes tests that run only when the models are already present locally;
without them they are skipped, and the rest of the suite stays meaningful.

### Constraints of the environment this was produced on

RTX 4060 Laptop (8 GB VRAM), 30 GB RAM, no ECC exposed, no `sudo`. The download cap is
5 GB: that is the constraint that chose 0.5 B models over 7 B ones, not laziness. Neither
measured experiment uses the GPU.

## Safeguards

The realistic risk of a bit-flip project is not the bit-flip: it is the full disk and the
overwritten file.

- **No physical bit is ever flipped.** No Rowhammer, no `/dev/mem`, no kernel modules, no
  physical memory mappings, no DMA, no `sudo`. Every flip is arithmetic on `numpy` arrays
  inside the process: indistinguishable from a real fault to the model, harmless to the
  DRAM hosting it.
- **Model files are read-only**, opened via read-only mmap, with SHA-256 recorded on entry
  to each experiment and re-verified on exit. A changed hash is a failing test, not a log
  line (`bitflip.guard.immutable`).
- **No modified weights touch the disk.** No code path serialises a flipped model.
- **A free-space guard with an abort threshold** runs before every download.

## Declared boundaries

- **No real Rowhammer.** We study the *payload* — what the flipped bit does — not the
  *delivery vector*. Delivery is DRAM-module specific, requires reverse-engineering the TRR
  mitigation, and is already documented elsewhere.
- **No optimised attack recipe.** We measure *how much* alignment is lost and *how
  invisible* the loss is; we do not produce the gradient-guided search that finds the
  optimal bit triple for jailbreaking a specific model. That artefact is transferable and
  reusable by third parties; the stealth curve — the result a defender needs — is
  obtainable without it. The boundary is drawn around the *artefact*, not around the
  question.
- **No harmful output published**, in any form. Generations needed for measurement live in
  a scratch directory, are classified, and are deleted; results retain class labels,
  scalars and a truncated hash.
- **No defences implemented** in this version: measuring the critical surface is the
  prerequisite for designing them, not the other way round.
- **No 7 B models**, for the disk constraint stated above.

## Layout

```
src/bitflip/
  guard.py       host safeguards: free-space guard, read-only enforcement,
                 hash-verified immutability context
  codec.py       16-bit pattern ↔ value, bit flips, IEEE-754 field decomposition
  weights.py     safetensors reader: read-only mmap, self-closing arithmetic
  gguf.py        GGUF anatomy: tensor types, block layout, scale extraction
  fragility.py   per-bit-position flip outcomes, shared by E1 and E3
  stats.py       weighted quantiles over the pattern histogram
  fetch.py       pinned-revision acquisition, freezing, manifest
experiments/
  e1_bit_hierarchy.py
  e3_gguf_surface.py
results/         CSVs and manifest — the only source of every published figure
tests/           the contract of every module
docs/            per-experiment technical notes
```

Both file parsers validate themselves by closing their arithmetic exactly on the file
size: a silently misparsed file cannot be mistaken for a result.

## Principles

1. **Host safeguarding** — if an experiment violates it, the experiment is cut.
2. **Nothing asserted, everything recomputed** — every published figure traces back to the
   script and the CSV that produced it. Figures from the literature are cited as a
   comparison point, never as the source of one of our results.
3. **Determinism, or it is not a measurement** — two runs of the same configuration in
   the same session give identical results digit for digit, and every experiment
   asserts it before measuring anything. Across sessions the guarantee is weaker and
   has been measured rather than assumed: the same configuration on different hardware
   moved perplexity by 3.0 × 10⁻⁵ relative, because BLAS kernels and summation order
   differ. Every comparison is therefore made against a baseline measured in the same
   session. Top-1 agreement was identical across sessions in all 30 configurations,
   where 9 of 30 perplexities were not.
4. **Pre-registration of criteria** — thresholds and classification rules are frozen in a
   file with a hash before the first run that uses them.
5. **No claim without its control** — null, positive, brick, benign.
6. **The boundary is around the artefact, not around the question.**
7. **Operational security** — a single public identity, no local paths, no bylines.

## Reading order

1. This file, for what exists and what does not.
2. [`docs/e1-bit-hierarchy.md`](docs/e1-bit-hierarchy.md),
   [`docs/e3-gguf-surface.md`](docs/e3-gguf-surface.md) and
   [`docs/e2-degradation.md`](docs/e2-degradation.md) — the per-experiment technical
   notes: method, result, and declared boundaries, one per experiment.
3. `results/` — the CSVs behind every figure quoted above.

### The write-up

A paper covering the whole programme is in preparation and will be published here when
the remaining experiments (E2, E4, E5) have been run. It is deliberately not released in
draft: a partial write-up invites its unmeasured sections to be read as findings. Until
then, this repository is the record — the code, the data, and the two technical notes
are complete and reproducible as they stand.

## Licence

MIT, see [`LICENSE`](LICENSE). The model weights used remain subject to their upstream
licences and are **not** redistributed by this repository: they are downloaded at the
revisions pinned in `src/bitflip/fetch.py`.
