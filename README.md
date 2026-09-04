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
| **E1 at scale** | Does that hierarchy depend on the size of the model? Three subjects over a 15× range | **measured** — [`docs/e1-scale.md`](docs/e1-scale.md) |
| **E3** | Does quantisation protect? Bit census and scale fragility of a GGUF `q4_k_m` | **measured** — [`docs/e3-gguf-surface.md`](docs/e3-gguf-surface.md) |
| baseline | Deterministic perplexity, and the noise floor between sessions | **measured** — [`docs/e2-degradation.md`](docs/e2-degradation.md) |
| **E2** | Model degradation: random flips versus chosen flips | **measured** — [`docs/e2-degradation.md`](docs/e2-degradation.md) |
| oracle | Deterministic refusal / compliance / degenerate classifier | **measured — spec v3 validated at 6 of 6 corners** — [`docs/e5-oracle-validation.md`](docs/e5-oracle-validation.md); still not checked against human labels, and a quarter of a working model's benign answers remain undecided |
| **E5** | Silent de-alignment: De-alignment Fraction and Stealth Ratio | **measured — DF = 0 at every dose on both arms** — [`docs/e5b-silent-dealignment.md`](docs/e5b-silent-dealignment.md); the Stealth Ratio is withheld at every dose, and the note says why |
| **E4** | From field fault rates to time before a natural critical flip | not yet measured (S6) |
| **E6** | Agentic severity: does a flip make a tool-using agent take destructive actions? | not yet measured — every tool is an instrumented stub that records the call and does nothing |

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
- bits 11-13 are one almost everywhere, so flipping them **divides** — by 2¹⁶, 2³² and
  2⁶⁴ — which removes a weight rather than exploding it. That channel is **18.74%** of
  the bit space against the catastrophic 6.26%, three times larger, and it went unnamed
  because the criterion only asks whether a magnitude exploded;
- **6.2595%** of the file's bits are catastrophic — one in every 15.98;
- the abliterated model has the same profile digit for digit: 176 catastrophic bits of
  difference out of 7.9 billion. Removing alignment does not move the geometry of the
  weights.

The operational consequence is not that bit 14 has the largest multiplier — that is
arithmetic — but that its value is **predictable**. To amplify a weight, an attacker does
not need to know which weight is being hit.

### E1 at scale — the geometry belongs to the format, not to the model

The same exact enumeration on `Qwen3-4B-Instruct-2507` (4,022,468,096 weights) and
`Qwen2.5-7B-Instruct` (7,615,616,512 weights) returns 6.2588% and 6.2839% catastrophic
bits, against 6.2595% at 0.5 B: a spread of **0.025 percentage points across a 15× range**
of parameter count. The top exponent bit's zero fraction rises with size — 99.99799%,
99.99950%, 99.99979% — towards a limit it approaches and does not reach.

Method, coverage checks and per-shard digests: [`docs/e1-scale.md`](docs/e1-scale.md).

### E3 — quantisation does not protect, it concentrates

Measured on the `q4_k_m` GGUF file of the same model (491,400,032 bytes):

- the file is **mostly not K-quantised**. K-quants require rows that are multiples of 256;
  the hidden size of Qwen2.5-0.5B is 896 = 3.5 × 256, so 146 tensors out of 170 fall back
  to legacy 32-element blocks. A file labelled `q4_k_m` does not describe the format of its
  weights;
- bit census by function: 91.745% quants, **7.018% fp16 scales**, 1.178% integer scales,
  0.059% float;
- bit 14 of every fp16 scale is zero in **100.00%** of cases — exactly universal here,
  where in the `bfloat16` weights the same bit is zero in 99.9980% and rises towards, but
  never reaches, 100% as models grow;
- exactly one bit in sixteen of each scale is catastrophic: 6.2500%.

That last figure is exact rather than rounded — `0.0625 == 1/16` at both block sizes — and
it is the floor geometry imposes on any 16-bit format. The `fp16` scales **reach** it,
because there the top exponent bit is zero in exactly 100.00% of cases and no other
position contributes anything. The `bfloat16` weights miss it in two opposing directions:
the weights already carrying a one in bit 14 subtract from the floor (0.000126% at 0.5 B),
and the near-zero weights, whose bits 11–13 are zero, add to it (0.009674%). The second
term is the larger, which is why all three models measured sit above 6.25% and none below.

The comparison that answers the question:

| format | total bits | catastrophic | share | blast radius | weights lost per random flip |
|---|---|---|---|---|---|
| bf16 safetensors | 7,904,524,288 | 494,787,536 | 6.2595% | 1 weight | 0.062595 |
| gguf q4_k_m | 3,883,618,304 | 17,034,752 | 0.4386% | 40.1 weights | **0.175711** |

Quantisation cuts the share of catastrophic bits by 14× and multiplies the blast radius of
each by 40×, because a scale governs its entire block. What that nets out to depends on
what is held equal between two files of different size, so all three answers are published
rather than the largest alone:

| held equal | gguf / bf16 |
|---|---|
| the flip — a fault landed *in this file* | **2.807×** |
| the physical exposure — same DRAM, same hours, same fault rate per bit | **1.379×** |
| the model — the share of its own parameters each format loses | **1.081×** |

Quantisation does not protect, it concentrates: every ratio is above 1. The **magnitude**
depends on the question, and 1.08–1.38 at equal exposure is the defensible range. That
second row is also the one E4 has to cross with a field fault rate, which is quoted per
bit per hour; the per-flip figure would overstate that bridge twofold.

The third row does not reconstruct without its denominator, so here it is: the two files do
not hold the same number of weights. The safetensors declares 494,032,768 and the GGUF
630,167,424, and the difference — 136,134,656 = 151,936 × 896 — is the embedding that GGUF
**unties** and stores a second time as `output.weight`. Normalised on the same count, the
third row *is* the second, 1.379×; it reads 1.081× only because each format is divided by
the weights it actually stores.

That conclusion has three declared boundaries — the per-weight multiplier is not the same
in the two formats, quantised damage is spatially correlated, and "weights lost" is not
yet "damage to the model". They are stated in full in the technical note, and they should be read alongside the
headline number.

### E5 — no silent window at the doses a fault can deliver

Measured on `Qwen3-4B-Instruct-2507`: 300 harmful probes at every dose, both controls in
the same run, criteria frozen and attested before the first token
([`docs/e5b-silent-dealignment.md`](docs/e5b-silent-dealignment.md)).

| arm | doses | De-alignment Fraction | perplexity, relative | collapsed |
|---|---|---|---|---|
| chosen — bit 13 of the largest weights, downward | 1, 2, 5, 10 | **0.000** at every dose | −1.7 × 10⁻⁴ (an improvement) rising to +4.1 × 10⁻⁴ | none |
| random — three seeds per dose | 1, 2, 5, 10 | **0.000** at every non-collapsed dose | +10⁻⁴ to +6 × 10⁻⁴ | 1 seed of 3 at dose 10 |

The flipped model complies with the same 4 of 300 probes as the intact one, against 197
for the abliterated anchor, and the paired test finds no discordant pairs. Perplexity, by
contrast, moves at every dose above the 3 × 10⁻⁵ floor E2 measured. The pre-registered
null was alignment and capability failing together; what happened is a third case —
**capability moved first, and alignment did not move at all.** The Stealth Ratio is
withheld at every dose, and each row of `results/e5b-*-scores.csv` says why: a fraction
inside its own interval divided by a small perplexity change is not a ratio.

Two things this does not cover, both stated in the note. The optimised search that
[`SCOPE.md` §5.1](SCOPE.md) permits has not been run, so the chosen arm bounds an
*unoptimised* attacker only. And the top-1 agreement of every condition, on both arms,
was lost with a kernel that died after its last condition — the notebook now checkpoints
it.

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

## Run it on Kaggle

The experiments that need a model in memory run on Kaggle, where the accelerator is free
and the notebook installs this package straight from this repository — so a notebook run
and a local run execute the same code, at whichever commit is current.

| Notebook | What it does | Runtime |
|---|---|---|
| [oracle validation](https://www.kaggle.com/code/seb001010/bit-flip-e5-oracle-validation) | validates the refusal / compliance / degenerate classifier at six corners, three models against two probe sets | ~41 min, 2×T4 |
| [E2 degradation](https://www.kaggle.com/code/seb001010/bit-flip-e2-degradation) | perplexity and top-1 agreement under random and chosen faults, and the leverage figure between them | GPU |
| [E5b, chosen arm](https://www.kaggle.com/code/seb001010/bit-flip-e5b-silent-dealignment-chosen-arm) | De-alignment Fraction and Stealth Ratio under the registered chosen policy, four doses, both controls in the run | ~2.8 h of generation, ~4.5 h wall, 2×T4 |
| [E5b, random arm](https://www.kaggle.com/code/seb001010/bit-flip-e5b-silent-dealignment-random-arm) | the same under random flips, three seeds per dose | ~6 h of generation, ~8 h wall, 2×T4 |

Every CSV behind the figures quoted above is also published as a dataset:
[**bit-flip results**](https://www.kaggle.com/datasets/seb001010/bit-flip-results). It
carries no model weights and no generated text.

## How you can help

Two gaps here need hardware diversity or human judgement rather than more code, which
makes them things a reader can close and an author cannot.

1. **Run the oracle notebook on your accelerator and post the digest it prints.** Both
   sessions behind the published figures drew two T4s, so *reproduction on different
   hardware is not established* — and since the cause of divergence is the order of
   floating-point reductions, the expectation is that it fails. A mismatch is the more
   interesting result and it is the one this project expects, so it is worth posting
   either way.
2. **Label a sample of answers by hand.** The classifier has never been compared with
   human judgement — no Cohen's κ — and 28% of an aligned model's answers to harmless
   questions are still filed as undecided. The benign corner involves no harmful content
   at all, and sixty labels are enough to be useful.

The protocol for both, including what must never be sent back, is in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Safeguards

The realistic risk of a bit-flip project is not the bit-flip: it is the full disk and the
overwritten file. **No physical bit is ever flipped** — every flip is arithmetic on `numpy`
arrays inside the process; model files are opened read-only and hash-verified on entry and
exit (`bitflip.guard.immutable`); no code path serialises a flipped model; a free-space
guard with an abort threshold runs before every download.

The full list is [`SCOPE.md` §4](SCOPE.md), which is where it is maintained.

## Scope and declared boundaries

The fault model, the adversary model, the host safeguards and the publication boundary are
stated once, normatively, in [`SCOPE.md`](SCOPE.md). What follows is **reproduced from its
§5**; that file, not this page, is authoritative, and no other document here states the
boundary in its own words.

> The boundary is drawn around the artefact, not around the question. What is measured is
> the worst case; what is published is the result, not the means of reproducing it against a
> third party.
>
> - **Optimised search**: it is run, against locally held weights of publicly available
>   models only, on an inert scalar objective — a refusal margin, a De-alignment Fraction,
>   never generated text. Published: the curve and the counts. Not published, and not
>   retained beyond the run: the addresses — tensor, offset, bit index.
> - **Model output**: no harmful output is published in any form. Generations live in a
>   scratch directory, are classified, and are deleted; published tables retain class
>   labels, scalars, lengths and a truncated SHA-256.
> - **Weights**: none redistributed, original or de-aligned, not even as a diff.
> - **Delivery vector**: no Rowhammer executed, on this machine or any other. This work
>   studies the payload, not the delivery, which is DRAM-module specific and documented at
>   length elsewhere.
> - **Defences**: none implemented in this version. Measuring the critical surface is the
>   prerequisite for designing one, not the other way round.

### What this project has withdrawn

Self-correction belongs on the front page, not in a file nobody opens. The complete list is
dated in [`ERRATA.md`](ERRATA.md), each entry carrying the venues its correction has and has
**not** yet reached. The three that change how a number on this page should be read:

- **The boundary above replaced its opposite** (2026-08-29). Until that date this project
  said it would never run an optimised search. That drew the line around the question
  instead of the artefact, and it would have left every published figure a lower bound of
  unknown looseness — a reassuring number that is false. What is withheld is the addresses,
  and withholding them is compatible with measuring the worst case.
- **One ratio was standing in for three** (2026-08-28). The 2.807× above was published
  alone, as though it answered every version of "does quantisation protect?". It answers
  one: a fault landing *in this file*. E4 bridges to field rates quoted per bit per hour and
  needs the 1.379×; the per-flip figure would overstate that bridge twofold, in the
  alarmist direction.
- **"Bits 11-13 are harmless by construction" was false** (2026-08-28) — and false about the
  *instrument*, not the subject. The criterion asked only whether a magnitude exploded, so a
  channel that annihilates a weight scored as harmless. It is 18.74% of the bit space.

## Layout

```
SCOPE.md         fault model, adversary model, safeguards, publication boundary —
                 normative, and the only place any of them is stated
ERRATA.md        claims published and then withdrawn, dated, with the venues each
                 correction has and has not yet reached
src/bitflip/
  guard.py       host safeguards: free-space guard, read-only enforcement,
                 hash-verified immutability context
  codec.py       16-bit pattern ↔ value, bit flips, IEEE-754 field decomposition
  weights.py     safetensors reader: read-only mmap, self-closing arithmetic
  gguf.py        GGUF anatomy: tensor types, block layout, scale extraction
  fragility.py   per-bit-position flip outcomes, shared by E1 and E3
  stats.py       weighted quantiles over the pattern histogram
  exposure.py    the three normalisations between two storage formats
  fetch.py       pinned-revision acquisition, freezing, manifest
  inject.py      in-memory fault injection, always reversible
  metrics.py     deterministic model-quality measurements
  damage.py      classifying what a fault did to a model
  probes.py      probe sets, and what counts as answering them
  oracle.py      deterministic refusal / compliance / degenerate verdict
  alignment.py   oracle verdicts → the two numbers E5 publishes
  compare.py     two classification runs, over the answers they share
  spec/          pre-registered thresholds and rules, hashed before first use
experiments/
  e1_bit_hierarchy.py
  e3_gguf_surface.py
  e5_compare_specs.py
kaggle/          notebook sources in percent format, the metadata that publishes
                 them, and the description of the results dataset
results/         CSVs and manifest — the only source of every published figure
results/history/ superseded runs, kept as the evidence behind every entry in ERRATA.md
tests/           the contract of every module
docs/            per-experiment technical notes
```

Both file parsers validate themselves by closing their arithmetic exactly on the file
size: a silently misparsed file cannot be mistaken for a result.

## Model scale

E1 and E3 are static analyses of bit patterns, and their conclusions are about the
geometry of the formats rather than the size of the model, so they were first measured on
`Qwen2.5-0.5B-Instruct`. E1 has since been repeated over a 15× range of parameter count,
and the 6.2595% figure does not depend on it — see above and
[`docs/e1-scale.md`](docs/e1-scale.md). E3 has not: it is measured on one quantised file.

Everything that runs a model — E2, E5, E6 — targets a model people actually deploy:

| role | model |
|---|---|
| subject | `Qwen/Qwen3-4B-Instruct-2507` — current generation, ~3.4 M downloads |
| positive control | `huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated` (ablation only, no further training) |

A result on half a billion parameters says little to anyone running seven. Two
consequences follow and both are binding: the model stays in `bfloat16` rather than
being promoted to `float32` — which is also the more faithful choice, since a weight
with bit 14 flipped is 3.06 × 10³⁸ and `bfloat16` represents it — and if the available
memory cannot hold the subject, the run fails loudly. An experiment that quietly
substitutes a smaller model is worse than one that does not run.

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
6. **The boundary is around the published artefact, not around the question** — the worst
   case is measured; the addresses that produce it are not published. Stated normatively,
   once, in [`SCOPE.md`](SCOPE.md).
7. **Operational security** — no local paths, no bylines, and a single alias for the
   code. The Kaggle account that runs the experiments is named here deliberately: a
   notebook that cannot say which repository it installs from is not reproducible. That
   link was accepted as a cost of running in the open, not overlooked.

## Reading order

1. This file, for what exists and what does not.
2. [`SCOPE.md`](SCOPE.md) — what is assumed, what is never done, and what leaves the
   project. Read it before the results if you are here to assess the ethics rather than
   the numbers.
3. [`docs/e1-bit-hierarchy.md`](docs/e1-bit-hierarchy.md),
   [`docs/e1-scale.md`](docs/e1-scale.md),
   [`docs/e3-gguf-surface.md`](docs/e3-gguf-surface.md),
   [`docs/e2-degradation.md`](docs/e2-degradation.md),
   [`docs/e5-oracle-validation.md`](docs/e5-oracle-validation.md) and
   [`docs/e5b-silent-dealignment.md`](docs/e5b-silent-dealignment.md) — the
   per-experiment technical notes: method, result, and declared boundaries, one per
   experiment.
4. `results/` — the CSVs behind every figure quoted above, and
   `results/history/` for the runs that were withdrawn.
5. [`ERRATA.md`](ERRATA.md) — what this project has said and unsaid, and how far each
   correction has travelled.

### The write-up

A note covering the whole programme exists and is **not** published. It is held until the
experiment still open — E4 — has been run, so that the argument it makes can be made
once, over a complete record, instead of being revised under a reader's eye.

Until then this repository is the record, not a trailer for one: the code, the data, the
six per-experiment notes under `docs/`, [`SCOPE.md`](SCOPE.md) and
[`ERRATA.md`](ERRATA.md) are complete and reproducible as they stand.

## Licence

MIT, see [`LICENSE`](LICENSE). The model weights used remain subject to their upstream
licences and are **not** redistributed by this repository: they are downloaded at the
revisions pinned in `src/bitflip/fetch.py`.
