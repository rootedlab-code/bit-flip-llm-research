# Contributing

This project measures what a flipped bit does to a language model. Two of its gaps are the
kind a community can close and a single author cannot, because closing them needs hardware
diversity or human judgement rather than more code. They are listed first, ahead of code
contributions, because they are worth more.

Everything below can be done without cloning anything: both tasks start from a Kaggle
notebook and a *Copy & Edit*.

---

## Task 1 — run it on your accelerator, post one block

**The gap.** Two sessions of the oracle-validation notebook produced byte-identical
answers to all 600 probes, so generation is reproducible under a fixed configuration.
Both sessions drew two Tesla T4s. Nothing in this repository shows that a run reproduces
on *different* hardware — and since the cause of divergence is the order in which
floating-point reductions happen, the expectation is that it does not. See
[`docs/environment-notes.md`](docs/environment-notes.md).

**What to do**, about 45 minutes of machine time and one minute of yours:

1. Open the notebook:
   <https://www.kaggle.com/code/seb001010/bit-flip-e5-oracle-validation>
2. *Copy & Edit*, then *Run All*. Nothing needs changing.
3. The last cell prints a block: the specification digest, two file digests, the
   accelerator and the torch version.
4. Paste that block into a comment on the notebook.

**What your number means.** If your verdict digest is the one this repository publishes,
greedy decoding reproduced across hardware and a caveat comes out of the research. If it
differs, that caveat is confirmed and — for the first time — quantified.

**A mismatch is the more interesting outcome, and it is the one this project expects.**
Please post it either way. A negative result nobody posts is a negative result nobody has.

The same applies to [the E2 notebook](https://www.kaggle.com/code/seb001010/bit-flip-e2-degradation),
where the claim under test is narrower and sharper: top-1 agreement was identical across
two machines in all 30 configurations while 9 of the 30 perplexities were not. If that
holds on a third machine, agreement is the metric every later experiment should be built
on.

---

## Task 2 — tell us whether the classifier agrees with a human

**The gap.** The oracle is validated against models whose behaviour is known in advance —
an aligned model, an abliterated one, a bricked one. It has **never been compared with
human judgement**: no Cohen's κ has been measured. Until it is, the share of answers it
files as `indeterminate` is the only honest statement of how much it does not know, and on
an aligned model answering harmless questions that share is **28%**.

Cohen's κ is the standard measure of agreement between two raters that corrects for
agreement occurring by chance. Here the two raters are the classifier and you.

### 2a — the benign corner (recommended, and involves no harmful content at all)

This is an aligned model answering ordinary instructions from Alpaca: *"give three tips for
staying healthy"*. There is nothing unpleasant to read, and it is where the instrument is
weakest.

1. *Copy & Edit* the oracle-validation notebook.
2. Add a cell that keeps the answers for the `base` / `benign` corner instead of
   discarding them, and prints them one at a time **without** the oracle's verdict.
3. For each, write down one of `compliance` (the model delivered an answer to the question
   asked), `refusal`, `degenerate` (not language), or `unsure`.
4. Emit a CSV with exactly three columns: `probe`, `answer_sha256`, `human_label`.
5. Open an issue or a pull request on this repository with that CSV **and** your run's
   `e5-verdicts.csv`.

Both files are needed, and the reason is the finding in Task 1: your run's answers are not
this repository's answers, so the labels are only meaningful against the verdicts of the
same text. They are paired on `answer_sha256`, and any row whose digest does not match is
dropped rather than counted.

Sixty labelled answers are enough to be useful. There is no need to do all one hundred.

### 2b — the harmful corners (opt in, only if you want to)

The same procedure on the `harmful` probe sets means reading what an abliterated model
produces in response to AdvBench prompts. It is the more valuable data and it is
unpleasant. Skip it without explanation if you would rather not; 2a alone is a real
contribution.

### What not to send, in either case

**Never post generated text** — not in an issue, not in a comment, not in a dataset. The
notebooks deliberately keep none; the digest and the label are enough. A pull request
containing model output will be closed rather than edited.

---

## What happens to what you send

Contributions are recorded in the technical note for the experiment they touch, with the
figure they changed and your name or handle as you give it. Say so if you would rather not
be named. If a contributed measurement contradicts a published figure here, the published
figure changes and the note says who found it.

---

## Contributing code

The project's conventions, in the order they matter:

1. **Calibrate against conditions whose answer is known in advance, never against the
   hypothesis under test.** This is the rule the whole programme rests on. When the oracle
   failed, it was diagnosed on `base`/`benign` — a condition with no flips, no ablation and
   no alignment in play — precisely because loosening a criterion until the interesting
   corner passes would inflate the quantity being measured.
2. **Pre-register criteria.** Thresholds and classification rules live in a versioned file
   with a digest, frozen before the run that uses them, together with the condition that
   would reject them.
3. **Nothing asserted, everything recomputed.** Every published figure traces back to the
   script and the CSV that produced it.
4. **A run records its own configuration.** Comparisons across runs are sound only when the
   generation configuration is identical, and "identical" has to be checkable — which is
   what `results/e5-run-manifest.json` is for.

Before opening a pull request:

```sh
uv sync
uv run pytest          # the contract of every module
uv run ruff check .
uv run ruff format --check .
```

Logic that produces a published number belongs in `src/bitflip/` under test, not in an
experiment script. The Kaggle notebooks are generated from the `.py` sources under
`kaggle/` — edit the `.py`, then regenerate:

```sh
uv run python kaggle/build_notebook.py kaggle/e5/e5_alignment.py kaggle/e5/e5-alignment.ipynb
```

## Reporting something that looks wrong

Open an issue with the file, the line and what you expected. A figure that cannot be
recomputed from the repository is a defect, and so is a stale document.
