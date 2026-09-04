# Scope — fault model, adversary, and publication boundary

**Status: normative.** This file is the single source for what this project assumes, what
it will not do, and what leaves it. Every other document — the README, the technical note,
the dataset description, the notebooks, `CONTRIBUTING.md` — **links here and does not
restate it**. A boundary written in its own words in a second place is a boundary that will
eventually disagree with itself; that failure has already happened once in this repository
and is recorded in [`ERRATA.md`](ERRATA.md).

Changes to this file are changes to the commitments of the project. They are dated in
`ERRATA.md`, never made silently.

---

## 1. What this project is

A **measurement study** of the critical surface of language-model weights, sitting between
two literatures that have studied the same physical event separately: systems reliability,
where nobody picks the address, and hardware security, where somebody does.

The unit of contribution is a **quantity**, not a capability: what fraction of a model's
bit space is critical, how that fraction moves with the storage format, and what the choice
of address is worth against a random fault. The central hypothesis is falsifiable in both
directions — if safety alignment degrades earlier than general capability, output-side
monitoring is blind by construction and integrity has to be enforced on the weights; if the
two fail together, the concern is misplaced. Both outcomes are published.

## 2. Fault model

The **single event upset**: one bit of the weight file, resident in memory, changes state.
Nothing else changes. Correlated multi-bit corruption within a word, transfer errors and
permanent cell failure are outside the model.

Two injection policies define the axis:

| policy | how the position is chosen | what it stands for |
|---|---|---|
| random | drawn uniformly over the bits of the file, under a declared seed | the natural fault |
| targeted | fixed in advance by a stated criterion | the attacker |

The **leverage factor** is the ratio between the number of random flips and the number of
targeted flips needed to produce the same damage, at equal damage metric.

**Every flip in this work is arithmetic on in-process memory arrays.** To the model, a
weight altered in memory is indistinguishable from one corrupted by a real fault; to the
DRAM hosting it, it is an ordinary write.

## 3. Adversary model

The adversary is assumed to be able to **choose which bit flips**, and nothing more. That
single capability is the whole subject: it is the one thing that separates a cosmic ray
from Rowhammer, and the work exists to price it.

The adversary is **not** assumed to have gradients, activations or logits from the victim's
running deployment, to read the victim's memory, or to control the victim's inputs. Where an
experiment grants a capability beyond address choice — for instance white-box access to a
locally held copy of a public model, in order to run the search of §5.1 — that grant is
stated in the experiment's own note, and it makes the resulting figure an **upper bound on
the attacker's efficiency**, which is the direction a defender needs.

## 4. Host safeguards

Not a footnote: a design constraint that, on conflict, cuts the experiment rather than the
rule.

- **No physical bit is ever flipped.** No Rowhammer, no `/dev/mem`, no kernel modules, no
  physical memory mappings, no DMA, no elevated privileges.
- **Model files are read-only**, opened via read-only mmap, with SHA-256 recorded on entry
  to every experiment and re-verified on exit, including when the experiment failed. A
  changed hash is a failing test, not a log line.
- **No modified weights touch the disk.** No code path serialises a flipped model.
- **Writes are confined** to the project directory, the scratch directory and the model
  cache; a free-space guard with an abort threshold runs before every download.

The realistic risk of a bit-flip project is not the bit-flip: it is the full disk and the
overwritten file. That is where the safeguard belongs.

## 5. Publication boundary

**The boundary is drawn around the artefact, not around the question.** What is measured is
the worst case; what is published is the result, not the means of reproducing it against a
third party.

### 5.1 Optimised search for the bits that matter

The search **is run**. Its objective is an **inert scalar** — a refusal margin, a
De-alignment Fraction — never generated text, and never a success rate against a named
harmful prompt. It is run only against weights of publicly available models held locally,
never against anyone's deployed system.

- **Published:** the curve and the counts — how the measured quantity moves as a function of
  the number of flips, and how many flips a policy needs to reach a stated threshold.
- **Not published, and not retained beyond the run that produced it:** the **addresses** —
  tensor, offset, bit index — and any ranking that would let a reader recover them.

The reason for measuring it at all: a boundary drawn around the *question* would leave the
programme reporting whatever an unoptimised policy happens to achieve, which is a lower
bound of unknown looseness. Publishing "N flips cost X% of alignment" when a guided search
needs three hands a defender a reassuring number that is false, and a defender who cannot
see the worst case cannot budget against it. The offensive uplift of measuring it here is
small — the method is published and peer-reviewed, and is cited as such in the technical
note — while the defensive value is not. The addresses are the transferable artefact; the
curve is the result.

*This supersedes an earlier formulation that excluded optimised search altogether. See
[`ERRATA.md`](ERRATA.md).*

### 5.2 Model output

**No harmful output is published, in any form** — not verbatim, not truncated, not redacted.
Generations needed for measurement live in a scratch directory, are classified, and are
deleted. Published tables retain class labels, scalars, lengths and a truncated SHA-256:
enough to re-verify a classification, not enough to redistribute anything.

### 5.3 Weights

**No weights are redistributed**, original or modified, not even as a diff. Artefacts are
downloaded by the reader at the revisions pinned in `src/bitflip/fetch.py`, and remain
subject to their upstream licences.

### 5.4 Delivery vector

**No Rowhammer is executed**, on this machine or any other. This work studies the
*payload* — what the flipped bit does — not the *delivery*. Delivery is DRAM-module
specific, is documented at length in the literature cited by the technical note, and
repeating it would add no knowledge while endangering the host.

### 5.5 Defences

None are implemented in this version. Measuring the critical surface is the prerequisite for
designing a defence, not the other way round. Nothing here should be read as a claim that a
defence exists.

## 6. The escalation rule

**If an experiment cannot run without crossing one of these boundaries, the experiment is
cut** — not the boundary. A boundary that yields to the first inconvenient result was a
preference, not a commitment.

Publication-boundary changes may only ever be made in the direction of *more* restraint
without ceremony. A change in the other direction is a decision, is argued in `ERRATA.md`
with the reason and the date, and lands in the same commit as the wording it changes.

## 7. Where this file is binding

This file binds any venue that **states the boundary of the research programme** — what
this project will and will not do:

- [`README.md`](README.md)
- the technical note in `docs/`
- the published notebooks, where they describe the programme rather than their own cell
- [`CONTRIBUTING.md`](CONTRIBUTING.md)

It does **not** bind a document that says what a particular artefact *contains*.
`kaggle/dataset/DESCRIPTION.md` §"What this dataset does not contain, and will not" is the
worked example: it uses much of the vocabulary of §5 about a different referent — the
fifteen CSVs a reader is deciding whether to download, not the experiments the project
forbids itself. Replacing it with a quotation from here would leave a dataset description
that no longer describes the dataset. Two pages that share a vocabulary and differ in
subject are not a divergence, and unifying them would manufacture the ambiguity this
section exists to prevent.

A venue that needs the boundary in front of its reader **reproduces §5 verbatim**, under a
line saying it is reproduced and that this file is authoritative. What no venue may do is
**paraphrase or argue** it in its own words: fresh wording creates a second owner, and two
owners eventually disagree. That is not a hypothetical — it is the defect this file was
created to close, recorded in `ERRATA.md` under 2026-08-29.
