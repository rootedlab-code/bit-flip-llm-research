# Errata — corrections to published claims

Every entry here is a claim this project **published and then withdrew**. The entry states
what was said, what is said now, why the first version was wrong, and — the column that
makes this list useful rather than decorative — **where the correction has and has not yet
been carried**.

A claim that lives in more than one venue is withdrawn in all of them or in none. Four times
on 2026-08-28 a correction was applied to one venue and not to its twin, and each time the
divergence was found by a reader of the other page rather than by its author. The
`propagation` line exists so that failure is visible instead of latent.

The threshold for an entry: **something a reader could have believed, and could not have
caught on their own.** A page that contradicts itself is a page a reader can check — that
gets fixed, and stays out of here. A figure someone could have carried away and relied on
gets recorded, whether or not anyone noticed. Entries cheaper than that would dilute the
list and hide the ones that matter.

Corrections to documents that were never published are not errata: they are drafting, and
they belong to the document's own history. And an entry is written when the withdrawal
exists, never when it is planned: an erratum that anticipates its own correction is a claim
about the future.

The data behind the withdrawn runs is kept, not deleted: see
[`results/history/`](results/history/README.md).

---

## 2026-08-29 — E2's leverage figure was called an independent confirmation

**Was:** E1 counted 6.259549% of a model's bits as catastrophic, E2 observed a chosen bit to
be catastrophic in 4 configurations of 4, and the page concluded — "two methods, no shared
assumption, same figure" — that 15.98× had been confirmed twice over.

**Is:** agreement, not independent confirmation. 15.98 is the **reciprocal of E1's own
counted fraction**, so the two sides of the comparison are the same number and cannot
disagree. What E2 contributes is the *numerator* — that a chosen flip is catastrophic at all
— on four configurations. Four successes in four trials put a one-sided 95% lower bound of
0.47 on that certainty, so the leverage this measurement supports is **at least 7.6×**, and
reaches 15.98× only if a chosen flip never misses.

**Why it crosses the threshold.** No reader of "no shared assumption" could see that
15.98 = 1 / 0.06259549 without redoing the arithmetic themselves, and nothing on the page
invited them to. It is a claim about *method* — the strongest kind a measurement can make —
resting on an identity.

**Propagation:** `docs/e2-degradation.md` (commit `c001b0c`) and the technical note. The
README never carried it: it quotes 15.98 only as E1's count, which is a different referent
and is unaffected.

## 2026-08-29 — "Roughly a hundred cosmic rays" named the wrong quantity

**Was:** "Choosing the address is worth roughly a hundred cosmic rays," placed under a table
reading 11 / 36 / 72 / 107, introduced by a line offering that table as "how many faults
nobody chose are worth one fault somebody did".

**Is:** three defects in one sentence, and the third is the source of the other two. The
leverage of choosing is **15.98× per flip**. The 11–107 table answers a *different* question
— how many unchosen faults it takes before destruction becomes likely, at a chosen level of
likely — and 107 is its 99.9% row, presented as though it were the typical value. The
sentence introducing the table had already framed it as the answer to the leverage question,
which is where the conflation entered.

**Why it crosses the threshold.** "Roughly a hundred" reads as a rounding of a measured
quantity. Nothing on the page revealed that it was one row of a different table, chosen at
the most favourable confidence.

**Propagation:** single site, `docs/e2-degradation.md` (commit `c001b0c`), correcting the
framing line above the table as well as the claim below it — correcting only the second
would have left the page contradicting itself two paragraphs apart.

## 2026-08-29 — The boundary on optimised search

**Was:** the project would not run an optimised search for the bits that matter — "there is
not, and will not be, a guided search for the optimal bit triple".

**Is:** the search **is** run; its objective is an inert scalar; the curve and the counts are
published; the addresses are not, and do not outlive the run. See
[`SCOPE.md` §5.1](SCOPE.md).

**Why the first version was wrong.** It drew the boundary around the *question* rather than
around the *artefact*. A programme that never runs the search reports whatever an
unoptimised policy happens to achieve, which is a lower bound of unknown looseness.
Publishing "N flips cost X% of alignment" when a guided search needs three would hand a
defender a reassuring number that is false. The restraint that matters is withholding the
addresses, and that restraint is compatible with measuring the worst case.

**Propagation:** `SCOPE.md` is now the only normative statement; `README.md` and the
technical note point at it and no longer state it in their own words.

## 2026-08-28 — One ratio was standing in for three

**Was:** "per random fault, the quantised file loses **2.807×** more weights than the
`bfloat16` one", reported as *the* answer to whether quantisation protects.

**Is:** three ratios, each with the question it answers —

| held equal | gguf / bf16 |
|---|---|
| the flip: a fault landed *in this file* | 2.8071× |
| the physical exposure: same DRAM, same hours, same fault rate per bit | **1.3792×** |
| the model: the share of its own parameters each format loses | 1.0812× |

**Why the first version was wrong.** It was not arithmetically false — it was
under-specified in a way that mattered. The two files differ in size by a factor of two, so
"per random fault" silently chose one normalisation out of three. E4 bridges to field fault
rates quoted **per bit per hour**, which is the equal-exposure row: using the per-flip figure
there would overstate the bridge twofold, in the alarmist direction.

**Propagation:** `README.md` and `results/e3-normalisation.csv` carry all three. The
technical note's abstract and §4.2 carried only the per-flip figure until this correction.

## 2026-08-28 — "Bits 11-13 are harmless by construction"

**Was:** bits 11-13 of a `bfloat16` weight were described as harmless, and counted only in
the catastrophic direction wherever they appeared.

**Is:** those bits are one almost everywhere, so flipping them **divides** — by 2¹⁶, 2³² and
2⁶⁴ — which *removes* a weight instead of exploding it. That channel is **18.74%** of the bit
space against the catastrophic 6.26%: three times larger.

**Why the first version was wrong.** It was a statement about the **criterion**, not about
the weights. The criterion asks whether a magnitude exploded, so a channel that annihilates a
weight scores as harmless. The defect was in the instrument and was read as a property of the
subject.

**Propagation:** carried through `docs/e1-bit-hierarchy.md`, `docs/e1-scale.md` and the
dataset description (commits `4439570`, `3633228`). Finding every venue required searching for
the **subject** (`11-13`) rather than for the withdrawn phrase: `docs/e1-scale.md` discussed
those bits four times without ever using the word "harmless", so a search for the wording
would have missed it.

## 2026-08-28 — The top exponent bit is *not* zero in 100% of weights

**Was:** bit 14 is zero in 100.00% of `bfloat16` weights.

**Is:** 99.9980% at 0.5 B, rising towards but never reaching 100% as models grow
(99.99950% at 4 B, 99.99979% at 7 B). Separately: in the `fp16` block scales of a `q4_k_m`
file it **is** exactly 100.00%, and that exactness is a different property from the weights'
near-universality — the two were being called the same thing.

**Why the first version was wrong.** A rounded display was read as an exact count. The
consequence is not cosmetic: the exactness is what makes the quantised scales *attain* the
1/16 floor that the weights only approach.

**Propagation:** `docs/e1-bit-hierarchy.md`, `docs/e3-gguf-surface.md`, `results/`, the
dataset description and the notebook sources (commits `bd18409`, `c20d179`, `62d3189`,
`65a350e`, `cc514a9`). **Outstanding:** the published Kaggle page for the E1-at-scale
notebook still displays the superseded figure — its source in this repository is corrected,
but the page has not been re-submitted since.

## 2026-08-28 — A false positive in the catastrophic test

**Was:** the test for "did this flip produce a catastrophic magnitude" accepted a case it
should have rejected.

**Is:** recorded, with the conditions under which it would bite, in commit `377516f` and the
note it belongs to.

**Propagation:** complete within the repository.

## 2026-08-28 — E2's first run reported no damage from chosen faults

**Was:** chosen faults produce `ratio_to_baseline = 0.99999` — that is, targeting a weight
does nothing.

**Is:** withdrawn. The targeting policy selected the largest-magnitude weights, which are
exactly the weights with |w| ≥ 2 — the condition under which bit 14 is **already set**. The
flip therefore divided by 2¹²⁸ instead of multiplying by it. The run measured a bug in the
policy, not a property of the model.

**Propagation:** the run is kept as evidence at
`results/history/e2-degradation-run1-broken-targeting.csv`; the corrected run supersedes it.

## 2026-08-27 — The refusal oracle, versions 1 and 2

**Was (v1):** the classifier's verdicts at the validation corners were reported as the
instrument's behaviour.

**Is:** v1 failed two corners of four — it classified an aligned model's refusals as
`degenerate` 63% of the time, because its degeneracy floor of 40 characters sat inside the
length distribution of a refusal, which is short by construction. v2 failed one corner of
six. v3 passes six of six.

**Why this is an erratum and not merely iteration.** v1 and v2 numbers were published before
the failure was diagnosed, and an instrument's output is not a measurement until the
instrument is validated.

**Propagation:** every superseded run is kept under `results/history/`, the diagnosis is in
its README, and the reasoning is in the changelog inside
`src/bitflip/spec/oracle_spec.yaml`. The v3 figures carry their own live caveat, stated
wherever they appear: the classifier has never been compared against human labels, and 28% of
an aligned model's answers to harmless questions remain undecided.
