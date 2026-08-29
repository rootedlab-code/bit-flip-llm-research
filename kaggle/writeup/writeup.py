# %% [markdown]
# # Who picks the bit
#
# ## A cosmic ray and an attacker do the same thing to a language model. What is the difference worth?
#
# A bit flips in a DRAM cell. It happens on its own — cosmic rays, thermal noise, a cell
# that leaks — and it happens on purpose, because Rowhammer lets a process on the same
# machine choose which cell. **The physical event is identical. Exactly one thing
# differs: who picks the address.**
#
# Two research communities study this and rarely talk. Reliability engineering asks how
# often a bit flips and what a machine should do about it. Security asks whether an
# attacker can steer one. Nobody has put the two on a single axis and measured the
# leverage: *what is it worth, in numbers, to choose?*
#
# This notebook is the introduction to that project. Every figure below is read from the
# published dataset, not typed in — if a number here disagrees with the CSV, the CSV wins
# and the notebook is wrong.
#
# **Repository:** <https://github.com/rootedlab-code/bit-flip-llm-research> ·
# **Author:** `rootedlab-code`

# %%
import json
import pathlib

import pandas as pd

CANDIDATES = [pathlib.Path("/kaggle/input/bit-flip-results"), pathlib.Path("results")]
DATA = next((p for p in CANDIDATES if p.exists()), None)
if DATA is None:
    raise SystemExit("attach the bit-flip results dataset, or run from the repository root")

read = lambda name: pd.read_csv(DATA / name)
load = lambda name: json.loads((DATA / name).read_text())

print(f"reading from {DATA}")
print(f"{len(list(DATA.glob('*')))} files")

# %% [markdown]
# ## What is measured, and what is not
#
# This project publishes status markers rather than a finished story, because a
# half-finished experiment described in the past tense is how a reader ends up believing
# something nobody measured. The table below is the honest state.
#
# | | Question | Status |
# |---|---|---|
# | **E1** | Which of the 16 bits of a weight matters, and how much? | **measured** |
# | **E3** | Does quantisation protect? | **measured** |
# | **E2** | How much does the model actually degrade? | **measured** |
# | oracle | Can a program tell refusal from compliance reliably? | **measured**, six corners of six |
# | **E5** | Does *alignment* fail before *capability*? | **running** |
# | **E4** | How long until a natural fault hits a critical bit? | not run |
# | **E6** | Does a flipped agent take destructive actions? | not run |
#
# Nothing in the dataset stands in for the two that have not run. There are no estimates,
# no interpolations, and no figures borrowed from other papers.

# %% [markdown]
# ## E1 — one bit in sixteen carries almost everything
#
# A `bfloat16` weight is 16 bits: 1 sign, 8 exponent, 7 mantissa. Flipping each one does
# something different, and the outcome depends **only on the 16-bit pattern** — not on
# which weight happens to carry it. So the histogram of all 65,536 patterns present in a
# file describes the entire model without loss.
#
# That is why every fraction here is a **count, not a sample estimate**. Nothing was
# sampled; all 494 million weights were enumerated.

# %%
bits = read("e1-bit-hierarchy-base.csv")
view = bits[["bit", "field", "zero_bit_fraction", "median_delta", "catastrophic_fraction"]]
print(view.to_string(index=False, float_format=lambda v: f"{v:.6g}"))

# %% [markdown]
# Read the last column. Fifteen positions are at or near zero and **bit 14 is at
# 0.99998**: flipping the top exponent bit is catastrophic in essentially every weight.
#
# The reason is in the third column. Bit 14 is *zero* in 99.998% of weights, because
# almost every weight is smaller than 1. So the flip is nearly always 0→1, which
# multiplies by 2¹²⁸.
#
# **The operational point is not that the multiplier is large — that is arithmetic. It is
# that the value is predictable.** To amplify a weight, an attacker does not need to know
# which weight is being hit. That is what makes the attacks in the literature practical.

# %%
summary = read("e1-summary.csv").iloc[0]
print(f"weights            {summary.weights:,}")
print(f"bits in the file   {summary.total_bits:,}")
print(f"catastrophic bits  {summary.catastrophic_bits:,}")
print(f"share              {summary.catastrophic_bit_fraction:.6%}  =  one bit in {summary.one_bit_in:.2f}")

# %% [markdown]
# ## The channel that a threshold cannot see
#
# "Catastrophic" above means *the magnitude exploded*. That predicate has nowhere to put
# a different outcome: **the weight is gone**. Flipping bit 11, 12 or 13 — which are `1`
# in almost every weight — divides by 2¹⁶, 2³² or 2⁶⁴. A typical weight of 0.02 becomes
# 3·10⁻⁷. It is not perturbed; it is removed.
#
# The technical note called those bits "harmless by construction" until the whole bit
# space was partitioned by **outcome** instead of thresholded:

# %%
spectrum = read("e1-perturbation-spectrum.csv")
base = spectrum[spectrum.model == "base"].sort_values("bit_share", ascending=False)
print(base[["outcome", "bit_share", "positions"]].to_string(index=False, float_format=lambda v: f"{v:.7%}"))
print(f"\nsum of the classes: {base.bit_share.sum():.10f}")

# %% [markdown]
# The channel that had no name is **three times** the size of the one that did — 18.74%
# against 6.26%. The seven classes sum to exactly 1, which is the check that nothing is
# unaccounted for.
#
# **What this does not license.** In the *random* arm this is noise: removing weights at
# random, one in five hundred million, is unstructured pruning at a sparsity the
# literature finds does nothing. The channel is interesting in the *chosen* arm, where it
# is pruning at the granularity of one bit, on a weight the attacker selected, without
# access to the file.
#
# **Why it is worth naming anyway.** A catastrophic flip produces NaN, and NaN is the
# loudest thing a model can do — the output collapses and any operator notices within one
# request. So *silent* de-alignment, which is what E5 exists to look for, cannot live
# inside that 6.26% **by construction**. If it exists, it is somewhere in the other 93.7%.

# %% [markdown]
# ## Is this a property of the model, or of the format?
#
# The obvious objection to a result measured on one small model is that it might be about
# that model. E1 was therefore repeated across a 15× range of size and two model
# generations.

# %%
scale = read("e1-scale-summary.csv")
print(scale[["model", "weights", "catastrophic_bit_fraction", "top_exponent_zero_fraction"]]
      .to_string(index=False, float_format=lambda v: f"{v:.6%}"))
print(f"\nspread across a 15x range: "
      f"{(scale.catastrophic_bit_fraction.max() - scale.catastrophic_bit_fraction.min()) * 100:.4f} "
      f"percentage points")

# %% [markdown]
# It is the **format**, not the model. All three sit just above the 6.25% floor that one
# bit in sixteen imposes by geometry.
#
# The last column narrows a claim this project got wrong once. The top exponent bit is
# *not* zero in 100% of weights — a two-decimal rounding once said so. It climbs towards
# universality with scale without ever arriving. The weights it misses are those with
# |w| ≥ 2, where flipping that bit **divides** by 2¹²⁸ instead of multiplying, which is
# why they matter more than their number suggests: an attacker who targets the largest
# weights hits exactly them, and measures nothing.

# %% [markdown]
# ## E3 — does quantisation protect?
#
# A quantised file stores 4-bit integers plus a per-block `fp16` **scale**. The scale is a
# small share of the bits and governs an entire block, so corrupting one damages every
# weight it covers. The question is whether the smaller, denser file is safer or not.
#
# It depends entirely on **what you hold equal**, and this project published the wrong one
# first.

# %%
norm = read("e3-normalisation.csv")
for _, row in norm.iterrows():
    mark = "  <-- the one a field fault rate must be crossed with" if row.feeds_fault_rate else ""
    print(f"{row.ratio:>7.4f}x   {row.key}{mark}")
    print(f"          {row.question}")

# %% [markdown]
# The headline was **2.807×** — the cost of a flip that has already landed *inside* the
# file. But the quantised file is 0.49 times the size, so at equal physical exposure it
# intercepts about half the faults, and the honest figure for a deployment is **1.379×**.
# Field fault rates are quoted per bit per hour, so using the per-flip number would
# overstate that bridge twofold, in the alarming direction.
#
# All three are above 1, so the direction survives every choice: quantisation does not
# protect on this metric, it **concentrates**. The magnitude does not survive.
#
# **And the other channel points the other way.** Measured on the same two populations,
# the annihilation channel is 18.74% of `bfloat16` weight bits against 0.0014% and 0.027%
# of the `fp16` scale bits — three to four orders of magnitude *less* exposed. Quantisation
# concentrates the exploding fault and very nearly eliminates the annihilating one. One
# threshold could not have told you that.

# %% [markdown]
# ## E2 — what actually happens to the model
#
# E1 and E3 measure a **surface**: which bits are dangerous, and how many. Neither says
# what happens to the model. This does.

# %%
dmg = read("e2-degradation.csv")
print(pd.crosstab(dmg.policy, dmg.damage_class).to_string())
print()
print(dmg[dmg.policy == "random"].groupby("flips").damage_class.value_counts().to_string())

# %% [markdown]
# **Degradation is not gradual.** The model either survives a fault untouched or stops
# being a model; across 29 configurations the middle is one case wide.
#
# The two ways of dying are different and are counted apart. A random fault settles
# perplexity at ≈151,936 — the vocabulary size, which is the perplexity of a uniform
# distribution: the model is numerically alive and still answers, but every token is
# equally likely to it. A chosen fault produces NaN: the weight stays finite by
# construction, but multiplying activations by 3·10³⁸ overflows anyway.
#
# A model that went uniform *responds*. One that went NaN does not. **Neither is a
# stealthy failure** — which is exactly what E5 has to look for underneath both.

# %% [markdown]
# ## The instrument, before the experiment
#
# E5 asks whether a model can lose alignment while still passing every check an operator
# would run. That question is only answerable with an instrument shown to work, so the
# classifier was validated first, at six corners of its own output space, against models
# whose behaviour was known in advance.

# %%
oracle = read("e5-oracle-validation.csv")
print(oracle.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# %% [markdown]
# Six corners of six pass. But the number that matters for anyone building on this is the
# last column: **28% of a working model's answers to harmless questions are still not
# recognised as delivery.** That is a floor under the resolution of any de-alignment
# figure built on this instrument, and it is published rather than hidden.
#
# **A declared validity condition is currently unsatisfied.** The protocol calls for
# agreement with human labels — Cohen's κ, with a threshold fixed in advance below which
# the oracle is not used. No labels exist, so no κ exists, so that threshold has never
# been tested. Six automated corners against known-truth models is a weaker guarantee than
# agreement with a person, and it is not a substitute for it.

# %% [markdown]
# ## How the work is kept honest
#
# Three rules, and each one has caught something:
#
# 1. **Nothing asserted, everything recomputed.** Every published figure traces to a CSV
#    in this dataset. The claim is checkable, and it has failed once — two figures were
#    quoted for models that were not listed among the artefacts. Both are now listed.
# 2. **Determinism, or it is not a measurement.** Two runs of the same configuration
#    produce identical output, and the experiments assert it before measuring anything.
#    The scope was narrowed by measurement rather than assumed: change the batch size and
#    only 341 of 600 greedy generations stay byte-identical, because padding width changes
#    the order of floating-point reductions and `argmax` is discontinuous.
# 3. **Pre-registration.** Classification criteria and statistics are frozen in versioned
#    files with digests, printed in the published output of a run **before** a single
#    token is generated.
#
# **What this dataset will never contain:** no model weights, original or modified; no
# text generated by a model under test, in any form; no reusable attack procedure — no
# optimal bit addresses, no DRAM hammering parameters. The project measures the *payload*,
# what the flipped bit does, not the *delivery vector*.

# %% [markdown]
# ## Two gaps a reader can close and the author cannot
#
# **1. Does a run reproduce on a different accelerator?** Two sessions produced
# byte-identical answers to all 600 probes — and both drew two Tesla T4s. Since the cause
# of divergence is the order of floating-point reductions, the expectation is that a
# different card does *not* reproduce. Expectation is not measurement. Copy & Edit the
# oracle-validation notebook, run it, and post the digest block its last cell prints. A
# mismatch is the more interesting result, so it is worth posting either way.
#
# **2. Has the classifier ever agreed with a human?** No κ has been measured. Sixty
# hand-labelled answers would be useful, and the benign half involves no harmful content
# at all. The protocol — including what must never be sent back — is in `CONTRIBUTING.md`
# in the repository.

# %%
manifest = load("models-manifest.json")
print("every artefact measured, pinned by revision and digest:\n")
for entry in manifest:
    print(f"  {entry['repo_id']:<46} @{entry['revision'][:8]}  {entry['bytes']:>13,} bytes")
print("\nreproduce:")
print("  uv sync && uv run pytest")
print("  uv run python -m bitflip.fetch")
print("  uv run python experiments/e1_bit_hierarchy.py")
print("  uv run python experiments/e3_gguf_surface.py")
print("\nthe regenerated CSVs must be byte-identical to the published ones.")
