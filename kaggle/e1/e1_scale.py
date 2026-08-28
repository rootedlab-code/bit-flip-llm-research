# %% [markdown]
# # E1 at scale — is the fragility a property of the format, or of the model?
#
# E1 measured, on `Qwen2.5-0.5B-Instruct`, that **6.2595% of a model file's bits are
# catastrophic** — one in every 15.98 — and that bit 14, the top exponent bit, is zero
# in 100.00% of the weights and lethal in 99.998% of them. That is the figure a fault
# rate gets crossed with, so it is the figure the whole quantitative argument leans on.
#
# It was measured on half a billion parameters. **Nothing in it has been shown to hold
# for a model anybody deploys**, and there are two different reasons it might not: the
# number could depend on the size of the model, or on the habits of one family.
#
# This notebook separates those two, by measuring three models along two axes:
#
# | subject | parameters | what it varies |
# |---|---|---|
# | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | the anchor — reproduces the published figure |
# | `Qwen/Qwen2.5-7B-Instruct` | 7.6B | **size**, at constant family |
# | `Qwen/Qwen3-4B-Instruct-2507` | 4B | **family and generation**, at comparable size |
#
# The 0.5B is measured again here rather than carried over, and that is deliberate:
# it goes through the same code path as the two models that have never been measured,
# so if the reader is to believe the new rows, the old one has to come back unchanged.
# The notebook checks that itself, against the committed figure, and prints the result.
#
# ## What this measures, and what it does not
#
# The statistics are **exact, not sampled**. The outcome of a flip depends only on the
# 16-bit pattern, not on which weight carries it, so the histogram of the 65,536
# patterns summarises a model without loss however large it is. Every fraction below is
# a count.
#
# **No model is executed.** This is static analysis of stored bytes: no GPU, no
# inference, no generated text, nothing that depends on an accelerator or a seed. It is
# the half of the programme that reproduces exactly, and it says nothing about how much
# a model *degrades* — that is E2's question, and it needs a GPU.
#
# The files are read under a guard that re-verifies their SHA-256 afterwards, and the
# digests are published with the results.
#
# Static half of the programme:
# **https://github.com/rootedlab-code/bit-flip-llm-research**

# %%
# ruff: noqa: E402
import subprocess
import sys

print(sys.version)

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-cache-dir",
        "git+https://github.com/rootedlab-code/bit-flip-llm-research",
    ],
    check=True,
)

# %%
import csv
import json
import shutil
from importlib.metadata import version
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from bitflip.codec import BF16
from bitflip.fragility import bit_rows, format_table, population_summary
from bitflip.guard import free_gib, immutable, require_free_space
from bitflip.weights import code_histogram, dtype_census, open_weights

OUTPUT = Path("/kaggle/working")
# The weights are scratch: they are read once and deleted, and they must not land in
# the 20 GB of persisted output, where 23 GB of downloads would not fit anyway.
SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp/bitflip")

# Only the stored tensors and the two files describing them. No tokenizer, no weights in
# other formats: this experiment reads bytes, not a runnable model.
WEIGHT_PATTERNS = ("*.safetensors", "model.safetensors.index.json", "config.json")

# Above the largest subject with room to spare. One model is held at a time and deleted
# before the next, so the peak is one model, not their sum.
MIN_FREE_GIB = 25.0

# Pinned by revision rather than by branch: an experiment that cannot say which bytes it
# measured is not reproducible, and a model repository is not immutable.
SUBJECTS = (
    (
        "qwen25-0.5b",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775",
    ),
    (
        "qwen3-4b",
        "Qwen/Qwen3-4B-Instruct-2507",
        "cdbee75f17c01a7cc42f958dc650907174af0554",
    ),
    ("qwen25-7b", "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
)

# The committed result this run has to reproduce, from `results/e1-summary.csv` at
# commit 0e1b787. Stated before the run, not read off it.
ANCHOR = "qwen25-0.5b"
PUBLISHED = {
    "weights": 494_032_768,
    "catastrophic_bits": 494_787_536,
    "catastrophic_bit_fraction": 0.06259548556908678,
}

print(
    f"scratch {SCRATCH} · {free_gib(SCRATCH):.1f} GiB free · bitflip {version('bitflip')}"
)

# %% [markdown]
# ## Measuring one model
#
# Download, read, delete. The histogram is 65,536 counters — half a megabyte — so what
# survives a subject is four orders of magnitude smaller than the subject.
#
# Two checks run on every model and neither is decoration:
#
# - **The histogram must total the BF16 parameter count** the headers declare. A shard
#   that failed to download would otherwise not raise anything; it would quietly produce
#   a smaller histogram, which is the one way a whole-population count can be wrong
#   while looking right.
# - **The share of the model actually stored as BF16 is published beside the fraction.**
#   The 0.5B is uniformly BF16, so on it the qualification never had to be made. Carrying
#   that assumption to a model nobody has looked at is exactly how a denominator shrinks
#   in silence, and the fraction inflates without anything failing.


# %%
def measure(name: str, repo: str, revision: str) -> tuple[list[dict], dict]:
    require_free_space(SCRATCH, MIN_FREE_GIB)
    directory = SCRATCH / name
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=directory,
        allow_patterns=list(WEIGHT_PATTERNS),
    )

    try:
        weights = open_weights(directory)
        with immutable(weights.paths) as digests:
            counts = code_histogram(weights, BF16)
            census = dtype_census(weights)

        stored = sum(row["parameters"] for row in census.values())
        covered = census.get("BF16", {}).get("parameters", 0)
        if int(counts.sum()) != covered:
            raise RuntimeError(f"histogram {int(counts.sum()):,} != BF16 {covered:,}")

        declared = json.loads((directory / "config.json").read_text()).get("torch_dtype")
        rows = bit_rows(counts, BF16)
        totals = (
            {"model": name, "repo": repo, "revision": revision}
            | population_summary(counts, rows, BF16)
            | {
                "stored_parameters": stored,
                "bf16_parameter_share": covered / stored,
                "declared_dtype": declared,
                "tensors": len(weights),
                "shards": len(weights.paths),
            }
        )
        totals_digests = {Path(path).name: digest for path, digest in digests.items()}
    finally:
        shutil.rmtree(directory, ignore_errors=True)

    print(f"\n=== {name}: {repo}@{revision[:7]} ===")
    print(
        f"{stored:,} stored parameters across {totals['tensors']} tensors in "
        f"{totals['shards']} shard(s) · declared {declared}"
    )
    for dtype, row in sorted(census.items()):
        print(
            f"  {dtype:<8} {row['tensors']:>4} tensors  {row['parameters']:>15,} params"
        )
    print(format_table(rows))
    print(
        f"weights with |w| < 1: {totals['fraction_below_one']:.4%} · "
        f"median exponent {totals['median_exponent']:.0f} (bias {BF16.bias})"
    )
    print(
        f"catastrophic bits: {totals['catastrophic_bits']:,} of "
        f"{totals['total_bits']:,} = {totals['catastrophic_bit_fraction']:.4%}, "
        f"one in {totals['one_bit_in']:.2f}"
    )
    return rows, totals | {"digests": totals_digests}


# %% [markdown]
# ## The three subjects
#
# One at a time, largest peak on disk about 15 GB, everything deleted before the next.

# %%
OUTPUT.mkdir(exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)

summaries = []
for name, repo, revision in SUBJECTS:
    rows, totals = measure(name, repo, revision)
    with (OUTPUT / f"e1-bit-hierarchy-{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries.append(totals)

# %% [markdown]
# ## Did the anchor come back unchanged?
#
# The 0.5B was measured through the same reader as the two models that had never been
# read, on a different machine and a different Python from the run that published it. If
# it does not reproduce, nothing else on this page can be believed, and the check says so
# here rather than being left to the reading afterwards.

# %%
anchor = next(row for row in summaries if row["model"] == ANCHOR)
mismatched = {
    field: (expected, anchor[field])
    for field, expected in PUBLISHED.items()
    if anchor[field] != expected
}
for field, expected in PUBLISHED.items():
    mark = "✗" if field in mismatched else "✓"
    print(f"{mark} {field}: published {expected}, measured {anchor[field]}")
print("\nANCHOR REPRODUCED" if not mismatched else f"\nANCHOR BROKEN: {mismatched}")

# %% [markdown]
# ## Format, or model?

# %%
print(
    f"{'model':<12} {'parameters':>15} {'bf16 share':>11} "
    f"{'|w|<1':>9} {'catastrophic':>13} {'one bit in':>11}"
)
for row in summaries:
    print(
        f"{row['model']:<12} {row['stored_parameters']:>15,} "
        f"{row['bf16_parameter_share']:>10.2%} {row['fraction_below_one']:>8.4%} "
        f"{row['catastrophic_bit_fraction']:>12.4%} {row['one_bit_in']:>11.2f}"
    )

spread = max(r["catastrophic_bit_fraction"] for r in summaries) - min(
    r["catastrophic_bit_fraction"] for r in summaries
)
print(f"\nspread across {len(summaries)} models, 0.5B to 7.6B: {spread:.4%} of all bits")
print(
    "top exponent bit is zero in: "
    + ", ".join(f"{r['model']} {r['fraction_below_one']:.4%}" for r in summaries)
)

# %%
fields = [key for key in summaries[0] if key != "digests"]
with (OUTPUT / "e1-scale-summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summaries)

# Read from the objects actually used, never restated by hand, so it cannot drift away
# from the run it describes.
manifest = {
    "format": {"name": "bf16", "total_bits": BF16.total_bits, "bias": BF16.bias},
    "subjects": [
        {
            "model": row["model"],
            "repo": row["repo"],
            "revision": row["revision"],
            "shards": row["shards"],
            "digests": row["digests"],
        }
        for row in summaries
    ],
    "environment": {
        "python": sys.version,
        "numpy": np.__version__,
        "bitflip": version("bitflip"),
    },
}
(OUTPUT / "e1-scale-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
