# %% [markdown]
# # E2 — How much does a language model degrade under bit flips?
#
# This notebook is the running half of a research programme whose static half lives at
# **https://github.com/rootedlab-code/bit-flip-llm-research**.
#
# The static half already established, exactly rather than by sampling, that **6.2595%
# of the bits** of a `bfloat16` model file are catastrophic when flipped, and that bit
# 14 — the top exponent bit — is zero in **100.00%** of weights, so an attacker never
# needs to know which weight is being hit.
#
# What that does *not* say is how much the **model** degrades. A weight driven to
# 6.8e+36 inside a rarely used tensor may not change a comma of the output. This
# notebook measures the difference between a fault nobody chose and a fault someone did.
#
# **Numerical note.** Kaggle's T4 has no native `bfloat16`. The fault is injected into
# the **stored bf16 representation** — which is what sits in DRAM — and the arithmetic
# is done in `float32`. This is exact, not a compromise: a bf16 weight with bit 14
# flipped is 6.8e+36, which `float32` represents and `float16` would turn into `inf`,
# erasing the very effect under study. A bf16 value promoted to `float32` keeps its
# pattern in the top 16 bits, so bit *b* of the stored bf16 is bit *b+16* of the
# `float32` holding it.

# %%
# ruff: noqa: E402
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "git+https://github.com/rootedlab-code/bit-flip-llm-research",
    ],
    check=True,
)

# %%
import csv
import itertools
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from bitflip.codec import BF16, from_float32
from bitflip.inject import (
    TOP_EXPONENT_BIT,
    flipped_model,
    largest_magnitude_flips,
    random_flips,
)
from bitflip.metrics import perplexity, set_determinism

BASE_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
CORPUS_REPO = "Salesforce/wikitext"
CORPUS_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"

WINDOW = 1024
STRIDE = 512
CORPUS_TOKENS = 32_768
RANDOM_COUNTS = (1, 10, 100, 1_000, 10_000)
TARGETED_COUNTS = (1, 2, 5, 10)
SEEDS = (0, 1, 2, 3, 4)

OUTPUT = Path("/kaggle/working")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")

# %% [markdown]
# ## Loading: bf16 patterns, float32 arithmetic

# %%
set_determinism(0)
tokenizer = AutoTokenizer.from_pretrained(BASE_REPO, revision=BASE_REVISION)
model = AutoModelForCausalLM.from_pretrained(
    BASE_REPO, revision=BASE_REVISION, dtype=torch.bfloat16
)
model = model.to(torch.float32).to(DEVICE).eval()

parameter_sizes = {name: p.numel() for name, p in model.named_parameters()}
print(
    f"{sum(parameter_sizes.values()):,} parameters across {len(parameter_sizes)} tensors"
)

# %% [markdown]
# ## Corpus
#
# WikiText-2 test split, so the absolute perplexity is comparable with the literature.
# Only the *relative* degradation is used in the conclusions, but an absolute number a
# reader can sanity-check is worth having.

# %%
import pyarrow.parquet as pq  # noqa: E402

corpus_path = hf_hub_download(CORPUS_REPO, CORPUS_FILE, repo_type="dataset")
text = "\n\n".join(pq.read_table(corpus_path).column("text").to_pylist())
token_ids = tokenizer(text, return_tensors="pt").input_ids[0][:CORPUS_TOKENS].to(DEVICE)
print(f"{token_ids.numel():,} tokens of corpus")


def score(chunk: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(chunk.unsqueeze(0)).logits[0]


def measure() -> float:
    return perplexity(score, token_ids, window=WINDOW, stride=STRIDE)


# %% [markdown]
# ## Baseline, and the null control
#
# Two runs of the same configuration must agree digit for digit, and zero flips must
# reproduce the baseline exactly. Without both, no later difference is attributable.

# %%
baseline = measure()
repeat = measure()
print(f"baseline perplexity : {baseline!r}")
print(f"second run          : {repeat!r}")
print(f"identical           : {baseline == repeat}")

with flipped_model(model, []):
    null_control = measure()
print(f"zero flips          : {null_control!r}  identical: {null_control == baseline}")
assert baseline == repeat, "the measurement is not deterministic; stop here"
assert null_control == baseline, "the null control does not reproduce the baseline"

# %% [markdown]
# ## Random faults — the cosmic-ray model
#
# Uniform over every bit of every weight, which means proportional to tensor size: a
# large tensor is hit more often because it offers more targets, not because it matters
# more.

# %%
rows = []
for count, seed in itertools.product(RANDOM_COUNTS, SEEDS):
    flips = random_flips(parameter_sizes, count, seed=seed)
    with flipped_model(model, flips):
        value = measure()
    rows.append(
        {
            "policy": "random",
            "flips": count,
            "seed": seed,
            "perplexity": value,
            "ratio_to_baseline": value / baseline,
        }
    )
    print(
        f"random {count:>6} flips, seed {seed}: "
        f"ppl {value:>12.4f}  x{value / baseline:.4f}"
    )

# %% [markdown]
# ## Chosen faults — the top exponent bit of the largest weights
#
# The policy the literature finds effective. E1 explains why it works without knowing
# the model: that bit is zero in 100% of weights, so the flip always amplifies. The only
# extra knowledge needed is where the large weights are.

# %%
codes = {
    name: from_float32(parameter.detach().float().cpu().numpy().reshape(-1), BF16)
    for name, parameter in model.named_parameters()
}

for count in TARGETED_COUNTS:
    flips = largest_magnitude_flips(codes, count, bit=TOP_EXPONENT_BIT)
    with flipped_model(model, flips):
        value = measure()
    rows.append(
        {
            "policy": "targeted",
            "flips": count,
            "seed": -1,
            "perplexity": value,
            "ratio_to_baseline": value / baseline,
        }
    )
    print(
        f"targeted {count:>4} flips        : ppl {value:>12.4f}  x{value / baseline:.4f}"
    )

# %% [markdown]
# ## Results

# %%
rows.append(
    {
        "policy": "baseline",
        "flips": 0,
        "seed": -1,
        "perplexity": baseline,
        "ratio_to_baseline": 1.0,
    }
)
OUTPUT.mkdir(exist_ok=True)
with (OUTPUT / "e2-degradation.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"{len(rows)} measurements written to e2-degradation.csv")

# %% [markdown]
# ## The leverage figure
#
# How many random faults it takes to do the damage of one chosen fault. That ratio is
# the price of an attacker being able to pick the address.

# %%
targeted_one = next(r for r in rows if r["policy"] == "targeted" and r["flips"] == 1)
by_count = {}
for row in rows:
    if row["policy"] == "random":
        by_count.setdefault(row["flips"], []).append(row["ratio_to_baseline"])

chosen_ratio = targeted_one["ratio_to_baseline"]
print(f"one chosen flip costs x{chosen_ratio:.4f} of baseline perplexity\n")
for count in sorted(by_count):
    median = sorted(by_count[count])[len(by_count[count]) // 2]
    print(f"{count:>6} random flips: median x{median:.4f}")
