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
# **About the accelerator.** Kaggle hands out whichever GPU is free. The P100 it often
# assigns is `sm_60`, and recent PyTorch wheels dropped Pascal to make room for
# Blackwell — the preinstalled build carries `sm_70` upwards only, so it cannot launch
# a single kernel on that card. This is a property of the build, not of the hardware:
# `torch==2.6.0+cu124` still ships `sm_60` and runs on it at about 5,400 tokens/s of
# scoring, against roughly 200 on the four vCPUs of a CPU session.
#
# The first cell therefore reads the compute capability *before* importing torch, and
# installs the matching stack when the card is older than the shipped build supports.
# `torchvision` has to come along: the image's copy is compiled against the torch being
# replaced, and leaving it in place makes `transformers` fail on import with
# `operator torchvision::nms does not exist`.
#
# **Numerical note.** A GPU without native `bfloat16`, such as a T4, also works
# here. The fault is injected into
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

print(sys.version)

LEGACY_TORCH = ("torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0")
LEGACY_INDEX = "https://download.pytorch.org/whl/cu124"
OLDEST_SHIPPED_CAPABILITY = 7.0


def compute_capability() -> float | None:
    """The GPU's compute capability, read without importing torch.

    It has to happen before the import: torch cannot be swapped underneath a process
    that has already loaded it, and by the time torch could tell us the card is
    unsupported it is too late to do anything about it.
    """
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        return None
    return float(probe.stdout.strip().splitlines()[0])


capability = compute_capability()
print(f"compute capability: {capability}")
if capability is not None and capability < OLDEST_SHIPPED_CAPABILITY:
    print(f"older than the shipped build supports; installing {LEGACY_TORCH[0]}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            *LEGACY_TORCH,
            "--index-url",
            LEGACY_INDEX,
        ],
        check=True,
    )

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "git+https://github.com/rootedlab-code/bit-flip-llm-research",
    ],
    check=True,
)

# %%
import csv
import itertools
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from bitflip.codec import BF16, from_float32
from bitflip.damage import INTACT, damage_class
from bitflip.inject import (
    TOP_EXPONENT_BIT,
    flipped_model,
    largest_magnitude_flips,
    random_flips,
)
from bitflip.metrics import agreement, evaluate, set_determinism

BASE_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
CORPUS_REPO = "Salesforce/wikitext"
CORPUS_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"

WINDOW = 1024
STRIDE = 512
VOCABULARY = 151_936  # Qwen2.5; a uniform output has exactly this perplexity
CORPUS_TOKENS_GPU = 32_768
CORPUS_TOKENS_CPU = 8_192
RANDOM_COUNTS = (1, 10, 100, 1_000, 10_000)
TARGETED_COUNTS = (1, 2, 5, 10)
SEEDS = (0, 1, 2, 3, 4)

OUTPUT = Path("/kaggle/working")


def usable_cuda() -> tuple[bool, str]:
    """Whether this GPU can actually run this torch build.

    Kaggle hands out whichever accelerator is free, and a P100 (sm_60) paired with a
    torch built for sm_70+ fails at the first kernel launch rather than at import. A
    notebook meant to be re-run by strangers should say so and carry on, not crash.
    """
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    supported = torch.cuda.get_arch_list()
    if f"sm_{major}{minor}" not in supported:
        return False, f"{name} is sm_{major}{minor}; this torch supports {supported}"
    return True, f"{name} (sm_{major}{minor})"


GPU_OK, GPU_REASON = usable_cuda()
DEVICE = "cuda" if GPU_OK else "cpu"
CORPUS_TOKENS = CORPUS_TOKENS_GPU if GPU_OK else CORPUS_TOKENS_CPU
if not GPU_OK:
    torch.set_num_threads(os.cpu_count() or 4)
print(f"device: {DEVICE} — {GPU_REASON}")
print(
    f"numpy {np.__version__} · torch {torch.__version__} · "
    f"corpus {CORPUS_TOKENS:,} tokens"
)

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
#
# The corpus is shortened when this falls back to CPU. That makes the absolute figure
# noisier, but every condition is measured on the same tokens, so the ratios that carry
# the conclusions are unaffected.

# %%
import pyarrow.parquet as pq  # noqa: E402

corpus_path = hf_hub_download(CORPUS_REPO, CORPUS_FILE, repo_type="dataset")
text = "\n\n".join(pq.read_table(corpus_path).column("text").to_pylist())
token_ids = tokenizer(text, return_tensors="pt").input_ids[0][:CORPUS_TOKENS].to(DEVICE)
print(f"{token_ids.numel():,} tokens of corpus")


def score(chunk: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(chunk.unsqueeze(0)).logits[0]


def measure() -> tuple[float, torch.Tensor]:
    return evaluate(score, token_ids, window=WINDOW, stride=STRIDE)


# %% [markdown]
# ## Baseline, and the null control
#
# Two runs of the same configuration must agree digit for digit, and zero flips must
# reproduce the baseline exactly. Without both, no later difference is attributable.

# %%
baseline, baseline_predictions = measure()
repeat, repeat_predictions = measure()
print(f"baseline perplexity : {baseline!r}")
print(f"second run          : {repeat!r}")
print(f"identical           : {baseline == repeat}")

with flipped_model(model, []):
    null_control, null_predictions = measure()
print(f"zero flips          : {null_control!r}  identical: {null_control == baseline}")
assert baseline == repeat, "the measurement is not deterministic; stop here"
assert null_control == baseline, "the null control does not reproduce the baseline"
assert agreement(baseline_predictions, repeat_predictions) == 1.0

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
        value, predictions = measure()
    kept = agreement(baseline_predictions, predictions)
    verdict = damage_class(value, kept, VOCABULARY)
    rows.append(
        {
            "policy": "random",
            "flips": count,
            "seed": seed,
            "perplexity": value,
            "ratio_to_baseline": value / baseline,
            "top1_agreement": kept,
            "damage_class": verdict,
        }
    )
    print(
        f"random {count:>6} flips, seed {seed}: ppl {value:>12.4f}  "
        f"agreement {kept:.4f}  {verdict}"
    )

# %% [markdown]
# ## Chosen faults — the top exponent bit, on a weight the flip can actually amplify
#
# The naive reading of E1 is a trap, and the first run of this notebook fell into it.
# "Bit 14 is zero in 100% of weights, so hit the largest weight" is wrong: the largest
# weights are precisely those with |w| ≥ 2, which is exactly the condition for bit 14 to
# be **already set**. Flipping it there divides by 2¹²⁸ and does nothing. In this model
# all 1000 largest weights are in that category.
#
# A second trap sits behind the first: among the weights the flip does amplify, the
# largest overflow. A weight in [1, 2) times 2¹²⁸ exceeds the bfloat16 maximum and
# becomes NaN — total destruction rather than corruption. `largest_magnitude_flips`
# therefore picks the largest weight the flip amplifies **while staying finite**.

# %%
codes = {
    name: from_float32(parameter.detach().float().cpu().numpy().reshape(-1), BF16)
    for name, parameter in model.named_parameters()
}

for count in TARGETED_COUNTS:
    flips = largest_magnitude_flips(codes, count, bit=TOP_EXPONENT_BIT)
    with flipped_model(model, flips):
        value, predictions = measure()
    kept = agreement(baseline_predictions, predictions)
    verdict = damage_class(value, kept, VOCABULARY)
    rows.append(
        {
            "policy": "targeted",
            "flips": count,
            "seed": -1,
            "perplexity": value,
            "ratio_to_baseline": value / baseline,
            "top1_agreement": kept,
            "damage_class": verdict,
        }
    )
    print(
        f"targeted {count:>4} flips        : ppl {value:>12.4f}  "
        f"agreement {kept:.4f}  {verdict}"
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
        "top1_agreement": 1.0,
        "damage_class": INTACT,
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
#
# Perplexity saturates at the vocabulary size — a destroyed model outputs a uniform
# distribution, and every degree of destruction then looks identical. Top-1 agreement
# with the undamaged model does not saturate, so it is reported alongside.
#
# The ratio to baseline is *not* used in this summary: chosen faults overflow the
# arithmetic and produce NaN, and a leverage figure computed from NaN is not a figure.
# The comparison runs on damage classes, which are defined for every outcome.

# %%
targeted_one = next(r for r in rows if r["policy"] == "targeted" and r["flips"] == 1)
by_count = {}
for row in rows:
    if row["policy"] == "random":
        by_count.setdefault(row["flips"], []).append(row["damage_class"])

print(f"one chosen flip: {targeted_one['damage_class']}\n")
print("share of runs that left the model usable, by fault count:")
for count in sorted(by_count):
    verdicts = by_count[count]
    intact = sum(1 for v in verdicts if v == INTACT)
    print(f"{count:>6} random flips: {intact}/{len(verdicts)} intact")

print("\nevery measurement, by damage class:")
for row in sorted(rows, key=lambda r: (r["policy"], r["flips"], r["seed"])):
    print(
        f"  {row['policy']:<9} {row['flips']:>6} flips seed {row['seed']:>2}: "
        f"agreement {row['top1_agreement']:.4f}  {row['damage_class']}"
    )
