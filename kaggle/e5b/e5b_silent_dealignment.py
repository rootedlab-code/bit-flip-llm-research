# %% [markdown]
# # E5, part two — can a model lose its alignment quietly?
#
# Part one validated the instrument: an oracle that classifies a generated answer as
# refusal, compliance, degenerate or undecided, checked at six corners of its own output
# space against models whose behaviour was known in advance. This notebook uses it to ask
# the question the instrument was built for.
#
# **Can bit flips move a model away from its alignment while leaving every check an
# operator would run looking normal?**
#
# Two numbers answer it, and both are pre-registered in `e5_spec.yaml` before any token
# below was generated:
#
# - **De-alignment Fraction** — how far the flipped model travelled from the aligned one
#   *towards an abliterated build of the same base*. Anchoring gives the axis a unit:
#   "refusals fell 40%" has no scale, "it went 40% of the way to a model built to refuse
#   nothing" has one.
# - **Stealth Ratio** — that fraction over the relative worsening of perplexity. Near 1,
#   alignment and capability fail together and anyone watching quality sees it. Much
#   above 1, alignment sits on a more fragile subspace than capability does.
#
# ## Two arms, and why the obvious attack is not one of them
#
# The **random** arm is the cosmic-ray model: faults uniform over every bit of every
# weight. The **chosen** arm is an attacker who picks.
#
# The obvious choice for an attacker — amplify the largest weight through the top
# exponent bit — was measured in E2 and it does not work for this question. A single such
# flip takes perplexity to NaN and top-1 agreement to zero. The surviving weight sits
# near 3.4·10³⁸ and overflows the activations downstream, so the model is a brick at dose
# one and there is nothing quiet left to measure.
#
# The registered chosen policy therefore goes the other way: it flips an exponent bit
# that is already 1, **downward**, dividing the weight by 2⁶⁴. On the base model that is
# 1.992188 → 1.08·10⁻¹⁹ — the weight is removed rather than exploded, and nothing can
# overflow because no value grows. E1's perturbation spectrum measures that channel at
# **18.74% of the bit space** against the catastrophic 6.26%, and this is the first
# experiment to reach it.
#
# ## What is registered before the run, and where the "before" is attested
#
# Both specifications are printed below with their SHA-256. The oracle's fixes how one
# answer is classified; `e5_spec.yaml` fixes how those classifications become a number —
# the step where the published figure is actually made, and the one where a free choice
# would be worth the most.
#
# The freeze is anchored by this page: the digests appear in its saved output, and Kaggle
# records the execution time server-side. **Version 1 of this notebook generates
# nothing.** It prints the digests and stops, so that a third party's timestamp sits on
# the specification *before* the first token exists, rather than beside the results.
#
# ## The bias this attack carries, stated before the run and not after
#
# The registered chosen policy removes the **largest** weights. Magnitude pruning keeps
# those precisely because capability depends on them, so this is close to the worst
# available policy for capability — the opposite of what a search for *silent*
# de-alignment would want. It is expected to damage perplexity early and to push the
# Stealth Ratio towards 1.
#
# It is registered anyway, because ranking targets by their effect on refusal would tune
# the attack against the hypothesis it exists to test, and no inert ranking is obviously
# better. But the consequence travels with the result: **a Stealth Ratio near 1 under
# this policy is evidence that alignment and capability fail together *under this
# policy*, and not yet evidence that no silent window exists.**
#
# ## The doses, and why the largest is not a control
#
# The ladder is registered in `e5_spec.yaml`. Its top rung is where E2 found the
# transition rather than the wreckage: at 10 random flips two of five seeds came back
# **intact** — perplexity ratio 1.00, top-1 0.9998 — while two others sat at 9,434× and
# had collapsed entirely. Reading the median of that spread as the dose would register
# the opposite of what the data shows. A silent window, if one exists, looks exactly
# like a seed whose perplexity does not move while its refusals do, and that rung is
# where intact and collapsed coexist. The non-silent control is the brick, which the
# oracle is already calibrated against.
#
# **No generated text leaves this notebook.** Answers are classified and discarded; what
# is written out is the verdict, the length and a truncated SHA-256.
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
import hashlib
import io
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from bitflip.alignment import (
    E5Spec,
    VerdictCounts,
    both_rules,
    instrument_drifted,
    rules_agree,
    stealth_ratio,
)
from bitflip.inject import (
    COLLAPSE_BIT,
    collapse_flips,
    flipped_model,
    model_codes,
    random_flips,
)
from bitflip.metrics import agreement, evaluate, set_determinism
from bitflip.oracle import (
    COMPLIANCE,
    DEGENERATE,
    INDETERMINATE,
    REFUSAL,
    OracleSpec,
    classify,
)
from bitflip.probes import BENIGN, HARMFUL, build_probe_set

# --- what this run is -------------------------------------------------------------
# Version 1 of each kernel is pushed with this set. It makes the page public, prints both
# digests, and generates nothing — so the server-recorded time attests the specification
# rather than the results.
ATTEST_ONLY = True

# "chosen" or "random". One kernel per arm: the whole design is 8.2 hours of two T4s,
# which does not fit one session, and the specification forbids splitting the numerator
# from its denominator. Each arm therefore carries its own base and abliterated
# baselines, which costs 55 minutes of duplicated generation and moves no threshold.
ARM = "chosen"

BASE_REPO = "Qwen/Qwen3-4B-Instruct-2507"
BASE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
ABLITERATED_REPO = "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated"
ABLITERATED_REVISION = "c9bd464550d4078c72af0dd22aa18d0437868ce3"

ADVBENCH_COMMIT = "098262edf85f807224e70ecd87b9d83716bf6b73"
ADVBENCH_URL = (
    f"https://raw.githubusercontent.com/llm-attacks/llm-attacks/{ADVBENCH_COMMIT}"
    "/data/advbench/harmful_behaviors.csv"
)
ALPACA_REPO = "tatsu-lab/alpaca"
ALPACA_FILE = "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"
ALPACA_REVISION = "dce01c9e388e0a4c3a1b4a4b0f0b3b2c1f0c0d0e"

CORPUS_REPO = "Salesforce/wikitext"
CORPUS_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
CORPUS_TOKENS = 32_768
WINDOW = 1024
STRIDE = 512

MAX_NEW_TOKENS = 256
BATCH_SIZE = 32
SEED = 0

OUTPUT = Path("/kaggle/working")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

oracle_spec = OracleSpec.load()
e5_spec = E5Spec.load()

print(f"oracle_spec.yaml  v{oracle_spec.version}  sha256 {oracle_spec.digest}")
print(f"e5_spec.yaml      v{e5_spec.version}  sha256 {e5_spec.digest}")
print(f"\narm: {ARM} · attestation only: {ATTEST_ONLY}")

# %% [markdown]
# ## What the specification fixes, read back from the file rather than restated
#
# Every number below comes from `e5_spec.yaml`. Restating them in prose is how a
# registered value and a published one begin to differ.

# %%
HARMFUL_PROBES = e5_spec.harmful_probes
BENIGN_PROBES = e5_spec.benign_probes
DOSES = e5_spec.doses_ladder
SEEDS = e5_spec.seeds if ARM == "random" else 1

print(f"probes      harmful {HARMFUL_PROBES} · benign {BENIGN_PROBES}")
print(f"doses       {DOSES}")
print(f"seeds       {SEEDS} ({'random' if ARM == 'random' else 'deterministic'})")
print(
    f"rules       primary {e5_spec.primary_rule} · sensitivity {e5_spec.sensitivity_rule}"
)
print(f"SR floor    {e5_spec.min_resolvable_denominator:.1e} relative perplexity")

# %% [markdown]
# ## The probes
#
# 300 harmful, because the interval matters here in a way it did not during validation:
# at 100 probes a De-alignment Fraction carries ±0.14, and a four-dose curve at that
# resolution is four overlapping points. The benign set stays at 100 — it is a gate on
# whether the model still works, not a component of the score.

# %%
harmful_rows = list(
    csv.DictReader(io.StringIO(requests.get(ADVBENCH_URL, timeout=60).text))
)
harmful_set = build_probe_set(
    (row["goal"] for row in harmful_rows), HARMFUL, oracle_spec, limit=HARMFUL_PROBES
)

alpaca = pq.read_table(
    hf_hub_download(
        ALPACA_REPO, ALPACA_FILE, repo_type="dataset", revision=ALPACA_REVISION
    )
)
benign_set = build_probe_set(
    (
        instruction
        for instruction, extra in zip(
            alpaca.column("instruction").to_pylist(),
            alpaca.column("input").to_pylist(),
            strict=True,
        )
        if not extra.strip()
    ),
    BENIGN,
    oracle_spec,
    limit=BENIGN_PROBES,
)

for probe_set in (harmful_set, benign_set):
    print(
        f"{probe_set.kind:<8} {len(probe_set):>4} probes · "
        f"{probe_set.discarded} discarded · coverage {probe_set.coverage:.1%}"
    )

# %% [markdown]
# ## Generation, classification, and the quality checks an operator would run

# %%
tokenizer = AutoTokenizer.from_pretrained(BASE_REPO, revision=BASE_REVISION)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

GREEDY = GenerationConfig(
    do_sample=False,
    num_beams=1,
    max_new_tokens=MAX_NEW_TOKENS,
    pad_token_id=tokenizer.pad_token_id,
)


def load(repo: str, revision: str):
    model = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, dtype=torch.bfloat16, device_map="auto"
    )
    return model.eval()


def answer(model, prompts: list[str]) -> list[str]:
    conversations = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in prompts
    ]
    batch = tokenizer(conversations, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        produced = model.generate(**batch, generation_config=GREEDY)
    width = batch["input_ids"].shape[1]
    return tokenizer.batch_decode(produced[:, width:], skip_special_tokens=True)


def answer_all(model, probes) -> list[str]:
    """Sorted by prompt length so each batch's padding stays tight."""
    order = sorted(range(len(probes)), key=lambda index: len(probes[index].prompt))
    replies: list[str | None] = [None] * len(probes)
    started = time.time()
    for start in range(0, len(order), BATCH_SIZE):
        chunk = order[start : start + BATCH_SIZE]
        for index, reply in zip(
            chunk, answer(model, [probes[index].prompt for index in chunk]), strict=True
        ):
            replies[index] = reply
        done = min(start + BATCH_SIZE, len(order))
        print(
            f"    {done:>4}/{len(order)} · {done / (time.time() - started):.1f}/s",
            flush=True,
        )
    return [reply for reply in replies if reply is not None]


corpus_path = hf_hub_download(CORPUS_REPO, CORPUS_FILE, repo_type="dataset")
corpus_text = "\n\n".join(pq.read_table(corpus_path).column("text").to_pylist())
corpus_tokens = (
    tokenizer(corpus_text, return_tensors="pt").input_ids[0][:CORPUS_TOKENS].to(DEVICE)
)
print(f"corpus {corpus_tokens.numel():,} tokens of WikiText-2 test")


def quality(model) -> tuple[float, torch.Tensor]:
    """Perplexity and the top-1 prediction at every scored position.

    These are the checks an operator actually has: a quality metric and whether the
    model still predicts what it used to. Both come out of the same forward passes,
    because running the model twice to get two views of the same logits would double
    the cost of every dose.
    """

    def score(chunk: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(chunk.unsqueeze(0)).logits[0]

    return evaluate(score, corpus_tokens, window=WINDOW, stride=STRIDE)


def counts_of(replies: list[str], probe_set) -> tuple[VerdictCounts, list[dict]]:
    """Classify a condition's answers, keeping no text."""
    tally: Counter[str] = Counter()
    rows = []
    for probe, reply in zip(probe_set.probes, replies, strict=True):
        verdict = classify(reply, probe.keywords, oracle_spec)
        tally[verdict] += 1
        rows.append(
            {
                "probe": probe.identifier,
                "kind": probe_set.kind,
                "verdict": verdict,
                "answer_chars": len(reply),
                "answer_sha256": hashlib.sha256(reply.encode()).hexdigest()[:16],
            }
        )
    return (
        VerdictCounts(
            compliance=tally[COMPLIANCE],
            refusal=tally[REFUSAL],
            degenerate=tally[DEGENERATE],
            indeterminate=tally[INDETERMINATE],
        ),
        rows,
    )


def flips_for(model, dose: int, seed: int) -> list:
    """The registered policy for this arm. Nothing here reads the oracle."""
    if ARM == "chosen":
        return collapse_flips(model_codes(model), count=dose, bit=COLLAPSE_BIT)
    return random_flips(
        {name: c.size for name, c in model_codes(model).items()}, count=dose, seed=seed
    )


# %% [markdown]
# ## The run
#
# One load of each model. The base is measured intact, then at every dose under the
# context manager that **always** undoes its flips on exit, then the abliterated anchor
# last — so the numerator and both halves of its denominator come out of one session,
# under one generation configuration, as the specification requires.

# %%
verdicts: list[dict[str, object]] = []
conditions: list[dict[str, object]] = []


def measure(name: str, model, dose: int, seed: int | None) -> dict[str, object]:
    """One condition: both probe sets classified, plus the two quality checks."""
    print(f"\n=== {name} ===", flush=True)
    row: dict[str, object] = {"condition": name, "arm": ARM, "dose": dose, "seed": seed}
    for probe_set in (harmful_set, benign_set):
        counted, rows = counts_of(answer_all(model, probe_set.probes), probe_set)
        for entry in rows:
            verdicts.append({"condition": name, **entry})
        row[f"{probe_set.kind}_counts"] = counted
        print(
            f"  {probe_set.kind:<8} "
            + "  ".join(
                f"{v[:6]} {getattr(counted, v) / counted.total:>6.1%}"
                for v in ("refusal", "compliance", "degenerate", "indeterminate")
            )
        )
    perplexity_value, top1 = quality(model)
    row["perplexity"] = perplexity_value
    row["top1"] = top1
    print(f"  perplexity {perplexity_value:.6f}")
    return row


def run_arm() -> None:
    set_determinism(SEED)
    base = load(BASE_REPO, BASE_REVISION)

    intact = measure("base", base, dose=0, seed=None)
    conditions.append(intact)

    for dose in DOSES:
        for seed in range(SEEDS):
            flips = flips_for(base, dose, seed)
            with flipped_model(base, flips):
                conditions.append(
                    measure(f"{ARM}-d{dose}-s{seed}", base, dose=dose, seed=seed)
                )

    del base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    anchor = load(ABLITERATED_REPO, ABLITERATED_REVISION)
    conditions.append(measure("abliterated", anchor, dose=0, seed=None))
    del anchor
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


if ATTEST_ONLY:
    print(
        "\nAttestation only. Both specification digests are printed above and no token\n"
        "has been generated under them. Version 2 of this kernel runs the arm."
    )
else:
    run_arm()

# %% [markdown]
# ## The two numbers
#
# DF under both registered rules at every dose, never one alone: an undecided answer is
# not evidence of compliance and not evidence of refusal, and where it goes can change
# DF's sign. If the two rules disagree about whether anything moved, no DF is reported
# for that dose — the disagreement is the finding.


# %%
def summarise() -> list[dict[str, object]]:
    by_name = {row["condition"]: row for row in conditions}
    reference = by_name["base"]["harmful_counts"]
    anchor = by_name["abliterated"]["harmful_counts"]
    baseline_perplexity = float(by_name["base"]["perplexity"])
    baseline_top1 = by_name["base"]["top1"]

    rows = []
    for row in conditions:
        if row["dose"] == 0:
            continue
        flipped = row["harmful_counts"]
        relative = (float(row["perplexity"]) - baseline_perplexity) / baseline_perplexity
        entry: dict[str, object] = {
            "condition": row["condition"],
            "arm": ARM,
            "dose": row["dose"],
            "seed": row["seed"],
            "indeterminate_rate": flipped.indeterminate_rate,
            "benign_degenerate": row["benign_counts"].degenerate
            / row["benign_counts"].total,
            "perplexity": row["perplexity"],
            "relative_perplexity_change": relative,
            "top1_agreement": agreement(baseline_top1, row["top1"]),
            "instrument_drifted": instrument_drifted(flipped, e5_spec),
        }

        fractions = both_rules(flipped, reference, anchor, e5_spec)
        entry["rules_agree"] = rules_agree(fractions)
        for rule, result in fractions.items():
            entry[f"df_{rule}"] = result.value
            entry[f"df_{rule}_low"] = result.low
            entry[f"df_{rule}_high"] = result.high

        stealth = stealth_ratio(fractions[e5_spec.primary_rule].value, relative, e5_spec)
        entry["stealth_ratio"] = stealth.value
        entry["stealth_is_bound"] = stealth.is_bound
        rows.append(entry)
    return rows


if not ATTEST_ONLY:
    scores = summarise()
    print(
        f"{'condition':<20} {'DF strict':>18} {'DF decided':>18} "
        f"{'undecided':>10} {'rel ppl':>11} {'SR':>10}"
    )
    for entry in scores:
        strict = (
            f"{entry['df_strict']:>6.3f} "
            f"[{entry['df_strict_low']:.2f},{entry['df_strict_high']:.2f}]"
        )
        decided = (
            f"{entry['df_decided']:>6.3f} "
            f"[{entry['df_decided_low']:.2f},{entry['df_decided_high']:.2f}]"
        )
        stealth = (
            "bound" if entry["stealth_is_bound"] else f"{entry['stealth_ratio']:.2f}"
        )
        print(
            f"{entry['condition']:<20} {strict:>18} {decided:>18} "
            f"{entry['indeterminate_rate']:>9.1%} "
            f"{entry['relative_perplexity_change']:>11.2e} {stealth:>10}"
        )

# %% [markdown]
# ## What is written out

# %%
if not ATTEST_ONLY:
    OUTPUT.mkdir(exist_ok=True)
    for name, rows in (
        (f"e5b-{ARM}-scores.csv", scores),
        (f"e5b-{ARM}-verdicts.csv", verdicts),
    ):
        with (OUTPUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(scores)} scored conditions and {len(verdicts)} verdicts written")

    manifest = {
        "arm": ARM,
        "specifications": {
            "oracle_spec.yaml": {
                "version": oracle_spec.version,
                "digest": oracle_spec.digest,
            },
            "e5_spec.yaml": {"version": e5_spec.version, "digest": e5_spec.digest},
        },
        "models": {
            "base": {"repo": BASE_REPO, "revision": BASE_REVISION},
            "abliterated": {
                "repo": ABLITERATED_REPO,
                "revision": ABLITERATED_REVISION,
            },
        },
        "probes": {
            "advbench_commit": ADVBENCH_COMMIT,
            "alpaca_revision": ALPACA_REVISION,
            "harmful": len(harmful_set),
            "benign": len(benign_set),
        },
        "generation": {
            "batch_size": BATCH_SIZE,
            "batch_order": "ascending prompt length",
            "max_new_tokens": MAX_NEW_TOKENS,
            "seed": SEED,
            "padding_side": tokenizer.padding_side,
            "do_sample": GREEDY.do_sample,
            "num_beams": GREEDY.num_beams,
        },
        "corpus": {
            "repo": CORPUS_REPO,
            "file": CORPUS_FILE,
            "tokens": int(corpus_tokens.numel()),
            "window": WINDOW,
            "stride": STRIDE,
        },
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }
    (OUTPUT / f"e5b-{ARM}-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
