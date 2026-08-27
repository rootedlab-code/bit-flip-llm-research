# Environment notes

Findings about the machines this research runs on, kept because each one cost a failed
run to establish and none of them is written down anywhere else. They are facts about
tooling, not about bit flips, but a reproduction that cannot start is not a reproduction.

## Kaggle assigns whichever accelerator is free, and says nothing

Requesting a specific accelerator through the API does not work, and **fails silently**:
the CLI accepts any string for `machine_shape` without complaint, and the server ignores
values it does not recognise. Several plausible spellings — `GpuT4x2`, `nvidiaTeslaT4x2`,
`T4x2` — were accepted and none took effect; the session kept arriving with a P100.

The value the server actually recognises is **`NvidiaTeslaT4`**, learned by reading it
back after setting the accelerator once in the notebook's UI. With it in
`kernel-metadata.json`, subsequent pushes preserve the setting.

Two T4s give **29.1 GiB** at compute capability **7.5**.

## Recent PyTorch dropped Pascal, which is not the same as the card being unusable

A P100 is `sm_60`. The image's torch (2.10.0+cu128) carries
`['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']` — Pascal was dropped
to make room for Blackwell — so it cannot launch a single kernel on that card. The
failure surfaces at the first CUDA operation, not at import, which is why it appears
several cells later than its cause.

`torch==2.6.0+cu124` still carries `sm_50` and `sm_60` and runs on a P100. Measured
there: **~5,400 tokens/s** of scoring and 32 tokens/s of batch-1 generation, against
roughly 200 tokens/s of scoring on the four vCPUs of a CPU session.

**`torchvision` must be replaced along with torch.** The image's copy is compiled
against the torch being displaced, and leaving it in place makes `transformers` fail on
import with `operator torchvision::nms does not exist` — an error that mentions neither
torch nor the GPU. The working set is `torch==2.6.0`, `torchvision==0.21.0`,
`torchaudio==2.6.0` from the `cu124` index.

The notebooks read the compute capability with `nvidia-smi` **before importing torch**,
because torch cannot be swapped underneath a process that has already loaded it.

## Greedy generation reproduces within a run, only partly across runs

Two runs of the same notebook, same pinned model revisions, same seed, same greedy
configuration and the same pair of T4s, produced **byte-identical answers for only 341 of
600 probes**. The runs differed in one thing: a batch of 8 in arrival order against a
batch of 32 sorted by prompt length.

Three measurements pin this down, and together they make it a result rather than an
explanation:

| what was held fixed | what varied | answers identical |
|---|---|---|
| everything, across two sessions | nothing | **600 / 600** |
| model, seed, decoding, hardware | batch size and ordering | **341 / 600** |
| model, seed, decoding, hardware, prompts | padding width alone, 17 → 31 tokens | **6 / 8** |

The first row is what makes the second interpretable. Greedy generation **does** reproduce
exactly across sessions — the verdict tables of two separate runs came out with the same
SHA-256, `1df6f109…` — so the 341 is not ambient session noise with nowhere to be traced.
The third row varies the suspected cause on its own and moves 2 answers out of 8.

The mechanism: with left padding, the batch a prompt lands in determines the padded width
it is generated under, which changes the reduction order inside attention, which moves
logits in their last bits. Greedy decoding takes an argmax, and an argmax is discontinuous
— one near-tie flips and the rest of the answer follows it somewhere else. The bricked
model is the exception, 200 of 200 identical across configurations, because a collapsed
output has no near-ties left to flip.

The determinism the notebook asserts is real but narrower than it looks: the same prompt
twice in one batch, and batched against unbatched, agree. Neither of those changes the
padded width, so neither would have caught this. The notebook measures the padding effect
directly instead, once per run, and records the result in `e5-run-manifest.json`.

**What is not established:** the 600 of 600 holds hardware fixed as well — both sessions
drew two T4s. Nothing here says a run reproduces on a different accelerator, and given
that the cause is arithmetic ordering, the expectation should be that it does not.

Consequences, and they are not small:

- **A comparison across runs is valid only when the generation configuration is
  identical**, and that is a claim someone has to be able to check. It is why every run
  writes `e5-run-manifest.json`: batch size, ordering, decoding parameters, revisions,
  devices. Two runs whose manifests differ in any of those fields did not answer the same
  probes with the same text, and their verdict shares are not comparable.
- **A verdict share carried over from a run with a different configuration is not a
  baseline**, including a pre-registered threshold. Comparing against one mixes the effect
  under study with regeneration noise, in an unknown proportion and an unknown direction.
  That is exactly what happened to the 87.0% floor in `docs/e5-oracle-validation.md`.
- Comparisons **within** one run are unaffected, which is where E5's own quantities live:
  base, brick and abliterated are generated in a single pass, under one configuration.
- The verdict tables store a SHA-256 of every answer precisely so the matched subset can
  be recovered when the configurations do differ. `experiments/e5_compare_specs.py` does
  that, and reports the matched fraction rather than assuming it.

## Dependency floors are a portability defect, not caution

Declaring `numpy>=2.2` — which was simply the version on the author's machine — made pip
upgrade Kaggle's numpy 2.0.2 and break every preinstalled package compiled against it,
`transformers` included. Nothing in this project needs a recent numpy. Floors are now
set to what the code actually requires.

Likewise `requires-python = ">=3.13"` made pip refuse the package outright on Kaggle,
which ships 3.12. The real floor is 3.11, where `hashlib.file_digest` arrived.

## Package data has to be inside the package

The pre-registered oracle criteria lived beside the source tree, so installing from git
left them behind and `OracleSpec.load()` failed. A specification absent from the
environment where the experiments run is not registered anywhere that matters. They now
ship as `bitflip/spec/oracle_spec.yaml`, verified present in the built wheel.

## numpy has no bfloat16

Model weights cannot be converted to numpy for target selection. They do not need to be:
a bf16 weight promoted to float32 keeps its pattern in the top 16 bits, so the codes are
one shift away from the integer view — the same identity the injection relies on.
