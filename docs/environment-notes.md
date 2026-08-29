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

## `grep` here respects `.gitignore`, so "no occurrences" can mean "did not look"

The `grep` on the shell used to develop this project is not `/usr/bin/grep`: it is a
wrapper around `ugrep --ignore-files`, which reads `.gitignore` and skips what it
excludes. Run from the repository root, a search therefore never enters `docs/public/`
or the `PLAN-*.md` files.

```
grep         -rn "twelve orders" .   ->  0 occurrences
command grep -rn "twelve orders" .   ->  4, including docs/public/paper.md:767
```

Both commands are correct; only one of them answers the question "is this figure still
anywhere". The failure mode is the one this file exists to collect: **the tool reports
success by finding nothing, and finding nothing is what a passing check looks like.** It
is the same shape as the Kaggle CLI accepting any accelerator name and the server
silently ignoring the ones it does not know.

It bites hardest exactly where it matters most here, because the write-up held back from
git — the one place a retracted figure can survive unnoticed and then be published — is
excluded by construction.

Use an explicit path (`grep -rn x docs/`) or `command grep` before writing "verified" or
"no residues" anywhere. A sweep run from the root and reported as complete is a claim
about the tracked files only, and should say so.

## Searching for a phrase assumes it sits on one line, and prose wraps

The note above fixes a search that was reading `.gitignore`. It does not fix the second
way the same claim goes wrong, which bit twice in one afternoon *after* the first was
understood:

While the README still carried the claim that has since been retracted:

```
command grep -c "harmless by construction" README.md   ->  0
command grep -c "harmless"                 README.md   ->  3
```

The phrase was there. It was split across two lines, because the file is wrapped at 90
columns and the words fell where they fell. Every sweep for a retired phrase in this
project runs against Markdown that wraps, so **searching for the phrase finds the
occurrences that happen to be short enough**, and reports the rest as absent.

The two blind spots are independent. A search can use `command grep` and still be wrong
this way, which is how the second one survived the fix for the first.

Search for the **shortest distinctive word** of the claim, not the claim: `harmless`, not
`harmless by construction`; `twelve`, not `twelve orders of magnitude`.

And a third level, which the word does not reach either: searching for the retired
**formulation** finds only the pages that used it. A page can give the same wrong half of
the picture in its own words and never contain the phrase at all — one did, and it was
found by searching the **subject**, `11-13`, rather than anything the retraction said.
So a retraction has three sweeps, not one: the phrase, its shortest distinctive word, and
the thing the claim was about. The extra hits
cost a moment to read; the missing ones cost a published figure. Where a phrase must be
matched as a whole, join the lines first (`command grep -z`, or a reader that strips the
wrapping) rather than trusting the line-oriented answer.

## Two sessions on one working tree: a clean `git diff` can mean the opposite of clean

When more than one session edits the same checkout, the obvious guard — look at
`git status --short`, and if a path is dirty, wait — has a case that reads backwards.
Reproduced in an isolated repository:

| what happened | `git status --short` | then `git diff --quiet <path>` |
|---|---|---|
| the file was only touched | clean | 0 |
| the file was rewritten with identical bytes | clean | 0 |
| **another session edited it, then committed between the two commands** | ` M path` | **0** |

The first two are harmless: `git status` refreshes the index cache itself, so a bare
`touch` never shows as dirty. The third is the one that matters. After the owner commits,
the working tree matches HEAD again, so `diff --quiet` reports no difference — while the
file on disk is not the file that was read a moment earlier and **HEAD has moved**.

A guard phrased as "exit 0 means it was only a timestamp, carry on" therefore says carry
on in exactly the case where the content changed underneath. The failure has the shape
this file collects: the check passes, and passing is indistinguishable from nothing
having happened.

The guard that holds: if `git status --short` showed ` M` on a path that is not yours,
exit 1 from `git diff --quiet` means the edit is still in flight — wait. Exit 0 means it
landed and was committed while you were looking — **re-read the file and check
`git log --oneline -1` before doing anything with what you read.** Never treat the second
case as permission to proceed.

A sharper consequence, learned by having it happen while writing this entry. The sessions
share the `.git` directory, not only the checkout, so **the history is one object**:

- another session's `git pull` moves *your* HEAD, without your having run anything;
- another session's `git push` publishes *your* local commits, because they are already
  in the shared history — a commit is public as soon as anyone pushes, not when its
  author decides to.

The second is the one to hold on to. A commit made locally, intended to be reviewed
before going out, leaves the machine on the next push by anyone. Where publication is a
decision — and in this project the public repository is a decision, with a
[declared boundary](../README.md) around what leaves it — that has to be planned for
rather than discovered: commit only what is ready to be public, or do not commit yet.

The visible symptom is a push rejected with `cannot lock ref ... is at X but expected Y`,
where X is a commit that already contains your own work.

## The Kaggle CLI ignores the `id` field and slugs the title instead

`kernel-metadata.json` declares an `id`. The CLI does not use it. Pushed with
`"id": "seb001010/bit-flip-e5b-chosen"` and
`"title": "bit-flip e5b silent dealignment (chosen arm)"`, it warns —

```
Your kernel title does not resolve to the specified id.
```

— and then creates the kernel at the **title's** slug,
`bit-flip-e5b-silent-dealignment-chosen-arm`. The push succeeds, the URL it prints is the
title's, and the declared id names nothing.

How it bites is worse than the mismatch:

```
kaggle kernels status seb001010/bit-flip-e5b-chosen
  -> Permission 'kernels.get' was denied ... It can also occur if the notebook is private.
```

That reads as a permissions or visibility problem. It is a kernel that **does not exist**.
Anyone trusting the message would conclude their page was private when it had never been
created. Same family as `machine_shape`: the field is accepted, the server does something
else, and the error points somewhere unrelated. Give a kernel a title whose slug *is* the
id, or align the id to the title's slug after the first push.

## An identifier is copied from its machine source, never from prose that truncates it

The E5b notebook was written with a dataset revision that does not exist. The eight
characters came from a technical note — `Alpaca @dce01c9b` — and the remaining
thirty-two were reconstructed, wrongly, including the eighth character:

```
results/e5-run-manifest.json   dce01c9b08f87459cf36a430d809084718273017   full, versioned
docs/e5-oracle-validation.md   dce01c9b                                   truncated for a reader
what was written               dce01c9e388e0a4c3a1b4a4b0f0b3b2c1f0c0d0e   fabricated
```

The true value was one file away, in the manifest that exists precisely so that a run
never has to be reconstructed. The source consulted was the one formatted for a human
instead of the one written for a machine. **An eight-character prefix in a document is a
citation, not a datum.**

This is not the ordinary lapse that more care would prevent, and that is the part worth
recording. **A forty-character hash is the most plausible-looking value there is.** It
contains nothing that could seem wrong, so nothing triggers a check. It is the limiting
case of the argument the two entries above make: verification cannot depend on a figure
looking strange, because the ones that matter never do.

## An attestation run pays twice

`e5_spec.yaml` requires the notebook carrying the specification digests to be public
*before* the first token exists, and pushing a Kaggle notebook runs it. So version 1 of
each E5b kernel prints the digests and generates nothing. That was its whole purpose.

It did a second job nobody designed. **A run that resolves every external identifier
while computing nothing is a dry run of the identifiers.** The fabricated revision above
surfaced there, as a 404 two minutes in, rather than after the 2.7 hours of generation
the chosen arm would have spent before reaching the same line.

The generalisation outlives the attestation requirement: a pass that resolves every model
revision, dataset revision and corpus pin without generating anything costs minutes and
moves configuration failures ahead of the expense. It is worth doing in any notebook that
downloads before it computes.

## A nested repository swallows a file-by-file add without a word

`docs/public/` is held out of the project's history on purpose and now carries a
repository of its own, with no remote, so that edits there can be diffed and reverted
without any path to publication. That solves the problem it was made for.

It leaves a trap for whoever later decides some of those files should be tracked after
all, and the dangerous half is not the obvious one. Measured on a throwaway pair of
repositories rather than reasoned about:

| command | output | what is staged |
|---|---|---|
| `git add sub/` | eleven lines of warning, naming `git rm --cached sub` | mode `160000`: a **gitlink** |
| `git add sub/paper.md` | **nothing at all**, exit 0 | **nothing at all** |

The directory form is loud. It says what it is doing and how to undo it, and the gitlink
it stages would at least show up as a change. The **file** form is the silent one: no
output, exit status 0, and the file it was asked to add stays untracked. Nothing in the
terminal distinguishes it from having worked.

That is the row that belongs in this file, and it is worse than it looks, because the
file-by-file add is exactly what a careful person would use. Releasing selected documents
from a held-back directory means naming them one at a time, which is precisely the form
that fails without saying so.

**The remedy has an order, and the obvious first step is wrong.** De-nesting first
(`rm -rf docs/public/.git`) destroys the only history that directory has ever had — the
thing it was created to protect. Save it first:

```sh
git -C docs/public bundle create /path/to/paper-history.bundle --all
rm -rf docs/public/.git
git add docs/public/<file>            # now stages the file
git status --short                    # two file rows, not one directory row
```

The bundle is a single file that `git clone` reads back into a full repository, verified
by round trip. Confirming `git status` afterwards is not ceremony: it is the only thing
that distinguishes the fixed case from the silent failure above.

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
