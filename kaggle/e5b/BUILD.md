# Building the two E5b kernels

One source, two kernels. The whole design is 8.2 hours of two T4s, which does not fit a
single session, and `e5_spec.yaml` forbids splitting a numerator from its denominator —
so each arm runs as its own kernel carrying its own base and abliterated baselines. That
costs 55 minutes of duplicated generation and moves no registered threshold.

The arm is a constant in the source. The `random` notebook is produced by substituting
it, so the two pages cannot drift apart in anything except that one line:

```sh
python kaggle/build_notebook.py kaggle/e5b/e5b_silent_dealignment.py \
    kaggle/e5b/chosen/e5b-chosen.ipynb

sed 's/^ARM = "chosen"$/ARM = "random"/' kaggle/e5b/e5b_silent_dealignment.py \
    > /tmp/e5b_random.py
python kaggle/build_notebook.py /tmp/e5b_random.py kaggle/e5b/random/e5b-random.ipynb
```

## Push order, which the specification fixes rather than leaves to taste

`attestation.published_before_first_generation` requires the page carrying both digests
to be public **before** the first token exists. Pushing a Kaggle notebook runs it, so the
first push of each kernel must be a version that generates nothing:

1. `ATTEST_ONLY = True` — push both kernels. Each prints the two specification digests,
   generates zero tokens, and finishes in about two minutes. Kaggle records the execution
   time server-side, and that public output is the anchor.
2. `ATTEST_ONLY = False` — push `chosen` first. It is the shorter run at 2.7 hours, and
   the more informative: if the collapse policy bricks the model at dose 1 the way the
   amplifying one does, that is known before spending 6.4 hours on the random arm.
3. Push `random` last.
