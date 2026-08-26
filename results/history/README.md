# Superseded runs, kept as evidence

Every claim in the technical notes about something that went **wrong** points at a file
here. A defect described without the data that showed it is an assertion, and this
project does not make assertions.

| file | what it is | what it proves |
|---|---|---|
| `e2-degradation-run1-broken-targeting.csv` | first completed E2 run | Chosen faults report `ratio_to_baseline = 0.99999`, i.e. no damage at all. The targeting policy was selecting the largest-magnitude weights, which are exactly those with \|w\| ≥ 2 — the condition for bit 14 to be **already set**, so the flip divided by 2¹²⁸ instead of multiplying. See `docs/e2-degradation.md`. |
| `e2-degradation-run2-session-a.csv` | E2 with the corrected policy | Paired with `../e2-degradation.csv` (session B) it is the evidence for the inter-session noise figure: the identical configuration on different hardware moved baseline perplexity by 3.030 × 10⁻⁵ relative, while top-1 agreement was identical in all 30 configurations. |
| `e5-oracle-validation-spec-v1.csv` | oracle validation under spec v1 | Two of four corners failed. The aligned model's refusals were classified `degenerate` 63% of the time. |
| `e5-verdicts-spec-v1.csv` | per-probe verdicts, spec v1 | The diagnosis: all 63 misclassified refusals are 40–48 characters long, against a median of 139 for those classified correctly. The degeneracy floor of 40 letters sat inside the refusal length distribution — a refusal is short by construction. This is what the changelog in `src/bitflip/spec/oracle_spec.yaml` refers to. |

No generated text is stored in any of these files: the verdict tables carry the class,
the answer length and a truncated SHA-256, which is enough to re-verify a classification
and not enough to redistribute anything.
