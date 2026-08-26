# E3 — The critical surface of a quantized model

Subject: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` @`9217f5d`, file `q4_k_m` (491,400,032 bytes).
Comparison: the same model in bf16 (E1).
Raw data: `results/e3-gguf-bit-census.csv`, `results/e3-gguf-scale-fragility.csv`,
`results/e3-comparison.csv`.

## Method

The same exact method as E1, applied to a different population: not the weights, but
the **scales**. The parser validates itself by closing the arithmetic on the file —
header 5,947,744 bytes + data = 491,400,032 bytes, exactly. The fp16 scales are
extracted by reshaping each tensor's byte range into (blocks × bytes per block) and
taking the column; their histogram then goes through the same per-bit computation.

## First result, unexpected: the file is hardly K-quantized at all

| type | tensors | ne[0] divisible by 256 |
|---|---|---|
| Q5_0 | 133 | no |
| Q8_0 | 13 | no |
| Q6_K | 12 | yes |
| Q4_K | 12 | yes |

The separation is **perfect, without exceptions**: K-quants require rows that are
multiples of 256, Qwen2.5-0.5B's hidden size is 896 = 3.5 × 256, and only `ffn_down`
(4864 = 19 × 256) meets the requirement. Everything else falls back to legacy
quantization in blocks of 32.

Practical consequence: a file labelled `q4_k_m` **does not describe the format of its
weights**. Anyone designing a defence on the assumption of 256-element super-blocks
gets it wrong on 146 tensors out of 170.

## Bit census by function

| role | bits | share |
|---|---|---|
| quants | 3,563,012,096 | 91.745% |
| **fp16 scales** | **272,556,032** | **7.018%** |
| integer scales | 45,760,512 | 1.178% |
| float (norms, biases) | 2,289,664 | 0.059% |

## The scales share the weights' universal weakness

In the scales of 32-weight blocks **and** in those of 256-weight blocks, bit 14 is zero
in **100.00%** of cases — the same property E1 found in the bf16 weights, with the same
consequence: whoever hits that bit does not need to know what value they are hitting.

In fp16 the top exponent bit multiplies by 2¹⁶ = 65,536 (against 2¹²⁸ in bf16, see
`tests/test_codec.py`). Exactly **one bit in sixteen** of every scale is catastrophic:
6.2500%, that is one per scale, always the same one.

## The comparison that answers the question

| format | total bits | catastrophic bits | share | radius | weights lost per random flip |
|---|---|---|---|---|---|
| bf16 safetensors | 7,904,524,288 | 494,787,536 | 6.2595% | 1 weight | 0.062595 |
| gguf q4_k_m | 3,883,618,304 | 17,034,752 | 0.4386% | 40.1 weights | **0.175711** |

Quantization cuts the share of catastrophic bits by **14×** — and multiplies the blast
radius of each by **40×**, because one scale governs its whole block. The balance:

> for the same random fault, the quantized file loses **2.807 times more weights** than
> the bf16 one.

Quantization does not protect: it **concentrates**. It moves the risk from a large
population of barely dangerous bits to a small population of very dangerous ones, and
the concentration worsens the expected value rather than improving it.

## Limits of this conclusion — stated

**The two weight populations are not the same.** The safetensors file declares
494,032,768 parameters, the GGUF 630,167,424: the difference, 136,134,656 =
151,936 × 896, is the embedding that GGUF **unties** and materialises a second time as
`output.weight`. The comparison above normalises per bit rather than per weight, so it
holds; but "weights lost" refers to two sets of different size, and that must be kept
in mind.

"Weights lost" is not "damage to the model", and the difference matters in three ways:

1. **The multiplier is not the same.** A hit bf16 weight is multiplied by 2¹²⁸, a
   quantized one by 2¹⁶: twelve orders of magnitude of difference in severity per
   weight. The count above treats them as equivalent because both cross the threshold —
   a debatable choice, and one that is stated.
2. **Quantized damage is correlated.** The 32 weights of a block are contiguous in the
   same row; 40 weights scattered across different tensors are another matter. Which of
   the two destructions weighs more on the output is not something this experiment says.
3. **Neither number is yet a measure of degradation.** That needs E2: perplexity and
   accuracy on a model that has actually been hit.

Until that point, E3's result should be read for what it is — a measure of **surface**,
not of consequence.
