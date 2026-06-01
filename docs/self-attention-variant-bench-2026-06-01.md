# Self-Attention Variant Bench: 2026-06-01

This slice checks whether 1024-token self-attention really needs QJL residual
correction. The prior trace showed that, for self-attention, packed attention
time is larger than encode time, unlike the text-key cross-attention path.

## Synthetic 1024-Token Bench

Command:

```bash
uv run shmoosh-self-attention-variant-bench \
  --device cuda \
  --dtype fp16 \
  --batch-size 1 \
  --heads 20 \
  --query-tokens 1024 \
  --key-tokens 1024 \
  --dim 64 \
  --variants 6:128,6:64,6:0,7:0 \
  --codebook-samples 80000 \
  --warmup-iters 3 \
  --iters 20 \
  --backend auto \
  --output-dir captures/self-attention-variant-bench-1024-k6-k7-qjl
```

Exact fp32 attention baseline in the same harness: `1.4558 ms/iter`.

| Variant | Total ms | Encode ms | Attention ms | Relative RMSE | Cosine error | Packed bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K6 + QJL128 | 3.8417 | 1.2061 | 2.6101 | 0.030060 | 0.000452 | 72 |
| K6 + QJL64 | 2.2470 | 0.9494 | 1.3067 | 0.042352 | 0.000896 | 64 |
| K6, no QJL | 1.4329 | 0.5602 | 0.9379 | 0.036941 | 0.000682 | 52 |
| K7, no QJL | 1.3117 | 0.5913 | 0.8348 | 0.023998 | 0.000288 | 60 |

Synthetic readout: K7/no-QJL is the best shape here. It is faster than K6/QJL128
and lower-error on the synthetic exact-attention comparison.

## Tile Sensitivity

Explicit `block_k=32`:

| Variant | Total ms | Encode ms | Attention ms | Relative RMSE |
| --- | ---: | ---: | ---: | ---: |
| K6 + QJL128 | 8.2894 | 1.0774 | 7.6473 | 0.030060 |
| K7, no QJL | 1.1473 | 0.6529 | 0.7431 | 0.023998 |

Explicit `block_k=64`:

| Variant | Result |
| --- | --- |
| K6 + QJL128 | fails Triton shared-memory limit: required 107520, hardware limit 101376 |
| K7, no QJL | 1.9330 ms total, 0.5474 ms encode, 1.2907 ms attention |

Tile readout: widening the streaming key tile helps no-QJL at `block_k=32`, but
hurts or fails with QJL128. This points toward separate kernel tuning for
QJL-heavy and no-QJL self-attention.

## Image-Level Validation

The synthetic result did not transfer at a 50% denoising gate:

| Policy | Output dir | PSNR | MSE | Baseline s | Shmoosh s |
| --- | --- | ---: | ---: | ---: | ---: |
| K7/no-QJL, exact first 50% | `captures/image-ab-juggernaut-up0-self-attn1-firstblocks-gated50pct-k7-noqjl-1024-reading-nook` | 48.29 dB | 0.00001482 | 11.7876 | 9.6714 |

Moving self-attention activation later rescued it:

| Policy | Output dir | PSNR | MSE | Baseline s | Shmoosh s |
| --- | --- | ---: | ---: | ---: | ---: |
| K7/no-QJL, exact first 70% | `captures/image-ab-juggernaut-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-1024-reading-nook` | 52.07 dB | 0.00000620 | 11.5258 | 9.6632 |

Three-case 1024 suite for the 70% self-attention policy:

| Case | Baseline s | Shmoosh s | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| `reading-nook-seed1-1024` | 11.6374 | 9.8205 | 1.185x | 52.07 dB |
| `maple-leaf-seed2-1024` | 8.7059 | 8.4474 | 1.031x | 52.12 dB |
| `misty-lake-seed3-1024` | 8.5973 | 8.5609 | 1.004x | 58.61 dB |

Aggregate:

- min PSNR: `52.07 dB`
- mean PSNR: `54.27 dB`
- max MSE: `0.00000620`
- mean speedup: `1.079x`

This is the best self-attention-only policy so far. It beats the K6/QJL128 50%
self-attention policy on quality and mean runtime, but only after the activation
window moves from 50% to 70%.

## Processor Trace

The 70% K7/no-QJL self-attention run was traced on the reading-nook case:

```text
captures/image-ab-juggernaut-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-1024-trace-reading-nook
```

Trace summary:

- baseline: `11.4399s`
- Shmoosh: `9.6055s`
- speedup: `1.191x`
- PSNR: `52.07 dB`
- scheduled exact calls: `42`
- scheduled quantized calls: `18`

The call counts match the intended 70% gate: three modules stay exact for the
first 14 of 20 denoising steps, then run quantized for the last 6 steps.

Compared with the older K6/QJL128 50% self-attention trace:

| Trace | Quantized calls | Packed encode s | Packed attention s | Quantized total s | Mean quantized ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| K6/QJL128, exact first 50% | 30 | 0.0750 | 0.1329 | 0.2348 | 7.8270 |
| K7/no-QJL, exact first 70% | 18 | 0.0202 | 0.0336 | 0.0732 | 4.0640 |

Trace readout: the quality win is not just coming from hiding the policy later.
The no-QJL path is materially lighter per quantized call too. The residual
projection and residual-sign packing phases disappear, and packed attention
drops from `4.4311ms` per call to `1.8646ms` per call.

## Cross + Self Composition

The K7/no-QJL 70% self-attention policy was also composed with the cached
cross-attention policy:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-cache-self-attn1-gated70pct-k5-k7-noqjl-policy.json
```

Three-case 1024 suite:

| Case | Baseline s | Shmoosh s | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| `reading-nook-seed1-1024` | 11.4652 | 9.8514 | 1.164x | 49.12 dB |
| `maple-leaf-seed2-1024` | 8.6853 | 8.6528 | 1.004x | 48.50 dB |
| `misty-lake-seed3-1024` | 8.8476 | 8.9046 | 0.994x | 56.72 dB |

Aggregate:

- min PSNR: `48.50 dB`
- mean PSNR: `51.44 dB`
- max MSE: `0.00001413`
- mean speedup: `1.058x`

Composition readout: this is faster than the K6/QJL128 70% self-attention
composition, but lower quality. Keep it as a speed/quality tradeoff, not as the
default policy.

## Restricted Composition

The processor trace ranked the K7/no-QJL self-attention modules by quantized
runtime:

| Module | Mean quantized ms |
| --- | ---: |
| `up_blocks.0.attentions.1.transformer_blocks.0.attn1` | 3.7710 |
| `up_blocks.0.attentions.2.transformer_blocks.0.attn1` | 3.8903 |
| `up_blocks.0.attentions.0.transformer_blocks.0.attn1` | 4.5308 |

Two restricted cross+self policies were tested:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-cache-self-attn1-gated70pct-k5-k7-noqjl-a1-policy.json
configs/underpaint-juggernaut-sdxl-up0-cross-cache-self-attn1-gated70pct-k5-k7-noqjl-a1-a2-policy.json
```

Three-case 1024 comparison:

| Policy | Mean speedup | Min PSNR | Mean PSNR | Max MSE |
| --- | ---: | ---: | ---: | ---: |
| cached cross-attention only | 1.046x | 49.40 dB | 52.59 dB | 0.00001148 |
| cross + K7/no-QJL self `a1` | 1.053x | 49.02 dB | 52.26 dB | 0.00001253 |
| cross + K7/no-QJL self `a1+a2` | 1.066x | 48.49 dB | 51.95 dB | 0.00001416 |
| cross + K7/no-QJL self `a0+a1+a2` | 1.058x | 48.50 dB | 51.44 dB | 0.00001413 |

Restricted composition readout: dropping the heaviest `a0` self-attention module
does help quality versus the full K7/no-QJL composition, and the two-module
variant is the fastest K7/no-QJL cross+self result so far. But neither
restricted self-attention policy beats cached cross-attention alone on fidelity.
For now, K7/no-QJL self-attention remains a strong standalone denoising-layer
policy, not the default add-on to the cross-cache policy.

## Interpretation

Three things are now clearer:

1. QJL is not universally worth its cost in 1024-token self-attention. K7/no-QJL
   is a serious candidate when activated late.
2. Synthetic attention metrics still cannot replace image-level trajectory
   tests. The same K7/no-QJL policy that looked excellent synthetically failed
   at 50% and succeeded at 70%.
3. Cross-cache and late self-attention are not simply additive yet. Restricted
   self-attention reduces the composition penalty, but cached cross-attention
   alone still has the better fidelity/runtime balance.

Next slice:

1. Treat cached cross-attention and late K7/no-QJL self-attention as separate
   policy modes until a better composition rule is found.
2. Consider a dedicated no-QJL streaming kernel tile default if repeated image
   traces keep favoring `block_k=32`.
3. Revisit composition through timestep separation, such as ending cross-cache
   before self-attention wakes up.
