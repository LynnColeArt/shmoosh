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

## Interpretation

Two things are now clearer:

1. QJL is not universally worth its cost in 1024-token self-attention. K7/no-QJL
   is a serious candidate when activated late.
2. Synthetic attention metrics still cannot replace image-level trajectory
   tests. The same K7/no-QJL policy that looked excellent synthetically failed
   at 50% and succeeded at 70%.

Next slice:

1. Trace the K7/no-QJL 70% self-attention image run to confirm the expected
   encode/attention split in the real processor.
2. Test a cross+self composition with cross attention held at the current
   cached policy but self-attention restricted to only the strongest one or two
   K7/no-QJL modules.
3. Consider a dedicated no-QJL streaming kernel tile default if repeated image
   traces keep favoring `block_k=32`.
