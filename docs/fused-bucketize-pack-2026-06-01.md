# Fused Bucketize Pack

This slice tested whether no-QJL packed-key encode can skip the materialized
code-index tensor. The target is the rotate/bucketize/pack part of the 1024
self-attention encode path.

## Kernel Shape

The new CUDA fast path is intentionally narrow:

```text
qjl_bits == 0
code_format == "packed"
bits == 7
CUDA + Triton available
```

It consumes the rotated normalized K tensor, performs codebook-boundary binary
search in Triton, and writes packed code bytes directly. K7 uses two natural
8-code pack groups per Triton program, so each program emits 14 packed bytes.

The PyTorch fallback remains the default for byte-code, QJL residual policies,
CPU, non-Triton environments, and K6.

## K6 Rejection

K6 looked promising in synthetic after grouping two natural pack groups per
program:

```text
captures/self-attention-variant-bench-1024-packed-k6-k7-noqjl-fused-bucketize-pack-group2-bk128-rerun
```

| Variant | Total ms | Encode ms | Attention ms | Relative RMSE |
| --- | ---: | ---: | ---: | ---: |
| K6/no-QJL fused | 0.9246 | 0.1719 | 0.7590 | 0.036941 |
| K7/no-QJL fused | 1.0850 | 0.2343 | 0.7506 | 0.023998 |

But K6 did not transfer cleanly to the image trace. The temporary K6 fused
suite at
`captures/image-policy-suite-juggernaut-up0-self-attn1-firstblocks-gated70pct-k6-noqjl-1024-fused-bucketize-pack-v2`
kept quality, but mean speedup fell to `1.061x` and mean packed encode was
`0.7759ms`, worse than the in-place-normalize K6 suite's `0.6357ms`.

K6 therefore stays on the existing PyTorch bucketize plus fast-pack path.

## K7 Synthetic Bench

Prior in-place-normalize synthetic run:

```text
captures/self-attention-variant-bench-1024-packed-k6-k7-noqjl-inplace-normalize-bk128
```

Fused bucketize/pack synthetic rerun:

```text
captures/self-attention-variant-bench-1024-packed-k6-k7-noqjl-fused-bucketize-pack-group2-bk128-rerun
```

| Variant | Encode before | Encode after | Total before | Total after | Relative RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| K7/no-QJL | 0.3738 | 0.2343 | 0.9382 | 1.0850 | 0.023998 |

The useful signal is encode-side. Whole synthetic total is noisy because the
attention phase moved in the opposite direction during the rerun.

## K7 Image Suite

Policy:

```text
configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-policy.json
```

Output:

```text
captures/image-policy-suite-juggernaut-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-1024-fused-bucketize-pack-v2
```

| Case | Baseline s | Shmoosh s | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| `reading-nook-seed1-1024` | 12.7151 | 10.8811 | 1.169x | 51.87 dB |
| `maple-leaf-seed2-1024` | 10.0700 | 9.7737 | 1.030x | 52.19 dB |
| `misty-lake-seed3-1024` | 9.5569 | 9.5684 | 0.999x | 57.82 dB |

Aggregate:

- minimum PSNR: `51.87 dB`
- mean PSNR: `53.96 dB`
- mean speedup: `1.070x`
- mean packed encode: `0.8821ms`
- mean packed attention: `1.4635ms`
- mean rotate/bucketize phase: `0.5178ms`
- mean pack-codes phase: `0.0039ms`

Compared with the in-place-normalize K7 suite, mean packed encode moved from
`0.9185ms` to `0.8821ms`. The larger internal shift is clearer in the subphase:
`encode_pack_codes` moved from `0.4281ms` to effectively zero because bucketize
and pack now happen inside the Triton kernel. The cost appears under the
historical `encode_rotate_bucketize` timing label.

## Readout

This is a real kernel-path simplification, but still a modest end-to-end win.
K7/no-QJL remains the preferred quality policy. K6/no-QJL remains a speed
tradeoff policy, but not through this fused bucketize/pack fast path.

The next encode lever is to reduce the remaining rotate/bucketize work itself:
either fuse rotation with bucketize/pack, or replace the rotation-plus-boundary
pipeline with a representation that the attention kernel can consume more
directly.

## Fused Rotation Follow-Up

The fused rotation follow-up is recorded in
`docs/fused-rotate-bucketize-pack-2026-06-01.md`. It moved the K7/no-QJL
rotation into the Triton encode kernel. The 1024 image suite stayed
quality-identical and mean packed encode moved from `0.8821ms` to `0.8342ms`.
