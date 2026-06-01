# Encode Normalize V2

This slice reduced packed-key encode overhead without changing the packed K
format or policy surface.

## Change

The encode path previously built a normalized `unit` tensor in addition to the
float32 key working copy. The current path normalizes the float32 working copy
in place, while preserving a raw float32 copy only when QJL residual correction
needs it.

That keeps no-QJL policies on the cheaper path:

```text
keys -> float32 working copy -> in-place normalize -> rotate -> bucketize
```

The fp32-input case still clones before in-place mutation when the dtype cast
would share storage with the caller's tensor.

## Rejected Attempt

A folded-math version also tested:

```text
rotate raw keys, then scale by sqrt(head_dim) / norm
```

It avoided the explicit normalization division but did not earn its keep. The
synthetic run at
`captures/self-attention-variant-bench-1024-packed-k6-k7-noqjl-folded-encode-bk128`
was not better overall:

| Variant | Total ms | Encode ms | Attention ms |
| --- | ---: | ---: | ---: |
| K6/no-QJL folded | 0.9470 | 0.2911 | 0.7618 |
| K7/no-QJL folded | 0.9840 | 0.4333 | 0.6585 |

The kept version is the in-place working-copy normalization instead.

## Synthetic 1024 Bench

Prior compact-K continuation-mask run:

```text
captures/self-attention-variant-bench-1024-packed-k6-k7-noqjl-contmask-bk128-seq
```

In-place normalize run:

```text
captures/self-attention-variant-bench-1024-packed-k6-k7-noqjl-inplace-normalize-bk128
```

| Variant | Total before | Total after | Encode before | Encode after | Attention before | Attention after | Relative RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K6/no-QJL | 0.9278 | 0.8917 | 0.2855 | 0.2509 | 0.6722 | 0.6402 | 0.036941 |
| K7/no-QJL | 0.9596 | 0.9382 | 0.3949 | 0.3738 | 0.6480 | 0.6850 | 0.023998 |

The useful signal is encode-side allocation/copy reduction. Attention movement
is expected to be noisy at this scale and is not the claimed lever.

## Image Suites

K7/no-QJL 70% policy:

```text
captures/image-policy-suite-juggernaut-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-1024-inplace-normalize
```

| Case | Baseline s | Shmoosh s | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| `reading-nook-seed1-1024` | 12.0737 | 10.2636 | 1.176x | 51.87 dB |
| `maple-leaf-seed2-1024` | 9.3647 | 9.0304 | 1.037x | 52.19 dB |
| `misty-lake-seed3-1024` | 8.9270 | 8.9200 | 1.001x | 57.82 dB |

Aggregate: `51.87 dB` minimum PSNR, `53.96 dB` mean PSNR, `1.076x` mean
speedup, `0.9185ms` mean packed encode per quantized call.

K6/no-QJL 70% policy:

```text
captures/image-policy-suite-juggernaut-up0-self-attn1-firstblocks-gated70pct-k6-noqjl-1024-inplace-normalize
```

| Case | Baseline s | Shmoosh s | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| `reading-nook-seed1-1024` | 12.2545 | 10.0365 | 1.221x | 51.58 dB |
| `maple-leaf-seed2-1024` | 9.0196 | 8.6959 | 1.037x | 50.38 dB |
| `misty-lake-seed3-1024` | 8.6107 | 8.6352 | 0.997x | 56.78 dB |

Aggregate: `50.38 dB` minimum PSNR, `52.91 dB` mean PSNR, `1.092x` mean
speedup, `0.6357ms` mean packed encode per quantized call.

## Readout

This is a small implementation cleanup, not a policy shift. K7/no-QJL remains
the preferred high-fidelity 1024 self-attention policy. K6/no-QJL remains the
explicit speed-mode tradeoff.

The next meaningful speed lever is still deeper than normalization: reduce the
rotate/bucketize path, fuse encode with attention, or remove Python/processor
overhead around fixed 1024 shapes.

## Fused Bucketize Pack Follow-Up

The next encode slice is recorded in
`docs/fused-bucketize-pack-2026-06-01.md`. It adds a K7/no-QJL Triton path that
performs boundary search and bit packing together. This reduced K7 synthetic
encode to `0.2343ms` and kept the 1024 image suite quality-identical, but the
end-to-end image gain stayed modest.
