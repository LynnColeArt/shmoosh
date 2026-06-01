# Fused Rotate Bucketize Pack

This slice extends the K7/no-QJL encode fast path from fused bucketize+pack to
fused rotation+bucketize+pack.

## Kernel Shape

The CUDA fast path remains narrow:

```text
qjl_bits == 0
code_format == "packed"
bits == 7
head_dim == 64
CUDA + Triton available
```

The kernel takes normalized K vectors, applies the codec rotation with
`tl.dot`, performs codebook-boundary binary search, and writes packed K7 bytes
directly. It uses `16` vectors per Triton program. A quick shape probe showed:

| Block vectors | K7 encode ms |
| ---: | ---: |
| 8 | 0.2503 |
| 16 | 0.1846 |
| 32 | 0.2197 |

The `16`-vector shape was kept.

## Correctness

CUDA unit coverage compares the fused rotation path against the existing
PyTorch reference pipeline:

```text
unit -> torch.matmul(unit, rotation.T) -> bucketize -> pack
```

The fused path produced byte-identical packed codes in the test case.

## Synthetic 1024 Bench

Prior fused bucketize+pack synthetic run:

```text
captures/self-attention-variant-bench-1024-packed-k6-k7-noqjl-fused-bucketize-pack-group2-bk128-rerun
```

Fused rotation+bucketize+pack reruns:

```text
captures/self-attention-variant-bench-1024-packed-k7-noqjl-fused-rotate-bucketize-pack-bk128-blockv16-rerun
captures/self-attention-variant-bench-1024-packed-k6-k7-noqjl-fused-rotate-bucketize-pack-bk128-blockv16-final
```

| Variant | Encode before | Encode after | Total after | Attention after | Relative RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| K7/no-QJL | 0.2343 | 0.2007 | 0.9973 | 0.7266 | 0.023998 |

The combined K6/K7 rerun recorded K7 encode at `0.2299ms`; the K7-only rerun
recorded `0.2007ms`. Treat the synthetic result as directionally positive but
noisy.

K6 remains on the existing path. The new fused rotation gate is K7-only.

## K7 Image Suite

Output:

```text
captures/image-policy-suite-juggernaut-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-1024-fused-rotate-bucketize-pack
```

| Case | Baseline s | Shmoosh s | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| `reading-nook-seed1-1024` | 12.6469 | 10.6287 | 1.190x | 51.87 dB |
| `maple-leaf-seed2-1024` | 9.6988 | 9.9376 | 0.976x | 52.19 dB |
| `misty-lake-seed3-1024` | 9.9336 | 9.7538 | 1.018x | 57.82 dB |

Aggregate:

- minimum PSNR: `51.87 dB`
- mean PSNR: `53.96 dB`
- mean speedup: `1.065x`
- mean packed encode: `0.8342ms`
- mean packed attention: `1.4519ms`
- mean rotate/bucketize phase: `0.4722ms`
- mean pack-codes phase: `0.0036ms`

Compared with the prior fused bucketize+pack image suite:

| Metric | Bucketize+pack | Rotate+bucketize+pack |
| --- | ---: | ---: |
| Mean packed encode | 0.8821ms | 0.8342ms |
| Mean rotate/bucketize | 0.5178ms | 0.4722ms |
| Mean pack-codes | 0.0039ms | 0.0036ms |
| Mean speedup | 1.070x | 1.065x |

The phase timing improved in the intended place. Whole-image speedup did not
move enough to treat this as a visible UX win.

## Readout

This validates that rotation can be fused into the encode kernel without
quality drift, and it removes another intermediate tensor from the K7/no-QJL
path. The remaining speed problem is now less about pack mechanics and more
about whether encode and attention should be fused together, or whether the
attention kernel should consume a representation that avoids this encode step
entirely.
