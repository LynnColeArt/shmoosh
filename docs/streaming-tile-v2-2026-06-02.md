# Streaming Tile V2

This slice tests a tile-local reuse lever without changing the packed
representation or adding a new unpack kernel.

The previous large-key no-QJL streaming default was:

```text
BLOCK_Q=32
BLOCK_K=32
```

For the preferred 1024 self-attention policy, the new narrow default is:

```text
bits == 7
qjl_bits == 0
code_format == "packed"
head_dim == 64
large-key streaming path

BLOCK_Q=64
BLOCK_K=16
```

The idea is to halve the number of query programs that reinterpret each packed
K stream while keeping the key tile small enough to avoid the register pressure
that hurt wider key tiles.

## Synthetic Sweep

All runs used:

```text
batch=1
heads=20
query_tokens=1024
key_tokens=1024
head_dim=64
dtype=fp16
bits=7
qjl_bits=0
```

| Shape | Encode ms | Attention ms | Total ms | Output |
| --- | ---: | ---: | ---: | --- |
| auto old shape, effectively `BQ32/BK32` | 0.1753 | 0.7009 | 0.8708 | `captures/self-attention-variant-bench-1024-packed-k7-noqjl-tile-bq32-bk32-baseline` |
| explicit `BQ64/BK32` | 0.2122 | 0.9390 | 1.1315 | `captures/self-attention-variant-bench-1024-packed-k7-noqjl-tile-bq64-bk32` |
| explicit `BQ32/BK64` | 0.1924 | 1.1821 | 1.3205 | `captures/self-attention-variant-bench-1024-packed-k7-noqjl-tile-bq32-bk64` |
| explicit `BQ32/BK16` | 0.1610 | 0.8464 | 0.9977 | `captures/self-attention-variant-bench-1024-packed-k7-noqjl-tile-bq32-bk16` |
| explicit `BQ64/BK16` | 0.1594 | 0.6009 | 0.7311 | `captures/self-attention-variant-bench-1024-packed-k7-noqjl-tile-bq64-bk16` |
| confirm old `BQ32/BK32` | 0.1574 | 0.6281 | 0.8165 | `captures/self-attention-variant-bench-1024-packed-k7-noqjl-tile-bq32-bk32-confirm` |
| confirm `BQ64/BK16` | 0.1793 | 0.6146 | 0.7844 | `captures/self-attention-variant-bench-1024-packed-k7-noqjl-tile-bq64-bk16-confirm` |
| `BQ128/BK16` | 0.2434 | 1.1935 | 1.3109 | `captures/self-attention-variant-bench-1024-packed-k7-noqjl-tile-bq128-bk16-confirm` |
| auto after promotion | 0.1816 | 0.6305 | 0.7923 | `captures/self-attention-variant-bench-1024-packed-k7-noqjl-auto-tile-bq64-bk16` |
| explicit old after promotion | 0.1873 | 0.6870 | 0.8810 | `captures/self-attention-variant-bench-1024-packed-k7-noqjl-explicit-old-tile-after-auto` |

The synthetic signal is small and noisy, but `BQ64/BK16` is the only adjacent
shape that repeatedly avoids a clear loss.

## Image Suite

Output:

```text
captures/image-policy-suite-juggernaut-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-1024-auto-tile-bq64-bk16
```

Validation used the same three-case 1024 Juggernaut suite as the prior K7/no-QJL
self-attention runs, with processor timing enabled.

| Case | Baseline s | Shmoosh s | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| `reading-nook-seed1-1024` | 12.6735 | 10.2530 | 1.236x | 52.07 dB |
| `maple-leaf-seed2-1024` | 9.2042 | 8.8459 | 1.041x | 52.12 dB |
| `misty-lake-seed3-1024` | 9.1809 | 9.1750 | 1.001x | 58.61 dB |

Aggregate:

- minimum PSNR: `52.07 dB`
- mean PSNR: `54.27 dB`
- mean speedup: `1.098x`
- mean scheduled quantized call: `3.1222ms`
- mean packed encode: `0.6860ms`
- mean packed attention: `1.3521ms`
- mean rotate/bucketize: `0.4206ms`
- mean pack-codes: `0.0039ms`

Compared with the prior fused rotation+bucketize+pack image suite,
`packed_attention` moved from `1.4519ms` to `1.3521ms` per quantized call. Treat
that as a small phase-level win; whole-image runtime remains noisy and prompt
dependent.

## Readout

Promote `BQ64/BK16` only for the narrow K7/no-QJL/head_dim=64 packed streaming
path. Keep other packed formats on their existing defaults:

```text
QJL path: BQ32/BK16
generic no-QJL path: BQ32/BK32
K7/no-QJL/head_dim=64: BQ64/BK16
```

This does not solve the larger speed problem. It slightly reduces packed
attention cost while preserving the current quality envelope. The next real
speed lever remains layout/reuse beyond one Triton program, encode+attention
fusion, or fixed-shape graph/compile overhead reduction.
