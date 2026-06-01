# Direct Rotated K Probe

This slice adds an opt-in `code_format="rotated"` path to the synthetic
self-attention benchmark. It is a diagnostic representation, not a compression
candidate.

## Representation

The rotated block stores:

```text
unit_rotated_k = normalize(K) @ rotation.T
norms = ||K||
```

The attention kernel rotates Q, streams over `unit_rotated_k`, multiplies scores
by the exact key norms, applies softmax, and accumulates exact V. This removes
bucketize, bit packing, bit unpacking, and codebook lookup from the measured
attention path.

Storage at SDXL self-attention shape `head_dim=64` with fp16 rotated keys:

| Format | Bytes/vector | Compression ratio vs fp16 K |
| --- | ---: | ---: |
| K7 packed + norm | 60 | 2.13x |
| direct rotated + norm | 132 | 0.97x |

So this probe intentionally trades density for a simpler attention kernel.

## Correctness

Added coverage:

- CPU torch rotated attention matches exact torch attention.
- CUDA Triton rotated attention matches the torch rotated fallback.
- The synthetic bench accepts `--code-format rotated`.

Validation:

```text
uv run python -m pytest
75 passed
```

## Synthetic 1024 Bench

Packed K7/no-QJL comparison:

```text
captures/self-attention-variant-bench-1024-packed-k7-noqjl-after-rotated-probe
```

Rotated probe:

```text
captures/self-attention-variant-bench-1024-rotated-k7-noqjl-probe
```

| Format | Encode ms | Attention ms | Total ms | Rel RMSE | Bytes/vector |
| --- | ---: | ---: | ---: | ---: | ---: |
| K7 packed | 0.1777 | 0.6771 | 0.8428 | 0.023998 | 60 |
| direct rotated | 0.0949 | 1.3738 | 1.4724 | 0.000310 | 132 |

The direct representation does solve the encode-prep problem: encode is about
`1.87x` faster than the current packed K7 path. But attention is about `2.03x`
slower because each key vector is much larger and the kernel rereads it across
streaming softmax tiles.

## Tile Sweep

The default direct kernel shape is `BLOCK_Q=32`, `BLOCK_K=32`. A quick tile
sweep did not expose a better production candidate:

| Shape | Encode ms | Attention ms | Total ms |
| --- | ---: | ---: | ---: |
| `BQ32/BK16` | 0.0994 | 2.4527 | 2.2362 |
| `BQ32/BK32` | 0.0949 | 1.3738 | 1.4724 |
| `BQ32/BK64` | 0.0919 | 1.6112 | 1.7758 |
| `BQ16/BK32` | 0.0928 | 2.5052 | 2.6558 |
| `BQ64/BK32` | 0.0952 | 1.3482 | 1.5498 |
| `BQ64/BK64` | 0.0944 | 10.3100 | 10.7406 |

`BQ64/BK32` slightly improves isolated attention time, but total time remains
well behind packed K7 and the larger tile variants spill badly.

## Readout

The result is a useful negative-positive:

```text
Direct rotated K removes encode mechanics and recovers near-exact quality.
It loses as a 1024 self-attention runtime format because density still wins.
```

This keeps bit-packed K7/no-QJL as the default for 1024 self-attention. The next
speed lever should preserve compact key reads while reducing repeated work:

- fuse encode and attention for cases where K is not reused enough to amortize
  encode;
- make the packed attention kernel consume compact data with less decode work;
- test fixed-shape CUDA graphs or compile once policy surfaces stop changing.
