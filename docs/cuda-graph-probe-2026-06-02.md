# CUDA Graph Probe

This slice adds an opt-in CUDA graph replay lane to
`shmoosh-self-attention-variant-bench`:

```bash
uv run shmoosh-self-attention-variant-bench \
  --variants 7:0 \
  --query-tokens 1024 \
  --key-tokens 1024 \
  --dim 64 \
  --heads 20 \
  --dtype fp16 \
  --device cuda \
  --backend triton \
  --code-format packed_t \
  --warmup-iters 12 \
  --iters 200 \
  --cuda-graph \
  --output-dir captures/self-attention-variant-bench-1024-packedt-k7-noqjl-cudagraph-20260602
```

The graph path captures two fixed-shape calls:

- attention only;
- encode plus attention.

This is a ceiling probe, not a production processor path. A production graph
processor would need static input buffers and device-to-device copies for
query, packed codes, norms, and values. That copy layer is only worth building
if launch overhead is a meaningful fraction of the remaining runtime.

## Result

Shape:

```text
batch=1
heads=20
query_tokens=1024
key_tokens=1024
head_dim=64
format=packed_t
policy=K7/no-QJL
```

Measured on the RTX 4070:

| Path | Eager ms | CUDA graph ms | Change |
| --- | ---: | ---: | ---: |
| attention | 0.6849 | 0.6645 | 1.031x |
| encode + attention | 0.8919 | 0.8768 | 1.017x |

Quality is unchanged from the current K7/no-QJL synthetic trace:

```text
relative_rmse=0.023998
cosine_error=0.000288
```

## Tile Check

The packed-transpose layout did not make larger tiles attractive. A quick
1024-token `packed_t` K7/no-QJL sweep showed:

| Tile | Attention ms | Total ms | Note |
| --- | ---: | ---: | --- |
| auto | 0.6241 | 0.7929 | current default |
| BQ32/BK32 | 0.6568 | 0.8643 | close, not better |
| BQ64/BK16 | 0.6398 | 0.8467 | explicit current tile |
| BQ32/BK16 | 0.8516 | 1.0754 | worse |

The auto path remains the best current choice for `packed_t`.

## Read

CUDA graphs shave some overhead, but not enough to justify adding a static-buffer
graph cache to the Diffusers processor yet. This result points away from
orchestration overhead as the main limiter and back toward:

1. reducing decode/index work inside packed attention;
2. fusing encode and attention more tightly;
3. trying a purpose-built CUDA extension if Triton scalar unpack overhead stays
   stubborn.
