# FP16 Norm Probe

This slice adds an opt-in packed-key norm storage dtype:

```text
norm_dtype="fp32"  # default
norm_dtype="fp16"  # opt-in
```

The preferred K7/no-QJL SDXL self-attention representation stores:

```text
packed codes: 56 bytes/vector
norm fp32:     4 bytes/vector
total:        60 bytes/vector
```

Using fp16 norms changes only the norm payload:

```text
packed codes: 56 bytes/vector
norm fp16:     2 bytes/vector
total:        58 bytes/vector
```

The fused Triton attention and score kernels now read the norm tensor in its
stored dtype and cast loaded values to fp32 before accumulation. The torch
fallback/reference paths still upcast norms for comparison math.

## Synthetic Result

Command shape:

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

| Norm dtype | Bytes/vector | Encode ms | Attention ms | Total ms | Relative RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 | 60 | 0.2355 | 0.7564 | 0.9166 | 0.023998 |
| fp16 | 58 | 0.2285 | 0.7686 | 0.9050 | 0.024005 |

The quality movement is tiny, but the attention phase did not improve. The
slightly better total time is not enough to trust because encode timing is
noisy and the kernel-side attention time moved the wrong way.

## Read

Keep `norm_dtype="fp16"` as an opt-in memory-format knob. Do not promote it to
the preferred K7/no-QJL policy yet.

This result says the remaining 1024 self-attention cost is not bottlenecked by
the two norm bytes saved per key vector. The next speed work should stay focused
on packed code interpretation, encode+attention fusion, or a lower-level CUDA
kernel.
