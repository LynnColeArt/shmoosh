# K7 Head64 Specialized Unpack Probe

This slice tested a narrower version of native packed attention:

```text
K7 only
no QJL residual
head_dim=64 only
CUDA/Triton streaming self-attention only
```

The prototype hardcoded the K7 bit layout. Instead of the generic per-dimension
unpack path loading a primary byte and, when needed, a continuation byte, it
loaded each 8-code group as 7 bytes and unpacked all 8 codes together.

## Result

Comparison at synthetic SDXL-style self-attention shape:

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

| Path | Encode ms | Attention ms | Total ms | Rel RMSE |
| --- | ---: | ---: | ---: | ---: |
| specialized K7/head64 unpack | 0.1574 | 1.5081 | 1.6919 | 0.023998 |
| generic packed K7 kernel | 0.1694 | 0.6445 | 0.8490 | 0.023998 |
| default after reverting probe | 0.1723 | 0.7429 | 0.9528 | 0.023998 |

The hardcoded unpack was correctness-equivalent, but it was much slower. The
likely cause is that the reduced byte-load count was overwhelmed by register
pressure and tensor assembly from rebuilding the full `64 x BLOCK_K` codebook
value tile through many `where` operations.

The prototype was not retained in runtime code. The current generic packed
kernel remains the default.

## Readout

The codebook-dot direction is already present in Shmoosh's generic packed
attention: the kernel consumes packed K codes, looks up scalar codebook values,
and accumulates attention scores without materializing decoded K in global
memory. This experiment says the next speedup is not a naive hardcoded unpack.

Better next candidates:

- tile-local decode reuse that avoids rebuilding the code-value tile per query
  program;
- a different packed layout designed for Triton matrix construction rather than
  just byte density;
- encode+attention fusion for modules where K is used only once;
- fixed-shape CUDA graphs or compile once the policy surface stabilizes.
