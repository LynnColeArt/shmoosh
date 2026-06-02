# Fused Norm Encode

This slice adds an explicit packed-key encode backend:

```text
key_encode_backend="split" # default, previous multi-stage encode
key_encode_backend="fused" # K7/head64/no-QJL fused Triton encode
key_encode_backend="auto"  # use fused when eligible, otherwise split
```

The fused path is intentionally narrow:

```text
bits=7
qjl_bits=0
head_dim=64
code_format in {"packed", "packed_t"}
CUDA Triton only
```

It computes key norms, normalizes K, rotates, bucketizes, and writes packed
codes in one Triton kernel. For `packed_t`, it writes the transposed
`(code_bytes, tokens)` layout directly instead of packing and then transposing
from Python.

## Synthetic

Shape:

```text
batch=1
heads=20
query_tokens=1024
key_tokens=1024
head_dim=64
policy=K7/no-QJL packed_t
precision=score+value tf32, rotation ieee
```

Sequential 4070 timing after gating:

| Encode backend | Encode | Attention | Total | Relative RMSE |
| --- | ---: | ---: | ---: | ---: |
| `split` | 0.2532 ms | 0.4258 ms | 0.5838 ms | 0.024059 |
| `fused` | 0.1356 ms | 0.4287 ms | 0.5831 ms | 0.024060 |

The encode phase improved by about `46%`, but synthetic total was effectively
flat in this run. Earlier before/after timing showed the same directional
encode win:

```text
split encode=0.2204ms total=0.5920ms
fused encode=0.1172ms total=0.5180ms
```

Treat the encode phase as the stable win and total timing as noise-sensitive.

## Image Compare

Output:

```text
captures/image-policy-compare-juggernaut-up0-self-attn1-k7-noqjl-1024-score-value-fused-encode-20260602
```

Same-process 1024 comparison against the current balanced score+value TF32
policy:

| Backend | Min PSNR | Mean PSNR | Mean speedup | Packed encode | Packed attention | Scheduled quantized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `split` | 52.0788 dB | 54.1603 dB | 1.060x | 0.9956 ms | 0.8851 ms | 3.3231 ms |
| `fused` | 51.8747 dB | 54.1765 dB | 1.126x | 0.5506 ms | 0.8082 ms | 2.8287 ms |

Per-case:

| Backend | Case | PSNR | Shmoosh seconds | Packed encode |
| --- | --- | ---: | ---: | ---: |
| `split` | reading-nook | 52.0788 dB | 11.2108 | 1.0154 ms |
| `fused` | reading-nook | 51.8747 dB | 10.0371 | 0.5664 ms |
| `split` | maple-leaf | 52.1602 dB | 9.6956 | 1.0862 ms |
| `fused` | maple-leaf | 52.1041 dB | 9.4340 | 0.5200 ms |
| `split` | misty-lake | 58.2421 dB | 9.8491 | 0.8851 ms |
| `fused` | misty-lake | 58.5508 dB | 9.4728 | 0.5653 ms |

## Decision

Keep split encode as the default and pin the existing preferred policies to:

```text
key_encode_backend="split"
```

Promote fused encode only as an opt-in speed tradeoff:

```text
configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-score-value-tf32-fused-encode-policy.json
```

The speed signal is real:

```text
packed_encode:        0.9956ms -> 0.5506ms
scheduled_quantized:  3.3231ms -> 2.8287ms
```

The hard-case quality cost is also real:

```text
reading-nook: 52.0788 dB -> 51.8747 dB
```

That makes fused encode useful for a user-selectable 4070 speed mode, not the
balanced default.
