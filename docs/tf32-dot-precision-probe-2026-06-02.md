# TF32 Dot Precision Probe

This slice adds an opt-in packed-attention precision knob:

```text
dot_precision="ieee"   # default, high-fidelity path
dot_precision="tf32"   # fast-mode candidate on Ampere/Ada
dot_precision="tf32x3" # Triton alternate, tested but not useful here
```

The current fused packed-attention kernels previously forced every `tl.dot` to
`input_precision="ieee"`. That includes:

- Q rotation;
- codebook-dot score accumulation;
- optional QJL correction;
- attention-weight times V accumulation.

For the K7/no-QJL `packed_t` 1024 self-attention path, switching those dots to
`tf32` lets the kernel use faster matrix hardware while still accumulating in
fp32.

## Synthetic Result

Shape:

```text
batch=1
heads=20
query_tokens=1024
key_tokens=1024
head_dim=64
format=packed_t
norm_dtype=fp32
policy=K7/no-QJL
```

Confirmation run on the RTX 4070:

| Dot precision | Encode ms | Attention ms | Total ms | Relative RMSE | Cosine error |
| --- | ---: | ---: | ---: | ---: | ---: |
| ieee | 0.1992 | 0.6232 | 0.8177 | 0.023998 | 0.000288 |
| tf32 | 0.1898 | 0.2831 | 0.4761 | 0.024101 | 0.000288 |

`tf32x3` was also tested in the shorter first pass and did not help:

```text
tf32x3 attention=0.7365ms
tf32x3 total=0.9475ms
```

## Image Suite

Output:

```text
captures/image-policy-suite-juggernaut-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-1024-packedt-tf32
```

| Case | Baseline s | Shmoosh s | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| reading-nook-seed1-1024 | 12.0448 | 10.2168 | 1.179x | 51.70 dB |
| maple-leaf-seed2-1024 | 8.9102 | 8.8393 | 1.008x | 52.16 dB |
| misty-lake-seed3-1024 | 9.0123 | 8.9922 | 1.002x | 58.66 dB |

Aggregate:

```text
min_psnr_db=51.7043
mean_psnr_db=54.1728
mean_speedup=1.0684x
```

Processor phase means:

```text
packed_attention=0.7293ms
packed_encode=0.7874ms
scheduled_quantized=2.5282ms
encode_rotate_bucketize=0.4161ms
encode_pack_codes=0.1241ms
```

Compared with the previous `packed_t` image suite:

```text
packed_attention:     1.2806ms -> 0.7293ms
scheduled_quantized:  2.9597ms -> 2.5282ms
min_psnr:             52.07 dB -> 51.70 dB
mean_psnr:            54.27 dB -> 54.17 dB
```

## Read

This is the largest packed-attention kernel-side improvement so far. It is also
the first speed knob with a visible, though still small, image-fidelity tradeoff
on the hardest 1024 case.

Keep `ieee` as the high-fidelity default. Treat `tf32` as an explicit fast mode:

```text
configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-tf32-policy.json
```

The next useful validation is broader image coverage, then a 3080 check once
that hardware is available. If the quality tax remains bounded, `tf32` may
become the practical 4070 fast policy.

Follow-up precision splitting is recorded in
`docs/tf32-dot-split-probe-2026-06-02.md`, with the same-process comparison in
`docs/precision-policy-compare-2026-06-02.md`. The best balanced 4070 fast-mode
candidate keeps Q rotation at `ieee` precision while using `tf32` for score and
value dots. It recovers reading-nook quality to `52.08 dB` while reducing mean
packed-attention time from `1.3043ms` to `0.8613ms` versus IEEE.
