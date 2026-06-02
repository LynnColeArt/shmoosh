# TF32 Dot Split Probe

This slice splits packed-attention dot precision into separate controls:

```text
dot_precision              # global default for every tl.dot
rotation_dot_precision     # Q rotation
score_dot_precision        # q_rot dot codebook values
value_dot_precision        # attention weights dot exact V
qjl_dot_precision          # QJL residual dots
```

Existing policies keep their behavior because `dot_precision` still fans out to
all dot sites when no split override is provided.

## Synthetic Sweep

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

First split sweep on the RTX 4070:

| Profile | Attention ms | Total ms | Relative RMSE |
| --- | ---: | ---: | ---: |
| all ieee | 0.7693 | 0.9789 | 0.023998 |
| all tf32 | 0.3303 | 0.5886 | 0.024101 |
| rotation tf32 | 0.6845 | 0.9209 | 0.024017 |
| score tf32 | 0.5105 | 0.7879 | 0.024044 |
| value tf32 | 0.7077 | 0.9046 | 0.024009 |
| rotation+score tf32 | 0.4617 | 0.6822 | 0.024083 |
| score+value tf32 | 0.3664 | 0.6155 | 0.024059 |
| rotation+value tf32 | 0.5265 | 0.7785 | 0.024030 |

Confirmation:

| Profile | Attention ms | Total ms | Relative RMSE | Cosine error |
| --- | ---: | ---: | ---: | ---: |
| all tf32 | 0.3394 | 0.5631 | 0.024101 | 0.000288 |
| score+value tf32 | 0.3904 | 0.6251 | 0.024059 | 0.000288 |

Read:

- `score_dot_precision=tf32` is the biggest single-dot win.
- `value_dot_precision=tf32` combines well with score TF32.
- leaving `rotation_dot_precision="ieee"` recovers some synthetic error while
  keeping most of the speedup.

## Image Suite

The balanced candidate uses:

```text
dot_precision="ieee"
score_dot_precision="tf32"
value_dot_precision="tf32"
```

Output:

```text
captures/image-policy-suite-juggernaut-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-1024-packedt-score-value-tf32
```

| Case | Baseline s | Shmoosh s | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| reading-nook-seed1-1024 | 20.2124 | 11.5171 | 1.755x | 52.08 dB |
| maple-leaf-seed2-1024 | 10.3752 | 10.4875 | 0.989x | 52.16 dB |
| misty-lake-seed3-1024 | 10.4310 | 10.5062 | 0.993x | 58.24 dB |

Aggregate:

```text
min_psnr_db=52.0788
mean_psnr_db=54.1603
mean_speedup=1.2617x
```

Processor phase means:

```text
packed_attention=0.9982ms
packed_encode=1.2763ms
scheduled_quantized=3.9065ms
```

The reading-nook PSNR recovered from the all-TF32 suite:

```text
all tf32 reading-nook:          51.70 dB
score+value tf32 reading-nook:  52.08 dB
high-fidelity packed_t:         52.07 dB
```

The timing result is mixed. The packed-attention phase is still better than the
high-fidelity packed_t suite (`1.2806ms -> 0.9982ms`), but this image run had
noisy encode and scheduled-quantized timings. The whole-image speedup is also
inflated by a slow first baseline pass.

## Read

All-TF32 remains the clearer fast mode. Score+value TF32 is the better
quality-recovery candidate and should be rerun before promotion:

```text
configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-score-value-tf32-policy.json
```

The useful finding is structural: the small all-TF32 quality tax appears to come
mostly from Q rotation precision, not from the score/value dots alone.
