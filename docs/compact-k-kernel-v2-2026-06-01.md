# Compact-K Kernel V2

This slice kept the runtime on bit-packed K and attacked the consumer side of
the packed attention kernel.

The byte-code slice proved that encode can be made cheaper, but larger K payloads
lose ground in 1024-token self-attention. The compact-K question is therefore:

```text
Can dense packed codes stay dense while becoming cheaper to consume?
```

## Kernel Change

The fused Triton attention kernels previously loaded a continuation byte for
every packed code, even when the code fit entirely inside the first byte.

For the active no-QJL policies:

- K6 crosses a byte boundary for about 50% of dimensions.
- K7 crosses a byte boundary for about 75% of dimensions.

The kernel now masks continuation-byte loads with:

```text
bit_offset + bits > 8
```

This keeps the existing dense packed representation and the existing `tl.dot`
score path, but avoids continuation-byte reads that cannot affect the decoded
code index.

## Synthetic 1024 Bench

Shape:

```text
batch=1 heads=20 query_tokens=1024 key_tokens=1024 dim=64 fp16
```

Pre-change:

```text
captures/self-attention-variant-bench-1024-packed-k6-k7-noqjl-pre-contmask
```

Post-change:

```text
captures/self-attention-variant-bench-1024-packed-k6-k7-noqjl-contmask-bk128-seq
```

| Variant | Total ms before | Total ms after | Encode ms after | Attention ms before | Attention ms after | Bytes/vector | Relative RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| K6/no-QJL | 0.9339 | 0.9278 | 0.2855 | 0.7080 | 0.6722 | 52 | 0.036941 |
| K7/no-QJL | 0.9892 | 0.9596 | 0.3949 | 0.6890 | 0.6480 | 60 | 0.023998 |

Synthetic timings are still noisy, but the valid sequential comparison points in
the right direction: the continuation mask helps the compact consumer path
without changing quality.

## K6 No-QJL Image Probe

Policy:

```text
configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated70pct-k6-noqjl-policy.json
```

Reading-nook smoke:

```text
captures/image-ab-juggernaut-up0-self-attn1-firstblocks-gated70pct-k6-noqjl-1024-trace-reading-nook
```

| Metric | Value |
| --- | ---: |
| PSNR | 51.58 dB |
| MSE | 0.00000695 |
| baseline | 12.3114s |
| Shmoosh | 10.2602s |
| speedup | 1.200x |
| packed encode | 0.0140s |
| packed attention | 0.0285s |
| scheduled quantized | 0.0648s |

The smoke passed, so K6/no-QJL moved to the three-case 1024 suite.

## Three-Case Suites

K6/no-QJL output:

```text
captures/image-policy-suite-juggernaut-up0-self-attn1-firstblocks-gated70pct-k6-noqjl-1024
```

K7/no-QJL same-code comparison output:

```text
captures/image-policy-suite-juggernaut-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-1024-contmask
```

| Policy | Min PSNR | Mean PSNR | Mean speedup | Mean baseline s | Mean Shmoosh s |
| --- | ---: | ---: | ---: | ---: | ---: |
| K6/no-QJL 70% | 50.38 dB | 52.91 dB | 1.082x | 10.5028 | 9.7051 |
| K7/no-QJL 70% | 51.87 dB | 53.96 dB | 1.066x | 10.4469 | 9.8021 |

Per-case K6/no-QJL:

| Case | Baseline s | Shmoosh s | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| `reading-nook-seed1-1024` | 12.5993 | 10.5046 | 1.199x | 51.58 dB |
| `maple-leaf-seed2-1024` | 9.4868 | 9.2486 | 1.026x | 50.38 dB |
| `misty-lake-seed3-1024` | 9.4223 | 9.3621 | 1.006x | 56.78 dB |

Per-case K7/no-QJL:

| Case | Baseline s | Shmoosh s | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| `reading-nook-seed1-1024` | 12.4530 | 10.5039 | 1.186x | 51.87 dB |
| `maple-leaf-seed2-1024` | 9.4894 | 9.4705 | 1.002x | 52.19 dB |
| `misty-lake-seed3-1024` | 9.3983 | 9.4320 | 0.996x | 57.82 dB |

## Readout

The continuation-byte mask is a small but sensible compact-K kernel cleanup. It
does not change the policy story by itself, but it lowers measured quantized
call time in the image traces.

K6/no-QJL is now a real speed-mode candidate:

- it uses only `52` bytes/vector versus K7's `60`;
- it roughly ties or slightly beats K7 on mean runtime in same-code image
  validation;
- it loses about `1.0 dB` mean PSNR and `1.5 dB` minimum PSNR versus K7.

K7/no-QJL remains the preferred 1024 self-attention default because the quality
margin is clearer than the K6 speed margin.

Next kernel pressure points:

1. Reduce or fuse the rotate/bucketize encode path.
2. Try a true grouped packed-code consumer only if it can keep `tl.dot`-class
   throughput.
3. Test fixed-shape CUDA graphs or compile once the processor stack stops
   changing.
