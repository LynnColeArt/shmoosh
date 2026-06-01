# Cross + Self-Attention Composition: 2026-06-01

This slice tests whether the cached cross-attention policy and the new
late-step self-attention policy compose at 1024x1024.

The key result: composition is viable, but the 50% self-attention gate is too
aggressive. Moving self-attention activation to 70% recovers most of the quality
while keeping a small mean runtime signal.

## Inputs

Cross-attention base policy:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated30pct-k5-k6-qjl128-policy.json
```

Self-attention source policy:

```text
configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated50pct-k6-qjl128-policy.json
```

The composed policies keep:

- cross-attention exact for the first 30% of denoising, resolving to step 6 in
  a 20-step run;
- self-attention exact for either the first 50% or first 70%, resolving to step
  10 or step 14 in a 20-step run;
- exact V everywhere;
- cached cross-attention K/V enabled.

## 50% Self Gate

Policy:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-cache-self-attn1-gated-k5-k6-qjl128-policy.json
```

Reading-nook smoke:

| Output dir | PSNR | MSE | Baseline s | Shmoosh s |
| --- | ---: | ---: | ---: | ---: |
| `captures/image-ab-juggernaut-up0-cross-cache-self-attn1-gated-k5-k6-1024-reading-nook` | 48.82 dB | 0.00001313 | 11.6855 | 9.8911 |

Three-case suite:

| Case | Baseline s | Shmoosh s | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| `reading-nook-seed1-1024` | 11.4412 | 10.4851 | 1.091x | 48.82 dB |
| `maple-leaf-seed2-1024` | 9.0457 | 9.2386 | 0.979x | 47.49 dB |
| `misty-lake-seed3-1024` | 9.0405 | 9.3791 | 0.964x | 56.20 dB |

Aggregate:

- min PSNR: `47.49 dB`
- mean PSNR: `50.84 dB`
- max MSE: `0.00001783`
- mean speedup: `1.015x`

Readout: this technically passes a loose image-delta gate, but it is not a good
candidate. It loses too much fidelity versus either source policy and barely
moves mean runtime.

## 70% Self Gate

Policy:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-cache-self-attn1-gated70pct-k5-k6-qjl128-policy.json
```

Reading-nook smoke:

| Output dir | PSNR | MSE | Baseline s | Shmoosh s |
| --- | ---: | ---: | ---: | ---: |
| `captures/image-ab-juggernaut-up0-cross-cache-self-attn1-gated70pct-k5-k6-1024-reading-nook` | 50.08 dB | 0.00000982 | 11.5301 | 9.9446 |

Three-case suite:

| Case | Baseline s | Shmoosh s | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| `reading-nook-seed1-1024` | 11.5301 | 9.9446 | 1.159x | 50.08 dB |
| `maple-leaf-seed2-1024` | 8.7118 | 8.7455 | 0.996x | 49.17 dB |
| `misty-lake-seed3-1024` | 8.5364 | 8.7596 | 0.975x | 56.72 dB |

Aggregate:

- min PSNR: `49.17 dB`
- mean PSNR: `51.99 dB`
- max MSE: `0.00001212`
- mean speedup: `1.048x`

Readout: this is the better composition policy. It recovers quality compared to
the 50% self gate, but it is still only marginally better on runtime than the
cached cross-attention policy alone.

## Interpretation

The composition result supports the current thesis:

- static "safe module" decisions are insufficient;
- timestep windows are a real control surface;
- self-attention can compose with cross-attention, but needs a later activation
  window than it needed in isolation;
- the combined policy should not be treated as a production default yet.

The next useful slice is not to add more modules. It is to improve the
underlying self-attention path:

1. Profile the streaming self-attention kernel directly.
2. Compare QJL64 and no-QJL K6/K7 variants for the self-attention modules.
3. Re-run the 70% composition after any kernel or policy-cost improvement.
