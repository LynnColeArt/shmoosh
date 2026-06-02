# Precision Policy Compare

This slice adds a same-process image comparison runner:

```text
shmoosh-image-policy-compare
```

The runner loads the pipeline once, renders one exact baseline per case, then
restores exact processors before each candidate policy. This removes the worst
one-policy-per-process comparison noise and makes precision-policy comparisons
share the same baseline image for each prompt/seed.

## Command

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1 uv run shmoosh-image-policy-compare \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config stabilityai/stable-diffusion-xl-base-1.0 \
  --component unet \
  --case-file configs/underpaint-juggernaut-validation-1024-cases.json \
  --output-dir captures/image-policy-compare-juggernaut-up0-self-attn1-k7-noqjl-1024-precision-20260602 \
  --dtype fp16 \
  --device cuda \
  --model-cpu-offload \
  --local-files-only \
  --attention-backend packed \
  --packed-backend auto \
  --code-format packed_t \
  --trace-processor-timing \
  --candidate ieee=configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-policy.json \
  --candidate all-tf32=configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-tf32-policy.json \
  --candidate score-value-tf32=configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-score-value-tf32-policy.json
```

Hardware:

```text
RTX 4070
1024x1024 SDXL/Juggernaut cases
20 denoising steps
exact first 70%, quantized final 30%
K7/no-QJL, packed_t, exact V
```

## Aggregate

Whole-image speedup includes one cold first baseline and should not be treated
as a steady-state UX promise. It is still useful for candidate-to-candidate
comparison because each candidate shares the same baseline image per case.

| Candidate | Min PSNR | Mean PSNR | Mean speedup | Packed attention | Packed encode | Scheduled quantized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ieee` | 52.0730 dB | 54.2689 dB | 1.753x | 1.3043 ms | 0.8649 ms | 3.5793 ms |
| `all_tf32` | 51.7043 dB | 54.1728 dB | 1.795x | 0.7883 ms | 0.9561 ms | 3.0088 ms |
| `score_value_tf32` | 52.0788 dB | 54.1603 dB | 1.815x | 0.8613 ms | 0.9565 ms | 3.2340 ms |

## Cases

| Candidate | Case | PSNR | Shmoosh seconds | Packed attention |
| --- | --- | ---: | ---: | ---: |
| `ieee` | reading-nook | 52.0730 dB | 10.7271 | 1.4003 ms |
| `all_tf32` | reading-nook | 51.7043 dB | 9.5609 | 0.8056 ms |
| `score_value_tf32` | reading-nook | 52.0788 dB | 9.3425 | 0.8801 ms |
| `ieee` | maple-leaf | 52.1226 dB | 9.6922 | 1.2439 ms |
| `all_tf32` | maple-leaf | 52.1556 dB | 9.6434 | 0.8323 ms |
| `score_value_tf32` | maple-leaf | 52.1602 dB | 9.5755 | 0.8804 ms |
| `ieee` | misty-lake | 58.6111 dB | 9.5516 | 1.2686 ms |
| `all_tf32` | misty-lake | 58.6584 dB | 10.0558 | 0.7269 ms |
| `score_value_tf32` | misty-lake | 58.2421 dB | 10.0280 | 0.8235 ms |

## Decision

Promote `score_value_tf32` as the RTX 4070 balanced fast-mode candidate:

```text
configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-score-value-tf32-policy.json
```

It keeps the hard-case reading-nook quality at the high-fidelity level
(`52.08 dB`) while reducing mean packed-attention time from `1.3043ms` to
`0.8613ms`, about a `34%` phase reduction.

Keep `ieee` as the high-fidelity reference/default:

```text
configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-policy.json
```

Keep `all_tf32` as the explicit maximum-speed precision mode with a known
hard-case PSNR tax:

```text
configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-tf32-policy.json
```

This is a 4070 decision, not yet a 3080 decision. The next hardware-specific
validation should repeat this same runner on the 3080.

## Next

The comparison runner should become the default way to compare image policies
that differ only in runtime format, precision, or kernel path. It is less useful
for absolute whole-pipeline speed claims until we add optional baseline warmup
or baseline repeat controls.

The next performance work should use `score_value_tf32` as the balanced
candidate and focus on reducing the remaining `scheduled_quantized` cost. The
current phase split says packed attention improved, but encode plus scheduling
still hides most of the kernel win at whole-image scale.
