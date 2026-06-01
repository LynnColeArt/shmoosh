# Late-Step Self-Attention Sweep: 2026-06-01

This slice starts testing the denoising-layer part of the three-layer speed
plan. The target is SDXL self-attention at 1024x1024, where spatial key counts
are large enough that packed K should matter more than the 77-token text
cross-attention path.

## Harness Update

`shmoosh-image-module-sweep` now accepts scheduled denoising windows:

- `--quantize-start-step`
- `--quantize-end-step`
- `--quantize-start-percent`
- `--quantize-end-percent`

The sweep records both the requested window and the resolved absolute step in
`summary.json`, `summary.csv`, per-module `metrics.json`, and
`suggested_policy.json`. This lets single-module sweeps test the same
trajectory-aware policy surface as full policy A/B runs.

## 1024 Runtime Note

The first 1024 run without `--model-cpu-offload` completed denoising but hit
CUDA OOM during VAE decode on the RTX 4070. `nvidia-smi` showed a local
`llama-server` using about 2.3 GiB, and the non-offloaded SDXL process had only
about 67 MiB free at decode time.

For local 12GB validation runs, use:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run shmoosh-image-module-sweep ... --model-cpu-offload
```

## Single-Module Probe

Command shape:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run shmoosh-image-module-sweep \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --config stabilityai/stable-diffusion-xl-base-1.0 \
  --pipeline-class sdxl \
  --component unet \
  --module-filter attn1 \
  --module-indices 24,34,44 \
  --prompt "a cozy reading nook with a green velvet chair, a small wooden table, warm window light, realistic photo" \
  --output-dir captures/image-module-sweep-juggernaut-self-attn1-up0-firstblocks-k6-gate50-1024 \
  --steps 20 \
  --height 1024 \
  --width 1024 \
  --guidance-scale 5.0 \
  --seed 1 \
  --dtype fp16 \
  --device cuda \
  --model-cpu-offload \
  --local-files-only \
  --bits 6 \
  --qjl-bits 128 \
  --attention-backend packed \
  --packed-backend auto \
  --quantize-start-percent 0.5 \
  --candidate-psnr-db 45
```

The `attn1`-filtered module indices map to these U-Net modules:

| Filtered index | Module | PSNR | MSE | Baseline s | Shmoosh s |
| --- | --- | ---: | ---: | ---: | ---: |
| 24 | `up_blocks.0.attentions.0.transformer_blocks.0.attn1` | 51.93 dB | 0.00000641 | 11.9902 | 10.0727 |
| 34 | `up_blocks.0.attentions.1.transformer_blocks.0.attn1` | 53.66 dB | 0.00000431 | 11.9902 | 9.0515 |
| 44 | `up_blocks.0.attentions.2.transformer_blocks.0.attn1` | 53.26 dB | 0.00000472 | 11.9902 | 8.8896 |

The speed numbers are directional only because module sweeps reuse one baseline
image and run candidates sequentially in the same loaded process. The important
first-pass signal is that all three single-module late-step self-attention
probes cleared the fidelity gate at native SDXL resolution.

## Composition Smoke

The three single-module candidates were promoted into:

```text
configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated50pct-k6-qjl128-policy.json
```

Combined A/B command shape:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run shmoosh-image-ab-smoke \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --config stabilityai/stable-diffusion-xl-base-1.0 \
  --pipeline-class sdxl \
  --component unet \
  --policy-file configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated50pct-k6-qjl128-policy.json \
  --prompt "a cozy reading nook with a green velvet chair, a small wooden table, warm window light, realistic photo" \
  --output-dir captures/image-ab-juggernaut-up0-self-attn1-firstblocks-gated50pct-k6-1024-reading-nook \
  --steps 20 \
  --height 1024 \
  --width 1024 \
  --guidance-scale 5.0 \
  --seed 1 \
  --dtype fp16 \
  --device cuda \
  --model-cpu-offload \
  --local-files-only \
  --bits 6 \
  --qjl-bits 128 \
  --attention-backend packed \
  --packed-backend auto
```

Result:

| Output dir | PSNR | MSE | MAE | Baseline s | Shmoosh s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `captures/image-ab-juggernaut-up0-self-attn1-firstblocks-gated50pct-k6-1024-reading-nook` | 50.57 dB | 0.00000878 | 0.00103344 | 11.7899 | 9.9105 |

The composed policy uses global module indices 48, 68, and 88 when selected
from the full U-Net attention list. Each module stays exact for the first 50%
of denoising, resolving to step 10 in a 20-step run.

## Three-Case 1024 Suite

The exploratory self-attention policy then ran against the native-resolution
validation suite:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run shmoosh-image-policy-suite \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --config stabilityai/stable-diffusion-xl-base-1.0 \
  --pipeline-class sdxl \
  --component unet \
  --policy-file configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated50pct-k6-qjl128-policy.json \
  --case-file configs/underpaint-juggernaut-validation-1024-cases.json \
  --output-dir captures/image-policy-suite-juggernaut-up0-self-attn1-firstblocks-gated50pct-k6-1024 \
  --dtype fp16 \
  --device cuda \
  --model-cpu-offload \
  --local-files-only \
  --bits 6 \
  --qjl-bits 128 \
  --attention-backend packed \
  --packed-backend auto
```

| Case | Baseline s | Shmoosh s | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| `reading-nook-seed1-1024` | 12.1679 | 10.2707 | 1.185x | 50.57 dB |
| `maple-leaf-seed2-1024` | 8.4584 | 8.5038 | 0.995x | 49.62 dB |
| `misty-lake-seed3-1024` | 8.4089 | 8.7074 | 0.966x | 57.67 dB |

Suite aggregate:

- min PSNR: `49.62 dB`
- mean PSNR: `52.62 dB`
- max MSE: `0.00001091`
- mean baseline: `9.6784s`
- mean Shmoosh: `9.1606s`
- mean speedup: `1.057x`

## Processor Trace

The composed self-attention policy was traced on the reading-nook 1024 case:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run shmoosh-image-ab-smoke \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --config stabilityai/stable-diffusion-xl-base-1.0 \
  --pipeline-class sdxl \
  --component unet \
  --policy-file configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated50pct-k6-qjl128-policy.json \
  --prompt "a cozy reading nook with a green velvet chair, a small wooden table, warm window light, realistic photo" \
  --output-dir captures/image-ab-juggernaut-up0-self-attn1-firstblocks-gated50pct-k6-1024-trace-reading-nook \
  --steps 20 \
  --height 1024 \
  --width 1024 \
  --guidance-scale 5.0 \
  --seed 1 \
  --dtype fp16 \
  --device cuda \
  --model-cpu-offload \
  --local-files-only \
  --bits 6 \
  --qjl-bits 128 \
  --attention-backend packed \
  --packed-backend auto \
  --trace-processor-timing
```

Trace summary:

```text
baseline_seconds=11.4621
shmoosh_seconds=9.6313
psnr=50.57dB
record_count=360
```

Processor timing:

| Phase | Calls | Seconds | Mean per call |
| --- | ---: | ---: | ---: |
| `packed_attention` | 30 | 0.1329 | 4.431 ms |
| `packed_encode` | 30 | 0.0750 | 2.500 ms |
| `scheduled_quantized` | 30 | 0.2348 | 7.827 ms |
| `scheduled_exact` | 30 | 0.0260 | 0.867 ms |
| `policy_dispatch` | 60 | 0.0002 | 0.003 ms |

Encode subphases:

| Phase | Calls | Seconds | Mean per call |
| --- | ---: | ---: | ---: |
| `encode_pack_residual_signs` | 30 | 0.0260 | 0.865 ms |
| `encode_pack_codes` | 30 | 0.0256 | 0.855 ms |
| `encode_residual_project` | 30 | 0.0121 | 0.403 ms |
| `encode_rotate_bucketize` | 30 | 0.0054 | 0.178 ms |
| `encode_normalize` | 30 | 0.0039 | 0.132 ms |

This flips the bottleneck shape from text-key cross-attention. In selected
cross-attention, tiny 77-token K made encode dominate. In 1024-token
self-attention, the fused streaming attention path is the larger measured
component, while encode remains meaningful but secondary.

## Readout

This is a real positive signal, but still not the full runtime answer:

- late-step K6/QJL128 self-attention can pass native 1024 fidelity on the first
  reading-nook prompt;
- composition across the first self-attention block in three `up_blocks.0`
  attention groups also passes the first prompt;
- the three-case 1024 suite cleared fidelity with a small mean runtime win;
- the processor trace shows self-attention needs attention-kernel work as much
  as encode work;
- the policy has not yet been mixed with the cached cross-attention policy.

Next slice:

1. Composition-test it with the cached cross-attention policy.
2. If the combined policy clears, run the three-case 1024 suite on the combined
   denoising policy.
3. After composition, profile the self-attention streaming kernel and compare
   QJL64/no-QJL variants for this 1024-token path.
