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

## Readout

This is a real positive signal, but still a small-scope one:

- late-step K6/QJL128 self-attention can pass native 1024 fidelity on the first
  reading-nook prompt;
- composition across the first self-attention block in three `up_blocks.0`
  attention groups also passes the first prompt;
- this policy is not accepted yet, because it has not cleared the three-case
  1024 suite and has not been mixed with the cached cross-attention policy.

Next slice:

1. Run the three-case 1024 suite for the self-attention policy.
2. Trace the composed self-attention run to split encode cost from fused
   attention cost.
3. If it clears, composition-test it with the cached cross-attention policy.
