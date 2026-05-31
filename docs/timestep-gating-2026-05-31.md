# Timestep-Gated Policy Sweep: 2026-05-31

## Code Path

Policy entries now support timestep windows:

```json
{
  "index": 49,
  "name": "up_blocks.0.attentions.0.transformer_blocks.0.attn2",
  "quantize_start_step": 4,
  "quantize_end_step": null
}
```

They also support percentage windows, which are resolved per run with
`ceil(total_steps * percent)`:

```json
{
  "index": 49,
  "name": "up_blocks.0.attentions.0.transformer_blocks.0.attn2",
  "quantize_start_percent": 0.2,
  "quantize_end_percent": null
}
```

The default window is step `0` through the end of the denoising trajectory. A
non-zero `quantize_start_step` leaves the original Diffusers processor installed
for early steps, then switches to the Shmoosh processor for later steps. A
percentage start window does the same thing, but scales with the requested
number of denoising steps.

The image A/B and policy-suite CLIs track the current denoising step through
Diffusers `callback_on_step_end`, seeding step `0` before the call and advancing
to `i + 1` after each step.

## Target Policy

The experiment retested the failed full mixed policy:

| Modules | Bits | Window |
| --- | --- | --- |
| 49, 59, 61, 65, 67 | K5 | varied |
| 79, 87 | K6 | varied |

Without timestep gating, this policy failed the compass A/B:

```text
mse=0.00369339
mae=0.03347438
psnr=24.33 dB
```

## Exact-First Sweep

At 20 denoising steps:

| Exact First Steps | Percent | Quantized Steps | MSE | MAE | PSNR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0% | 20 | 0.00369339 | 0.03347438 | 24.33 dB |
| 2 | 10% | 18 | 0.00092729 | 0.01709632 | 30.33 dB |
| 4 | 20% | 16 | 0.00002115 | 0.00186997 | 46.75 dB |
| 6 | 30% | 14 | 0.00001388 | 0.00163395 | 48.57 dB |
| 10 | 50% | 10 | 0.00000363 | 0.00072372 | 54.40 dB |

The tracked candidate uses exact-first 4 steps:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated20-k5-k6-qjl128-policy.json
```

The horizon-scaled version uses exact-first 20%:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated20pct-k5-k6-qjl128-policy.json
```

## Validation Suite

The exact-first-4 policy was run through the same three-case validation suite:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run shmoosh-image-policy-suite \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --policy-file configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated20-k5-k6-qjl128-policy.json \
  --case-file configs/underpaint-juggernaut-validation-cases.json \
  --steps 20 \
  --height 512 \
  --width 512 \
  --model-cpu-offload \
  --local-files-only \
  --output-dir captures/image-policy-suite-juggernaut-up0-cross-mixed-gated20-20step
```

| Case | Seed | MSE | MAE | PSNR |
| --- | ---: | ---: | ---: | ---: |
| reading-nook-seed1 | 1 | 0.00004122 | 0.00297608 | 43.85 dB |
| maple-leaf-seed2 | 2 | 0.00003032 | 0.00180282 | 45.18 dB |
| misty-lake-seed3 | 3 | 0.00000297 | 0.00040782 | 55.27 dB |

Aggregate:

```text
mean_psnr=48.10 dB
min_psnr=43.85 dB
max_mse=0.00004122
```

## Percentage Window Stress

The exact-first-20% policy reproduced the absolute step-4 result at 20 steps
and stayed stable when the denoising horizon changed to 30 steps:

| Check | Resolved Start Step | MSE | MAE | PSNR |
| --- | ---: | ---: | ---: | ---: |
| 20 steps, 512x512 | 4 | 0.00002115 | 0.00186997 | 46.75 dB |
| 30 steps, 512x512 | 6 | 0.00002689 | 0.00236529 | 45.70 dB |
| 20 steps, 768x512 | 4 | 0.00011053 | 0.00330194 | 39.57 dB |

The 30-step run confirms that the gate is not a hard-coded 20-step artifact.
The non-square run has a larger delta, but still remains far above the failed
ungated mixed policy.

## Native-Resolution Follow-Up

The 512px policy was retested at 1024x1024 in
`docs/1024-policy-validation-2026-05-31.md`. At native SDXL size, the 30% gate
is much stronger than the 20% gate:

| Exact First | Resolved Start Step | PSNR |
| ---: | ---: | ---: |
| 10% | 2 | 38.27 dB |
| 20% | 4 | 35.49 dB |
| 30% | 6 | 46.89 dB |

The accepted 1024 policy is:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated30pct-k5-k6-qjl128-policy.json
```

It cleared the three-case 1024 validation suite with `min_psnr=48.48 dB` and
`mean_psnr=52.15 dB`, then held at 30 steps with `min_psnr=48.55 dB` and
`mean_psnr=52.55 dB`.

## Interpretation

This is the strongest policy result so far. Exact-first gating rescued a mixed
policy that failed under static activation, while still quantizing seven modules
for 16 of 20 denoising steps.

The control surface is now:

```text
module, key bit depth, QJL sketch size, timestep window
```

The result supports the trajectory-aware thesis: diffusion quantization needs to
respect early denoising sensitivity, not just per-layer sensitivity.

## Next Slice

1. Start the packed-K production design against the accepted 1024 policy.
2. Estimate packed K5/K6 plus QJL-128 bandwidth savings for the selected modules.
3. Prototype a Torch-side packed-key metadata format before attempting kernels.
