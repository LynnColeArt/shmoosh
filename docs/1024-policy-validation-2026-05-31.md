# 1024 Policy Validation: 2026-05-31

## Rationale

The 512px sweeps are useful for wiring and fast rejection, but they are not the
right quality gate for SDXL-class local workflows. Native-resolution testing
needs to happen at 1024px, where the model's denoising behavior and attention
token counts better match real use.

## Gate Sweep

The full mixed K5/K6 policy was retested at 1024x1024 with percentage timestep
gates:

| Exact First | Resolved Start Step | MSE | MAE | PSNR |
| ---: | ---: | ---: | ---: | ---: |
| 10% | 2 | 0.00014880 | 0.00378267 | 38.27 dB |
| 20% | 4 | 0.00028238 | 0.00555338 | 35.49 dB |
| 30% | 6 | 0.00002047 | 0.00134793 | 46.89 dB |

The native-resolution curve is not monotonic: 20% was worse than 10% on the
compass prompt, while 30% recovered strongly. This reinforces the
trajectory-aware interpretation. The useful control surface is still:

```text
module, key bit depth, QJL sketch size, timestep window
```

but the timestep window has to be selected at the target resolution.

## Accepted 1024 Candidate

```text
configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated30pct-k5-k6-qjl128-policy.json
```

This policy leaves the first 30% of denoising exact and quantizes the selected
seven modules for the remaining 70% of steps.

## Validation Suite

Command:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run shmoosh-image-policy-suite \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --policy-file configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated30pct-k5-k6-qjl128-policy.json \
  --case-file configs/underpaint-juggernaut-validation-1024-cases.json \
  --model-cpu-offload \
  --local-files-only \
  --output-dir captures/image-policy-suite-juggernaut-up0-cross-mixed-gated30pct-1024-20step
```

Results:

| Case | Seed | MSE | MAE | PSNR |
| --- | ---: | ---: | ---: | ---: |
| reading-nook-seed1-1024 | 1 | 0.00001419 | 0.00129445 | 48.48 dB |
| maple-leaf-seed2-1024 | 2 | 0.00001135 | 0.00121853 | 49.45 dB |
| misty-lake-seed3-1024 | 3 | 0.00000141 | 0.00030250 | 58.52 dB |

Aggregate:

```text
mean_psnr=52.15 dB
min_psnr=48.48 dB
max_mse=0.00001419
```

## 30-Step Horizon Transfer

The same exact-first-30% policy was then tested at 30 denoising steps. The
percentage gate resolved to start step 9.

Compass A/B:

```text
mse=0.00007670
mae=0.00257249
psnr=41.15 dB
```

Validation command:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run shmoosh-image-policy-suite \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --policy-file configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated30pct-k5-k6-qjl128-policy.json \
  --case-file configs/underpaint-juggernaut-validation-1024-30step-cases.json \
  --model-cpu-offload \
  --local-files-only \
  --output-dir captures/image-policy-suite-juggernaut-up0-cross-mixed-gated30pct-1024-30step
```

Results:

| Case | Seed | MSE | MAE | PSNR |
| --- | ---: | ---: | ---: | ---: |
| reading-nook-seed1-1024-30step | 1 | 0.00000928 | 0.00098883 | 50.33 dB |
| maple-leaf-seed2-1024-30step | 2 | 0.00001396 | 0.00123418 | 48.55 dB |
| misty-lake-seed3-1024-30step | 3 | 0.00000133 | 0.00027147 | 58.76 dB |

Aggregate:

```text
mean_psnr=52.55 dB
min_psnr=48.55 dB
max_mse=0.00001396
```

## Interpretation

The 30% native-resolution gate is the best policy candidate so far. It validates
the scientific thesis more cleanly than the 512px runs: early denoising has to
stay exact long enough for high-resolution structure to settle, after which
attention-key compression can activate without large image drift.

This is still correctness evidence, not production speed evidence. The current
processor is a slow NumPy reference path. Practical value on 4070/3080 hardware
depends on packed K storage and a fused attention path.

## Next Slice

Packed-K design is recorded in `docs/packed-k-design-2026-05-31.md`. Under SDXL
cross-attention assumptions, the accepted policy has a `1.93x` packed-key ratio
during quantized steps and saves `26.65 MiB` of selected-key payload across a
30-step horizon.

The accepted policy has since cleared the packed-backend parity lane at both
20 and 30 steps; see `docs/packed-backend-validation-2026-05-31.md`.

Next:

1. Cache codec resources per module/backend instead of rebuilding them per
   attention call.
2. Warm Triton kernels before timing.
3. Fuse or reduce query-side projection overhead.
4. Avoid materializing full score tensors once the score kernel is stable.
