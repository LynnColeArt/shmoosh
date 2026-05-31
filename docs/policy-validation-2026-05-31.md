# Policy Validation: 2026-05-31

## Setup

This validation suite tested the K5 candidate policy across additional
prompt/seed cases:

```text
configs/underpaint-juggernaut-sdxl-k5-qjl128-policy.json
configs/underpaint-juggernaut-validation-cases.json
```

Command:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run turbo-d-image-policy-suite \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --policy-file configs/underpaint-juggernaut-sdxl-k5-qjl128-policy.json \
  --case-file configs/underpaint-juggernaut-validation-cases.json \
  --steps 20 \
  --height 512 \
  --width 512 \
  --model-cpu-offload \
  --local-files-only \
  --output-dir captures/image-policy-suite-juggernaut-k5-20step
```

The suite loads the pipeline once, runs each case as an exact baseline, then
installs the policy processor for a same-seed Turbo-D run. Outputs are written
per case, with aggregate summaries here:

```text
captures/image-policy-suite-juggernaut-k5-20step/summary.csv
captures/image-policy-suite-juggernaut-k5-20step/summary.json
```

## Results

| Case | Seed | MSE | MAE | PSNR | Visual Check |
| --- | ---: | ---: | ---: | ---: | --- |
| reading-nook-seed1 | 1 | 0.00021487 | 0.00763441 | 36.68 dB | closest visible differences around chair/window texture |
| maple-leaf-seed2 | 2 | 0.00005280 | 0.00330658 | 42.77 dB | structure preserved |
| misty-lake-seed3 | 3 | 0.00000360 | 0.00051561 | 54.44 dB | nearly identical |

Aggregate over the three additional cases:

```text
mean_mse=0.00009042
mean_mae=0.00381886
mean_psnr=44.63 dB
min_psnr=36.68 dB
```

Together with the original compass prompt from
`docs/precision-sweep-2026-05-31.md`, the K5 candidate has now held across four
prompt/seed pairs at 20 steps.

## Interpretation

The K5 policy is no longer just a single-prompt accident. It remains narrow:
only module 49 is enabled, only 512px SDXL/Juggernaut has been tested, and only
20-step image smoke quality is covered. But this is enough to justify expanding
laterally to neighboring up-block cross-attention modules.

The current processor is still a NumPy behavioral path, so runtime timings and
VRAM numbers are not optimization evidence.

## Combined Up-Block Policy

The later combined up-block candidate is tracked at:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-k5-qjl128-policy.json
```

That policy enables modules `49,59,61,65,67` in
`up_blocks.0.attentions.0` and leaves the neighboring failed modules exact. It
was run through the same three-case suite:

```text
captures/image-policy-suite-juggernaut-up0-cross-k5-20step/summary.csv
captures/image-policy-suite-juggernaut-up0-cross-k5-20step/summary.json
```

| Case | Seed | MSE | MAE | PSNR |
| --- | ---: | ---: | ---: | ---: |
| reading-nook-seed1 | 1 | 0.00011296 | 0.00511375 | 39.47 dB |
| maple-leaf-seed2 | 2 | 0.00007037 | 0.00383828 | 41.53 dB |
| misty-lake-seed3 | 3 | 0.00001206 | 0.00106764 | 49.19 dB |

Aggregate over the three additional cases:

```text
mean_psnr=43.39 dB
min_psnr=39.47 dB
max_mse=0.00011296
```

The combined candidate is therefore stronger than the single-prompt compass
result alone suggested. It still needs broader seed, resolution, and step-count
coverage before being treated as a production policy.

## K6 Up-Block Attention.1 Policy

The next up-block group needed a stricter policy:

```text
configs/underpaint-juggernaut-sdxl-up0-attn1-cross-k6-qjl128-policy.json
```

The K5 single-module gate found modules `79,85,87`, but the combined K5 policy
failed (`psnr=25.66 dB`). The accepted candidate uses K6 for modules `79,87`
and leaves module `85` exact. Validation outputs:

```text
captures/image-policy-suite-juggernaut-up0-attn1-cross-k6-20step/summary.csv
captures/image-policy-suite-juggernaut-up0-attn1-cross-k6-20step/summary.json
```

| Case | Seed | MSE | MAE | PSNR |
| --- | ---: | ---: | ---: | ---: |
| reading-nook-seed1 | 1 | 0.00005419 | 0.00333361 | 42.66 dB |
| maple-leaf-seed2 | 2 | 0.00002223 | 0.00199130 | 46.53 dB |
| misty-lake-seed3 | 3 | 0.00000559 | 0.00072002 | 52.52 dB |

Aggregate over the three additional cases:

```text
mean_psnr=47.24 dB
min_psnr=42.66 dB
max_mse=0.00005419
```

This is the first clear mixed-precision signal: one nearby up-block group has a
K5 candidate, while this group needs K6 for safe composition.

## Mixed Bridge Policy

Per-module precision support enabled a cross-group bridge policy:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-mixed-bridge-k5-k6-qjl128-policy.json
```

It enables module `67` at K5 and module `87` at K6. Larger mixed policies did
not compose on the compass prompt, but this bridge validated across the
three-case suite:

| Case | Seed | MSE | MAE | PSNR |
| --- | ---: | ---: | ---: | ---: |
| reading-nook-seed1 | 1 | 0.00016397 | 0.00631037 | 37.85 dB |
| maple-leaf-seed2 | 2 | 0.00006172 | 0.00346452 | 42.10 dB |
| misty-lake-seed3 | 3 | 0.00000933 | 0.00092828 | 50.30 dB |

Aggregate over the three additional cases:

```text
mean_psnr=43.42 dB
min_psnr=37.85 dB
max_mse=0.00016397
```

The mixed bridge confirms the code path and reveals the next missing policy
axis: timestep windows.

## Timestep-Gated Full Mixed Policy

The full mixed policy failed without timestep gating (`psnr=24.33 dB`), but
passed when the first 4 of 20 denoising steps stayed exact:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated20-k5-k6-qjl128-policy.json
```

Compass A/B:

```text
mse=0.00002115
mae=0.00186997
psnr=46.75 dB
```

Validation suite:

| Case | Seed | MSE | MAE | PSNR |
| --- | ---: | ---: | ---: | ---: |
| reading-nook-seed1 | 1 | 0.00004122 | 0.00297608 | 43.85 dB |
| maple-leaf-seed2 | 2 | 0.00003032 | 0.00180282 | 45.18 dB |
| misty-lake-seed3 | 3 | 0.00000297 | 0.00040782 | 55.27 dB |

Aggregate over the three additional cases:

```text
mean_psnr=48.10 dB
min_psnr=43.85 dB
max_mse=0.00004122
```

This is the strongest evidence so far that the missing policy axis is
trajectory-aware activation, not just bit depth.

## Next Slice

1. Add percentage-based timestep windows so policies scale beyond 20-step runs.
2. Stress the gated policy at more seeds and at a non-512 size.
3. Start a production-path design note for packed key storage and a Torch/Triton
   attention kernel against the timestep-aware policy surface.
