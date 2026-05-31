# Mixed-Precision Policy Support: 2026-05-31

## Code Path

Policy loading now supports per-module processor overrides. The top-level
`turbo_policy` remains the default, and each `quantized_modules` entry can
override processor fields directly or with a nested `turbo_policy` object.

Example:

```json
{
  "turbo_policy": {
    "bits": 5,
    "qjl_bits": 128,
    "quantize_keys": true,
    "quantize_values": false
  },
  "quantized_modules": [
    {
      "index": 67,
      "name": "up_blocks.0.attentions.0.transformer_blocks.9.attn2"
    },
    {
      "index": 87,
      "name": "up_blocks.0.attentions.1.transformer_blocks.9.attn2",
      "bits": 6
    }
  ]
}
```

The image A/B and policy-suite CLIs now report mixed processor metadata with
per-module precision assignments.

## Mixed Bridge Candidate

The accepted mixed bridge policy is:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-mixed-bridge-k5-k6-qjl128-policy.json
```

It enables:

| Module | Group | Bits |
| ---: | --- | ---: |
| 67 | up_blocks.0.attentions.0.transformer_blocks.9.attn2 | 5 |
| 87 | up_blocks.0.attentions.1.transformer_blocks.9.attn2 | 6 |

Compass A/B:

```text
mse=0.00006197
mae=0.00310667
psnr=42.08 dB
```

## Failed Expansion Probes

The larger mixed policies did not compose on the compass prompt:

| Modules | Policy | MSE | MAE | PSNR |
| --- | --- | ---: | ---: | ---: |
| 49,59,61,65,67,79,87 | K5 for attention.0, K6 for 79/87 | 0.00369339 | 0.03347438 | 24.33 dB |
| 49,59,61,65,67,79,87 | K6 for all selected modules | 0.00148685 | 0.02303079 | 28.28 dB |
| 49,59,65,67,79,87 | dropped borderline module 61 | 0.00402291 | 0.03433094 | 23.95 dB |
| 59,67,79,87 | high-confidence four-module core | 0.00315270 | 0.03110624 | 25.01 dB |
| 67,79,87 | bridge plus module 79 | 0.00248407 | 0.02741794 | 26.05 dB |
| 59,67,87 | bridge plus module 59 | 0.00298553 | 0.03133324 | 25.25 dB |

This means group-local validation does not imply cross-group composition. The
first safe cross-group policy is narrow: one module from each group.

## Validation Suite

The bridge policy was run through the three-case validation suite:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run turbo-d-image-policy-suite \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --policy-file configs/underpaint-juggernaut-sdxl-up0-cross-mixed-bridge-k5-k6-qjl128-policy.json \
  --case-file configs/underpaint-juggernaut-validation-cases.json \
  --steps 20 \
  --height 512 \
  --width 512 \
  --model-cpu-offload \
  --local-files-only \
  --output-dir captures/image-policy-suite-juggernaut-up0-cross-mixed-bridge-20step
```

| Case | Seed | MSE | MAE | PSNR |
| --- | ---: | ---: | ---: | ---: |
| reading-nook-seed1 | 1 | 0.00016397 | 0.00631037 | 37.85 dB |
| maple-leaf-seed2 | 2 | 0.00006172 | 0.00346452 | 42.10 dB |
| misty-lake-seed3 | 3 | 0.00000933 | 0.00092828 | 50.30 dB |

Aggregate:

```text
mean_psnr=43.42 dB
min_psnr=37.85 dB
max_mse=0.00016397
```

## Interpretation

Mixed precision is necessary but not sufficient. It lets us express the K5/K6
policy the experiments are asking for, but denoising error still compounds
across attention groups. The next policy mechanism should be timestep-aware
module gating, so a module can be exact during fragile early steps and quantized
later.

## Next Slice

1. Add timestep-window support to policy entries.
2. Re-test the rejected larger mixed policies with early-step exact fallback.
3. Continue the up-block sweep only after the policy machinery can express
   composition constraints.
