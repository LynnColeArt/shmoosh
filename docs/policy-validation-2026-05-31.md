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

## Next Slice

1. Sweep neighboring up-block cross-attention modules with `K5 + QJL-128`.
2. Promote any modules that clear the 20-step validation gate into a combined
   policy.
3. Start a production-path design note for packed key storage and a Torch/Triton
   attention kernel.
