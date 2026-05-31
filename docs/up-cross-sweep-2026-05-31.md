# Up-Block Cross-Attention Sweep: 2026-05-31

## Setup

This sweep expands laterally from the validated module-49 K5 policy into
neighboring cross-attention modules in:

```text
up_blocks.0.attentions.0
```

Source checkpoint, used read-only:

```text
/home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors
```

Command:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run turbo-d-image-module-sweep \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --prompt "a restored archival photo of a brass compass on a workbench" \
  --steps 20 \
  --height 512 \
  --width 512 \
  --module-indices 49,51,53,55,57,59,61,63,65,67 \
  --bits 5 \
  --qjl-bits 128 \
  --model-cpu-offload \
  --local-files-only \
  --codebook-samples 20000 \
  --candidate-psnr-db 30 \
  --output-dir captures/image-module-sweep-juggernaut-up0-attn0-cross-k5-20step
```

## Single-Module Results

| Module | Block | MSE | MAE | PSNR | Decision |
| ---: | --- | ---: | ---: | ---: | --- |
| 67 | transformer_blocks.9.attn2 | 0.00004664 | 0.00330807 | 43.31 dB | quantize |
| 59 | transformer_blocks.5.attn2 | 0.00006002 | 0.00394770 | 42.22 dB | quantize |
| 49 | transformer_blocks.0.attn2 | 0.00050738 | 0.01085879 | 32.95 dB | quantize |
| 65 | transformer_blocks.8.attn2 | 0.00053629 | 0.01251233 | 32.71 dB | quantize |
| 61 | transformer_blocks.6.attn2 | 0.00085147 | 0.01329195 | 30.70 dB | quantize, borderline |
| 63 | transformer_blocks.7.attn2 | 0.00166511 | 0.02451780 | 27.79 dB | exact |
| 55 | transformer_blocks.3.attn2 | 0.00298297 | 0.03143149 | 25.25 dB | exact |
| 51 | transformer_blocks.1.attn2 | 0.00337255 | 0.03222524 | 24.72 dB | exact |
| 53 | transformer_blocks.2.attn2 | 0.00387312 | 0.03391086 | 24.12 dB | exact |
| 57 | transformer_blocks.4.attn2 | 0.00396148 | 0.03413988 | 24.02 dB | exact |

The safe modules are not contiguous. A simple "quantize all cross-attention in
this up block" policy would include several bad modules. Policy needs explicit
module selection.

## Combined Policy

The generated gate-30 policy enables:

```text
49, 59, 61, 65, 67
```

Combined A/B command:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run turbo-d-image-ab-smoke \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --prompt "a restored archival photo of a brass compass on a workbench" \
  --steps 20 \
  --height 512 \
  --width 512 \
  --policy-file configs/underpaint-juggernaut-sdxl-up0-cross-k5-qjl128-policy.json \
  --model-cpu-offload \
  --local-files-only \
  --output-dir captures/image-ab-juggernaut-up0-cross-k5-combined-20step
```

Combined result:

```text
mse=0.00050956
mae=0.01113889
psnr=32.93 dB
```

The combined policy preserved image structure on the compass prompt and did not
compound errors beyond the single module-49 K5 result in a meaningful way.

Tracked policy:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-k5-qjl128-policy.json
```

## Validation Suite

The tracked combined policy was then run through the same three-case validation
suite used for the single-module K5 policy:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run turbo-d-image-policy-suite \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --policy-file configs/underpaint-juggernaut-sdxl-up0-cross-k5-qjl128-policy.json \
  --case-file configs/underpaint-juggernaut-validation-cases.json \
  --steps 20 \
  --height 512 \
  --width 512 \
  --model-cpu-offload \
  --local-files-only \
  --output-dir captures/image-policy-suite-juggernaut-up0-cross-k5-20step
```

| Case | Seed | MSE | MAE | PSNR |
| --- | ---: | ---: | ---: | ---: |
| reading-nook-seed1 | 1 | 0.00011296 | 0.00511375 | 39.47 dB |
| maple-leaf-seed2 | 2 | 0.00007037 | 0.00383828 | 41.53 dB |
| misty-lake-seed3 | 3 | 0.00001206 | 0.00106764 | 49.19 dB |

Aggregate:

```text
mean_psnr=43.39 dB
min_psnr=39.47 dB
max_mse=0.00011296
```

## Interpretation

This is the first evidence that Turbo-D can cover multiple SDXL attention
modules in one image run without visible collapse. It is still narrow evidence:
four prompt/seed pairs, 512px, 20 denoising steps, and only one U-Net up-block
attention group.

The result strongly supports explicit module policy. Within the same attention
group and same kind of attention, some modules are safe at K5 and others are
not.

## Next Slice

1. Add percentage-based timestep windows so policies scale beyond 20-step runs.
2. Draft the production-path design for packed K storage and a Torch/Triton
   attention kernel.
3. Add a stricter stress lane once the next attention group has a candidate:
   more seeds, more steps, and at least one non-512 size.
