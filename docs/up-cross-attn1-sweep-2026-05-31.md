# Up-Block Attention.1 Cross-Attention Sweep: 2026-05-31

## Setup

This sweep expands from `up_blocks.0.attentions.0` into the next SDXL up-block
attention group:

```text
up_blocks.0.attentions.1
```

The global cross-attention module indices are:

```text
69, 71, 73, 75, 77, 79, 81, 83, 85, 87
```

Source checkpoint, used read-only:

```text
/home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors
```

K5 single-module sweep:

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
  --module-indices 69,71,73,75,77,79,81,83,85,87 \
  --bits 5 \
  --qjl-bits 128 \
  --model-cpu-offload \
  --local-files-only \
  --codebook-samples 20000 \
  --candidate-psnr-db 30 \
  --output-dir captures/image-module-sweep-juggernaut-up0-attn1-cross-k5-20step
```

## K5 Single-Module Results

| Module | Block | MSE | MAE | PSNR | Decision |
| ---: | --- | ---: | ---: | ---: | --- |
| 87 | transformer_blocks.9.attn2 | 0.00004575 | 0.00363065 | 43.40 dB | candidate |
| 79 | transformer_blocks.5.attn2 | 0.00005094 | 0.00297832 | 42.93 dB | candidate |
| 85 | transformer_blocks.8.attn2 | 0.00084888 | 0.01409394 | 30.71 dB | candidate, borderline |
| 75 | transformer_blocks.3.attn2 | 0.00178149 | 0.02076316 | 27.49 dB | exact |
| 73 | transformer_blocks.2.attn2 | 0.00184625 | 0.02203867 | 27.34 dB | exact |
| 81 | transformer_blocks.6.attn2 | 0.00242064 | 0.02901060 | 26.16 dB | exact |
| 71 | transformer_blocks.1.attn2 | 0.00297058 | 0.03098887 | 25.27 dB | exact |
| 83 | transformer_blocks.7.attn2 | 0.00315262 | 0.03123195 | 25.01 dB | exact |
| 69 | transformer_blocks.0.attn2 | 0.00317857 | 0.03225932 | 24.98 dB | exact |
| 77 | transformer_blocks.4.attn2 | 0.00356783 | 0.03426987 | 24.48 dB | exact |

## Composition Probes

The single-module K5 gate did not compose:

| Modules | Bits | MSE | MAE | PSNR | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 79, 85, 87 | 5 | 0.00271782 | 0.02927642 | 25.66 dB | reject |
| 79, 87 | 5 | 0.00293874 | 0.02997779 | 25.32 dB | reject |
| 79, 87 | 6 | 0.00005620 | 0.00355609 | 42.50 dB | accept |
| 79, 85, 87 | 6 | 0.00134227 | 0.01892779 | 28.72 dB | reject |

The accepted tracked policy is therefore the K6 pair:

```text
configs/underpaint-juggernaut-sdxl-up0-attn1-cross-k6-qjl128-policy.json
```

## Validation Suite

The tracked K6 pair policy was run through the same three-case validation
suite:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run turbo-d-image-policy-suite \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --policy-file configs/underpaint-juggernaut-sdxl-up0-attn1-cross-k6-qjl128-policy.json \
  --case-file configs/underpaint-juggernaut-validation-cases.json \
  --steps 20 \
  --height 512 \
  --width 512 \
  --model-cpu-offload \
  --local-files-only \
  --output-dir captures/image-policy-suite-juggernaut-up0-attn1-cross-k6-20step
```

| Case | Seed | MSE | MAE | PSNR |
| --- | ---: | ---: | ---: | ---: |
| reading-nook-seed1 | 1 | 0.00005419 | 0.00333361 | 42.66 dB |
| maple-leaf-seed2 | 2 | 0.00002223 | 0.00199130 | 46.53 dB |
| misty-lake-seed3 | 3 | 0.00000559 | 0.00072002 | 52.52 dB |

Aggregate:

```text
mean_psnr=47.24 dB
min_psnr=42.66 dB
max_mse=0.00005419
```

## Interpretation

This group shows that single-module screening is necessary but not sufficient.
Modules 79 and 87 each looked safe at K5, but together they caused a large
image-level drift at K5. One extra key bit recovered the pair cleanly. Module 85
stays exact even at K6 because it destabilized the three-module composition.

The result points toward mixed-precision policy: `up_blocks.0.attentions.0`
currently has a K5 candidate, while `up_blocks.0.attentions.1` needs K6 for the
accepted pair.

## Next Slice

1. Add percentage-based timestep windows so policies scale beyond 20-step runs.
2. Stress the timestep-gated mixed policy at more seeds and at a non-512 size.
3. Continue sweeping `up_blocks.0.attentions.2` against the timestep-aware
   policy surface.
