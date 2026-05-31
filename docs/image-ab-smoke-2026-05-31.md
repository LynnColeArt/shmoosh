# Image A/B Smoke: 2026-05-31

## Setup

Source checkpoint, used read-only:

```text
/home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors
```

The first image-level smoke installs `ShmooshAttnProcessor` into one SDXL U-Net
self-attention module:

```text
008 down_blocks.2.attentions.0.transformer_blocks.0.attn1
```

The policy is the current best captured-tensor policy:

```text
K3 + QJL-128 + exact V
```

This is not a speed benchmark. The processor routes through the NumPy reference
codec and is only meant to answer whether the same-seed generation remains
stable when a chosen attention module uses Shmoosh key compression.

## Commands

Quantized-key image A/B:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run shmoosh-image-ab-smoke \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --prompt "a restored archival photo of a brass compass on a workbench" \
  --steps 2 \
  --height 512 \
  --width 512 \
  --module-indices 8 \
  --model-cpu-offload \
  --local-files-only \
  --codebook-samples 20000 \
  --output-dir captures/image-ab-juggernaut-module-008-2step
```

Exact-processor calibration:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run shmoosh-image-ab-smoke \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --prompt "a restored archival photo of a brass compass on a workbench" \
  --steps 2 \
  --height 512 \
  --width 512 \
  --module-indices 8 \
  --model-cpu-offload \
  --local-files-only \
  --exact-keys \
  --codebook-samples 20000 \
  --output-dir captures/image-ab-juggernaut-module-008-2step-exact-processor
```

## Results

Image outputs:

```text
captures/image-ab-juggernaut-module-008-2step/baseline.png
captures/image-ab-juggernaut-module-008-2step/shmoosh.png
captures/image-ab-juggernaut-module-008-2step/diff_heatmap.png
```

| Run | Quantized K | Quantized V | MSE | MAE | PSNR |
| --- | --- | --- | ---: | ---: | ---: |
| Exact processor calibration | no | no | 0.00001533 | 0.00232714 | 48.14 dB |
| K3 + QJL-128 + exact V | yes | no | 0.00433298 | 0.04297079 | 23.63 dB |

The exact-processor calibration is tight enough to show the harness is not
creating large drift merely by replacing Diffusers' default attention path.
The quantized-key run does create a visible image delta at 2 denoising steps,
but the generation does not collapse.

## Next Slice

1. Verify `configs/underpaint-juggernaut-sdxl-k5-qjl128-policy.json` through
   `--policy-file` at 20 steps.
2. Test the K5 policy on additional prompts/seeds.
3. Expand the sweep over more up-block modules before touching down-block
   attention broadly.
4. After a stable image policy emerges, replace the NumPy processor with a
   Torch/Triton implementation that can actually reduce runtime memory traffic.
