# Image Module Sweep: 2026-05-31

## Setup

Source checkpoint, used read-only:

```text
/home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors
```

This sweep reuses one same-seed 8-step baseline image, installs
`TurboDAttnProcessor` into one SDXL U-Net attention module at a time, and
records per-module image deltas.

The tested policy is:

```text
K3 + QJL-128 + exact V
```

Command:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run turbo-d-image-module-sweep \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --prompt "a restored archival photo of a brass compass on a workbench" \
  --steps 8 \
  --height 512 \
  --width 512 \
  --module-indices 8,9,48,49 \
  --model-cpu-offload \
  --local-files-only \
  --codebook-samples 20000 \
  --output-dir captures/image-module-sweep-juggernaut-8step
```

Outputs:

```text
captures/image-module-sweep-juggernaut-8step/baseline.png
captures/image-module-sweep-juggernaut-8step/summary.csv
captures/image-module-sweep-juggernaut-8step/summary.json
captures/image-module-sweep-juggernaut-8step/suggested_policy.json
```

## Results

| Module | Kind | MSE | MAE | PSNR | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 49 | up cross-attn | 0.00049776 | 0.01052006 | 33.03 dB | lowest drift in this set |
| 48 | up self-attn | 0.00110237 | 0.01719020 | 29.58 dB | near the first-pass threshold |
| 8 | down self-attn | 0.00131457 | 0.02321757 | 28.81 dB | visible but stable delta |
| 9 | down cross-attn | 0.00173393 | 0.02461394 | 27.61 dB | most sensitive in this set |

Quick visual inspection matched the metrics: module 49 stayed close to the
baseline, while module 9 changed the compass/glass geometry more noticeably.
None of the single-module runs collapsed.

Using a first-pass `30 dB` PSNR threshold, the generated policy suggestion keeps
module 49 as the only quantized candidate and leaves modules 8, 9, and 48 exact.
That tracked seed policy is stored at:

```text
configs/underpaint-juggernaut-sdxl-k3-qjl128-policy.json
```

## Interpretation

The small sweep hints that cross-attention is not uniformly fragile or safe:
the up-block cross-attention module tolerated K compression best, while the
down-block cross-attention module drifted most. That argues for module-specific
policy rather than a blanket "quantize all cross-attn" or "quantize all
self-attn" rule.

The current memory numbers are not meaningful as optimization evidence because
the processor is still a reference NumPy path and does not pack or fuse the
codes. The image deltas are meaningful as a policy search signal.

## Next Slice

1. Teach the image A/B harness to consume the policy file and run all candidate
   modules together.
2. Run a 12-20 step image A/B with module 49 enabled.
3. Expand the sweep over more up-block modules before touching down-block
   attention broadly.
