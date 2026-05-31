# Checkpoint: 2026-05-31

## What Exists

- Empty workspace converted into a small research repo.
- NumPy reference implementation of a TurboQuant-inspired diffusion attention codec.
- Synthetic attention probe comparing Turbo-D against plain scalar quantization.
- Optional Diffusers capture script for real Q/K/V tensor fixtures.
- Unit tests for codec behavior and metrics.

## Local Hardware

Detected during setup:

- NVIDIA GeForce RTX 4070
- 12,282 MiB VRAM
- driver 580.159.04

This matches the target consumer-GPU class for the project.

## Verification

Commands run:

```bash
uv sync --extra dev
uv run python -m compileall -q src experiments tests
uv run pytest
uv run turbo-d-attention-probe --tokens 128 --dim 64 --heads 4 --bits 3 --qjl-bits 64 --seed 11
```

Test result:

```text
6 passed
```

Synthetic 3-bit attention probe:

```text
turbo_d: score_mse=0.0643849 softmax_kl=0.0322994 output_cosine_error=0.0477741 output_mse=0.00323452
scalar:  score_mse=0.103598  softmax_kl=0.0571544 output_cosine_error=0.0976257 output_mse=0.00873431
```

Underpaint SDXL single-file smoke capture:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python experiments/capture_diffusers_attention.py \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --prompt "a restored archival photo of a brass compass on a workbench" \
  --steps 1 \
  --height 512 \
  --width 512 \
  --max-modules 1 \
  --max-captures-per-module 1 \
  --model-cpu-offload \
  --output-dir captures/underpaint-juggernaut-smoke \
  --local-files-only
```

Capture produced:

```text
captures/underpaint-juggernaut-smoke/capture_000.npz
q: (20, 1024, 64) float32
k: (20, 1024, 64) float32
v: (20, 1024, 64) float32
module: down_blocks.1.attentions.0.transformer_blocks.0.attn1
```

Real captured attention, 4-bit probe:

```text
turbo_d: score_mse=0.0183694 softmax_kl=0.00903013 output_cosine_error=0.000531458 output_mse=0.000145862
scalar:  score_mse=0.0235968 softmax_kl=0.0117142  output_cosine_error=0.000606888 output_mse=0.000155691
```

## Interpretation

This is not evidence of diffusion image quality yet. It is evidence that the first reference codec is wired correctly enough to test, and on a seeded synthetic attention workload with outliers and timestep-like drift, the geometric codec preserves attention behavior better than a plain scalar quantizer at the same nominal bit width.

The first real Underpaint-derived SDXL tensor shows the same direction as the synthetic probe: Turbo-D preserved attention behavior better than plain scalar quantization at the same nominal bit width. This is still a single block and a single denoising step, so the next step is a multi-layer and multi-timestep capture.

## Multi-Block Sweep

A 30-capture Underpaint/Juggernaut sweep is recorded in `docs/underpaint-sweep-2026-05-31.md`.

Headline result:

```text
bits=3 qjl=128: score_wins=30/30 kl_wins=30/30 out_cos_wins=30/30 mean_score_ratio=0.4817 mean_kl_ratio=0.5616
bits=4 qjl=128: score_wins=30/30 kl_wins=29/30 out_cos_wins=29/30 mean_score_ratio=0.5971 mean_kl_ratio=0.7367
```

The sweep also showed that smaller QJL corrections are not currently viable: 16 and 32 residual signs were consistently worse than scalar, and 64 signs was borderline. With the current estimator, QJL-128 is the first stable setting.

## Compatibility Note

The Quickie Video environment was inspected but not modified. Its `diffusers 0.38.0` plus `transformers 5.9.0` stack failed SDXL single-file conversion because Diffusers expected the older `CLIPTextModel.text_model` wrapper shape. Turbo-D now installs its own isolated optional stack with `diffusers 0.37.1` and `transformers 4.57.6`, which successfully loaded the Underpaint checkpoint.
