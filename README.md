# Turbo-D

Turbo-D is a research sandbox for adapting TurboQuant-style vector quantization to diffusion workflows, with an explicit focus on local consumer GPUs such as RTX 4070 and RTX 3080 class cards.

The working hypothesis is narrow:

- diffusion models are not autoregressive LLMs, so TurboQuant should not be ported as "KV cache compression";
- diffusion attention still relies on high-dimensional inner products, especially in DiT image and video models;
- a rotated vector quantizer plus residual sign correction may preserve attention geometry better than plain low-bit scalar quantization;
- diffusion needs timestep-aware and layer-aware precision policy because quantization errors accumulate through the denoising trajectory.

## Current Checkpoint

This repo starts with a CPU reference implementation, not a production kernel:

- `turbo_d.quantization.TurboDCodec`: rotation, Lloyd-style scalar codebook quantization, optional QJL-style residual sign correction.
- `turbo_d.metrics`: attention-score and attention-output error metrics.
- `turbo-d-attention-probe`: synthetic attention probe for early algorithm checks.
- `docs/research-brief.md`: the research thesis and related work map.
- `docs/experiment-plan.md`: the path toward 4070/3080 useful benchmarks.

## Quickstart

```bash
uv sync --extra dev
uv run pytest
uv run turbo-d-attention-probe --tokens 256 --dim 128 --bits 4 --qjl-bits 128
```

The reference path is intentionally NumPy-only so it can be tested without installing PyTorch or downloading a diffusion checkpoint.

To capture real attention tensors from a Diffusers model:

```bash
uv sync --extra dev --extra diffusers
uv run python experiments/capture_diffusers_attention.py \
  --model-id <diffusers-model-id> \
  --prompt "a detailed photo of a brass compass on a workbench" \
  --steps 4 \
  --output-dir captures
uv run turbo-d-attention-probe --npz captures/capture_000.npz --bits 4 --qjl-bits 128
```

Underpaint's local Juggernaut checkpoint can be used read-only:

```bash
uv run python experiments/capture_diffusers_attention.py \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --prompt "a restored archival photo of a brass compass on a workbench" \
  --steps 4 \
  --height 768 \
  --width 768 \
  --model-cpu-offload \
  --output-dir captures/underpaint-juggernaut
```

List attention modules before choosing capture targets:

```bash
uv run python experiments/capture_diffusers_attention.py \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --list-modules \
  --local-files-only
```

Sweep a capture directory:

```bash
uv run turbo-d-sweep-captures captures/underpaint-juggernaut \
  --bits 3,4 \
  --qjl-bits 0,128
```

Sweep runtime K/V policies:

```bash
uv run turbo-d-policy-sweep captures/underpaint-juggernaut \
  --policies k_only,v_only,kv \
  --bits 3 \
  --qjl-bits 128
```

Run the runtime-style attention smoke:

```bash
uv run turbo-d-runtime-smoke captures/underpaint-juggernaut/capture_000.npz \
  --bits 3 \
  --qjl-bits 128
```

Use exact values to test K/score compression separately from V compression:

```bash
uv run turbo-d-runtime-smoke captures/underpaint-juggernaut/capture_000.npz \
  --bits 3 \
  --qjl-bits 128 \
  --exact-values
```

Run a same-seed baseline vs Turbo-D image smoke test:

```bash
uv run turbo-d-image-ab-smoke \
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
  --output-dir captures/image-ab-juggernaut-module-008
```

The default image smoke policy is `K3 + QJL-128 + exact V`. Add
`--quantize-values` when testing value compression too.

Sweep several modules with one baseline image:

```bash
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
  --output-dir captures/image-module-sweep-juggernaut-8step
```

The module sweep writes `summary.csv`, `summary.json`, and a first-pass
`suggested_policy.json`. The current tracked seed policy is
`configs/underpaint-juggernaut-sdxl-k3-qjl128-policy.json`.

## GPU Target

The practical target is the 10-12GB VRAM band:

- RTX 3080 10GB/12GB
- RTX 4070 12GB
- nearby laptop/workstation cards with similar memory pressure

The first real diffusion benchmark should answer one question before anything else:

> Can attention-only Turbo-D reduce memory bandwidth or VRAM pressure without visibly destabilizing same-seed generation?

If the answer is yes, the next step is a fused Torch/Triton attention path.
