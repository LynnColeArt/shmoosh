# Shmoosh

Shmoosh is a research sandbox for adapting TurboQuant-style vector quantization to diffusion workflows, with an explicit focus on local consumer GPUs such as RTX 4070 and RTX 3080 class cards.

The working hypothesis is narrow:

- diffusion models are not autoregressive LLMs, so TurboQuant should not be ported as "KV cache compression";
- diffusion attention still relies on high-dimensional inner products, especially in DiT image and video models;
- a rotated vector quantizer plus residual sign correction may preserve attention geometry better than plain low-bit scalar quantization;
- diffusion needs timestep-aware and layer-aware precision policy because quantization errors accumulate through the denoising trajectory.

## Current Checkpoint

This repo starts with a CPU reference implementation, not a production kernel:

- `shmoosh.quantization.ShmooshCodec`: rotation, Lloyd-style scalar codebook quantization, optional QJL-style residual sign correction.
- `shmoosh.metrics`: attention-score and attention-output error metrics.
- `shmoosh-attention-probe`: synthetic attention probe for early algorithm checks.
- `docs/research-brief.md`: the research thesis and related work map.
- `docs/experiment-plan.md`: the path toward 4070/3080 useful benchmarks.

## Quickstart

```bash
uv sync --extra dev
uv run pytest
uv run shmoosh-attention-probe --tokens 256 --dim 128 --bits 4 --qjl-bits 128
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
uv run shmoosh-attention-probe --npz captures/capture_000.npz --bits 4 --qjl-bits 128
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
uv run shmoosh-sweep-captures captures/underpaint-juggernaut \
  --bits 3,4 \
  --qjl-bits 0,128
```

Sweep runtime K/V policies:

```bash
uv run shmoosh-policy-sweep captures/underpaint-juggernaut \
  --policies k_only,v_only,kv \
  --bits 3 \
  --qjl-bits 128
```

Run the runtime-style attention smoke:

```bash
uv run shmoosh-runtime-smoke captures/underpaint-juggernaut/capture_000.npz \
  --bits 3 \
  --qjl-bits 128
```

Use exact values to test K/score compression separately from V compression:

```bash
uv run shmoosh-runtime-smoke captures/underpaint-juggernaut/capture_000.npz \
  --bits 3 \
  --qjl-bits 128 \
  --exact-values
```

Run a same-seed baseline vs Shmoosh image smoke test:

```bash
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
  --output-dir captures/image-ab-juggernaut-module-008
```

The default image smoke policy is `K3 + QJL-128 + exact V`. Add
`--quantize-values` when testing value compression too.

Sweep several modules with one baseline image:

```bash
uv run shmoosh-image-module-sweep \
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

Run the tracked seed policy directly:

```bash
uv run shmoosh-image-ab-smoke \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --prompt "a restored archival photo of a brass compass on a workbench" \
  --steps 12 \
  --height 512 \
  --width 512 \
  --policy-file configs/underpaint-juggernaut-sdxl-k3-qjl128-policy.json \
  --model-cpu-offload \
  --local-files-only \
  --output-dir captures/image-ab-juggernaut-policy49-12step
```

For 20-step image tests, use the K5 candidate policy:

```bash
uv run shmoosh-image-ab-smoke \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --prompt "a restored archival photo of a brass compass on a workbench" \
  --steps 20 \
  --height 512 \
  --width 512 \
  --policy-file configs/underpaint-juggernaut-sdxl-k5-qjl128-policy.json \
  --model-cpu-offload \
  --local-files-only \
  --output-dir captures/image-ab-juggernaut-policy49-k5-20step
```

Validate a policy across prompt/seed cases:

```bash
uv run shmoosh-image-policy-suite \
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

The current K5 policy cleared the three-case validation suite with
`min_psnr=36.68 dB` and `mean_psnr=44.63 dB`; see
`docs/policy-validation-2026-05-31.md`.

The current combined up-block cross-attention candidate is:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-k5-qjl128-policy.json
```

It enables modules `49,59,61,65,67` and cleared the 20-step compass A/B with
`psnr=32.93 dB`. It also cleared the three-case validation suite with
`min_psnr=39.47 dB` and `mean_psnr=43.39 dB`; see
`docs/up-cross-sweep-2026-05-31.md`.

A second up-block candidate is:

```text
configs/underpaint-juggernaut-sdxl-up0-attn1-cross-k6-qjl128-policy.json
```

It enables modules `79,87` at K6. The K5 single-module gate found modules
`79,85,87`, but they did not compose. The K6 pair cleared the 20-step compass
A/B with `psnr=42.50 dB` and the three-case validation suite with
`min_psnr=42.66 dB` and `mean_psnr=47.24 dB`; see
`docs/up-cross-attn1-sweep-2026-05-31.md`.

Policy files can now override processor precision per module. The first mixed
bridge policy is:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-mixed-bridge-k5-k6-qjl128-policy.json
```

It enables module `67` at K5 and module `87` at K6. Larger mixed policies did
not compose, but this bridge cleared the 20-step compass A/B with
`psnr=42.08 dB` and the three-case validation suite with `min_psnr=37.85 dB`;
see `docs/mixed-policy-2026-05-31.md`.

Policy entries can also declare timestep windows. The first timestep-gated
candidate is:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated20-k5-k6-qjl128-policy.json
```

It leaves the first 4 of 20 denoising steps exact, then quantizes seven modules
for the remaining 16 steps. This rescued the failed full mixed policy from
`psnr=24.33 dB` to `psnr=46.75 dB`, and cleared the three-case validation suite
with `min_psnr=43.85 dB`; see `docs/timestep-gating-2026-05-31.md`.

The horizon-scaled form is:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated20pct-k5-k6-qjl128-policy.json
```

It leaves the first 20% of denoising exact. That reproduced the 20-step result
at `psnr=46.75 dB`, held at 30 steps with `psnr=45.70 dB`, and passed a
`768x512` shape stress at `psnr=39.57 dB`.

Native SDXL-size tests use 1024px as the meaningful quality gate. At 1024x1024,
the best current candidate is:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated30pct-k5-k6-qjl128-policy.json
```

It leaves the first 30% of denoising exact, then quantizes the selected seven
modules. It cleared the 1024 compass A/B with `psnr=46.89 dB` and a three-case
1024 validation suite with `min_psnr=48.48 dB`. At 30 steps, the same policy
cleared the 1024 validation suite with `min_psnr=48.55 dB`; see
`docs/1024-policy-validation-2026-05-31.md`.

## GPU Target

The practical target is the 10-12GB VRAM band:

- RTX 3080 10GB/12GB
- RTX 4070 12GB
- nearby laptop/workstation cards with similar memory pressure

The first real diffusion benchmark should answer one question before anything else:

> Can attention-only Shmoosh reduce memory bandwidth or VRAM pressure without visibly destabilizing same-seed generation?

If the answer is yes, the next step is a fused Torch/Triton attention path.

Estimate the packed-key storage target for the accepted 1024 policy:

```bash
uv run shmoosh-packed-policy-estimate \
  --policy-file configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated30pct-k5-k6-qjl128-policy.json \
  --steps 20 \
  --steps 30
```

With SDXL cross-attention assumptions, the selected modules save `1.27 MiB` of
key payload per quantized step (`1.93x` packed-key ratio). Across the validated
30-step horizon, that is `26.65 MiB` of selected-key payload; see
`docs/packed-k-design-2026-05-31.md`.
