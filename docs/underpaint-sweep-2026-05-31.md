# Underpaint Sweep: 2026-05-31

## Setup

Source checkpoint, used read-only:

```text
/home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors
```

Capture command:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python experiments/capture_diffusers_attention.py \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --prompt "a restored archival photo of a brass compass on a workbench" \
  --steps 3 \
  --height 512 \
  --width 512 \
  --module-indices 0,1,8,9,48,49,108,109,120,121 \
  --max-captures-per-module 3 \
  --max-tokens 4096 \
  --model-cpu-offload \
  --output-dir captures/underpaint-juggernaut-sweep \
  --local-files-only
```

Captured fixture:

```text
30 files
129 MB compressed
5 modules x self-attention x 3 invocations
5 modules x cross-attention x 3 invocations
```

The selected modules cover down, mid, and up U-Net attention blocks.

## Main Sweep

Command:

```bash
uv run turbo-d-sweep-captures captures/underpaint-juggernaut-sweep \
  --bits 3,4 \
  --qjl-bits 0,128 \
  --codebook-samples 80000 \
  --csv captures/underpaint-juggernaut-sweep/results.csv \
  --json captures/underpaint-juggernaut-sweep/results.json
```

Ratios below are `Turbo-D metric / scalar metric`; lower is better.

| Bits | QJL Signs | Score Wins | KL Wins | Output Cos Wins | Mean Score Ratio | Mean KL Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0 | 23/30 | 26/30 | 28/30 | 0.8787 | 0.8785 |
| 3 | 128 | 30/30 | 30/30 | 30/30 | 0.4817 | 0.5616 |
| 4 | 0 | 15/30 | 6/30 | 14/30 | 0.9909 | 1.1581 |
| 4 | 128 | 30/30 | 29/30 | 29/30 | 0.5971 | 0.7367 |

Split by attention kind:

| Kind | Bits | QJL Signs | Score Wins | KL Wins | Output Cos Wins | Mean Score Ratio | Mean KL Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| self-attn | 3 | 128 | 15/15 | 15/15 | 15/15 | 0.5824 | 0.5740 |
| self-attn | 4 | 128 | 15/15 | 15/15 | 15/15 | 0.7331 | 0.7345 |
| cross-attn | 3 | 128 | 15/15 | 15/15 | 15/15 | 0.3811 | 0.5492 |
| cross-attn | 4 | 128 | 15/15 | 14/15 | 14/15 | 0.4610 | 0.7389 |

## QJL Size Sweep

Command:

```bash
uv run turbo-d-sweep-captures captures/underpaint-juggernaut-sweep \
  --bits 3,4 \
  --qjl-bits 16,32,64,128 \
  --codebook-samples 80000 \
  --csv captures/underpaint-juggernaut-sweep/qjl_sweep.csv \
  --json captures/underpaint-juggernaut-sweep/qjl_sweep.json
```

| Bits | QJL Signs | Score Wins | KL Wins | Output Cos Wins | Mean Score Ratio | Mean KL Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 16 | 0/30 | 0/30 | 2/30 | 3.7780 | 4.3063 |
| 3 | 32 | 0/30 | 0/30 | 12/30 | 1.9235 | 2.2763 |
| 3 | 64 | 14/30 | 5/30 | 23/30 | 0.9458 | 1.1265 |
| 3 | 128 | 30/30 | 30/30 | 30/30 | 0.4817 | 0.5616 |
| 4 | 16 | 0/30 | 0/30 | 0/30 | 4.6918 | 5.6736 |
| 4 | 32 | 0/30 | 0/30 | 4/30 | 2.3813 | 2.9108 |
| 4 | 64 | 12/30 | 0/30 | 12/30 | 1.1678 | 1.4543 |
| 4 | 128 | 30/30 | 29/30 | 29/30 | 0.5971 | 0.7367 |

## Interpretation

The important result is not simply "rotation helps." In this fixture, the QJL residual correction is the difference between a sometimes-good codec and a consistently-good one.

The lower-sign QJL settings are not useful yet. With the current estimator and scaling, 16 and 32 signs inject too much correction noise, and 64 signs is only borderline. At 128 signs, Turbo-D becomes consistently better than scalar quantization across both self-attention and cross-attention.

For 64-dimensional attention vectors, the theoretical packed storage per vector is:

| Config | Codes | Norms | Residual Signs | Residual Norm | Total | FP16 Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3-bit, no QJL | 24 B | 4 B | 0 B | 0 B | 28 B | 4.57x smaller |
| 3-bit, QJL-128 | 24 B | 4 B | 16 B | 4 B | 48 B | 2.67x smaller |
| 4-bit, QJL-128 | 32 B | 4 B | 16 B | 4 B | 56 B | 2.29x smaller |

The current NumPy implementation does not bit-pack residual signs; these are theoretical runtime/storage targets for a production implementation.

## Next Questions

1. Does QJL-128 still win at 768px and 1024px, where attention memory pressure is more relevant?
2. Does a timestep-aware policy need QJL everywhere, or only in the fragile modules?
3. Can the residual correction be rescaled or regularized so QJL-64 becomes useful?
4. Does the same metric advantage survive inside an actual attention processor during image generation?

## Runtime Processor Prototype

Follow-up implementation added:

- `turbo_d.runtime_attention.turbo_d_attention_output`
- `turbo_d.diffusers_processor.TurboDAttnProcessor`
- `turbo-d-runtime-smoke`

This is a deliberately slow behavioral path. It mirrors Diffusers `AttnProcessor2_0`, but moves post-projection Q/K/V tensors through the NumPy Turbo-D reference codec. It is useful for correctness and policy experiments, not speed.

Smoke results on representative captures:

```text
self-attn capture_000, 3-bit QJL-128, quantized V:
output_mse=0.0008593021
output_cosine_error=0.0060882026

self-attn capture_000, 3-bit QJL-128, exact V:
output_mse=0.00052545276
output_cosine_error=0.0038128125

cross-attn capture_003, 3-bit QJL-128, quantized V:
output_mse=0.0012851002
output_cosine_error=0.50838081

cross-attn capture_003, 3-bit QJL-128, exact V:
output_mse=0.00041547219
output_cosine_error=0.50249268
active_output_cosine_error=0.00498536 active_rows=10240/20480
```

The cross-attention raw cosine value is misleading because half the rows in that capture have zero-norm reference outputs, likely from the unconditional half of classifier-free guidance. Filtering to active rows shows a small cosine error. Exact V improves MSE substantially, so early runtime policies should test K-only or higher-precision V variants before compressing V as aggressively as K.

## K/V Runtime Policy Sweep

Runtime-style policy command:

```bash
uv run turbo-d-policy-sweep captures/underpaint-juggernaut-sweep \
  --policies k_only,v_only,kv \
  --bits 3 \
  --qjl-bits 128 \
  --codebook-samples 80000 \
  --csv captures/underpaint-juggernaut-sweep/policy_sweep.csv \
  --json captures/underpaint-juggernaut-sweep/policy_sweep.json
```

Mean output metrics over the 30-capture fixture:

| Policy | K Bits | V Bits | Mean MSE | Mean Active Cosine Error |
| --- | ---: | ---: | ---: | ---: |
| K-only | 3 | exact | 0.00103543 | 0.00271623 |
| V-only | exact | 3 | 0.00155107 | 0.00870124 |
| K/V | 3 | 3 | 0.00256393 | 0.01136582 |
| V-only | exact | 4 | 0.00040175 | 0.00234521 |
| K/V | 3 | 4 | 0.00143154 | 0.00505935 |
| V-only | exact | 5 | 0.00011942 | 0.00068766 |
| K/V | 3 | 5 | 0.00115048 | 0.00340044 |

By attention kind, K-only is especially strong for cross-attention:

| Policy | Kind | Mean MSE | Mean Active Cosine Error |
| --- | --- | ---: | ---: |
| K-only | self-attn | 0.00195509 | 0.00325901 |
| K-only | cross-attn | 0.00011578 | 0.00217346 |
| K/V 3/5 | self-attn | 0.00214272 | 0.00351473 |
| K/V 3/5 | cross-attn | 0.00015824 | 0.00328615 |

Interpretation: on this fixture, key compression with QJL-128 causes less output drift than value compression. Raising V precision helps, but even K3/V5 trails K-only. The first image-generation prototype should therefore start with `K3 + QJL-128 + exact V`, then test whether V4/V5 is acceptable later.

## Image A/B Follow-Up

The first image-level A/B smoke is recorded in `docs/image-ab-smoke-2026-05-31.md`.

At 512px, 2 denoising steps, and one SDXL U-Net self-attention module
(`down_blocks.2.attentions.0.transformer_blocks.0.attn1`), the exact-processor
calibration produced `mse=0.00001533` and `psnr=48.14 dB`, while the
`K3 + QJL-128 + exact V` run produced `mse=0.00433298` and `psnr=23.63 dB`.
That means the harness itself is stable, and the first real image delta is
coming from key compression rather than processor plumbing.

An 8-step module sweep is recorded in `docs/image-module-sweep-2026-05-31.md`.
In a four-module probe, up-block cross-attention module 49 had the lowest image
delta (`mse=0.00049776`, `psnr=33.03 dB`), while down-block cross-attention
module 9 had the largest delta (`mse=0.00173393`, `psnr=27.61 dB`). The first
tracked seed policy is `configs/underpaint-juggernaut-sdxl-k3-qjl128-policy.json`.

Policy A/B results are recorded in `docs/policy-ab-2026-05-31.md`. Module 49
remained acceptable at 12 steps (`psnr=31.49 dB`), but the same K3 policy caused
large composition drift at 20 steps (`psnr=22.70 dB`). The 20-step exact
processor calibration was much closer (`psnr=34.16 dB`), so the long-run drift
comes from key compression rather than merely replacing the attention processor.

The focused precision sweep in `docs/precision-sweep-2026-05-31.md` found that
K4 still drifted at 20 steps (`psnr=25.10 dB`), while K5 recovered most of the
exact-processor behavior (`psnr=32.95 dB`). The first 20-step candidate policy
is `configs/underpaint-juggernaut-sdxl-k5-qjl128-policy.json`.

Additional policy validation is recorded in `docs/policy-validation-2026-05-31.md`.
Across three new prompt/seed cases, module 49 with K5 stayed above the rough
quality gate (`min_psnr=36.68 dB`, `mean_psnr=44.63 dB`).
