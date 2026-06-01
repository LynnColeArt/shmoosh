# Packed Backend Validation: 2026-05-31

## Scope

This validates the packed attention backend against the accepted native SDXL
policy:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated30pct-k5-k6-qjl128-policy.json
```

The goal is parity, not speed. The packed backend uses the Torch/Triton
packed-key exact-value attention path:

```bash
--attention-backend packed --packed-backend auto
```

The policy still leaves the first 30% of denoising steps exact, then quantizes
the selected seven up-block cross-attention modules.

## Commands

20-step 1024 validation:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run shmoosh-image-policy-suite \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --policy-file configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated30pct-k5-k6-qjl128-policy.json \
  --case-file configs/underpaint-juggernaut-validation-1024-cases.json \
  --model-cpu-offload \
  --local-files-only \
  --attention-backend packed \
  --packed-backend auto \
  --output-dir captures/image-policy-suite-juggernaut-up0-cross-mixed-gated30pct-1024-20step-packed
```

30-step 1024 validation:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run shmoosh-image-policy-suite \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --policy-file configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated30pct-k5-k6-qjl128-policy.json \
  --case-file configs/underpaint-juggernaut-validation-1024-30step-cases.json \
  --model-cpu-offload \
  --local-files-only \
  --attention-backend packed \
  --packed-backend auto \
  --output-dir captures/image-policy-suite-juggernaut-up0-cross-mixed-gated30pct-1024-30step-packed
```

## 20-Step Results

| Case | Reference PSNR | Packed PSNR | Packed MSE | Packed MAE |
| --- | ---: | ---: | ---: | ---: |
| reading-nook-seed1-1024 | 48.48 dB | 50.52 dB | 0.00000888 | 0.00112699 |
| maple-leaf-seed2-1024 | 49.45 dB | 49.31 dB | 0.00001171 | 0.00128130 |
| misty-lake-seed3-1024 | 58.52 dB | 58.61 dB | 0.00000138 | 0.00029679 |

Aggregate:

```text
reference_mean_psnr=52.15 dB
packed_mean_psnr=52.81 dB
reference_min_psnr=48.48 dB
packed_min_psnr=49.31 dB
packed_max_mse=0.00001171
```

## 30-Step Results

| Case | Reference PSNR | Packed PSNR | Packed MSE | Packed MAE |
| --- | ---: | ---: | ---: | ---: |
| reading-nook-seed1-1024-30step | 50.33 dB | 49.20 dB | 0.00001201 | 0.00105927 |
| maple-leaf-seed2-1024-30step | 48.55 dB | 48.65 dB | 0.00001364 | 0.00115821 |
| misty-lake-seed3-1024-30step | 58.76 dB | 59.12 dB | 0.00000122 | 0.00025663 |

Aggregate:

```text
reference_mean_psnr=52.55 dB
packed_mean_psnr=52.33 dB
reference_min_psnr=48.55 dB
packed_min_psnr=48.65 dB
packed_max_mse=0.00001364
```

## Performance Notes

The packed backend is not a speed result yet. It still materializes score
tensors and precomputes query-side codec projections in Torch. The first packed
20-step case also paid visible Triton compile overhead.

Observed packed candidate times:

| Horizon | Case | Baseline Seconds | Packed Seconds | Packed Peak Allocated |
| --- | --- | ---: | ---: | ---: |
| 20 | reading-nook | 12.11 | 44.03 | 5232 MiB |
| 20 | maple-leaf | 8.84 | 21.45 | 5232 MiB |
| 20 | misty-lake | 8.49 | 21.80 | 5232 MiB |
| 30 | reading-nook | 15.50 | 34.01 | 5232 MiB |
| 30 | maple-leaf | 12.10 | 31.94 | 5232 MiB |
| 30 | misty-lake | 11.79 | 31.68 | 5232 MiB |

These numbers confirm that the packed backend runs through the real image
pipeline without quality regression. They do not yet show production value for
4070/3080 workflows.

## Interpretation

Packed backend parity holds at native SDXL resolution for both accepted
horizons. The 20-step and 30-step suites keep minimum PSNR above the previous
reference-backend acceptance floor.

## First Performance Slice

The first performance slice after this validation added:

1. Per-processor codec caches keyed by head dimension, bit depth, QJL width,
   seed, and codebook settings.
2. Per-device packed score resource caches for rotation, codebook, and QJL
   matrices.
3. Automatic packed-backend warmup in image A/B, module sweep, and policy-suite
   runs before measured Shmoosh generation starts.
4. Token-count-independent Triton score kernel compilation, so a tiny warmup can
   compile the same kernel family later used by larger 1024 attention shapes.

A same-process microcheck on the RTX 4070 warmed `D=64`, `K5`, `QJL-128` once,
then ran a `1x20x64x64` query against `77` packed text keys:

```text
warm_seconds=10.3540
large_packed_attention_seconds=0.0646
codec_cache_entries=1
resource_cache_entries=1
```

This is still not an end-to-end speed claim.

## Fused Output Slice

The next slice introduced a fused Triton packed-K attention output path. The
fused path computes packed scores, softmax weights, and exact-V output
accumulation inside Triton kernels, so it does not materialize the full
`(batch, heads, query_tokens, key_tokens)` score tensor. CUDA `auto` now uses a
fast single-tile kernel for text-key attention and a streaming softmax kernel
for larger key sets.

The follow-up kernel slice folded query rotation and QJL projection into that
same fused kernel for fused-compatible head dimensions. This removes the
host-side `q_rot` and `q_proj` tensors from the fused path. The fused path now
falls back for `head_dim < 16`, non-power-of-two head dimensions,
non-power-of-two QJL widths, or CPU runs.

A same-process CUDA microcheck on the RTX 4070 compared fused output against
the materialized Triton score path for a `1x20x64x64` query and `77` packed text
keys:

```text
max_delta=0.000244140625
text77_public_auto_ms_per_iter=0.0840
text77_materialized_ms_per_iter=0.2032
```

After tile tuning, the streaming fused kernel is also faster than the
materialized Triton fallback on larger-key microchecks:

```text
multi257_cross_public_auto_ms_per_iter=0.0841
multi257_cross_materialized_ms_per_iter=0.1531
self257_public_auto_ms_per_iter=0.2274
self257_materialized_ms_per_iter=0.3990
self1024_4h_public_auto_ms_per_iter=0.5691
self1024_4h_materialized_ms_per_iter=1.0371
```

## Torch Encode Slice

The first 1024 image run with fused output exposed a new bottleneck: packed
attention was no longer dominated by score materialization, but `encode_packed_keys`
still copied K to CPU and used the NumPy reference codec before returning packed
Torch tensors. The next slice added an optional Torch/CUDA encode path that
reuses cached packed-score resources when the Diffusers processor is already on
the packed backend.

A CUDA encode microcheck on the RTX 4070 used `2x20x77x64` fp16 keys with `K5`
and `QJL-128`:

```text
numpy_cpu_encode_ms_per_iter=108.0557
torch_resource_encode_ms_per_iter=14.4400
codes_equal=True
norms_max_delta=0.00000095
residual_norms_max_delta=0.00000072
```

That removes the largest remaining host-side encode cost for text-key attention.
On one 1024 image A/B run, the accepted 30% gated policy became faster than the
exact baseline while keeping image drift small:

```text
baseline_seconds=12.4544
shmoosh_seconds=11.4222
psnr=50.54dB
mse=0.00000883
```

The three-case 1024 policy suite is still mixed on end-to-end time:

| case | baseline seconds | shmoosh seconds | speed ratio | PSNR |
| --- | ---: | ---: | ---: | ---: |
| reading-nook-seed1-1024 | 12.5654 | 11.4895 | 1.094x | 50.54 dB |
| maple-leaf-seed2-1024 | 9.1381 | 10.4339 | 0.876x | 49.40 dB |
| misty-lake-seed3-1024 | 9.1993 | 10.4704 | 0.879x | 57.84 dB |

Mean time was `10.3009s` baseline versus `10.7980s` Shmoosh across the suite.
The performance result is therefore no longer "packed is too slow"; it is
"packed can win on a longer attention-heavy case, but the fixed packed-path
overhead still loses on shorter cases." The next optimization target should
split measured time inside the processor so encode, fused attention, fallback
attention, and policy overhead are visible per module and timestep.
