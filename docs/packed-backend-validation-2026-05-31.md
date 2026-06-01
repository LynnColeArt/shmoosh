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

The next slice introduced a fused Triton packed-K attention output path for
text-key tiles up to `128` tokens. The fused path computes packed scores,
softmax weights, and exact-V output accumulation inside one Triton kernel, so it
does not materialize the full `(batch, heads, query_tokens, key_tokens)` score
tensor. Larger key sets, CPU runs, and explicit `torch` backend runs continue
through the materialized-score fallback.

The follow-up kernel slice folded query rotation and QJL projection into that
same fused kernel for fused-compatible head dimensions. This removes the
host-side `q_rot` and `q_proj` tensors from the fused path. The fused path now
falls back for `head_dim < 16`, non-power-of-two head dimensions,
non-power-of-two QJL widths, or key sets larger than the fixed text-key tile.

A same-process CUDA microcheck on the RTX 4070 compared fused output against
the materialized Triton score path for a `1x20x64x64` query and `77` packed text
keys:

```text
max_delta=0.000244140625
text77_public_auto_ms_per_iter=0.0873
text77_materialized_ms_per_iter=0.1987
```

An experimental streaming fused kernel now handles larger key sets without
materializing the score tensor, but it is not yet the production `auto` path. On
the same 4070, a `257` key-token microcheck was correct but slower than the
materialized Triton fallback:

```text
multi257_public_auto_ms_per_iter=0.2461
multi257_materialized_ms_per_iter=0.2353
multi257_direct_streaming_ms_per_iter=0.3769
multi257_streaming_delta=0.0001220703125
```
