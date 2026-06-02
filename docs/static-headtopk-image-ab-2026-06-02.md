# Static Head Top-K Image A/B - 2026-06-02

## Slice

Added an image-level sparse oracle runtime for static per-head top-k budgets.
This path is intentionally not the production kernel:

```text
packed K scores -> per-head top-k mask -> exact V attention output
```

It materializes the score tensor so we can test image fidelity before building a
fused sparse packed attention kernel.

## Runtime Policy Support

Policy JSON can now pass `static_head_topk_budgets` globally or per module.
When present on a packed K/exact V processor, the processor uses static
per-head top-k masking instead of the fused packed attention output.

Added three policies:

```text
configs/underpaint-juggernaut-sdxl-up0-self-attn1-static-headtopk-topp95-q50-k7-noqjl-policy.json
configs/underpaint-juggernaut-sdxl-up0-self-attn1-static-headtopk-topp98-q50-k7-noqjl-policy.json
configs/underpaint-juggernaut-sdxl-up0-self-attn1-static-headtopk-topp95-q90-k7-noqjl-policy.json
```

Each uses the same late 70% self-attention window as the prior K7/no-QJL
packed self-attention policy.

## Run

```bash
uv run python -m shmoosh.cli.image_policy_compare \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --case-file configs/underpaint-juggernaut-validation-1024-cases.json \
  --output-dir captures/image-policy-compare-juggernaut-static-headtopk-1024-20260602 \
  --device cuda \
  --dtype fp16 \
  --model-cpu-offload \
  --local-files-only \
  --trace-processor-timing \
  --candidate packed_k7=configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-score-value-tf32-bk32-policy.json \
  --candidate static_topp95_q50=configs/underpaint-juggernaut-sdxl-up0-self-attn1-static-headtopk-topp95-q50-k7-noqjl-policy.json \
  --candidate static_topp98_q50=configs/underpaint-juggernaut-sdxl-up0-self-attn1-static-headtopk-topp98-q50-k7-noqjl-policy.json \
  --candidate static_topp95_q90=configs/underpaint-juggernaut-sdxl-up0-self-attn1-static-headtopk-topp95-q90-k7-noqjl-policy.json
```

## Result

| Candidate | Min PSNR | Mean PSNR | Mean speedup | Attention phase |
| --- | ---: | ---: | ---: | ---: |
| packed K7 control | 52.10 dB | 54.21 dB | 1.071x | 0.747 ms fused packed |
| static top-p 0.95 q=0.50 | 41.56 dB | 44.81 dB | 1.049x | 13.417 ms sparse materialized |
| static top-p 0.98 q=0.50 | 43.65 dB | 46.56 dB | 1.076x | 14.169 ms sparse materialized |
| static top-p 0.95 q=0.90 | 47.30 dB | 49.46 dB | 1.085x | 14.160 ms sparse materialized |

Per case:

| Candidate | Reading nook | Maple leaf | Misty lake |
| --- | ---: | ---: | ---: |
| packed K7 control | 52.10 dB | 52.42 dB | 58.13 dB |
| static top-p 0.95 q=0.50 | 43.39 dB | 41.56 dB | 49.47 dB |
| static top-p 0.98 q=0.50 | 44.61 dB | 43.65 dB | 51.43 dB |
| static top-p 0.95 q=0.90 | 47.51 dB | 47.30 dB | 53.57 dB |

## Read

Static sparse budgets are now wired through the actual image stack, but the
median-budget policies are too destructive at image level. The conservative
top-p 0.95 q=0.90 policy is the best of this slice, yet it still trails the
packed K7 control by about 4.8 dB at minimum PSNR.

The speed numbers should not be treated as a sparse-kernel forecast. The sparse
path materializes full packed score tensors and spends roughly 13-14 ms per
quantized attention call, versus about 0.75 ms for the existing fused packed K7
attention path.

The useful result is negative-positive:

```text
Positive:
  static head budgets integrate through image A/B and preserve image structure.

Negative:
  current static budgets are not good enough to justify a sparse kernel yet.
```

## Next Slice

Do not build the sparse kernel yet. The next better slice is to isolate where
the quality loss enters:

1. Test static budgets only on `up_blocks.0.attentions.1` and `.2`, leaving
   `.0` exact.
2. Try a hybrid mask oracle: mandatory local window plus per-head static top-k.
3. Compare static q=0.90 against dynamic top-p on captures for the same module
   mapping to see whether the image loss is from static budgets or from sparse
   masking itself.

If the restricted-module run recovers quality, a sparse kernel might target only
the lighter late self-attention modules first.
