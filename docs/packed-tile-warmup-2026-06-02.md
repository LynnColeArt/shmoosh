# Packed Tile Warmup And BK32 Probe

## Context

The first post-direct-`packed_t` tile probe showed a synthetic win for `BQ64/BK32`, but the image timing was ambiguous. We added explicit packed-attention tile controls so the image policy harness can compare tile choices in the same loaded SDXL process.

New policy/config surface:

- `packed_block_q`
- `packed_block_k`

These can come from CLI defaults, global `shmoosh_policy`, or per-module policy entries.

## Warmup Finding

The first same-process image tile A/B exposed a measurement artifact: processor warmup used a 1-token key, which only compiled the single-tile Triton path. The first quantized self-attention step then paid the streaming-kernel compile cost inside the measured render.

The CUDA warmup now runs both:

- a 1-token shape for the single-key path
- a small streaming shape using the requested tile

That removed the first-step compile spike from the candidate timing.

## 1024 Image Tile A/B

Command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1 uv run shmoosh-image-policy-compare \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config stabilityai/stable-diffusion-xl-base-1.0 \
  --component unet \
  --case-file configs/underpaint-juggernaut-validation-1024-cases.json \
  --output-dir captures/image-policy-compare-juggernaut-up0-self-attn1-k7-noqjl-1024-score-value-tile-ab-warmed-20260602 \
  --dtype fp16 \
  --device cuda \
  --model-cpu-offload \
  --local-files-only \
  --attention-backend packed \
  --packed-backend auto \
  --code-format packed_t \
  --trace-processor-timing \
  --candidate bk16=configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-score-value-tf32-bk16-policy.json \
  --candidate bk32=configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-score-value-tf32-bk32-policy.json
```

Aggregate results:

| Candidate | Min PSNR | Mean PSNR | Mean speedup | Mean packed attention | Mean packed encode |
| --- | ---: | ---: | ---: | ---: | ---: |
| BK16 | 52.0788 dB | 54.1603 dB | 1.0870x | 0.7738 ms | 0.7197 ms |
| BK32 | 52.0976 dB | 54.2148 dB | 1.0977x | 0.7390 ms | 0.6872 ms |

Per-case speedup:

| Case | BK16 | BK32 |
| --- | ---: | ---: |
| reading-nook-seed1-1024 | 1.2099x | 1.3080x |
| maple-leaf-seed2-1024 | 1.0458x | 1.0000x |
| misty-lake-seed3-1024 | 0.9903x | 0.9823x |

## Conclusion

Keep `BQ64/BK32` as the preferred `packed_t` tile for the K7/head64/no-QJL self-attention path on the RTX 4070. The win is modest, but it survives same-process 1024 image validation once the streaming kernel is warmed.

The tile policy should still be rerun on the 3080 before treating it as universal.
