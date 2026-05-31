# Precision Sweep: 2026-05-31

## Setup

This sweep tests whether the 20-step drift from `K3 + QJL-128 + exact V` can be
fixed by raising key precision on the same module:

```text
049 up_blocks.0.attentions.0.transformer_blocks.0.attn2
```

Source checkpoint, used read-only:

```text
/home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors
```

Command shape:

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run shmoosh-image-ab-smoke \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --prompt "a restored archival photo of a brass compass on a workbench" \
  --steps 20 \
  --height 512 \
  --width 512 \
  --module-indices 49 \
  --bits <3|4|5> \
  --qjl-bits 128 \
  --model-cpu-offload \
  --local-files-only \
  --codebook-samples 20000 \
  --output-dir captures/<run-dir>
```

## Results

| Run | K Bits | MSE | MAE | PSNR | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `captures/image-ab-juggernaut-policy49-20step` | 3 | 0.00537460 | 0.04197849 | 22.70 dB | composition drift |
| `captures/image-ab-juggernaut-module49-k4-20step` | 4 | 0.00309126 | 0.03192301 | 25.10 dB | improved, still too much drift |
| `captures/image-ab-juggernaut-module49-k5-20step` | 5 | 0.00050738 | 0.01085879 | 32.95 dB | first acceptable 20-step candidate |
| `captures/image-ab-juggernaut-module49-20step-exact-processor` | exact | 0.00038338 | 0.00880545 | 34.16 dB | custom processor reference |

K5 keeps the 20-step composition close to the baseline in this seed and prompt.
K4 improves over K3 but still moves the scene too much. The exact-processor
reference is only slightly better than K5 by image MSE, which makes K5 the first
credible precision setting for module 49 at 20 steps.

For 64-dimensional attention vectors, the theoretical packed per-vector storage
for `K5 + QJL-128` is:

| Component | Bytes |
| --- | ---: |
| 5-bit codes | 40 |
| vector norm | 4 |
| residual signs | 16 |
| residual norm | 4 |
| total | 64 |

Compared with a 64-dim FP16 vector at 128 bytes, this is still a theoretical
`2.0x` compression ratio for encoded keys. The current NumPy processor does not
bit-pack or reduce runtime memory traffic yet.

## Candidate Policy

The 20-step K5 candidate is tracked at:

```text
configs/underpaint-juggernaut-sdxl-k5-qjl128-policy.json
```

The tracked config was verified through `--policy-file`:

```text
captures/image-ab-juggernaut-policy49-k5-20step
mse=0.00050738
mae=0.01085879
psnr=32.95 dB
```

This should supersede the K3 policy for 20-step tests, while the K3 policy
remains useful as evidence for the short-horizon failure mode.

Follow-up validation across three additional prompt/seed pairs is recorded in:

```text
docs/policy-validation-2026-05-31.md
```

All three additional cases cleared the rough 30 dB PSNR gate, with minimum
`psnr=36.68 dB`.

## Next Slice

1. Add percentage-based timestep windows so policies scale beyond 20-step runs.
2. Start designing production packing around timestep-aware policies.
3. Add a stricter validation lane once another attention group has a candidate.
