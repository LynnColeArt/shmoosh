# Underpaint Model Inventory

Read-only source inspected on 2026-05-31:

```text
/home/lynn/.underpaint/models
```

Useful diffusion candidate:

```text
/home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors
size: about 6.7 GB
```

Supporting Underpaint assets:

- Fooocus inpaint patch and LaMa assets under `/home/lynn/.underpaint/models/inpaint/fooocus`
- SAM-HQ and BiRefNet segmentation assets under `/home/lynn/.underpaint/models/segmentation`
- YOLO/adetailer detection assets under `/home/lynn/.underpaint/models/detail` and `/home/lynn/.underpaint/models/detection`
- Qwen GGUF prompt-helper models under `/home/lynn/.underpaint/models/prompt`

For Shmoosh's first attention experiment, use the Juggernaut checkpoint read-only through Diffusers single-file loading. Do not modify Underpaint's libraries or model store.

Suggested capture command:

```bash
uv sync --extra dev --extra diffusers
uv run python experiments/capture_diffusers_attention.py \
  --single-file /home/lynn/.underpaint/models/checkpoints/juggernaut-x-v10/Juggernaut-X-RunDiffusion-NSFW.safetensors \
  --pipeline-class sdxl \
  --config /home/lynn/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b \
  --prompt "a restored archival photo of a brass compass on a workbench" \
  --steps 4 \
  --height 768 \
  --width 768 \
  --max-modules 4 \
  --max-captures-per-module 2 \
  --model-cpu-offload \
  --output-dir captures/underpaint-juggernaut
```

The optional Diffusers dependency set pins `transformers<5`. The Quickie Video environment was read for compatibility clues only and had `transformers 5.9.0`, which failed SDXL single-file conversion because Diffusers expected the older `CLIPTextModel.text_model` wrapper shape.

Then run:

```bash
uv run shmoosh-attention-probe \
  --npz captures/underpaint-juggernaut/capture_000.npz \
  --bits 4 \
  --qjl-bits 128
```
