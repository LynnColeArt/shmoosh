# Experiment Plan

## Phase 0: Reference Codec

Goal: verify that the geometric codec behaves sensibly before involving diffusion checkpoints.

Tasks:

- implement rotated vector quantization in NumPy;
- implement QJL-style residual sign correction for dot-product estimates;
- build synthetic attention probes with controllable outliers and timestep-like scale drift;
- compare against plain low-bit scalar quantization.

Exit criteria:

- codec round-trips shapes correctly;
- higher bit-width reduces reconstruction error;
- attention metrics can be generated from synthetic tensors;
- probe output is reproducible with a seed.

## Phase 1: Tensor Capture From Diffusers

Goal: collect real Q/K/V tensors from a diffusion model on a 12GB card without modifying kernels.

Candidate model order:

1. small DiT/PixArt-class model if it fits comfortably;
2. SDXL attention blocks if DiT is too heavy;
3. video model only after image experiments are stable.

Captured metadata:

- model id;
- prompt and seed;
- resolution;
- scheduler and step count;
- denoising timestep;
- layer/block/head;
- tensor shape and dtype;
- baseline peak VRAM.

Exit criteria:

- a `.npz` fixture with Q/K/V tensors from at least 3 layers and 3 timesteps;
- attention probe runs on captured tensors;
- first sensitivity table exists.

## Phase 2: Diffusion-Aware Policy

Goal: find a mixed-precision policy that avoids obvious quality regressions.

Policy dimensions:

- K bit-width;
- V bit-width;
- QJL residual enabled for K score estimation;
- skip early/late timesteps;
- skip sensitive layers/heads;
- native precision fallback for pathological activations.

Metrics:

- attention-score MSE;
- softmax KL;
- attention-output cosine error;
- image delta against same-seed baseline;
- CLIP score or aesthetic proxy if available;
- peak VRAM and latency.

Exit criteria:

- at least one policy beats simple INT4 on attention-output error;
- same-seed images do not show immediate visible collapse;
- measured VRAM or attention-time benefit is plausible enough to justify GPU kernels.

## Phase 3: Local-GPU Runtime

Goal: make RTX 4070/3080 workflows measurably more useful.

Runtime path:

- start with a Torch attention processor;
- avoid full dequantization where possible;
- move hot loops to Triton only after profiling;
- benchmark against xFormers/SDPA/FlashAttention where available.

Consumer GPU success criteria:

- fits a resolution, batch size, or model variant that baseline cannot fit;
- or reduces peak VRAM enough to keep workflows responsive with ControlNet/LoRA/hires steps;
- or improves attention latency without visible quality loss.
