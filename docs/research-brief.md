# Research Brief

## Core Claim

TurboQuant is best understood as high-dimensional vector quantization, not as an LLM-only method. Its LLM use case is KV cache compression, but the transferable mechanism is geometric:

1. normalize high-dimensional vectors;
2. rotate them into a distribution that is easier to scalar-quantize;
3. quantize coordinates with a compact codebook;
4. correct residual inner-product bias with sign sketches.

Diffusion models do not have the same autoregressive cache structure as decoder LLMs. A direct KV-cache port is therefore the wrong first target. The plausible diffusion target is attention geometry: preserving `QK^T`, softmax structure, and attention output while using lower precision for K/V-like tensors or cached attention states.

## Why Diffusion Is Different

Diffusion quantization papers repeatedly identify two failure modes that do not appear in the same shape for LLM KV cache work:

- activation distributions vary strongly over denoising timesteps;
- errors accumulate through the sampling trajectory, so local tensor error is not the whole story.

For that reason, Turbo-D should not be a single global bit-width. It should become a policy:

- bits by timestep;
- bits by layer/block/head;
- optional native precision for fragile steps;
- residual correction where attention-score bias is visible.

## Related Work Anchors

- TurboQuant: online vector quantization with random rotation, scalar codebooks, and QJL residual correction.
- QJL: 1-bit sign-sketch residual correction for unbiased inner-product estimation.
- Q-Diffusion: timestep-aware calibration and split shortcut quantization for diffusion models.
- PTQ4DiT: salient-channel and temporal variability issues in diffusion transformers.
- QNCD and DMQ: quantization noise correction and outlier handling in diffusion.
- CacheQuant: joint caching and quantization can compound diffusion speedups.
- Diffusers caching and vLLM-Omni diffusion attention quantization: runtime attention/caching surfaces are already emerging.

## First Scientific Test

The first test should not be end-to-end image quality. It should be the smaller mechanistic question:

> Given Q, K, and V tensors from diffusion attention at specific layers and timesteps, does Turbo-D preserve attention scores and outputs better than simple INT4/FP8-style baselines at the same or lower effective bit budget?

Useful early metrics:

- `QK^T` mean squared error;
- row-wise softmax KL divergence;
- attention output cosine error;
- per-layer and per-timestep sensitivity maps;
- same-seed image deltas after a promising policy is found.

## Non-Goals For The First Prototype

- No claims of model-wide compression.
- No custom CUDA before the reference algorithm earns it.
- No one-size-fits-all diffusion bit-width.
- No quality claims from synthetic data alone.
