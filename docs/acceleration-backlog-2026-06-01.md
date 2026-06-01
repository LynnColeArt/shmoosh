# Acceleration Backlog: 2026-06-01

This note parks speed ideas that should survive the next focused kernel slice.
The current result is not a failure state: Shmoosh has a positive 1024 suite
signal, but the gain is still too small to feel transformative on a 4070/3080.

## Three-Layer Framing

Shmoosh should be evaluated across all three practical diffusion layers, not
only inside selected U-Net cross-attention modules:

1. Prompt and conditioning layer.
   - Cache text-derived cross-attention K/V across denoising steps.
   - Keep this conservative first because prompt conditioning is stable.
   - Test byte-aligned or unpacked code formats for tiny 77-token text keys,
     where bit packing may cost more than it saves.
2. Denoising layer.
   - Continue the current timestep-gated attention work.
   - Add late-step self-attention policy sweeps at 1024 because spatial key
     counts are much larger than text-key cross-attention.
   - Test smaller QJL settings, including `QJL64` and no-QJL higher-bit K
     policies, to see whether the correction cost still earns its keep.
   - Revisit V compression only after K policy behavior is stable; start late
     and high precision.
3. Decode and final image layer.
   - Profile VAE decode separately before treating it as Shmoosh scope.
   - If decode is material on local cards, test Shmoosh-adjacent wins such as
     fp16-safe VAE settings, compile, tiling, or quantized conv paths.

## Parked Techniques

- Feature or output caching across denoising steps, inspired by DeepCache-style
  and TeaCache-style work.
- Token merging or token downsampling for redundant spatial tokens.
- `torch.compile(..., mode="reduce-overhead")` or CUDA graphs for fixed 1024
  shapes once processor hooks settle.
- Fused encode-plus-attention kernels for text-key modules if launch overhead
  remains the limiter.
- Separate runtime formats by key count: byte-aligned codes for short text keys,
  bit-packed codes for large self-attention keys.
- Policy composition tests that include these acceleration techniques together,
  not only one-at-a-time local A/B checks.

## Resume After Next Slice

The immediate next slice remains:

1. Trace a short losing case, such as maple-leaf, with vectorized packing.
2. Identify whether the remaining overhead is launch count, scheduled wrapper
   time, packed attention, encode, or fallback behavior.
3. Then choose between prompt-layer K/V cache, late-step self-attention sweep,
   or launch-overhead reduction as the next implementation target.
