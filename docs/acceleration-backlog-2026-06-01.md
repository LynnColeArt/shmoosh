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

The maple short-case trace, prompt-layer K/V cache slice, and first late-step
self-attention smoke slice are now done. The cache is correct and modestly
useful, but selected text-key cross-attention is too small to be the main speed
lever. The self-attention smoke is recorded in
`docs/self-attention-sweep-2026-06-01.md`: three K6/QJL128 `up_blocks.0.attn1`
modules passed single-module 1024 probes with exact-first 50% activation, and
their three-module composition passed the reading-nook 1024 prompt at
`50.57 dB` PSNR. The same self-attention policy also cleared the three-case
1024 suite with `49.62 dB` minimum PSNR and a `1.057x` mean runtime signal.
The self-attention trace shows `packed_attention` at `0.1329s` and
`packed_encode` at `0.0750s` across 30 quantized calls, so the 1024-token path
needs streaming attention work as well as encode work.

Cross+self composition is recorded in
`docs/composition-policy-2026-06-01.md`. The 50% self-attention composition
passed but degraded quality and speed. The 70% self-attention composition is
the better exploratory policy, clearing the three-case 1024 suite with
`49.17 dB` minimum PSNR and a `1.048x` mean runtime signal, but it is still only
marginally better than the cached cross-attention policy alone.

The next slice should be:

1. Keep the cached cross-attention policy as the baseline policy layer.
2. Profile the self-attention streaming kernel directly.
3. Compare QJL64 and no-QJL K6/K7 variants for the self-attention modules.
4. Re-run the 70% composition after any kernel or policy-cost improvement.
