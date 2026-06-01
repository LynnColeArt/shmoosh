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

The QJL/no-QJL self-attention variant slice is recorded in
`docs/self-attention-variant-bench-2026-06-01.md`. Synthetic 1024-token
self-attention favored `K7/no-QJL`, but the 50% image gate failed. Moving the
same K7/no-QJL policy to a 70% self-attention gate produced the best
self-attention-only result so far: `52.07 dB` minimum PSNR and `1.079x` mean
runtime across the three-case 1024 suite. Composing that policy with cached
cross-attention improved speed slightly versus the K6/QJL128 composition but
lowered quality, so it remains a tradeoff policy rather than a default.
The K7/no-QJL 70% processor trace confirmed the policy shape: 42 exact calls
and 18 quantized calls, with packed encode at `0.0202s`, packed attention at
`0.0336s`, and scheduled quantized time at `0.0732s`. Compared with the older
K6/QJL128 self-attention trace, mean quantized call time fell from `7.8270ms`
to `4.0640ms`.
Restricted cross+self composition then tested the lightest one and two
K7/no-QJL self-attention modules. One-module composition improved quality over
the full three-self-module composition, but cached cross-attention alone still
had the better fidelity/runtime balance: `49.40 dB` min PSNR and `1.046x` mean
speedup for cross-cache only versus `49.02 dB` and `1.053x` for one-module
self composition.
The non-overlapping handoff policy, with cross-attention active from 30%-70%
and self-attention active from 70%-100%, was faster at `1.088x` mean speedup
but still lower quality at `48.67 dB` minimum PSNR. Composition drift is
therefore not only same-step overlap.
The no-QJL streaming attention auto default now uses `BLOCK_K=32` for large-key
no-QJL attention while keeping QJL on `BLOCK_K=16`. Synthetic 1024-token
K7/no-QJL attention improved from `0.8348ms` to `0.6753ms`, but the image trace
only moved packed attention from `0.0336s` to `0.0276s` while scheduled
quantized time stayed flat at about `0.073s`.
Fast bit packing then reduced the preferred K7/no-QJL synthetic encode time
from `0.5865ms` to `0.3949ms`. In the image trace, `encode_pack_codes` dropped
from `0.0147s` to `0.0100s`, and packed encode dropped from `0.0252s` to
`0.0195s`. Whole scheduled quantized time remained noisy at about this scale.
The three-case 1024 suite after this change cleared at `51.87 dB` minimum PSNR
and `1.084x` mean speedup.
Caching codebook bucket boundaries removed repeated constant setup from encode,
but the synthetic rerun was noisy and did not produce a separate K7 speed win.

The next slice should be:

1. Keep the cached cross-attention policy as the baseline policy layer.
2. Treat late K7/no-QJL self-attention as a separate high-fidelity policy mode,
   not a default add-on to cached cross-attention.
3. Look for the next self-attention speed lever beyond bit packing:
   fuse encode+attention, reduce projection/bucketize overhead, or try CUDA
   graphs/compile for fixed 1024 runs.
4. Revisit cross+self composition only after adding a new control surface, such
   as per-prompt policy choice or a stricter image-quality gate.
