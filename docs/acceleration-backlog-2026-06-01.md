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
The byte-code runtime V2 slice is recorded in
`docs/self-attention-runtime-v2-2026-06-01.md`. It validated an opt-in
`code_format="byte"` path end-to-end, including fused Triton attention and
Diffusers policy selection. Synthetic 1024-token K7/no-QJL self-attention
improved total time from `1.1133ms` packed to `1.0223ms` byte-code by cutting
encode from `0.4330ms` to `0.1552ms`, but fused attention slowed from
`0.7665ms` to `0.8826ms` because byte-code stores `68` bytes/vector instead of
`60`. The three-case image suite passed with identical quality to fast
bit-packed K7/no-QJL (`51.87 dB` minimum PSNR, `53.96 dB` mean PSNR), but mean
speedup fell to `1.057x` versus `1.084x`. Keep byte-code as an opt-in format
for shorter-key experiments; keep bit-packed K as the current 1024
self-attention default.
The compact-K kernel V2 slice is recorded in
`docs/compact-k-kernel-v2-2026-06-01.md`. It kept bit-packed K and avoided
continuation-byte loads when a packed code does not cross a byte boundary.
Synthetic K6/K7 no-QJL attention improved directionally in the valid sequential
comparison. Image validation then tested K6/no-QJL against K7/no-QJL at the
same 70% gate. K6/no-QJL is a viable speed-mode candidate at `1.082x` mean
speedup, but loses quality (`50.38 dB` min PSNR, `52.91 dB` mean PSNR) versus
K7/no-QJL (`51.87 dB` min PSNR, `53.96 dB` mean PSNR). Keep K7/no-QJL as the
preferred 1024 self-attention default; keep K6/no-QJL as an explicit speed
tradeoff policy.
The encode-normalize V2 slice is recorded in
`docs/encode-normalize-v2-2026-06-01.md`. It replaced an extra normalized tensor
with in-place normalization on the float32 working copy, while preserving raw
keys only for QJL residual correction. Synthetic K6/K7 no-QJL encode improved
from `0.2855ms` to `0.2509ms` for K6 and from `0.3949ms` to `0.3738ms` for K7.
The three-case 1024 suites stayed quality-correct: K7/no-QJL at `51.87 dB`
minimum PSNR and `1.076x` mean speedup; K6/no-QJL at `50.38 dB` minimum PSNR
and `1.092x` mean speedup. This is a small encode cleanup, not a new default.
The fused bucketize/pack slice is recorded in
`docs/fused-bucketize-pack-2026-06-01.md`. It adds a K7/no-QJL Triton fast path
that writes packed codes directly after boundary search, avoiding the
materialized code-index tensor. K7 synthetic encode improved from `0.3738ms` to
`0.2343ms`; the K7 image suite stayed quality-identical at `51.87 dB` minimum
PSNR and `1.070x` mean speedup, with mean packed encode moving from `0.9185ms`
to `0.8821ms`. K6 was tested but rejected for this fast path because the image
suite did not preserve the synthetic encode win.
The fused rotation+bucketize+pack slice is recorded in
`docs/fused-rotate-bucketize-pack-2026-06-01.md`. It moved K7/no-QJL rotation
into the Triton encode kernel, kept byte-identical packed-code correctness in a
CUDA unit test, and kept the 1024 image suite quality-identical. Mean packed
encode moved from `0.8821ms` to `0.8342ms`; mean rotate/bucketize moved from
`0.5178ms` to `0.4722ms`. This is a real kernel simplification, still not a
visible whole-pipeline UX win by itself.
The direct rotated-K probe is recorded in
`docs/direct-rotated-k-probe-2026-06-01.md`. It added an opt-in synthetic
`code_format="rotated"` path that stores normalized rotated K plus exact norms,
then consumes that representation in a streaming Triton attention kernel.
Encode improved versus packed K7/no-QJL (`0.0949ms` versus `0.1777ms`), and
quality was near-exact (`0.000310` relative RMSE), but attention slowed sharply
(`1.3738ms` versus `0.6771ms`) because the representation reads `132`
bytes/vector instead of packed K7's `60`. Keep this as a diagnostic path; do
not promote direct rotated K as the 1024 self-attention runtime format.
External prior art is summarized in
`docs/external-prior-art-2026-06-01.md`. The key correction is that Shmoosh
already computes first-order codebook-dot attention from packed K inside
Triton; the next useful question is whether the K7/no-QJL/head_dim=64 path can
be made less generic and cheaper to interpret.
The K7/head_dim=64 specialized unpack probe is recorded in
`docs/k7-head64-unpack-probe-2026-06-01.md`. A hardcoded 7-byte/8-code unpack
kernel was correctness-equivalent, but much slower than the generic packed
kernel (`1.5081ms` attention versus `0.6445ms`), so it was not retained in
runtime code. The likely culprit is register pressure and rebuilding the full
code-value tile through many `where` operations.
The streaming tile V2 slice is recorded in
`docs/streaming-tile-v2-2026-06-02.md`. It promotes a narrow
K7/no-QJL/head_dim=64 streaming tile choice from `BQ32/BK32` to `BQ64/BK16`.
Synthetic wins were small and noisy, but the three-case 1024 image suite kept
quality unchanged (`52.07 dB` minimum PSNR, `54.27 dB` mean PSNR) and moved mean
packed attention from the prior fused-rotate `1.4519ms` per call to `1.3521ms`.
The packed transpose layout slice is recorded in
`docs/packed-transpose-layout-2026-06-02.md`. It adds opt-in
`code_format="packed_t"`, storing packed code bytes as `(code_bytes, tokens)` so
Triton can read each byte lane contiguously across key tokens. The preferred
K7/no-QJL 1024 self-attention policy now uses `packed_t`: quality stayed
unchanged (`52.07 dB` minimum PSNR), and mean processor phases moved from
`1.3521ms` to `1.2806ms` packed attention and from `3.1222ms` to `2.9597ms`
scheduled quantized time.

The next slice should be:

1. Keep the cached cross-attention policy as the baseline policy layer.
2. Treat late K7/no-QJL self-attention as a separate high-fidelity policy mode,
   not a default add-on to cached cross-attention.
3. Look for the next self-attention speed lever beyond bit packing and
   normalize/bucketize/rotation cleanup: fuse encode+attention while preserving
   compact K reads, reduce decode work inside packed attention, or try CUDA
   graphs/compile for fixed 1024 runs.
4. Revisit cross+self composition only after adding a new control surface, such
   as per-prompt policy choice or a stricter image-quality gate.

The CUDA graph probe is recorded in `docs/cuda-graph-probe-2026-06-02.md`.
Fixed-shape graph replay gave a small synthetic gain for the current
`packed_t` K7/no-QJL path: attention moved from `0.6849ms` to `0.6645ms`, while
encode+attention moved from `0.8919ms` to `0.8768ms`. This is useful evidence,
but too small to justify a production static-buffer graph cache before deeper
packed attention or encode+attention kernel work.

The fp16 norm probe is recorded in `docs/fp16-norm-probe-2026-06-02.md`.
It added opt-in `norm_dtype="fp16"`, reducing K7/no-QJL `packed_t` storage from
`60` to `58` bytes/vector. Synthetic quality barely moved
(`0.023998` to `0.024005` relative RMSE), but attention time did not improve
(`0.7564ms` fp32 norms versus `0.7686ms` fp16 norms), so fp16 norms stay an
experimental memory-format knob rather than the preferred self-attention
default.
