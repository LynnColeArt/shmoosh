# Sparse Attention Research Leads - 2026-06-03

## Current Read

The 2026-06-02 static head-top-k image A/B batch is quality and plumbing
validation, not proof of an end-to-end sparse attention speed win.

The original printed mean speedups were contaminated by the first measured
baseline render:

```text
reading baseline: 11.806 s
maple baseline:    8.552 s
misty baseline:    8.600 s
```

The warmup-safe rerun in
`captures/image-policy-compare-juggernaut-static-headtopk-1024-warmup-20260602`
is the better runtime read:

```text
packed_k7:          1.025x, min 52.10 dB, mean 54.21 dB
static_topp95_q50:  1.002x, min 41.56 dB, mean 44.81 dB
static_topp98_q50:  1.009x, min 43.65 dB, mean 46.56 dB
static_topp95_q90:  1.020x, min 47.30 dB, mean 49.46 dB
```

Candidate ranking from this slice:

1. `packed_k7`: strongest quality candidate and the best control/default path.
2. `static_topp95_q90`: best static sparse quality, but still too lossy.
3. `static_topp98_q50`: middle quality, no convincing speed advantage.
4. `static_topp95_q50`: most aggressive and lowest quality.

The important timing clue is architectural: the materialized static sparse path
costs about 13-14 ms per quantized call, while packed K7 fused attention is
about 0.7 ms. The current sparse implementation is useful as an oracle, not as a
kernel target.

## External Leads

Sources checked:

- DeepSeek-V3.2 / DSA: <https://arxiv.org/abs/2512.02556>
- MISA: <https://arxiv.org/abs/2605.07363>
- NSA: <https://arxiv.org/abs/2502.11089>
- MHA2MLA: <https://arxiv.org/abs/2502.14837>
- MHA2MLA-VLM: <https://arxiv.org/abs/2601.11464>
- Sakana robust-kbench: <https://pub.sakana.ai/static/paper.pdf>

### DSA

DeepSeek-V3.2 introduces DeepSeek Sparse Attention as a dynamic sparse attention
mechanism for reducing long-context attention complexity while preserving model
behavior. The useful transfer is not the long-context KV-cache setting itself;
it is the idea that the sparse set should be query-dependent and selected by a
cheap side mechanism.

Shmoosh implication: static per-head budgets are probably the wrong final
sparsity shape. If we try sparse attention again, try dynamic token or block
selection against diffusion captures before building a kernel.

### MISA

MISA is especially relevant because it attacks the selector cost in DSA. Its
router uses cheap block-level statistics to activate only a few indexer heads
before expensive token-level scoring. That matches Shmoosh's failure mode: the
sparse selector/masking path is more expensive than the work it avoids.

Shmoosh implication: the next sparse oracle should be block-first. A cheap
block-level candidate set can be tested before any fine token top-k/top-p work.
If block routing cannot recover image quality in an oracle, a sparse kernel is
not worth building.

### NSA

Native Sparse Attention combines coarse-grained token compression with
fine-grained token selection and frames the design around hardware-aligned
arithmetic intensity. The details are LLM-training-specific, but the high-level
shape matches what Shmoosh needs: hierarchical sparse structure, not independent
static head budgets.

Shmoosh implication: test local-window plus block-selected global tokens as an
oracle. This is a better next sparse family than static top-k alone.

### MHA2MLA / MLA

MHA2MLA converts existing attention into a latent K/V representation using
partial-RoPE handling and low-rank approximations. MHA2MLA-VLM extends that
direction to multimodal models with modality-aware low-rank compression.

Shmoosh implication: this is not an immediate runtime drop-in for diffusion
self-attention, but it suggests a separate low-rank K/V probe:

```text
for selected SDXL attention modules:
  capture K and V tensors across prompts/timesteps
  fit per-module low-rank projections or SVD bases
  replay attention with low-rank K/V reconstruction
  measure output error and image A/B quality
```

This should be treated as a policy/quality experiment before any kernel work.

### Sakana-Style Evolutionary Search

The robust-kbench paper is less about Shmoosh's attention math and more about
how to avoid fooling ourselves. It emphasizes robust benchmark design,
verification, varied inputs, and evolutionary optimization under correctness
filters. That is directly relevant after the warmup artifact we just found.

Shmoosh implication: once benchmark semantics are stable, policy search should
optimize against warmed median speed, quality floors, memory, and stability. It
should not chase one noisy mean speedup.

## Next Best Slices

1. Keep `packed_k7` as the warmup-safe quality/speed control.
2. Add a small policy-search manifest generator that enumerates conservative
   policies over module set, timestep window, K bits, QJL bits, tile size,
   precision mode, and exact fallback windows.
3. Add a block-first sparse oracle over captured attention tensors:
   local window plus block-selected global tokens, with top-p/token reranking
   only inside selected blocks.
4. Run the oracle on captures before image A/B. Only promote policies that meet
   a strict tensor-quality floor.
5. Re-run image A/B with the warmup-safe compare CLI and report median and
   per-case speed, not only mean speed.
6. Defer sparse kernel work until an oracle policy recovers packed-K7-adjacent
   quality and has a selector shape that can plausibly be cheaper than dense
   packed attention.

## Working Thesis

```text
Shmoosh should not chase static sparse attention kernels yet.

The active speed path is:
  packed K7 as control/default
  warmup-safe benchmark semantics
  cheap routed/block sparse oracles
  evolutionary policy search after the metric is trustworthy
```
