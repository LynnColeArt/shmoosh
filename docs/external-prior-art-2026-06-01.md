# External Prior Art to Raid

This note records external work that directly informs the next Shmoosh kernel
slices. The shared theme is that compact KV representations only help when the
attention backend consumes them natively enough to avoid giving the memory win
back to decode overhead.

## AXELRAM / LOOKAT Shape

AXELRAM frames the desired systems shape cleanly: quantize KV cache entries
once, then compute attention scores from quantized indices without rebuilding
full-precision vectors on every read. The SRAM macro is not directly portable
to a 4070 Triton kernel, but the algorithmic lesson is portable:

```text
fixed codebook
transform on write
table lookup on read
score accumulation from compact indices
```

LOOKAT is also relevant because it applies lookup-optimized key attention with
product-quantized keys and lookup tables, again making attention consume the
compressed form directly instead of treating compression as a storage-only
layer.

Shmoosh already does the first-order version of this in fused packed attention:
the Triton kernel unpacks K indices, looks up scalar codebook values, and
accumulates `q_rot dot code_values` without materializing decoded K in global
memory. The remaining question is therefore narrower:

```text
Can the K7/no-QJL/head_dim=64 path become less generic and cheaper to interpret?
```

## Backend Warning

SGLang's quantized KV-cache documentation makes the systems warning explicit:
quantized KV can be slow if dequantization is not fused with attention or if the
selected backend lacks native quantized-KV support. That matches Shmoosh's
direct-rotated and byte-code experiments:

```text
byte-code/direct-rotated:
  cheaper encode or simpler representation
  worse attention because the read-side format cost dominates

bit-packed K7:
  denser reads win at 1024 even when encode is harder
```

## Quantization Policy Clues

KIVI and KVQuant both reinforce that K and V should not be treated as symmetric
quantization targets. KIVI's asymmetric KV-cache quantization and KVQuant's
per-channel/pre-RoPE key treatment are LLM-specific, but the useful Shmoosh
lesson is general: key grouping, scaling, and transform placement are policy
surfaces, not incidental implementation details.

For Shmoosh, the closest analogues are:

- quantize before or after the random rotation;
- fold norm/scale handling into the query or score path;
- test per-dimension or grouped key policies if K6/K7 quality-speed tradeoffs
  plateau.

## Diffusion-Specific Bias

The video-diffusion paper "Quantized Keys Steal Attention" is especially worth
tracking for lower-bit Shmoosh policies. Its central warning is that key
quantization noise can systematically inflate softmax attention weights, and it
derives an on-the-fly correction from quantization step size and query norm.

This does not directly solve the 4070 runtime problem, but it is a likely
quality-control lever if Shmoosh pushes below K7/no-QJL or sees composition
drift that looks like attention-weight inflation.

## Systems Architecture

FlashInfer is less about the exact Shmoosh quantizer and more about production
attention-engine shape: format-specialized kernels, JIT specialization,
heterogeneous KV storage, and scheduling around the real cache layout. This
supports keeping Shmoosh's next kernels narrow and measurable instead of trying
to make one generic kernel serve every representation.

## Next Kernel Slice

The current best target is not a new compression format. It is a more native
packed-attention computation:

```text
K7/no-QJL only
head_dim=64 only
CUDA/Triton only
self-attention 1024 first
hardcode the K7 bit layout
compare attention time against the current generic packed kernel
```

If this wins, specialize further around tile reuse. If it loses, the generic
kernel is already close enough and the next practical lever should be
encode+attention fusion or fixed-shape graph/compile overhead reduction.

## K7 Specialization Probe

The first hardcoded K7/no-QJL/head_dim=64 unpack probe lost badly and is
recorded in `docs/k7-head64-unpack-probe-2026-06-01.md`. That narrows the
lesson: native packed attention is still the right area, but the next attempt
should improve reuse or layout instead of reconstructing the same
`64 x BLOCK_K` code-value tile with more scalar control flow.

## Sources

- AXELRAM: https://arxiv.org/abs/2604.02638
- LOOKAT: https://arxiv.org/abs/2601.10155
- SGLang quantized KV cache docs:
  https://sgl-project.github.io/advanced_features/quantized_kv_cache.html
- KIVI: https://arxiv.org/abs/2402.02750
- KVQuant: https://arxiv.org/abs/2401.18079
- Quantized Keys Steal Attention: https://arxiv.org/abs/2605.26266
- FlashInfer: https://arxiv.org/abs/2501.01005
- TurboQuant: https://arxiv.org/abs/2504.19874
