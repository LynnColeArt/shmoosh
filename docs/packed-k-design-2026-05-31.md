# Packed-K Design: 2026-05-31

## Scope

This note targets the accepted native-resolution policy:

```text
configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated30pct-k5-k6-qjl128-policy.json
```

That policy quantizes keys only, keeps values exact, uses QJL-128, and activates
after the first 30% of denoising steps. It covers five K5 modules and two K6
modules in `up_blocks.0` cross-attention.

The current NumPy attention processor is only a correctness harness. A
production implementation has to keep keys packed until score computation;
if a kernel expands packed keys back to full fp16 tensors before attention, the
memory-bandwidth value disappears.

## Estimator

The repeatable estimator is:

```bash
uv run shmoosh-packed-policy-estimate \
  --policy-file configs/underpaint-juggernaut-sdxl-up0-cross-mixed-gated30pct-k5-k6-qjl128-policy.json \
  --steps 20 \
  --steps 30
```

Assumptions for the SDXL cross-attention path:

| Field | Value |
| --- | ---: |
| classifier-free-guidance batch | 2 |
| heads per selected module | 20 |
| text key tokens | 77 |
| head dim | 64 |
| exact dtype bytes | 2 |
| norm bytes | 4 |
| residual norm bytes | 4 |

These assumptions match the metadata observed in the 1024 runs for the selected
modules: 20 heads, 64-dimensional heads, and 2048-dimensional cross-attention
conditioning.

## Packed Vector Format

Per key vector:

| Format | Codes | Norm | QJL Signs | Residual Norm | Total | Exact FP16 Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact fp16 K | 128 B | 0 B | 0 B | 0 B | 128 B | 1.00x |
| K5 + QJL-128 | 40 B | 4 B | 16 B | 4 B | 64 B | 2.00x |
| K6 + QJL-128 | 48 B | 4 B | 16 B | 4 B | 72 B | 1.78x |

For the accepted mixed policy, the weighted packed-key ratio is `1.93x` during
quantized steps.

## Policy Estimate

Across the selected seven modules:

| Window | Exact K | Packed K | Saved | Ratio |
| --- | ---: | ---: | ---: | ---: |
| per quantized step | 2.63 MiB | 1.36 MiB | 1.27 MiB | 1.93x |
| 20-step quantized window | 36.85 MiB | 19.08 MiB | 17.76 MiB | 1.93x |
| 30-step quantized window | 55.27 MiB | 28.62 MiB | 26.65 MiB | 1.93x |

Across the full denoising horizon, exact early steps reduce the whole-run ratio:

| Horizon | All-Exact Selected K | Scheduled Packed K | Saved | Ratio |
| --- | ---: | ---: | ---: | ---: |
| 20 steps | 52.64 MiB | 34.87 MiB | 17.76 MiB | 1.51x |
| 30 steps | 78.96 MiB | 52.31 MiB | 26.65 MiB | 1.51x |

This is a lower-bound payload estimate. It counts materialized K payload bytes,
not every memory transaction in a fused attention kernel. If the kernel reloads
K tiles repeatedly, direct packed-K consumption can amplify the practical
bandwidth benefit.

## Torch-Side Metadata Shape

The first Torch-side prototype expresses packed keys as an explicit container,
not as an overloaded tensor. The implemented contract is
`shmoosh.packed_keys.PackedKeyBlock`:

```text
PackedKeyBlock
  codes: uint8[B, H, T, ceil(D * bits / 8)]
  norms: fp32[B, H, T]
  residual_signs: uint8[B, H, T, ceil(qjl_bits / 8)]
  residual_norms: fp32[B, H, T]
  bits: int
  qjl_bits: int
  head_dim: int
  seed: int
  codebook_samples: int
  lloyd_iters: int
```

The block currently stores deterministic codec parameters instead of materialized
resource IDs. A fused kernel should lower those parameters into immutable
per-module/per-bit resources:

| Resource | Scope |
| --- | --- |
| codebook | `bits`, `head_dim`, `seed`, `codebook_samples` |
| rotation | `head_dim`, `seed` |
| QJL projection | `qjl_bits`, `head_dim`, `seed` |

The policy already uses deterministic `processor_seed=11`, so the first packed
prototype can reuse stable resource IDs instead of storing matrices inside every
block.

The debug smoke command is:

```bash
uv run shmoosh-packed-key-smoke \
  --batch-size 1 \
  --heads 20 \
  --tokens 77 \
  --dim 64 \
  --bits 5 \
  --qjl-bits 128
```

It validates the tensor shapes, byte counts, and reference decode path. It does
not claim production speed, because the decode path still expands K before
attention.

## Packed Score Prototype

The first score path is `shmoosh.packed_scores.packed_key_scores`:

```text
query: fp16/fp32[B, H, Q, D]
packed_key_block: PackedKeyBlock[B, H, T, D]
scores: fp32[B, H, Q, T]
```

The Torch reference and Triton prototype share the same math:

1. Rotate query vectors into codec space with the deterministic codec rotation.
2. Unpack K code indices from `PackedKeyBlock.codes`.
3. Gather codebook centroids and accumulate the base dot product in codec space.
4. Add the QJL residual correction from packed sign bits and residual norms.

The Triton path still precomputes query-side projections in Torch:

```text
q_rot = Q @ rotation.T
q_proj = Q @ qjl_matrix.T
```

This is deliberate for the first kernel slice. It isolates the K-side packed
memory contract and proves that a GPU kernel can consume packed codes and sign
bytes directly without decoding full model-space K.

Smoke it with:

```bash
uv run shmoosh-packed-score-smoke \
  --batch-size 1 \
  --heads 20 \
  --query-tokens 64 \
  --key-tokens 77 \
  --dim 64 \
  --bits 5 \
  --qjl-bits 128 \
  --backend auto
```

## Packed Attention Prototype

The next bridge is `shmoosh.packed_attention.packed_key_attention_output`:

```text
query: fp16/fp32[B, H, Q, D]
packed_key_block: PackedKeyBlock[B, H, T, D]
value: fp16/fp32[B, H, T, D]
output: fp16/fp32[B, H, Q, D]
```

This keeps V exact and reuses `packed_key_scores` for the logits:

```text
scores = packed_key_scores(query, packed_key_block)
weights = softmax(scores / sqrt(D))
output = weights @ value
```

The smoke command compares the packed Torch/Triton path against the existing
NumPy runtime reference with `quantize_values=False`:

```bash
uv run shmoosh-packed-attention-smoke \
  --batch-size 1 \
  --heads 20 \
  --query-tokens 64 \
  --key-tokens 77 \
  --dim 64 \
  --bits 5 \
  --qjl-bits 128 \
  --backend auto
```

This is the first end-to-end attention primitive on the production side of the
research line: packed K, exact V, softmax, and attention output. It is still not
a Diffusers replacement because masks, layout integration, and output projection
plumbing remain outside this primitive.

## Diffusers Processor Hook

`ShmooshAttnProcessor` can now opt into the packed attention primitive:

```text
attention_backend = "reference" | "packed"
packed_backend = "auto" | "torch" | "triton"
```

The packed processor path activates only when the resolved policy is
`quantize_keys=true` and `quantize_values=false`. Other policies keep using the
reference path so value quantization and exact-key calibration behavior stay
unchanged.

The image CLIs expose the same knobs:

```bash
uv run shmoosh-image-ab-smoke ... \
  --attention-backend packed \
  --packed-backend auto
```

Policy files can also set those fields at top level inside `shmoosh_policy` or
inside a per-module `shmoosh_policy` override. This lets the accepted 1024
policy run through the packed-K exact-V path without changing the policy module
selection or timestep window semantics.

The processor now caches codec objects and packed score resources per module,
bit depth, head dimension, device, and seed. The image CLIs also warm packed
processors before timed generation. This removes repeated codebook/rotation/QJL
resource construction from the hot image path and moves Triton compilation out
of the measured Shmoosh run.

The packed backend now also routes CUDA `auto`/`triton` attention through a
fused Triton output kernel when the key sequence fits in the fixed `128` token
text-key tile. That kernel keeps V exact, consumes packed K directly, applies
softmax, and writes the final attention output without allocating a full score
tensor. Larger key sets keep using the materialized Triton score fallback.

## Kernel Direction

The current production path is a staged Torch/Triton design:

1. Encode K after `to_k` for selected modules and selected timesteps.
2. In the attention score kernel, consume packed codes directly.
3. For text-key tiles, fuse softmax and exact-V output accumulation into the
   packed-K Triton kernel.
4. For larger key sets, keep V exact and use existing attention output
   accumulation as a fallback.

The score computation can avoid reconstructing full K vectors by rotating query
tiles into codec space, accumulating codebook dot products, then adding the QJL
residual correction. A temporary reconstruct-then-attend kernel is useful for
debugging, but it should not be treated as the production target because it
reintroduces the fp16 K bandwidth.

The next kernel step is to fold query rotation and QJL projection into the
fused kernel, reducing the extra projected-query tensors that still surround the
current fused output launch.

## Acceptance

The next implementation slice should not promise speed yet. It should prove the
data path:

1. A Torch-side packed key object can represent K5/K6 + QJL-128 keys for the
   accepted policy.
2. Its byte accounting matches `shmoosh-packed-policy-estimate`.
3. A debug decode path reconstructs keys closely enough to match the current
   NumPy reference output.
4. The fused-score path can then be introduced behind the same container.
