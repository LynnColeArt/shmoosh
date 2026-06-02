# Attention Sparsity Oracle - 2026-06-02

## Slice

Added an exact-attention sparsity oracle for captured or synthetic Q/K/V tensors:

```bash
uv run python -m shmoosh.cli.attention_sparsity_oracle \
  captures/underpaint-juggernaut-sweep \
  --self-attn-only \
  --min-key-tokens 1024 \
  --limit 24 \
  --device auto \
  --dtype fp16 \
  --top-k 64,128,256 \
  --top-p 0.95,0.98 \
  --local-windows 9,17,33 \
  --csv captures/attention-sparsity-oracle/1024-self-2026-06-02.csv \
  --json captures/attention-sparsity-oracle/1024-self-2026-06-02.json
```

This is not a runtime speed claim. It asks which sparse masks preserve dense
attention output well enough to justify a real packed sparse kernel.

## Result

Run target: six 1024-token self-attention captures from the Juggernaut sweep,
fp16 inputs on CUDA.

| Mask | Mean kept keys | Mean attention mass | Mean relative RMSE | Mean cosine error |
| --- | ---: | ---: | ---: | ---: |
| top-k 64 | 6.25% | 0.6085 | 0.3118 | 0.032003 |
| top-k 128 | 12.50% | 0.7364 | 0.2126 | 0.017113 |
| top-k 256 | 25.00% | 0.8581 | 0.1262 | 0.006984 |
| top-p 0.95 | 38.80% | 0.9503 | 0.0405 | 0.000481 |
| top-p 0.98 | 50.85% | 0.9801 | 0.0177 | 0.000094 |
| local 9x9 | 6.85% | 0.3039 | 0.5824 | 0.114538 |
| local 17x17 | 21.25% | 0.4789 | 0.3882 | 0.062751 |
| local 33x33 | 58.62% | 0.7398 | 0.2250 | 0.025028 |

Best single row:

```text
capture_020.npz top-p 0.98
kept_key_fraction=0.5540
relative_rmse=0.012820
cosine_error=0.000064
```

## Read

Top-p is the first sparsity family here that looks like a plausible next
runtime target. It keeps too many keys to be a dramatic attention reduction by
itself, but it preserves dense attention far better than fixed top-k or pure
local windows in this capture set.

Pure local windows are not enough for SDXL-style 1024 self-attention. Even a
33x33 window keeps 58.6% of keys and still has much worse error than top-p
0.98. That suggests spatial locality should be treated as a helper constraint
or block layout, not as the whole policy.

Fixed top-k is useful as a lower-bound pressure test. K=256 only keeps 25% of
keys, but the output error is still high enough that it should not be the first
production sparse target unless an image-level run proves the model tolerates
the drift.

## Prior Art Direction

Recent sparse diffusion attention papers point at the same control surface:

- [HASTE](https://arxiv.org/abs/2605.14513) argues for head-wise adaptive
  top-p budgets and mask reuse based on query-key drift.
- [SpargeAttention2](https://arxiv.org/abs/2602.13515) uses hybrid top-k/top-p
  masking and reports that the two common rules fail in different regimes.
- [PISA](https://arxiv.org/abs/2602.01077) warns that discarding non-critical
  blocks can degrade quality, and instead approximates some of the discarded
  span.
- [FlashInfer](https://arxiv.org/abs/2501.01005) is useful systems precedent:
  sparse attention needs storage-layout-aware, format-specialized kernels.
- [Quantized Keys Steal Attention](https://arxiv.org/abs/2605.26266) remains a
  warning for aggressive packed/quantized K: key quantization can bias softmax
  mass, so bias correction may become relevant if sparse plus quantized K
  starts showing image drift.

## Next Slice

The most promising next implementation slice is a top-p calibration pass over
real captures:

1. Add per-head stats to the oracle.
2. Find per-head top-p budgets that hit a global kept-key target.
3. Export a static budget table for the 1024 self-attn modules.
4. Compare that table against top-p 0.98 and fixed top-k 256 in image A/B.

That gives us a policy candidate before building a sparse packed kernel.
