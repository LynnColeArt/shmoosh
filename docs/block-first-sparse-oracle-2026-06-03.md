# Block-First Sparse Oracle - 2026-06-03

## Slice

Added block-first sparse mask families to `shmoosh-attention-sparsity-oracle`.
These are tensor replay oracles, not runtime kernels.

New modes:

```text
spatial_block_top_k:
  select top spatial key tiles per query
  keep every key in those selected tiles

local_spatial_block_top_k:
  keep a square local window
  union it with top spatial key tiles per query
```

The selector is for square diffusion self-attention only. Cross-attention and
non-square attention captures skip these modes automatically.

## Commands

Initial block budget sweep:

```bash
uv run python -m shmoosh.cli.attention_sparsity_oracle captures/underpaint-juggernaut-sweep \
  --self-attn-only \
  --min-key-tokens 1024 \
  --device cuda \
  --dtype fp16 \
  --top-k "" \
  --top-p "0.95,0.98" \
  --local-windows "" \
  --spatial-block-top-k "4:4,4:8,4:16,8:1,8:2,8:4" \
  --local-spatial-block-top-k "4:4:9,4:8:9,4:8:17,8:1:9,8:2:9,8:2:17" \
  --csv captures/attention-sparsity-block-first-1024-20260603.csv \
  --json captures/attention-sparsity-block-first-1024-20260603.json
```

Wider block budget sweep:

```bash
uv run python -m shmoosh.cli.attention_sparsity_oracle captures/underpaint-juggernaut-sweep \
  --self-attn-only \
  --min-key-tokens 1024 \
  --device cuda \
  --dtype fp16 \
  --top-k "" \
  --top-p "0.95,0.98" \
  --local-windows "" \
  --spatial-block-top-k "4:16,4:24,4:32,8:4,8:6,8:8" \
  --local-spatial-block-top-k "4:16:9,4:24:9,4:24:17,8:4:9,8:6:9,8:6:17" \
  --csv captures/attention-sparsity-block-first-wide-1024-20260603.csv \
  --json captures/attention-sparsity-block-first-wide-1024-20260603.json
```

Both runs used six 1024-token self-attention captures:

```text
down_blocks.1.attn1 captures 0, 1, 2
up_blocks.1.attn1 captures 0, 1, 2
```

## Aggregate Results

Initial sweep:

| Mode | Setting | Kept | Mass | Mean rel RMSE | Worst rel RMSE | Mean cos err |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| top-p | 0.95 | 0.3880 | 0.9503 | 0.04054 | 0.05111 | 0.000481 |
| top-p | 0.98 | 0.5085 | 0.9801 | 0.01774 | 0.02219 | 0.000094 |
| spatial blocks | 4:8 | 0.1250 | 0.5760 | 0.29831 | 0.49476 | 0.032140 |
| spatial blocks | 4:16 | 0.2500 | 0.7336 | 0.17553 | 0.29363 | 0.012899 |
| local + spatial blocks | 4:8:17 | 0.2732 | 0.6897 | 0.19833 | 0.31217 | 0.016067 |
| spatial blocks | 8:4 | 0.2500 | 0.6503 | 0.23218 | 0.38700 | 0.022527 |

Wider sweep:

| Mode | Setting | Kept | Mass | Mean rel RMSE | Worst rel RMSE | Mean cos err |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| top-p | 0.95 | 0.3880 | 0.9503 | 0.04054 | 0.05111 | 0.000481 |
| top-p | 0.98 | 0.5085 | 0.9801 | 0.01774 | 0.02219 | 0.000094 |
| spatial blocks | 4:24 | 0.3750 | 0.8272 | 0.11512 | 0.19387 | 0.006059 |
| spatial blocks | 4:32 | 0.5000 | 0.8908 | 0.07668 | 0.12967 | 0.002881 |
| local + spatial blocks | 4:24:17 | 0.4536 | 0.8598 | 0.09405 | 0.15323 | 0.004113 |
| spatial blocks | 8:8 | 0.5000 | 0.8360 | 0.10540 | 0.17920 | 0.005499 |

## Read

Block-first routing works as plumbing, but this specific tile selector is not a
quality win. It needs far more retained keys than token top-p to approach the
dense output, and even at similar kept fractions it is much worse:

```text
top-p 0.95:
  kept 0.3880, rel RMSE 0.04054

spatial blocks 4:24:
  kept 0.3750, rel RMSE 0.11512

top-p 0.98:
  kept 0.5085, rel RMSE 0.01774

spatial blocks 4:32:
  kept 0.5000, rel RMSE 0.07668
```

The local-window union did not rescue block routing in this slice. It often
improves over coarser `8x8` tiles, but the best results still trail token-level
top-p by a wide margin.

This is a useful negative result:

```text
static per-head top-k:
  image-plausible but too lossy/slow

spatial block-first top-k:
  cheap-selector-shaped, but tensor quality is too weak

dynamic token top-p:
  still the strongest sparse oracle by quality
  still hard to make cheap enough at runtime
```

## Next Slice

Do not build a block sparse kernel from this selector.

The better next slice is policy search around the existing packed K7 path:

1. Generate conservative policy manifests over module set, timestep window,
   K bits, QJL bits, tile size, and precision mode.
2. Score them with the warmup-safe image compare CLI.
3. Use quality floors and per-case/median speed, not one mean speedup.

If sparse work continues, the next sparse oracle should test a two-stage
selector that uses coarse tiles only to narrow candidates, then performs
token-level top-p or top-k inside selected tiles. The current whole-tile keep
policy is too blunt.
