# Attention Budget Calibration - 2026-06-02

## Slice

Added a static per-head budget calibrator:

```bash
uv run python -m shmoosh.cli.attention_budget_calibrate \
  captures/underpaint-juggernaut-sweep \
  --self-attn-only \
  --min-key-tokens 1024 \
  --limit 24 \
  --device auto \
  --dtype fp16 \
  --top-p 0.95,0.98 \
  --budget-quantiles 0.50,0.90,0.95,0.99,1.0 \
  --csv captures/attention-budget-calibration/1024-self-static-2026-06-02.csv \
  --json captures/attention-budget-calibration/1024-self-static-2026-06-02.json \
  --head-csv captures/attention-budget-calibration/1024-self-heads-2026-06-02.csv \
  --head-json captures/attention-budget-calibration/1024-self-heads-2026-06-02.json \
  --budget-json captures/attention-budget-calibration/1024-self-policies-2026-06-02.json
```

The calibrator converts dynamic top-p masks into static per-head top-k budgets:

1. For each capture/head/query, compute the number of keys needed to reach a
   top-p mass threshold.
2. For each module/head, take a quantile of those kept-counts as the static K
   budget.
3. Re-evaluate that static per-head top-k policy against dense exact attention.

This is still an oracle. It does not claim runtime speed. It narrows the sparse
kernel target from "dynamic top-p attention" to "per-head static K budgets".

## Aggregate Result

Run target: six 1024-token self-attention captures from the Juggernaut sweep,
fp16 inputs on CUDA.

| Calibrated from | Budget quantile | Mean kept keys | Mean attention mass | Mean relative RMSE | Mean cosine error |
| --- | ---: | ---: | ---: | ---: | ---: |
| top-p 0.95 | 0.50 | 38.64% | 0.9338 | 0.050867 | 0.001158 |
| top-p 0.95 | 0.90 | 55.75% | 0.9775 | 0.021214 | 0.000204 |
| top-p 0.95 | 0.95 | 58.77% | 0.9813 | 0.018337 | 0.000156 |
| top-p 0.95 | 0.99 | 63.65% | 0.9863 | 0.014515 | 0.000102 |
| top-p 0.95 | 1.00 | 70.23% | 0.9913 | 0.010357 | 0.000054 |
| top-p 0.98 | 0.50 | 50.96% | 0.9678 | 0.026605 | 0.000364 |
| top-p 0.98 | 0.90 | 69.13% | 0.9916 | 0.009117 | 0.000039 |
| top-p 0.98 | 0.95 | 71.99% | 0.9932 | 0.007697 | 0.000029 |
| top-p 0.98 | 0.99 | 76.41% | 0.9953 | 0.005845 | 0.000017 |
| top-p 0.98 | 1.00 | 82.01% | 0.9972 | 0.004029 | 0.000009 |

## Module Differences

Median budget policies show that the two 1024 self-attention modules do not
want the same sparsity:

| Module | Calibrated from | Mean budget | Min budget | Max budget | Kept keys |
| --- | ---: | ---: | ---: | ---: | ---: |
| down_blocks.1.attn1 | top-p 0.95 | 483.4 | 254 | 780 | 47.21% |
| down_blocks.1.attn1 | top-p 0.98 | 622.0 | 384 | 889 | 60.74% |
| up_blocks.1.attn1 | top-p 0.95 | 307.9 | 50 | 633 | 30.07% |
| up_blocks.1.attn1 | top-p 0.98 | 421.7 | 90 | 778 | 41.18% |

That is the strongest policy result in this slice: a single global K is leaving
real module/head structure on the floor.

## Read

Static median budgets are the interesting runtime-facing candidates:

- top-p 0.95, q=0.50 keeps 38.64% of keys and has 0.0509 relative RMSE.
- top-p 0.98, q=0.50 keeps 50.96% of keys and has 0.0266 relative RMSE.

Compared with dynamic top-p from the previous oracle:

| Policy | Mean kept keys | Mean relative RMSE |
| --- | ---: | ---: |
| dynamic top-p 0.95 | 38.80% | 0.0405 |
| static top-p 0.95 median budget | 38.64% | 0.0509 |
| dynamic top-p 0.98 | 50.85% | 0.0177 |
| static top-p 0.98 median budget | 50.96% | 0.0266 |

So static budgets pay a quality penalty at the same kept-key fraction, but the
penalty is not catastrophic. That makes them worth image A/B testing before a
kernel build.

The 0.90+ budget quantiles are probably too expensive as speed candidates.
They are useful as safe reference points, especially top-p 0.95 q=0.90:
55.75% kept keys and 0.0212 relative RMSE.

## Next Slice

Run image A/B with three sparse policy candidates:

1. Static median budget from top-p 0.95: quality-risky, speed-interesting.
2. Static median budget from top-p 0.98: likely safer, still simpler than
   dynamic top-p.
3. Static q=0.90 budget from top-p 0.95: conservative reference.

Compare against dense, packed K7/no-QJL, and the prior fixed top-k 256 oracle
control. If the image suite accepts one of the median policies, the kernel
target becomes per-head static top-k selection for 1024 self-attention.
