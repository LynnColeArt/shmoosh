# Packed Attention Perf Guard

## Purpose

The packed streaming kernel is sensitive to small Triton source changes. This guard gives us a fast local tripwire for the current 4070 target path before trying deeper kernel work.

Guard profile:

```text
batch=1
heads=20
query_tokens=1024
key_tokens=1024
head_dim=64
bits=7
qjl_bits=0
code_format=packed_t
key_encode_backend=split
dot_precision=ieee
score_dot_precision=tf32
value_dot_precision=tf32
```

Default thresholds:

| Metric | Threshold |
| --- | ---: |
| median packed attention | <= 0.45 ms |
| median encode+attention total | <= 0.75 ms |
| max relative RMSE | <= 0.03 |
| min cosine | >= 0.999 |

The latency thresholds are intentionally guard bands, not record-setting targets. They should catch real regressions without failing on ordinary GPU timing variance.

## Command

```bash
uv run shmoosh-packed-attention-perf-guard \
  --samples 3 \
  --iters 200 \
  --warmup-iters 12 \
  --output-json captures/packed-attention-perf-guard/k7-packedt-1024-20260602.json
```

## First 4070 Run

Output:

```text
packed attention perf guard passed
attention=0.3915ms median (max 0.3939ms), total=0.5988ms median, rel_rmse=0.024060 max
```

Aggregate:

| Metric | Min | Median | Mean | Max |
| --- | ---: | ---: | ---: | ---: |
| attention ms | 0.3564 | 0.3915 | 0.3806 | 0.3939 |
| encode ms | 0.1875 | 0.1915 | 0.2039 | 0.2326 |
| total ms | 0.5575 | 0.5988 | 0.5890 | 0.6108 |
| relative RMSE | 0.024060 | 0.024060 | 0.024060 | 0.024060 |
| cosine | 0.999712 | 0.999712 | 0.999712 | 0.999712 |

Use this before and after packed streaming kernel experiments. If a candidate fails the guard, either revert it or document why the threshold needs to change.
