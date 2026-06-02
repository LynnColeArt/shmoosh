# Packed Encode Parity

This slice adds a repeatable split-vs-fused encode parity probe:

```text
shmoosh-packed-encode-parity
```

It compares:

```text
split encode block
fused encode block
code indices
stored norms
packed attention output
fused codes with split norms
split codes with fused norms
```

The goal is to explain the fused encode image delta without running the full
SDXL image suite every time.

## Command

```bash
uv run shmoosh-packed-encode-parity captures/underpaint-juggernaut-sweep \
  --self-attn-only \
  --min-key-tokens 128 \
  --dtype fp16 \
  --bits 7 \
  --qjl-bits 0 \
  --code-format packed_t \
  --attention-backend torch \
  --csv captures/packed-encode-parity-self-attn-k7-packedt-20260602.csv \
  --json captures/packed-encode-parity-self-attn-k7-packedt-20260602.json
```

The same run passed as a QA gate with:

```bash
--max-code-diff-rate 0.00001
--max-output-mse 0.00000003
```

## Result

Across 15 captured self-attention tensors:

| Metric | Value |
| --- | ---: |
| Total code index differences | 30 |
| Worst code diff count | 8 |
| Worst code diff rate | 0.00000763 |
| Worst output MSE | 0.0000000211 |
| Norm max abs delta | 0.00000191 |

Worst capture:

```text
capture_014.npz
module=up_blocks.0.attentions.0.transformer_blocks.0.attn1
key_tokens=256
code_diff_count=5
code_diff_rate=0.00000763
output_max_abs=0.03381348
output_mean_abs=0.00000398
output_mse=0.0000000211
```

The isolation check on the worst capture:

| Variant | Output MSE vs split |
| --- | ---: |
| fused codes + fused norms | 0.0000000211 |
| fused codes + split norms | 0.0000000211 |
| split codes + fused norms | 0.0000000000177 |

## Interpretation

Fused encode does not have a broad layout or norm bug. The divergence is almost
entirely from extremely rare bucket-boundary flips introduced by doing
normalize+rotate+bucketize in one Triton kernel instead of materializing the
split path's normalized tensor first.

That means the previous fused-encode image result should be read as:

```text
fast path is numerically very close locally
but diffusion can amplify tiny boundary decisions over repeated denoising
```

This supports the current policy decision:

```text
split encode = balanced default
fused encode = opt-in speed mode with a parity QA gate
```

The next kernel-side quality recovery target is not norm dtype. It is exact
bucket-boundary compatibility, or a hybrid path that avoids those rare code
flips while preserving most of the fused encode speed.
