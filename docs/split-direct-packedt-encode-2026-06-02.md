# Split Direct Packed-T Encode

This slice optimizes the balanced split encode path without changing its
numerical policy.

Previously, the K7/head64/no-QJL split path did:

```text
PyTorch normalize
Triton rotate + bucketize + pack as (tokens, code_bytes)
PyTorch transpose to packed_t (code_bytes, tokens)
```

The rotate/bucketize/pack Triton kernel now writes `packed_t` directly when
`code_format="packed_t"`. The path still uses the same split PyTorch
normalization, so it preserves the split path's bucket decisions.

## Synthetic

Shape:

```text
batch=1
heads=20
query_tokens=1024
key_tokens=1024
head_dim=64
policy=K7/no-QJL packed_t
precision=score+value tf32, rotation ieee
```

Output:

```text
captures/self-attention-variant-bench-k7-packedt-score-value-tf32-split-directt-20260602
```

| Backend | Encode | Attention | Total | Relative RMSE |
| --- | ---: | ---: | ---: | ---: |
| previous split | 0.2532 ms | 0.4258 ms | 0.5838 ms | 0.024059 |
| split direct packed_t | 0.1834 ms | 0.4269 ms | 0.5871 ms | 0.024059 |
| fused norm encode | 0.1149 ms | 0.3773 ms | 0.4473 ms | 0.024060 |

The synthetic encode phase improves by about `27.6%` versus the previous
sequential split run while preserving the split error metrics.

## Capture Parity

The parity gate was rerun after the direct packed_t change:

```text
captures/packed-encode-parity-self-attn-k7-packedt-direct-split-20260602.json
```

It matched the previous split-vs-fused parity profile:

| Metric | Value |
| --- | ---: |
| Captures | 15 |
| Total code index differences | 30 |
| Worst code diff rate | 0.00000763 |
| Worst output MSE | 0.0000000211 |

That confirms the direct packed_t write did not change split-path numerical
behavior.

## Image Compare

Output:

```text
captures/image-policy-compare-juggernaut-up0-self-attn1-k7-noqjl-1024-score-value-directt-encode-20260602
```

Same-process 1024 comparison:

| Candidate | Min PSNR | Mean PSNR | Mean speedup | Packed encode | Scheduled quantized |
| --- | ---: | ---: | ---: | ---: | ---: |
| split direct packed_t | 52.0788 dB | 54.1603 dB | 1.049x | 0.7281 ms | 2.8211 ms |
| fused norm encode | 51.8747 dB | 54.1765 dB | 1.079x | 0.5068 ms | 2.4800 ms |

For the split path, the previous same-process fused-encode comparison measured:

```text
packed_encode:        0.9956 ms
scheduled_quantized:  3.3231 ms
```

So the direct packed_t write recovers a large chunk of the fused encode speed
while retaining the balanced split policy's hard-case quality:

```text
reading-nook: 52.0788 dB
```

## Decision

Keep this as the default split behavior for `packed_t`.

Fused norm encode remains useful as an opt-in speed tradeoff, but direct packed_t
write is the quality-safe improvement: it removes the transpose tax without
introducing the rare bucket-boundary flips that explain fused's hard-case PSNR
drop.
