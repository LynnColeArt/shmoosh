# Packed Transpose Layout

This slice tests a compact layout change instead of a new attention algorithm.

The normal packed code tensor is shaped like:

```text
(batch, heads, key_tokens, code_bytes)
```

For a fixed packed byte lane, Triton reads across key tokens with a stride of
`code_bytes`. At K7/head_dim=64, that stride is `56` bytes.

The new opt-in `code_format="packed_t"` stores the same bit-packed payload as:

```text
(batch, heads, code_bytes, key_tokens)
```

That lets the attention kernel read one packed byte lane contiguously across a
key tile. Storage size and quantization math are unchanged.

## Implementation

- `PackedKeyBlock` now accepts `code_format="packed_t"`.
- Fused K7 encode can still produce the packed code bytes, then transpose the
  compact code tensor.
- Torch decode/score fallbacks transpose back before unpacking.
- Triton packed-score and packed-attention kernels accept a `TRANSPOSED_CODES`
  constexpr and switch address math for packed-code loads.
- `packed_t` is exposed through the synthetic bench and image policy CLIs.
- Plain `packed` remains the global default; `packed_t` is preferred only by the
  validated K7/no-QJL 1024 self-attention policy.

## Synthetic 1024 Bench

All runs used:

```text
batch=1
heads=20
query_tokens=1024
key_tokens=1024
head_dim=64
dtype=fp16
bits=7
qjl_bits=0
backend=auto
```

First comparison:

| Format | Encode ms | Attention ms | Total ms | Output |
| --- | ---: | ---: | ---: | --- |
| `packed` | 0.2181 | 0.6829 | 0.7634 | `captures/self-attention-variant-bench-1024-packed-k7-noqjl-layout-compare-packed` |
| `packed_t` | 0.1824 | 0.5988 | 0.8214 | `captures/self-attention-variant-bench-1024-packedt-k7-noqjl-layout-compare` |

Confirmation:

| Format | Encode ms | Attention ms | Total ms | Output |
| --- | ---: | ---: | ---: | --- |
| `packed` | 0.1693 | 0.5775 | 0.7607 | `captures/self-attention-variant-bench-1024-packed-k7-noqjl-layout-confirm-packed` |
| `packed_t` | 0.1837 | 0.5561 | 0.7389 | `captures/self-attention-variant-bench-1024-packedt-k7-noqjl-layout-confirm` |

The synthetic result is still noisy, but the attention phase favors
`packed_t`.

## Image Suite

Output:

```text
captures/image-policy-suite-juggernaut-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-1024-packedt
```

| Case | Baseline s | Shmoosh s | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| `reading-nook-seed1-1024` | 11.6581 | 9.8974 | 1.178x | 52.07 dB |
| `maple-leaf-seed2-1024` | 8.9354 | 8.9190 | 1.002x | 52.12 dB |
| `misty-lake-seed3-1024` | 8.8122 | 8.8353 | 0.997x | 58.61 dB |

Aggregate:

- minimum PSNR: `52.07 dB`
- mean PSNR: `54.27 dB`
- mean speedup: `1.063x`
- mean scheduled quantized call: `2.9597ms`
- mean packed encode: `0.6482ms`
- mean packed attention: `1.2806ms`
- mean rotate/bucketize: `0.3990ms`
- mean pack-codes: `0.0037ms`

Compared with the previous `BQ64/BK16` packed layout suite, `packed_t` moved:

```text
packed_attention:     1.3521ms -> 1.2806ms
packed_encode:        0.6860ms -> 0.6482ms
scheduled_quantized:  3.1222ms -> 2.9597ms
```

Whole-image speed remains prompt/noise dominated, but the processor phase moved
in the intended direction and quality stayed identical.

## Readout

This is the first layout change that makes the packed attention kernel a little
cheaper without sacrificing compactness.

Keep `packed` as the general default because it is simpler and broadly tested.
Use `packed_t` for the preferred K7/no-QJL/head_dim=64 1024 self-attention
policy, where the stride-vs-contiguous key-byte read matters.
