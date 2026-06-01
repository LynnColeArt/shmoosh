# Self-Attention Runtime V2: Byte Codes

This slice tested a second runtime code layout for packed-K exact-V
self-attention. The existing format stores K7/no-QJL codes bit-packed:

```text
56 code bytes + 4 norm bytes = 60 bytes per 64-dim key vector
```

The byte-code format stores one code byte per head dimension:

```text
64 code bytes + 4 norm bytes = 68 bytes per 64-dim key vector
```

The hypothesis was that removing bit packing from encode and bit extraction from
the fused attention kernel might outweigh the 13% larger K payload for the
late-step 1024-token K7/no-QJL self-attention policy.

## Implementation

Code changes:

- `PackedKeyBlock.code_format` now supports `packed` and `byte`.
- `encode_packed_keys(..., code_format="byte")` stores raw uint8 code indices.
- Torch score and attention paths decode byte-code blocks directly.
- The fused Triton attention kernels have a byte-code branch that loads code
  indices directly instead of extracting bit fields.
- Diffusers processors and image policy JSON can select `code_format`.
- `shmoosh-self-attention-variant-bench` records and compares code formats.

The existing bit-packed format remains the default.

## Validation

Focused tests:

```text
uv run python -m pytest \
  tests/test_packed_keys.py \
  tests/test_packed_scores.py \
  tests/test_packed_attention.py \
  tests/test_diffusers_processor.py \
  tests/test_image_policy.py \
  tests/test_self_attention_variant_bench.py
```

Result:

```text
54 passed
```

Compile check:

```text
uv run python -m compileall -q src experiments tests
```

Result: clean.

## Synthetic 1024 Bench

Byte-code run:

```text
captures/self-attention-variant-bench-1024-byte-k7-noqjl-100iters
```

Packed comparison run:

```text
captures/self-attention-variant-bench-1024-packed-k7-noqjl-byte-slice-compare-100iters
```

Shape:

```text
batch=1 heads=20 query_tokens=1024 key_tokens=1024 dim=64 fp16
```

| Format | Total ms | Encode ms | Attention ms | Bytes/vector | Relative RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| byte | 1.0223 | 0.1552 | 0.8826 | 68 | 0.023998 |
| packed | 1.1133 | 0.4330 | 0.7665 | 60 | 0.023998 |

Synthetic readout:

- byte-code cuts encode time by about 64%;
- packed remains faster inside attention;
- byte-code wins total synthetic time by about 8% in this run;
- quality is identical because both formats encode the same code indices.

## Image Smoke

Policy:

```text
configs/underpaint-juggernaut-sdxl-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-byte-policy.json
```

Trace output:

```text
captures/image-ab-juggernaut-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-byte-1024-trace-reading-nook
```

| Metric | Value |
| --- | ---: |
| PSNR | 51.87 dB |
| MSE | 0.00000651 |
| baseline | 14.2857s |
| Shmoosh | 12.4606s |
| speedup | 1.146x |
| packed encode | 0.0170s |
| packed attention | 0.0374s |
| scheduled quantized | 0.0942s |

The quality result matches the fast bit-packed K7/no-QJL trace. The runtime
shape does not: byte-code makes `encode_pack_codes` tiny, but the larger code
payload increases fused attention time.

## Three-Case Suite

Output:

```text
captures/image-policy-suite-juggernaut-up0-self-attn1-firstblocks-gated70pct-k7-noqjl-byte-1024
```

| Case | Baseline s | Shmoosh s | Speedup | PSNR |
| --- | ---: | ---: | ---: | ---: |
| `reading-nook-seed1-1024` | 14.4324 | 12.8105 | 1.127x | 51.87 dB |
| `maple-leaf-seed2-1024` | 11.4405 | 11.2621 | 1.016x | 52.19 dB |
| `misty-lake-seed3-1024` | 11.3259 | 11.1200 | 1.019x | 57.82 dB |

Aggregate:

- min PSNR: `51.87 dB`
- mean PSNR: `53.96 dB`
- max MSE: `0.00000651`
- mean speedup: `1.057x`

Compared with the prior fast bit-packed suite:

| Format | Min PSNR | Mean PSNR | Mean speedup |
| --- | ---: | ---: | ---: |
| byte-code | 51.87 dB | 53.96 dB | 1.057x |
| fast bit-packed | 51.87 dB | 53.96 dB | 1.084x |

## Readout

Byte-code is correct and worth keeping as an opt-in runtime format, but it is
not the preferred 1024 self-attention format. For large spatial key sets, the
saved encode work is smaller than the fused-attention bandwidth penalty at the
image level.

The result sharpens the next kernel direction:

1. Keep bit-packed K for 1024 self-attention.
2. Keep byte-code available for shorter key-count experiments, especially
   prompt-layer text keys where launch overhead and bit extraction may dominate.
3. The next serious self-attention speed lever is not "unpack the codes"; it is
   fusing encode work closer to attention or reducing rotate/bucketize overhead.
