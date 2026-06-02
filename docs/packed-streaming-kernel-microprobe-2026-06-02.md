# Packed Streaming Kernel Microprobe

## Target

This slice probed the current winning 4070 self-attention path:

```text
code_format=packed_t
bits=7
qjl_bits=0
head_dim=64
packed_block_q=64
packed_block_k=32
score_dot_precision=tf32
value_dot_precision=tf32
rotation_dot_precision=ieee
```

The goal was to find a small Triton-kernel improvement without changing the packed representation or image policy.

## Probes

### Hoist Packed Bit Layout Math

The streaming kernel recomputed these values inside each key tile:

```python
bit_position = dim_offsets[:, None] * BITS
byte_index = bit_position // 8
bit_offset = bit_position % 8
needs_next_byte = (bit_offset + BITS) > 8
```

These depend only on the dimension offsets and bit depth, so they now live outside the key-tile loop.

Synthetic results:

| Run | Total | Encode | Attention | rel_rmse |
| --- | ---: | ---: | ---: | ---: |
| BK32 prior capture | 0.5083 ms | 0.1783 ms | 0.3145 ms | 0.024060 |
| bit-hoist | 0.5108 ms | 0.1788 ms | 0.3182 ms | 0.024060 |
| bit-hoist confirm | 0.4817 ms | 0.1696 ms | 0.3075 ms | 0.024060 |

Readout: neutral. Keep as a small cleanup, but do not count it as a real performance win.

### Collapse No-QJL Logit Scaling

Tried folding:

```text
scores *= key_norms / sqrt(dim)
logits = scores / sqrt(dim)
```

into a single no-QJL logit expression. This looked cheaper in source but made Triton generate a slower kernel.

Synthetic results:

| Run | Total | Encode | Attention | rel_rmse |
| --- | ---: | ---: | ---: | ---: |
| scale-collapse | 0.5934 ms | 0.2276 ms | 0.4313 ms | 0.024060 |
| scale-collapse confirm | 0.5577 ms | 0.2091 ms | 0.3618 ms | 0.024060 |

Readout: reject.

### 8-Warp Streaming Launch

Tried `num_warps=8` for the narrow `packed_t` K7/head64/no-QJL streaming path.

Synthetic result:

| Run | Total | Encode | Attention | rel_rmse |
| --- | ---: | ---: | ---: | ---: |
| 8-warps | 0.7631 ms | 0.1812 ms | 0.5901 ms | 0.024060 |

Readout: reject. The current 4-warp launch remains the better shape.

## Conclusion

The current `packed_t` streaming kernel is sensitive to seemingly harmless source changes. Source-level "less work" does not necessarily become faster Triton code.

No new speed claim from this slice. The only retained code change is the invariant bit-layout hoist, which is neutral in synthetic timing and keeps the hot loop simpler.

Next useful kernel slices:

1. Add a small regression benchmark guard for the 1024 K7/packed_t path.
2. Profile generated Triton IR/PTX/register pressure before attempting another hardcoded specialization.
3. Try a separate representation-level idea instead of further source reshuffling inside this kernel.
