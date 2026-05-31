from __future__ import annotations

from math import sqrt
from typing import Any, Literal

from shmoosh.packed_keys import PackedKeyBlock, encode_packed_keys
from shmoosh.packed_scores import (
    PackedScoreResources,
    packed_key_scores,
)


def packed_key_attention_output(
    query: Any,
    block: PackedKeyBlock,
    value: Any,
    *,
    resources: PackedScoreResources | None = None,
    backend: Literal["auto", "torch", "triton"] = "auto",
    output_dtype: Any | None = None,
) -> Any:
    """Compute attention output from packed keys and exact values.

    Inputs use Diffusers-style shape `(batch, heads, tokens, head_dim)`.
    Values are intentionally exact in this first production-path slice.
    """

    torch = _load_torch()
    _validate_query_value_block(query, value, block)
    target_dtype = query.dtype if output_dtype is None else output_dtype
    scores = packed_key_scores(
        query,
        block,
        resources=resources,
        backend=backend,
    )
    weights = torch.softmax(scores / sqrt(block.head_dim), dim=-1)
    output = torch.einsum(
        "bhqt,bhtd->bhqd",
        weights,
        value.to(device=query.device, dtype=torch.float32),
    )
    return output.to(dtype=target_dtype)


def encode_and_attention_output(
    query: Any,
    key: Any,
    value: Any,
    *,
    bits: int,
    qjl_bits: int,
    seed: int,
    backend: Literal["auto", "torch", "triton"] = "auto",
    codebook_samples: int = 80_000,
    lloyd_iters: int = 80,
    output_dtype: Any | None = None,
) -> Any:
    """Encode K into a packed block, then run packed-K exact-V attention."""

    block = encode_packed_keys(
        key,
        bits=bits,
        qjl_bits=qjl_bits,
        seed=seed,
        codebook_samples=codebook_samples,
        lloyd_iters=lloyd_iters,
    )
    return packed_key_attention_output(
        query,
        block,
        value,
        backend=backend,
        output_dtype=output_dtype,
    )


def _validate_query_value_block(
    query: Any,
    value: Any,
    block: PackedKeyBlock,
) -> None:
    if query.ndim != 4 or value.ndim != 4:
        raise ValueError("query/value must have shape (batch, heads, tokens, head_dim)")
    batch, heads, _query_tokens, head_dim = (int(size) for size in query.shape)
    value_batch, value_heads, value_tokens, value_dim = (
        int(size) for size in value.shape
    )
    key_batch, key_heads, key_tokens, key_dim = block.shape
    if (batch, heads, head_dim) != (key_batch, key_heads, key_dim):
        raise ValueError(
            "query and packed keys must share batch, heads, and head_dim"
        )
    if (value_batch, value_heads, value_tokens, value_dim) != (
        key_batch,
        key_heads,
        key_tokens,
        key_dim,
    ):
        raise ValueError("value must match packed key batch, heads, tokens, and dim")


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch is required for packed attention; install with "
            "`uv sync --extra dev --extra diffusers`"
        ) from exc
    return torch
