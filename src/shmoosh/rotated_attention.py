from __future__ import annotations

from math import sqrt
from typing import Any, Literal

from shmoosh.packed_scores import PackedScoreResources, triton, tl
from shmoosh.rotated_keys import RotatedKeyBlock

_ROTATED_TRITON_QUERY_TILE = 32
_ROTATED_TRITON_KEY_TILE = 32


def rotated_key_attention_output(
    query: Any,
    block: RotatedKeyBlock,
    value: Any,
    *,
    resources: PackedScoreResources,
    backend: Literal["auto", "torch", "triton"] = "auto",
    output_dtype: Any | None = None,
) -> Any:
    """Compute exact-V attention from direct rotated keys."""

    _validate_query_value_block(query, value, block)
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be one of: auto, torch, triton")

    target_dtype = query.dtype if output_dtype is None else output_dtype
    if backend == "torch":
        return torch_rotated_key_attention_output(
            query,
            block,
            value,
            resources=resources,
            output_dtype=target_dtype,
        )
    if _can_launch_triton_rotated_attention(query, block, value):
        return triton_rotated_key_attention_output(
            query,
            block,
            value,
            resources=resources,
            output_dtype=target_dtype,
        )
    if backend == "triton" and not getattr(query, "is_cuda", False):
        raise ValueError("triton rotated-key attention requires a CUDA query tensor")
    return torch_rotated_key_attention_output(
        query,
        block,
        value,
        resources=resources,
        output_dtype=target_dtype,
    )


def torch_rotated_key_attention_output(
    query: Any,
    block: RotatedKeyBlock,
    value: Any,
    *,
    resources: PackedScoreResources,
    output_dtype: Any | None = None,
) -> Any:
    """Materialized-score direct rotated-K attention fallback."""

    torch = _load_torch()
    _validate_query_value_block(query, value, block)
    target_dtype = query.dtype if output_dtype is None else output_dtype
    rotation = resources.rotation.to(device=query.device, dtype=torch.float32)
    query_f = query.to(dtype=torch.float32)
    q_rot = torch.matmul(query_f, rotation.T)
    scores = torch.einsum(
        "bhqd,bhtd,bht->bhqt",
        q_rot,
        block.rotated_keys.to(device=query.device, dtype=torch.float32),
        block.norms.to(device=query.device, dtype=torch.float32),
    )
    weights = torch.softmax(scores / sqrt(block.head_dim), dim=-1)
    output = torch.einsum(
        "bhqt,bhtd->bhqd",
        weights,
        value.to(device=query.device, dtype=torch.float32),
    )
    return output.to(dtype=target_dtype)


def triton_rotated_key_attention_output(
    query: Any,
    block: RotatedKeyBlock,
    value: Any,
    *,
    resources: PackedScoreResources,
    output_dtype: Any | None = None,
    block_q: int = _ROTATED_TRITON_QUERY_TILE,
    block_k: int = _ROTATED_TRITON_KEY_TILE,
) -> Any:
    """Streaming Triton attention for direct rotated K."""

    torch = _load_torch()
    if triton is None or _rotated_key_attention_streaming_kernel is None:
        raise RuntimeError("triton is required for fused rotated-key attention")
    if not _can_launch_triton_rotated_attention(
        query,
        block,
        value,
        block_q=block_q,
        block_k=block_k,
    ):
        raise ValueError(
            "fused Triton rotated-key attention requires CUDA tensors, "
            "power-of-two dimensions, and valid tile sizes"
        )

    _validate_query_value_block(query, value, block)
    target_dtype = query.dtype if output_dtype is None else output_dtype
    batch, heads, q_tokens, head_dim = (int(size) for size in query.shape)
    key_tokens = int(block.shape[2])
    head_like = batch * heads
    query_input = query.contiguous()
    rotation = resources.rotation.to(device=query.device, dtype=torch.float32)
    rotated_keys = block.rotated_keys.to(device=query.device).contiguous().reshape(
        head_like,
        key_tokens,
        head_dim,
    )
    norms = block.norms.to(device=query.device, dtype=torch.float32).contiguous().reshape(
        head_like,
        key_tokens,
    )
    value_input = value.to(device=query.device).contiguous().reshape(
        head_like,
        key_tokens,
        head_dim,
    )
    output = torch.empty(
        (head_like, q_tokens, head_dim),
        device=query.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(q_tokens, block_q), head_like)
    _rotated_key_attention_streaming_kernel[grid](
        query_input,
        rotation,
        rotated_keys,
        norms,
        value_input,
        output,
        q_tokens,
        key_tokens,
        HEAD_DIM=head_dim,
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
        ATTENTION_SCALE=1.0 / sqrt(head_dim),
        num_warps=4,
    )
    return output.reshape(batch, heads, q_tokens, head_dim).to(dtype=target_dtype)


def _validate_query_value_block(
    query: Any,
    value: Any,
    block: RotatedKeyBlock,
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
            "query and rotated keys must share batch, heads, and head_dim"
        )
    if (value_batch, value_heads, value_tokens, value_dim) != (
        key_batch,
        key_heads,
        key_tokens,
        key_dim,
    ):
        raise ValueError("value must match rotated key batch, heads, tokens, and dim")


def _can_launch_triton_rotated_attention(
    query: Any,
    block: RotatedKeyBlock,
    value: Any,
    *,
    block_q: int = _ROTATED_TRITON_QUERY_TILE,
    block_k: int = _ROTATED_TRITON_KEY_TILE,
) -> bool:
    return (
        triton is not None
        and _rotated_key_attention_streaming_kernel is not None
        and getattr(query, "is_cuda", False)
        and getattr(value, "is_cuda", False)
        and getattr(block.rotated_keys, "is_cuda", False)
        and block.rotated_keys.device == query.device
        and value.device == query.device
        and block.head_dim >= 16
        and _is_power_of_two(block.head_dim)
        and block_q >= 16
        and _is_power_of_two(block_q)
        and block_k >= 16
        and _is_power_of_two(block_k)
    )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch is required for rotated attention; install with "
            "`uv sync --extra dev --extra diffusers`"
        ) from exc
    return torch


if triton is not None and tl is not None:

    @triton.jit(do_not_specialize=["q_tokens", "key_tokens"])
    def _rotated_key_attention_streaming_kernel(
        query_ptr,
        rotation_ptr,
        rotated_key_ptr,
        norms_ptr,
        value_ptr,
        out_ptr,
        q_tokens,
        key_tokens,
        HEAD_DIM: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ATTENTION_SCALE: tl.constexpr,
    ):
        q_offsets = tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
        head_like = tl.program_id(1)
        k_offsets = tl.arange(0, BLOCK_K)
        dim_offsets = tl.arange(0, HEAD_DIM)
        q_mask = q_offsets < q_tokens

        query_values = tl.load(
            query_ptr
            + head_like * q_tokens * HEAD_DIM
            + q_offsets[:, None] * HEAD_DIM
            + dim_offsets[None, :],
            mask=q_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        rotation_t = tl.load(
            rotation_ptr
            + dim_offsets[None, :] * HEAD_DIM
            + dim_offsets[:, None]
        )
        q_rot = tl.dot(query_values, rotation_t, input_precision="ieee")

        m_i = tl.full((BLOCK_Q, 1), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_Q, 1), dtype=tl.float32)
        acc = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)

        for key_start in tl.range(0, key_tokens, BLOCK_K):
            tile_k_offsets = key_start + k_offsets
            tile_k_mask = tile_k_offsets < key_tokens
            rotated_keys = tl.load(
                rotated_key_ptr
                + head_like * key_tokens * HEAD_DIM
                + tile_k_offsets[None, :] * HEAD_DIM
                + dim_offsets[:, None],
                mask=tile_k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.dot(q_rot, rotated_keys, input_precision="ieee")
            key_norms = tl.load(
                norms_ptr + head_like * key_tokens + tile_k_offsets,
                mask=tile_k_mask,
                other=0.0,
            )
            logits = scores * (key_norms * ATTENTION_SCALE)[None, :]
            logits = tl.where(
                q_mask[:, None] & tile_k_mask[None, :],
                logits,
                -float("inf"),
            )

            m_tile = tl.max(logits, axis=1, keep_dims=True)
            m_new = tl.maximum(m_i, m_tile)
            m_new = tl.where(q_mask[:, None], m_new, 0.0)
            alpha = tl.exp(m_i - m_new)
            weights = tl.exp(logits - m_new)
            l_i = l_i * alpha + tl.sum(weights, axis=1, keep_dims=True)

            values = tl.load(
                value_ptr
                + head_like * key_tokens * HEAD_DIM
                + tile_k_offsets[:, None] * HEAD_DIM
                + dim_offsets[None, :],
                mask=tile_k_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            acc = acc * alpha + tl.dot(weights, values, input_precision="ieee")
            m_i = m_new

        output = acc / tl.where(l_i > 0.0, l_i, 1.0)
        tl.store(
            out_ptr
            + head_like * q_tokens * HEAD_DIM
            + q_offsets[:, None] * HEAD_DIM
            + dim_offsets[None, :],
            output,
            mask=q_mask[:, None],
        )

else:  # pragma: no cover
    _rotated_key_attention_streaming_kernel = None
