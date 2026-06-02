from __future__ import annotations

from math import pi, sqrt
from typing import Any, Literal

from shmoosh.packed_keys import PackedKeyBlock, encode_packed_keys
from shmoosh.packed_scores import (
    PackedScoreResources,
    build_score_resources,
    packed_key_scores,
    triton,
    tl,
)

DotPrecision = Literal["ieee", "tf32", "tf32x3"]

_FUSED_TRITON_SINGLE_KEY_TILE = 128
_FUSED_TRITON_QUERY_TILE = 16
_FUSED_TRITON_STREAMING_QUERY_TILE = 32
_FUSED_TRITON_STREAMING_K7_HEAD64_QUERY_TILE = 64
_FUSED_TRITON_STREAMING_QJL_KEY_TILE = 16
_FUSED_TRITON_STREAMING_NO_QJL_KEY_TILE = 32
_FUSED_TRITON_STREAMING_K7_HEAD64_PACKED_KEY_TILE = 16
_FUSED_TRITON_STREAMING_K7_HEAD64_PACKED_T_KEY_TILE = 32
_FUSED_TRITON_KEY_TILE = _FUSED_TRITON_SINGLE_KEY_TILE


def packed_key_attention_output(
    query: Any,
    block: PackedKeyBlock,
    value: Any,
    *,
    resources: PackedScoreResources | None = None,
    backend: Literal["auto", "torch", "triton"] = "auto",
    output_dtype: Any | None = None,
    dot_precision: DotPrecision = "ieee",
    rotation_dot_precision: DotPrecision | None = None,
    score_dot_precision: DotPrecision | None = None,
    value_dot_precision: DotPrecision | None = None,
    qjl_dot_precision: DotPrecision | None = None,
    block_q: int | None = None,
    block_k: int | None = None,
) -> Any:
    """Compute attention output from packed keys and exact values.

    Inputs use Diffusers-style shape `(batch, heads, tokens, head_dim)`.
    Values are intentionally exact in this first production-path slice.
    """

    _validate_query_value_block(query, value, block)
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be one of: auto, torch, triton")
    dot_precisions = _resolved_dot_precisions(
        dot_precision=dot_precision,
        rotation_dot_precision=rotation_dot_precision,
        score_dot_precision=score_dot_precision,
        value_dot_precision=value_dot_precision,
        qjl_dot_precision=qjl_dot_precision,
    )
    triton_block_q = _FUSED_TRITON_QUERY_TILE if block_q is None else block_q
    triton_block_k = _FUSED_TRITON_KEY_TILE if block_k is None else block_k

    target_dtype = query.dtype if output_dtype is None else output_dtype
    if backend == "torch":
        return torch_packed_key_attention_output(
            query,
            block,
            value,
            resources=resources,
            output_dtype=target_dtype,
        )
    if _can_use_fused_triton_attention(
        query,
        block,
        value,
        block_q=triton_block_q,
        block_k=triton_block_k,
    ):
        return triton_packed_key_attention_output(
            query,
            block,
            value,
            resources=resources,
            output_dtype=target_dtype,
            block_q=triton_block_q,
            block_k=triton_block_k,
            dot_precision=dot_precision,
            **dot_precisions,
        )
    if backend == "triton" and not getattr(query, "is_cuda", False):
        raise ValueError("triton packed-key attention requires a CUDA query tensor")
    return torch_packed_key_attention_output(
        query,
        block,
        value,
        resources=resources,
        backend=backend,
        output_dtype=target_dtype,
    )


def torch_packed_key_attention_output(
    query: Any,
    block: PackedKeyBlock,
    value: Any,
    *,
    resources: PackedScoreResources | None = None,
    backend: Literal["auto", "torch", "triton"] = "auto",
    output_dtype: Any | None = None,
) -> Any:
    """Materialized-score packed-key attention fallback."""

    torch = _load_torch()
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


def triton_packed_key_attention_output(
    query: Any,
    block: PackedKeyBlock,
    value: Any,
    *,
    resources: PackedScoreResources | None = None,
    output_dtype: Any | None = None,
    block_q: int = _FUSED_TRITON_QUERY_TILE,
    block_k: int = _FUSED_TRITON_KEY_TILE,
    dot_precision: DotPrecision = "ieee",
    rotation_dot_precision: DotPrecision | None = None,
    score_dot_precision: DotPrecision | None = None,
    value_dot_precision: DotPrecision | None = None,
    qjl_dot_precision: DotPrecision | None = None,
) -> Any:
    """Fused Triton packed-K attention.

    This path never materializes the full `(batch, heads, query_tokens,
    key_tokens)` score tensor. It uses a fast single-tile kernel for text-key
    attention and a streaming softmax kernel for larger key sets.
    """

    torch = _load_torch()
    dot_precisions = _resolved_dot_precisions(
        dot_precision=dot_precision,
        rotation_dot_precision=rotation_dot_precision,
        score_dot_precision=score_dot_precision,
        value_dot_precision=value_dot_precision,
        qjl_dot_precision=qjl_dot_precision,
    )
    if (
        triton is None
        or _packed_key_attention_output_kernel is None
        or _packed_key_attention_streaming_kernel is None
    ):
        raise RuntimeError("triton is required for fused packed-key attention")
    if not _can_launch_fused_triton_attention(query, block, value, block_k=block_k):
        raise ValueError(
            "fused Triton packed-key attention requires CUDA tensors, "
            "fused-compatible dimensions, and a valid tile size"
        )

    _validate_query_value_block(query, value, block)
    resources = _resources_for(query, block, resources)
    target_dtype = query.dtype if output_dtype is None else output_dtype

    batch, heads, q_tokens, head_dim = (int(size) for size in query.shape)
    key_tokens = int(block.shape[2])
    head_like = batch * heads
    query_input = query.contiguous()
    rotation = resources.rotation.to(device=query.device, dtype=torch.float32)
    codebook = resources.codebook.to(device=query.device, dtype=torch.float32)

    effective_qjl_bits = _effective_qjl_bits(block, resources)
    if effective_qjl_bits:
        qjl_matrix = resources.qjl_matrix.to(device=query.device, dtype=torch.float32)
        residual_signs = block.residual_signs.contiguous().reshape(
            head_like,
            key_tokens,
            block.qjl_sign_bytes_per_vector,
        )
        residual_norms = block.residual_norms.to(
            dtype=torch.float32
        ).contiguous().reshape(head_like, key_tokens)
    else:
        qjl_matrix = torch.empty((1,), device=query.device, dtype=torch.float32)
        residual_signs = torch.empty((1,), device=query.device, dtype=torch.uint8)
        residual_norms = torch.empty((1,), device=query.device, dtype=torch.float32)

    if block.code_format == "packed_t":
        codes = block.codes.contiguous().reshape(
            head_like,
            block.code_bytes_per_vector,
            key_tokens,
        )
    else:
        codes = block.codes.contiguous().reshape(
            head_like,
            key_tokens,
            block.code_bytes_per_vector,
        )
    norms = block.norms.contiguous().reshape(
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
    query_tile = block_q
    key_tile = block_k
    kernel = _packed_key_attention_output_kernel
    if key_tokens > block_k:
        query_tile = _select_streaming_query_tile(
            block_q,
            block,
            effective_qjl_bits,
        )
        key_tile = _select_streaming_key_tile(
            block_k,
            effective_qjl_bits,
            block=block,
        )
        kernel = _packed_key_attention_streaming_kernel
    grid = (triton.cdiv(q_tokens, query_tile), head_like)
    kernel[grid](
        query_input,
        rotation,
        qjl_matrix,
        codes,
        norms,
        residual_signs,
        residual_norms,
        codebook,
        value_input,
        output,
        q_tokens,
        key_tokens,
        HEAD_DIM=head_dim,
        BITS=block.bits,
        BYTE_CODES=block.code_format == "byte",
        TRANSPOSED_CODES=block.code_format == "packed_t",
        QJL_BITS=effective_qjl_bits,
        CODE_BYTES=block.code_bytes_per_vector,
        SIGN_BYTES=block.qjl_sign_bytes_per_vector if effective_qjl_bits else 1,
        BLOCK_Q=query_tile,
        BLOCK_K=key_tile,
        INV_SQRT_D=1.0 / sqrt(head_dim),
        ATTENTION_SCALE=1.0 / sqrt(head_dim),
        QJL_SCALE=(
            0.0
            if effective_qjl_bits == 0
            else sqrt(pi / 2.0) / float(effective_qjl_bits)
        ),
        ROTATION_DOT_INPUT_PRECISION=dot_precisions["rotation_dot_precision"],
        SCORE_DOT_INPUT_PRECISION=dot_precisions["score_dot_precision"],
        VALUE_DOT_INPUT_PRECISION=dot_precisions["value_dot_precision"],
        QJL_DOT_INPUT_PRECISION=dot_precisions["qjl_dot_precision"],
        num_warps=4,
    )
    return output.reshape(batch, heads, q_tokens, head_dim).to(dtype=target_dtype)


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
    codec: Any | None = None,
    resources: PackedScoreResources | None = None,
    key_encode_backend: Literal["split", "fused", "auto"] = "split",
    code_format: Literal["packed", "byte", "packed_t"] = "packed",
    norm_dtype: Literal["fp32", "fp16"] = "fp32",
    dot_precision: DotPrecision = "ieee",
    rotation_dot_precision: DotPrecision | None = None,
    score_dot_precision: DotPrecision | None = None,
    value_dot_precision: DotPrecision | None = None,
    qjl_dot_precision: DotPrecision | None = None,
) -> Any:
    """Encode K into a packed block, then run packed-K exact-V attention."""

    block = encode_packed_keys(
        key,
        bits=bits,
        qjl_bits=qjl_bits,
        seed=seed,
        codebook_samples=codebook_samples,
        lloyd_iters=lloyd_iters,
        codec=codec,
        resources=resources,
        key_encode_backend=key_encode_backend,
        code_format=code_format,
        norm_dtype=norm_dtype,
    )
    return packed_key_attention_output(
        query,
        block,
        value,
        resources=resources,
        backend=backend,
        output_dtype=output_dtype,
        dot_precision=dot_precision,
        rotation_dot_precision=rotation_dot_precision,
        score_dot_precision=score_dot_precision,
        value_dot_precision=value_dot_precision,
        qjl_dot_precision=qjl_dot_precision,
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


def _resources_for(
    query: Any,
    block: PackedKeyBlock,
    resources: PackedScoreResources | None,
) -> PackedScoreResources:
    if resources is not None:
        return resources
    return build_score_resources(block, device=query.device)


def _effective_qjl_bits(block: PackedKeyBlock, resources: PackedScoreResources) -> int:
    if (
        block.qjl_bits > 0
        and block.residual_signs is not None
        and block.residual_norms is not None
        and resources.qjl_matrix is not None
    ):
        return block.qjl_bits
    return 0


def _can_use_fused_triton_attention(
    query: Any,
    block: PackedKeyBlock,
    value: Any,
    *,
    block_q: int = _FUSED_TRITON_QUERY_TILE,
    block_k: int = _FUSED_TRITON_KEY_TILE,
) -> bool:
    return _can_launch_fused_triton_attention(
        query,
        block,
        value,
        block_q=block_q,
        block_k=block_k,
    )


def _can_launch_fused_triton_attention(
    query: Any,
    block: PackedKeyBlock,
    value: Any,
    *,
    block_q: int = _FUSED_TRITON_QUERY_TILE,
    block_k: int = _FUSED_TRITON_KEY_TILE,
) -> bool:
    return (
        triton is not None
        and _packed_key_attention_output_kernel is not None
        and _packed_key_attention_streaming_kernel is not None
        and getattr(query, "is_cuda", False)
        and getattr(value, "is_cuda", False)
        and getattr(block.codes, "is_cuda", False)
        and block.codes.device == query.device
        and value.device == query.device
        and block.head_dim >= 16
        and _is_power_of_two(block.head_dim)
        and _supports_fused_qjl_width(block.qjl_bits)
        and block_q >= 16
        and _is_power_of_two(block_q)
        and block_k >= 16
        and _is_power_of_two(block_k)
    )


def _supports_fused_qjl_width(qjl_bits: int) -> bool:
    return qjl_bits == 0 or _is_power_of_two(qjl_bits)


def _select_streaming_query_tile(
    block_q: int,
    block: PackedKeyBlock,
    effective_qjl_bits: int,
) -> int:
    if block_q != _FUSED_TRITON_QUERY_TILE:
        return block_q
    if _uses_k7_head64_compact_tile(block, effective_qjl_bits):
        return _FUSED_TRITON_STREAMING_K7_HEAD64_QUERY_TILE
    return _FUSED_TRITON_STREAMING_QUERY_TILE


def _select_streaming_key_tile(
    block_k: int,
    effective_qjl_bits: int,
    *,
    block: PackedKeyBlock | None = None,
) -> int:
    if block_k != _FUSED_TRITON_KEY_TILE:
        return block_k
    if block is not None and _uses_k7_head64_compact_tile(block, effective_qjl_bits):
        if block.code_format == "packed_t":
            return _FUSED_TRITON_STREAMING_K7_HEAD64_PACKED_T_KEY_TILE
        return _FUSED_TRITON_STREAMING_K7_HEAD64_PACKED_KEY_TILE
    if effective_qjl_bits:
        return _FUSED_TRITON_STREAMING_QJL_KEY_TILE
    return _FUSED_TRITON_STREAMING_NO_QJL_KEY_TILE


def _uses_k7_head64_compact_tile(
    block: PackedKeyBlock,
    effective_qjl_bits: int,
) -> bool:
    return (
        effective_qjl_bits == 0
        and block.code_format in {"packed", "packed_t"}
        and block.bits == 7
        and block.head_dim == 64
    )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _resolved_dot_precisions(
    *,
    dot_precision: DotPrecision,
    rotation_dot_precision: DotPrecision | None,
    score_dot_precision: DotPrecision | None,
    value_dot_precision: DotPrecision | None,
    qjl_dot_precision: DotPrecision | None,
) -> dict[str, DotPrecision]:
    _validate_dot_precision("dot_precision", dot_precision)
    for name, value in (
        ("rotation_dot_precision", rotation_dot_precision),
        ("score_dot_precision", score_dot_precision),
        ("value_dot_precision", value_dot_precision),
        ("qjl_dot_precision", qjl_dot_precision),
    ):
        if value is not None:
            _validate_dot_precision(name, value)
    return {
        "rotation_dot_precision": rotation_dot_precision or dot_precision,
        "score_dot_precision": score_dot_precision or dot_precision,
        "value_dot_precision": value_dot_precision or dot_precision,
        "qjl_dot_precision": qjl_dot_precision or score_dot_precision or dot_precision,
    }


def _validate_dot_precision(name: str, value: str) -> None:
    if value not in {"ieee", "tf32", "tf32x3"}:
        raise ValueError(f"{name} must be one of: ieee, tf32, tf32x3")


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch is required for packed attention; install with "
            "`uv sync --extra dev --extra diffusers`"
        ) from exc
    return torch


if triton is not None and tl is not None:

    @triton.jit(do_not_specialize=["q_tokens", "key_tokens"])
    def _packed_key_attention_output_kernel(
        query_ptr,
        rotation_ptr,
        qjl_matrix_ptr,
        codes_ptr,
        norms_ptr,
        residual_signs_ptr,
        residual_norms_ptr,
        codebook_ptr,
        value_ptr,
        out_ptr,
        q_tokens,
        key_tokens,
        HEAD_DIM: tl.constexpr,
        BITS: tl.constexpr,
        BYTE_CODES: tl.constexpr,
        TRANSPOSED_CODES: tl.constexpr,
        QJL_BITS: tl.constexpr,
        CODE_BYTES: tl.constexpr,
        SIGN_BYTES: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        INV_SQRT_D: tl.constexpr,
        ATTENTION_SCALE: tl.constexpr,
        QJL_SCALE: tl.constexpr,
        ROTATION_DOT_INPUT_PRECISION: tl.constexpr,
        SCORE_DOT_INPUT_PRECISION: tl.constexpr,
        VALUE_DOT_INPUT_PRECISION: tl.constexpr,
        QJL_DOT_INPUT_PRECISION: tl.constexpr,
    ):
        q_offsets = tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
        head_like = tl.program_id(1)
        k_offsets = tl.arange(0, BLOCK_K)
        dim_offsets = tl.arange(0, HEAD_DIM)
        q_mask = q_offsets < q_tokens
        k_mask = k_offsets < key_tokens
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
        q_rot = tl.dot(
            query_values,
            rotation_t,
            input_precision=ROTATION_DOT_INPUT_PRECISION,
        )

        if BYTE_CODES:
            code = tl.load(
                codes_ptr
                + head_like * key_tokens * CODE_BYTES
                + k_offsets[None, :] * CODE_BYTES
                + dim_offsets[:, None],
                mask=k_mask[None, :],
                other=0,
            ).to(tl.uint32)
        else:
            bit_position = dim_offsets[:, None] * BITS
            byte_index = bit_position // 8
            bit_offset = bit_position % 8
            needs_next_byte = (bit_offset + BITS) > 8
            if TRANSPOSED_CODES:
                code_byte = tl.load(
                    codes_ptr
                    + head_like * key_tokens * CODE_BYTES
                    + byte_index * key_tokens
                    + k_offsets[None, :],
                    mask=k_mask[None, :],
                    other=0,
                ).to(tl.uint32)
                next_byte = tl.load(
                    codes_ptr
                    + head_like * key_tokens * CODE_BYTES
                    + (byte_index + 1) * key_tokens
                    + k_offsets[None, :],
                    mask=(
                        k_mask[None, :]
                        & needs_next_byte
                        & ((byte_index + 1) < CODE_BYTES)
                    ),
                    other=0,
                ).to(tl.uint32)
            else:
                code_byte = tl.load(
                    codes_ptr
                    + head_like * key_tokens * CODE_BYTES
                    + k_offsets[None, :] * CODE_BYTES
                    + byte_index,
                    mask=k_mask[None, :],
                    other=0,
                ).to(tl.uint32)
                next_byte = tl.load(
                    codes_ptr
                    + head_like * key_tokens * CODE_BYTES
                    + k_offsets[None, :] * CODE_BYTES
                    + byte_index
                    + 1,
                    mask=(
                        k_mask[None, :]
                        & needs_next_byte
                        & ((byte_index + 1) < CODE_BYTES)
                    ),
                    other=0,
                ).to(tl.uint32)
            combined = code_byte | (next_byte << 8)
            code = (combined >> bit_offset) & ((1 << BITS) - 1)
        code_values = tl.load(codebook_ptr + code, mask=k_mask[None, :], other=0.0)
        scores = tl.dot(q_rot, code_values, input_precision=SCORE_DOT_INPUT_PRECISION)

        key_norms = tl.load(
            norms_ptr + head_like * key_tokens + k_offsets,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        scores *= (key_norms * INV_SQRT_D)[None, :]

        if QJL_BITS > 0:
            qjl_offsets = tl.arange(0, QJL_BITS)
            qjl_t = tl.load(
                qjl_matrix_ptr
                + qjl_offsets[None, :] * HEAD_DIM
                + dim_offsets[:, None]
            )
            q_proj = tl.dot(
                query_values,
                qjl_t,
                input_precision=QJL_DOT_INPUT_PRECISION,
            )
            sign_byte_index = qjl_offsets[:, None] // 8
            sign_bit_offset = qjl_offsets[:, None] % 8
            sign_byte = tl.load(
                residual_signs_ptr
                + head_like * key_tokens * SIGN_BYTES
                + k_offsets[None, :] * SIGN_BYTES
                + sign_byte_index,
                mask=k_mask[None, :],
                other=0,
            ).to(tl.uint32)
            sign_bit = (sign_byte >> sign_bit_offset) & 1
            signs = tl.where(sign_bit == 1, 1.0, -1.0)
            correction = tl.dot(q_proj, signs, input_precision=QJL_DOT_INPUT_PRECISION)
            residual_norm_values = tl.load(
                residual_norms_ptr + head_like * key_tokens + k_offsets,
                mask=k_mask,
                other=0.0,
            )
            scores += correction * (residual_norm_values * QJL_SCALE)[None, :]

        logits = tl.where(k_mask[None, :], scores * ATTENTION_SCALE, -float("inf"))
        weights = tl.softmax(logits, dim=1, keep_dims=True)
        values = tl.load(
            value_ptr
            + head_like * key_tokens * HEAD_DIM
            + k_offsets[:, None] * HEAD_DIM
            + dim_offsets[None, :],
            mask=k_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        output = tl.dot(weights, values, input_precision=VALUE_DOT_INPUT_PRECISION)
        tl.store(
            out_ptr
            + head_like * q_tokens * HEAD_DIM
            + q_offsets[:, None] * HEAD_DIM
            + dim_offsets[None, :],
            output,
            mask=q_mask[:, None],
        )

    @triton.jit(do_not_specialize=["q_tokens", "key_tokens"])
    def _packed_key_attention_streaming_kernel(
        query_ptr,
        rotation_ptr,
        qjl_matrix_ptr,
        codes_ptr,
        norms_ptr,
        residual_signs_ptr,
        residual_norms_ptr,
        codebook_ptr,
        value_ptr,
        out_ptr,
        q_tokens,
        key_tokens,
        HEAD_DIM: tl.constexpr,
        BITS: tl.constexpr,
        BYTE_CODES: tl.constexpr,
        TRANSPOSED_CODES: tl.constexpr,
        QJL_BITS: tl.constexpr,
        CODE_BYTES: tl.constexpr,
        SIGN_BYTES: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        INV_SQRT_D: tl.constexpr,
        ATTENTION_SCALE: tl.constexpr,
        QJL_SCALE: tl.constexpr,
        ROTATION_DOT_INPUT_PRECISION: tl.constexpr,
        SCORE_DOT_INPUT_PRECISION: tl.constexpr,
        VALUE_DOT_INPUT_PRECISION: tl.constexpr,
        QJL_DOT_INPUT_PRECISION: tl.constexpr,
    ):
        q_offsets = tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
        head_like = tl.program_id(1)
        k_offsets = tl.arange(0, BLOCK_K)
        dim_offsets = tl.arange(0, HEAD_DIM)
        q_mask = q_offsets < q_tokens
        k_mask = k_offsets < key_tokens
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
        q_rot = tl.dot(
            query_values,
            rotation_t,
            input_precision=ROTATION_DOT_INPUT_PRECISION,
        )

        if QJL_BITS > 0:
            qjl_offsets = tl.arange(0, QJL_BITS)
            qjl_t = tl.load(
                qjl_matrix_ptr
                + qjl_offsets[None, :] * HEAD_DIM
                + dim_offsets[:, None]
            )
            q_proj = tl.dot(
                query_values,
                qjl_t,
                input_precision=QJL_DOT_INPUT_PRECISION,
            )
        else:
            q_proj = tl.zeros((BLOCK_Q, 1), dtype=tl.float32)

        m_i = tl.full((BLOCK_Q, 1), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_Q, 1), dtype=tl.float32)
        acc = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)

        for key_start in tl.range(0, key_tokens, BLOCK_K):
            tile_k_offsets = key_start + k_offsets
            tile_k_mask = tile_k_offsets < key_tokens
            if BYTE_CODES:
                code = tl.load(
                    codes_ptr
                    + head_like * key_tokens * CODE_BYTES
                    + tile_k_offsets[None, :] * CODE_BYTES
                    + dim_offsets[:, None],
                    mask=tile_k_mask[None, :],
                    other=0,
                ).to(tl.uint32)
            else:
                bit_position = dim_offsets[:, None] * BITS
                byte_index = bit_position // 8
                bit_offset = bit_position % 8
                needs_next_byte = (bit_offset + BITS) > 8
                if TRANSPOSED_CODES:
                    code_byte = tl.load(
                        codes_ptr
                        + head_like * key_tokens * CODE_BYTES
                        + byte_index * key_tokens
                        + tile_k_offsets[None, :],
                        mask=tile_k_mask[None, :],
                        other=0,
                    ).to(tl.uint32)
                    next_byte = tl.load(
                        codes_ptr
                        + head_like * key_tokens * CODE_BYTES
                        + (byte_index + 1) * key_tokens
                        + tile_k_offsets[None, :],
                        mask=(
                            tile_k_mask[None, :]
                            & needs_next_byte
                            & ((byte_index + 1) < CODE_BYTES)
                        ),
                        other=0,
                    ).to(tl.uint32)
                else:
                    code_byte = tl.load(
                        codes_ptr
                        + head_like * key_tokens * CODE_BYTES
                        + tile_k_offsets[None, :] * CODE_BYTES
                        + byte_index,
                        mask=tile_k_mask[None, :],
                        other=0,
                    ).to(tl.uint32)
                    next_byte = tl.load(
                        codes_ptr
                        + head_like * key_tokens * CODE_BYTES
                        + tile_k_offsets[None, :] * CODE_BYTES
                        + byte_index
                        + 1,
                        mask=(
                            tile_k_mask[None, :]
                            & needs_next_byte
                            & ((byte_index + 1) < CODE_BYTES)
                        ),
                        other=0,
                    ).to(tl.uint32)
                combined = code_byte | (next_byte << 8)
                code = (combined >> bit_offset) & ((1 << BITS) - 1)
            code_values = tl.load(
                codebook_ptr + code,
                mask=tile_k_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(q_rot, code_values, input_precision=SCORE_DOT_INPUT_PRECISION)

            key_norms = tl.load(
                norms_ptr + head_like * key_tokens + tile_k_offsets,
                mask=tile_k_mask,
                other=0.0,
            ).to(tl.float32)
            scores *= (key_norms * INV_SQRT_D)[None, :]

            if QJL_BITS > 0:
                sign_byte_index = qjl_offsets[:, None] // 8
                sign_bit_offset = qjl_offsets[:, None] % 8
                sign_byte = tl.load(
                    residual_signs_ptr
                    + head_like * key_tokens * SIGN_BYTES
                    + tile_k_offsets[None, :] * SIGN_BYTES
                    + sign_byte_index,
                    mask=tile_k_mask[None, :],
                    other=0,
                ).to(tl.uint32)
                sign_bit = (sign_byte >> sign_bit_offset) & 1
                signs = tl.where(sign_bit == 1, 1.0, -1.0)
                correction = tl.dot(q_proj, signs, input_precision=QJL_DOT_INPUT_PRECISION)
                residual_norm_values = tl.load(
                    residual_norms_ptr + head_like * key_tokens + tile_k_offsets,
                    mask=tile_k_mask,
                    other=0.0,
                )
                scores += correction * (residual_norm_values * QJL_SCALE)[None, :]

            logits = tl.where(
                q_mask[:, None] & tile_k_mask[None, :],
                scores * ATTENTION_SCALE,
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
            acc = acc * alpha + tl.dot(
                weights,
                values,
                input_precision=VALUE_DOT_INPUT_PRECISION,
            )

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
    _packed_key_attention_output_kernel = None
    _packed_key_attention_streaming_kernel = None
