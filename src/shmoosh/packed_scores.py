from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Any, Literal

from shmoosh.packed_keys import PackedKeyBlock, _unpack_bits
from shmoosh.quantization import ShmooshCodec

try:  # pragma: no cover - exercised by CUDA-specific tests when available.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


@dataclass(frozen=True)
class PackedScoreResources:
    """Deterministic codec resources used by packed-key score paths."""

    rotation: Any
    codebook: Any
    boundaries: Any
    qjl_matrix: Any | None


def build_score_resources(
    block: PackedKeyBlock,
    *,
    device: Any | None = None,
    dtype: Any | None = None,
    codec: ShmooshCodec | None = None,
) -> PackedScoreResources:
    torch = _load_torch()
    target_device = block.codes.device if device is None else torch.device(device)
    target_dtype = torch.float32 if dtype is None else dtype
    if codec is None:
        codec = ShmooshCodec(
            dim=block.head_dim,
            bits=block.bits,
            qjl_bits=block.qjl_bits,
            seed=block.seed,
            codebook_samples=block.codebook_samples,
            lloyd_iters=block.lloyd_iters,
        )
    elif (
        codec.dim != block.head_dim
        or codec.bits != block.bits
        or codec.qjl_bits != block.qjl_bits
        or codec.seed != block.seed
        or codec.codebook_samples != block.codebook_samples
        or codec.lloyd_iters != block.lloyd_iters
    ):
        raise ValueError("codec parameters do not match packed score block")
    return score_resources_from_codec(
        codec,
        device=target_device,
        dtype=target_dtype,
    )


def score_resources_from_codec(
    codec: ShmooshCodec,
    *,
    device: Any,
    dtype: Any | None = None,
) -> PackedScoreResources:
    torch = _load_torch()
    target_device = torch.device(device)
    target_dtype = torch.float32 if dtype is None else dtype
    codebook = torch.from_numpy(codec.codebook).to(
        device=target_device,
        dtype=target_dtype,
    )
    qjl_matrix = (
        None
        if codec.qjl_matrix is None
        else torch.from_numpy(codec.qjl_matrix).to(
            device=target_device, dtype=target_dtype
        )
    )
    return PackedScoreResources(
        rotation=torch.from_numpy(codec.rotation).to(
            device=target_device,
            dtype=target_dtype,
        ),
        codebook=codebook,
        boundaries=((codebook[:-1] + codebook[1:]) * 0.5).contiguous(),
        qjl_matrix=qjl_matrix,
    )


def packed_key_scores(
    query: Any,
    block: PackedKeyBlock,
    *,
    resources: PackedScoreResources | None = None,
    backend: Literal["auto", "torch", "triton"] = "auto",
) -> Any:
    """Estimate unscaled attention scores `QK^T` from packed keys.

    The score path consumes `PackedKeyBlock.codes` directly. It reconstructs
    codebook coordinates for the dot product, but it does not decode packed keys
    back into full model-space K vectors.
    """

    if backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be one of: auto, torch, triton")
    if backend == "torch":
        return torch_packed_key_scores(query, block, resources=resources)
    if backend == "triton":
        return triton_packed_key_scores(query, block, resources=resources)
    if _can_use_triton(query):
        return triton_packed_key_scores(query, block, resources=resources)
    return torch_packed_key_scores(query, block, resources=resources)


def torch_packed_key_scores(
    query: Any,
    block: PackedKeyBlock,
    *,
    resources: PackedScoreResources | None = None,
) -> Any:
    """Torch reference score path for packed keys."""

    torch = _load_torch()
    _validate_query_block(query, block)
    resources = _resources_for(query, block, resources)

    query_f = query.to(dtype=torch.float32)
    rotation = resources.rotation.to(device=query.device, dtype=torch.float32)
    codebook = resources.codebook.to(device=query.device, dtype=torch.float32)
    q_rot = torch.matmul(query_f, rotation.T)

    indices = _unpack_bits(
        block.codes.to(device=query.device),
        bits=block.bits,
        value_count=block.head_dim,
    )
    code_values = codebook[indices] * (1.0 / sqrt(block.head_dim))
    scores = torch.einsum(
        "bhqd,bhtd,bht->bhqt",
        q_rot,
        code_values,
        block.norms.to(device=query.device, dtype=torch.float32),
    )

    if (
        block.qjl_bits > 0
        and block.residual_signs is not None
        and block.residual_norms is not None
        and resources.qjl_matrix is not None
    ):
        qjl_matrix = resources.qjl_matrix.to(device=query.device, dtype=torch.float32)
        projected_q = torch.matmul(query_f, qjl_matrix.T)
        sign_bits = _unpack_bits(
            block.residual_signs.to(device=query.device),
            bits=1,
            value_count=block.qjl_bits,
        )
        signs = torch.where(
            sign_bits > 0,
            torch.tensor(1.0, device=query.device),
            torch.tensor(-1.0, device=query.device),
        )
        correction = torch.einsum(
            "bhqr,bhtr,bht->bhqt",
            projected_q,
            signs,
            block.residual_norms.to(device=query.device, dtype=torch.float32),
        )
        scores = scores + correction * (sqrt(pi / 2.0) / float(block.qjl_bits))

    return scores


def triton_packed_key_scores(
    query: Any,
    block: PackedKeyBlock,
    *,
    resources: PackedScoreResources | None = None,
    block_q: int = 16,
    block_k: int = 16,
) -> Any:
    """Minimal Triton score kernel that unpacks packed K codes in-kernel."""

    torch = _load_torch()
    if triton is None or _packed_key_score_kernel is None:
        raise RuntimeError("triton is required for backend='triton'")
    if not query.is_cuda:
        raise ValueError("triton packed-key scores require a CUDA query tensor")

    _validate_query_block(query, block)
    resources = _resources_for(query, block, resources)
    if block.codes.device != query.device:
        raise ValueError("block tensors must live on the same CUDA device as query")

    batch, heads, q_tokens, head_dim = (int(size) for size in query.shape)
    key_tokens = int(block.shape[2])
    head_like = batch * heads
    query_f = query.to(dtype=torch.float32)
    rotation = resources.rotation.to(device=query.device, dtype=torch.float32)
    codebook = resources.codebook.to(device=query.device, dtype=torch.float32)
    q_rot = torch.matmul(query_f, rotation.T).contiguous().reshape(
        head_like, q_tokens, head_dim
    )

    effective_qjl_bits = (
        block.qjl_bits
        if (
            block.qjl_bits > 0
            and block.residual_signs is not None
            and block.residual_norms is not None
            and resources.qjl_matrix is not None
        )
        else 0
    )

    if effective_qjl_bits:
        qjl_matrix = resources.qjl_matrix.to(device=query.device, dtype=torch.float32)
        q_proj = torch.matmul(query_f, qjl_matrix.T).contiguous().reshape(
            head_like, q_tokens, effective_qjl_bits
        )
        residual_signs = block.residual_signs.contiguous().reshape(
            head_like, key_tokens, block.qjl_sign_bytes_per_vector
        )
        residual_norms = block.residual_norms.to(dtype=torch.float32).contiguous().reshape(
            head_like, key_tokens
        )
    else:
        q_proj = torch.empty((1,), device=query.device, dtype=torch.float32)
        residual_signs = torch.empty((1,), device=query.device, dtype=torch.uint8)
        residual_norms = torch.empty((1,), device=query.device, dtype=torch.float32)

    codes = block.codes.contiguous().reshape(
        head_like, key_tokens, block.code_bytes_per_vector
    )
    norms = block.norms.to(dtype=torch.float32).contiguous().reshape(
        head_like, key_tokens
    )
    output = torch.empty(
        (head_like, q_tokens, key_tokens),
        device=query.device,
        dtype=torch.float32,
    )
    grid = (
        triton.cdiv(q_tokens, block_q),
        triton.cdiv(key_tokens, block_k),
        head_like,
    )
    _packed_key_score_kernel[grid](
        q_rot,
        q_proj,
        codes,
        norms,
        residual_signs,
        residual_norms,
        codebook,
        output,
        q_tokens,
        key_tokens,
        HEAD_DIM=head_dim,
        BITS=block.bits,
        QJL_BITS=effective_qjl_bits,
        CODE_BYTES=block.code_bytes_per_vector,
        SIGN_BYTES=block.qjl_sign_bytes_per_vector if effective_qjl_bits else 1,
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
        INV_SQRT_D=1.0 / sqrt(head_dim),
        QJL_SCALE=(
            0.0
            if effective_qjl_bits == 0
            else sqrt(pi / 2.0) / float(effective_qjl_bits)
        ),
        num_warps=4,
    )
    return output.reshape(batch, heads, q_tokens, key_tokens)


def _validate_query_block(query: Any, block: PackedKeyBlock) -> None:
    if query.ndim != 4:
        raise ValueError("query must have shape (batch, heads, query_tokens, head_dim)")
    batch, heads, _tokens, head_dim = (int(size) for size in query.shape)
    key_batch, key_heads, _key_tokens, key_dim = block.shape
    if (batch, heads, head_dim) != (key_batch, key_heads, key_dim):
        raise ValueError(
            "query and packed keys must share batch, heads, and head_dim"
        )


def _resources_for(
    query: Any,
    block: PackedKeyBlock,
    resources: PackedScoreResources | None,
) -> PackedScoreResources:
    if resources is not None:
        return resources
    return build_score_resources(block, device=query.device)


def _can_use_triton(query: Any) -> bool:
    return (
        triton is not None
        and _packed_key_score_kernel is not None
        and getattr(query, "is_cuda", False)
    )


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch is required for packed score paths; install with "
            "`uv sync --extra dev --extra diffusers`"
        ) from exc
    return torch


if triton is not None and tl is not None:

    @triton.jit(do_not_specialize=["q_tokens", "key_tokens"])
    def _packed_key_score_kernel(
        q_rot_ptr,
        q_proj_ptr,
        codes_ptr,
        norms_ptr,
        residual_signs_ptr,
        residual_norms_ptr,
        codebook_ptr,
        out_ptr,
        q_tokens,
        key_tokens,
        HEAD_DIM: tl.constexpr,
        BITS: tl.constexpr,
        QJL_BITS: tl.constexpr,
        CODE_BYTES: tl.constexpr,
        SIGN_BYTES: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        INV_SQRT_D: tl.constexpr,
        QJL_SCALE: tl.constexpr,
    ):
        q_offsets = tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
        k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        head_like = tl.program_id(2)
        q_mask = q_offsets < q_tokens
        k_mask = k_offsets < key_tokens
        accum = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

        for dim_index in tl.static_range(0, HEAD_DIM):
            q_values = tl.load(
                q_rot_ptr + head_like * q_tokens * HEAD_DIM + q_offsets * HEAD_DIM + dim_index,
                mask=q_mask,
                other=0.0,
            )[:, None]
            bit_position = dim_index * BITS
            byte_index = bit_position // 8
            bit_offset = bit_position % 8
            code_byte = tl.load(
                codes_ptr
                + head_like * key_tokens * CODE_BYTES
                + k_offsets * CODE_BYTES
                + byte_index,
                mask=k_mask,
                other=0,
            ).to(tl.uint32)
            combined = code_byte
            if bit_offset + BITS > 8:
                next_byte = tl.load(
                    codes_ptr
                    + head_like * key_tokens * CODE_BYTES
                    + k_offsets * CODE_BYTES
                    + byte_index
                    + 1,
                    mask=k_mask & ((byte_index + 1) < CODE_BYTES),
                    other=0,
                ).to(tl.uint32)
                combined = combined | (next_byte << 8)
            code = (combined >> bit_offset) & ((1 << BITS) - 1)
            code_values = tl.load(codebook_ptr + code, mask=k_mask, other=0.0)
            accum += q_values * code_values[None, :]

        key_norms = tl.load(
            norms_ptr + head_like * key_tokens + k_offsets,
            mask=k_mask,
            other=0.0,
        )
        accum *= (key_norms * INV_SQRT_D)[None, :]

        if QJL_BITS > 0:
            correction = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
            for qjl_index in tl.static_range(0, QJL_BITS):
                projected_q = tl.load(
                    q_proj_ptr
                    + head_like * q_tokens * QJL_BITS
                    + q_offsets * QJL_BITS
                    + qjl_index,
                    mask=q_mask,
                    other=0.0,
                )[:, None]
                sign_byte_index = qjl_index // 8
                sign_bit_offset = qjl_index % 8
                sign_byte = tl.load(
                    residual_signs_ptr
                    + head_like * key_tokens * SIGN_BYTES
                    + k_offsets * SIGN_BYTES
                    + sign_byte_index,
                    mask=k_mask,
                    other=0,
                ).to(tl.uint32)
                sign_bit = (sign_byte >> sign_bit_offset) & 1
                signs = tl.where(sign_bit == 1, 1.0, -1.0)
                correction += projected_q * signs[None, :]
            residual_norm_values = tl.load(
                residual_norms_ptr + head_like * key_tokens + k_offsets,
                mask=k_mask,
                other=0.0,
            )
            accum += correction * (residual_norm_values * QJL_SCALE)[None, :]

        tl.store(
            out_ptr
            + head_like * q_tokens * key_tokens
            + q_offsets[:, None] * key_tokens
            + k_offsets[None, :],
            accum,
            mask=q_mask[:, None] & k_mask[None, :],
        )

else:  # pragma: no cover
    _packed_key_score_kernel = None
