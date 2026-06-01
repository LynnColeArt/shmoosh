from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Literal

import numpy as np

from shmoosh.quantization import EncodedVectors, ShmooshCodec

try:  # pragma: no cover - exercised by CUDA-specific tests when available.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


@dataclass(frozen=True)
class PackedKeyBlock:
    """Kernel-facing packed representation for Shmoosh key tensors.

    This is a data-path contract, not a fused kernel. The debug decode path
    reconstructs through the NumPy reference codec so we can validate packed
    metadata before writing Triton/CUDA score kernels.
    """

    codes: Any
    norms: Any
    residual_signs: Any | None
    residual_norms: Any | None
    bits: int
    qjl_bits: int
    head_dim: int
    seed: int
    codebook_samples: int = 80_000
    lloyd_iters: int = 80
    code_format: Literal["packed", "byte"] = "packed"

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (*tuple(int(size) for size in self.norms.shape), self.head_dim)

    @property
    def vector_count(self) -> int:
        batch, heads, tokens, _dim = self.shape
        return batch * heads * tokens

    @property
    def code_bytes_per_vector(self) -> int:
        if self.code_format == "byte":
            return self.head_dim
        return _ceil_div(self.head_dim * self.bits, 8)

    @property
    def qjl_sign_bytes_per_vector(self) -> int:
        return _ceil_div(self.qjl_bits, 8)

    @property
    def packed_bytes_per_vector(self) -> int:
        residual_bytes = (
            self.qjl_sign_bytes_per_vector + self.residual_norms.element_size()
            if self.residual_norms is not None
            else 0
        )
        return (
            self.code_bytes_per_vector
            + self.norms.element_size()
            + residual_bytes
        )

    def exact_key_bytes(self, *, dtype_bytes: int = 2) -> int:
        return self.vector_count * self.head_dim * dtype_bytes

    def packed_key_bytes(self) -> int:
        tensors = [self.codes, self.norms]
        if self.residual_signs is not None:
            tensors.append(self.residual_signs)
        if self.residual_norms is not None:
            tensors.append(self.residual_norms)
        return sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors)

    def compression_ratio(self, *, dtype_bytes: int = 2) -> float:
        return self.exact_key_bytes(dtype_bytes=dtype_bytes) / self.packed_key_bytes()

    def decode(self, *, dtype: Any | None = None, device: Any | None = None) -> Any:
        torch = _load_torch()
        target_device = self.codes.device if device is None else torch.device(device)
        target_dtype = torch.float32 if dtype is None else dtype
        codec = ShmooshCodec(
            dim=self.head_dim,
            bits=self.bits,
            qjl_bits=self.qjl_bits,
            seed=self.seed,
            codebook_samples=self.codebook_samples,
            lloyd_iters=self.lloyd_iters,
        )
        encoded = self.to_encoded_vectors()
        decoded = codec.decode(encoded)
        return torch.from_numpy(decoded).to(device=target_device, dtype=target_dtype)

    def to_encoded_vectors(self) -> EncodedVectors:
        if self.code_format == "byte":
            indices = self.codes.to(dtype=_load_torch().int64)
        else:
            indices = _unpack_bits(
                self.codes,
                bits=self.bits,
                value_count=self.head_dim,
            )
        residual_signs = None
        if self.residual_signs is not None:
            sign_bits = _unpack_bits(
                self.residual_signs, bits=1, value_count=self.qjl_bits
            )
            residual_signs = (
                sign_bits.reshape(-1, self.qjl_bits)
                .to(device="cpu", dtype=_load_torch().int8)
                .numpy()
            )
            residual_signs = np.where(residual_signs > 0, 1, -1).astype(np.int8)

        residual_norms = None
        if self.residual_norms is not None:
            residual_norms = (
                self.residual_norms.detach()
                .to(device="cpu", dtype=_load_torch().float32)
                .reshape(-1)
                .numpy()
            )

        return EncodedVectors(
            indices=indices.to(device="cpu").numpy().astype(np.uint8),
            norms=self.norms.detach().to(device="cpu", dtype=_load_torch().float32).numpy(),
            original_shape=self.shape,
            residual_signs=residual_signs,
            residual_norms=residual_norms,
        )


def encode_packed_keys(
    keys: Any,
    *,
    bits: int,
    qjl_bits: int,
    seed: int,
    codebook_samples: int = 80_000,
    lloyd_iters: int = 80,
    codec: ShmooshCodec | None = None,
    resources: Any | None = None,
    timing_recorder: Any | None = None,
    timing_module: str | None = None,
    step_state: Any | None = None,
    code_format: Literal["packed", "byte"] = "packed",
) -> PackedKeyBlock:
    torch = _load_torch()
    if keys.ndim != 4:
        raise ValueError("keys must have shape (batch, heads, tokens, head_dim)")
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in the range 1..8 for packed key blocks")
    if qjl_bits < 0:
        raise ValueError("qjl_bits must be non-negative")
    if code_format not in {"packed", "byte"}:
        raise ValueError("code_format must be one of: packed, byte")

    device = keys.device
    batch, heads, tokens, head_dim = (int(size) for size in keys.shape)
    if codec is None:
        codec = ShmooshCodec(
            dim=head_dim,
            bits=bits,
            qjl_bits=qjl_bits,
            seed=seed,
            codebook_samples=codebook_samples,
            lloyd_iters=lloyd_iters,
        )
    elif (
        codec.dim != head_dim
        or codec.bits != bits
        or codec.qjl_bits != qjl_bits
        or codec.seed != seed
        or codec.codebook_samples != codebook_samples
        or codec.lloyd_iters != lloyd_iters
    ):
        raise ValueError("codec parameters do not match requested packed key block")

    if resources is not None:
        return _encode_packed_keys_torch(
            keys,
            bits=bits,
            qjl_bits=qjl_bits,
            seed=seed,
            codebook_samples=codebook_samples,
            lloyd_iters=lloyd_iters,
            resources=resources,
            timing_recorder=timing_recorder,
            timing_module=timing_module,
            step_state=step_state,
            code_format=code_format,
        )

    encoded = codec.encode(
        keys.detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    indices = torch.from_numpy(encoded.indices.astype(np.int64))
    codes = _encode_codes(indices, bits=bits, code_format=code_format).to(device=device)
    norms = torch.from_numpy(encoded.norms.astype(np.float32)).to(device=device)

    residual_signs = None
    residual_norms = None
    if encoded.residual_signs is not None and encoded.residual_norms is not None:
        sign_bits = np.where(
            encoded.residual_signs.reshape(batch, heads, tokens, qjl_bits) > 0,
            1,
            0,
        ).astype(np.int64)
        residual_signs = _pack_bits(torch.from_numpy(sign_bits), bits=1).to(device=device)
        residual_norms = torch.from_numpy(
            encoded.residual_norms.reshape(batch, heads, tokens).astype(np.float32)
        ).to(device=device)

    return PackedKeyBlock(
        codes=codes,
        norms=norms,
        residual_signs=residual_signs,
        residual_norms=residual_norms,
        bits=bits,
        qjl_bits=qjl_bits,
        head_dim=head_dim,
        seed=seed,
        codebook_samples=codebook_samples,
        lloyd_iters=lloyd_iters,
        code_format=code_format,
    )


def _encode_packed_keys_torch(
    keys: Any,
    *,
    bits: int,
    qjl_bits: int,
    seed: int,
    codebook_samples: int,
    lloyd_iters: int,
    resources: Any,
    timing_recorder: Any | None = None,
    timing_module: str | None = None,
    step_state: Any | None = None,
    code_format: Literal["packed", "byte"] = "packed",
) -> PackedKeyBlock:
    torch = _load_torch()
    device = keys.device
    _batch, _heads, _tokens, head_dim = (int(size) for size in keys.shape)
    rotation = resources.rotation.to(device=device, dtype=torch.float32)
    codebook = resources.codebook.to(device=device, dtype=torch.float32)
    boundaries = resources.boundaries.to(device=device, dtype=torch.float32)
    if rotation.shape != (head_dim, head_dim):
        raise ValueError("score resources rotation does not match key head dimension")
    if codebook.numel() != (1 << bits):
        raise ValueError("score resources codebook does not match key bit depth")
    if boundaries.numel() != codebook.numel() - 1:
        raise ValueError("score resources boundaries do not match key bit depth")

    timing_metadata = {
        "bits": bits,
        "qjl_bits": qjl_bits,
        "code_format": code_format,
        "head_dim": head_dim,
        "key_tokens": int(keys.shape[2]),
        "heads": int(keys.shape[1]),
        "fused_bucketize_pack": False,
    }
    with _timing_span(
        timing_recorder,
        "encode_normalize",
        keys,
        timing_module,
        step_state,
        timing_metadata,
    ):
        keys_f = keys.detach().to(dtype=torch.float32)
        if keys_f.data_ptr() == keys.data_ptr():
            keys_f = keys_f.clone()
        raw_keys_f = keys_f if qjl_bits == 0 else keys_f.clone()
        norms = torch.linalg.vector_norm(keys_f, dim=-1)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        keys_f.div_(safe_norms.unsqueeze(-1))
    with _timing_span(
        timing_recorder,
        "encode_rotate_bucketize",
        keys,
        timing_module,
        step_state,
        timing_metadata,
    ):
        rotated = torch.matmul(keys_f, rotation.T)
        normalized = rotated * sqrt(head_dim)
        normalized = normalized.contiguous()
        indices = None
        codes = None
        if _can_use_triton_bucketize_pack(
            normalized,
            boundaries,
            bits=bits,
            qjl_bits=qjl_bits,
            code_format=code_format,
        ):
            timing_metadata["fused_bucketize_pack"] = True
            codes = _triton_bucketize_pack_codes(
                normalized,
                boundaries,
                bits=bits,
            )
        else:
            indices = torch.bucketize(
                normalized,
                boundaries,
            ).to(dtype=torch.int64)
    with _timing_span(
        timing_recorder,
        "encode_pack_codes",
        keys,
        timing_module,
        step_state,
        timing_metadata,
    ):
        if codes is None:
            codes = _encode_codes(
                indices,
                bits=bits,
                code_format=code_format,
                validate=False,
            )

    residual_signs = None
    residual_norms = None
    if qjl_bits > 0:
        with _timing_span(
            timing_recorder,
            "encode_residual_project",
            keys,
            timing_module,
            step_state,
            timing_metadata,
        ):
            qjl_matrix = resources.qjl_matrix
            if qjl_matrix is None:
                raise ValueError("score resources are missing QJL matrix")
            qjl_matrix = qjl_matrix.to(device=device, dtype=torch.float32)
            if qjl_matrix.shape != (qjl_bits, head_dim):
                raise ValueError("score resources QJL matrix does not match key block")
            code_values = codebook[indices] * (1.0 / sqrt(head_dim))
            decoded_unit = torch.matmul(code_values, rotation)
            residual = raw_keys_f - decoded_unit * norms.unsqueeze(-1)
            residual_norms = torch.linalg.vector_norm(residual, dim=-1)
            sign_bits = (torch.matmul(residual, qjl_matrix.T) >= 0).to(
                dtype=torch.int64
            )
        with _timing_span(
            timing_recorder,
            "encode_pack_residual_signs",
            keys,
            timing_module,
            step_state,
            timing_metadata,
        ):
            residual_signs = _pack_bits(sign_bits, bits=1, validate=False)

    return PackedKeyBlock(
        codes=codes,
        norms=norms,
        residual_signs=residual_signs,
        residual_norms=residual_norms,
        bits=bits,
        qjl_bits=qjl_bits,
        head_dim=head_dim,
        seed=seed,
        codebook_samples=codebook_samples,
        lloyd_iters=lloyd_iters,
        code_format=code_format,
    )


def _encode_codes(
    indices: Any,
    *,
    bits: int,
    code_format: str,
    validate: bool = True,
) -> Any:
    torch = _load_torch()
    if code_format not in {"packed", "byte"}:
        raise ValueError("code_format must be one of: packed, byte")
    indices = indices.to(dtype=torch.int64)
    if (
        validate
        and indices.numel()
        and (indices.min() < 0 or indices.max() >= (1 << bits))
    ):
        raise ValueError("value is out of range for requested bit width")
    if code_format == "byte":
        return indices.to(dtype=torch.uint8)
    return _pack_bits(indices, bits=bits, validate=False)


def _pack_bits(values: Any, *, bits: int, validate: bool = True) -> Any:
    torch = _load_torch()
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in the range 1..8")
    values = values.to(dtype=torch.int64)
    if (
        validate
        and values.numel()
        and (values.min() < 0 or values.max() >= (1 << bits))
    ):
        raise ValueError("value is out of range for requested bit width")

    value_count = int(values.shape[-1])
    byte_count = _ceil_div(value_count * bits, 8)
    if value_count == 0:
        return torch.zeros(
            (*values.shape[:-1], byte_count),
            dtype=torch.uint8,
            device=values.device,
        )

    fast_packed = _pack_bits_fast(values, bits=bits)
    if fast_packed is not None:
        return fast_packed

    packed = torch.zeros(
        (*values.shape[:-1], byte_count),
        dtype=torch.int64,
        device=values.device,
    )

    positions = torch.arange(value_count, device=values.device, dtype=torch.int64) * bits
    byte_indices = positions // 8
    bit_offsets = positions % 8
    widths = torch.minimum(
        8 - bit_offsets,
        torch.full_like(bit_offsets, bits),
    )
    vector_shape = (1,) * (values.ndim - 1) + (value_count,)
    masks = torch.bitwise_left_shift(torch.ones_like(widths), widths) - 1
    low_chunks = (values & masks.reshape(vector_shape)) << bit_offsets.reshape(vector_shape)
    low_indices = byte_indices.reshape(vector_shape).expand_as(values)
    packed.scatter_add_(-1, low_indices, low_chunks)

    high_widths = bits - widths
    crosses = high_widths > 0
    high_values = values[..., crosses] >> widths[crosses].reshape(
        (1,) * (values.ndim - 1) + (-1,)
    )
    high_indices = (byte_indices[crosses] + 1).reshape(
        (1,) * (values.ndim - 1) + (-1,)
    )
    packed.scatter_add_(-1, high_indices.expand_as(high_values), high_values)

    return packed.to(dtype=torch.uint8)


def _pack_bits_fast(values: Any, *, bits: int) -> Any | None:
    torch = _load_torch()
    value_count = int(values.shape[-1])
    if bits == 8:
        return values.to(dtype=torch.uint8)

    if bits == 1 and value_count % 8 == 0:
        grouped = values.reshape(*values.shape[:-1], value_count // 8, 8)
        shifts = torch.arange(8, device=values.device, dtype=torch.int64)
        return torch.sum(grouped << shifts, dim=-1).to(dtype=torch.uint8)

    if bits == 4 and value_count % 2 == 0:
        grouped = values.reshape(*values.shape[:-1], value_count // 2, 2)
        packed = grouped[..., 0] | (grouped[..., 1] << 4)
        return packed.to(dtype=torch.uint8)

    if bits == 5 and value_count % 8 == 0:
        grouped = values.reshape(*values.shape[:-1], value_count // 8, 8)
        v = [grouped[..., index] for index in range(8)]
        packed = torch.stack(
            (
                v[0] | ((v[1] & 0x07) << 5),
                (v[1] >> 3) | (v[2] << 2) | ((v[3] & 0x01) << 7),
                (v[3] >> 1) | ((v[4] & 0x0F) << 4),
                (v[4] >> 4) | (v[5] << 1) | ((v[6] & 0x03) << 6),
                (v[6] >> 2) | (v[7] << 3),
            ),
            dim=-1,
        )
        return packed.reshape(*values.shape[:-1], value_count * 5 // 8).to(
            dtype=torch.uint8
        )

    if bits == 6 and value_count % 4 == 0:
        grouped = values.reshape(*values.shape[:-1], value_count // 4, 4)
        v = [grouped[..., index] for index in range(4)]
        packed = torch.stack(
            (
                v[0] | ((v[1] & 0x03) << 6),
                (v[1] >> 2) | ((v[2] & 0x0F) << 4),
                (v[2] >> 4) | (v[3] << 2),
            ),
            dim=-1,
        )
        return packed.reshape(*values.shape[:-1], value_count * 6 // 8).to(
            dtype=torch.uint8
        )

    if bits == 7 and value_count % 8 == 0:
        grouped = values.reshape(*values.shape[:-1], value_count // 8, 8)
        v = [grouped[..., index] for index in range(8)]
        packed = torch.stack(
            (
                v[0] | ((v[1] & 0x01) << 7),
                (v[1] >> 1) | ((v[2] & 0x03) << 6),
                (v[2] >> 2) | ((v[3] & 0x07) << 5),
                (v[3] >> 3) | ((v[4] & 0x0F) << 4),
                (v[4] >> 4) | ((v[5] & 0x1F) << 3),
                (v[5] >> 5) | ((v[6] & 0x3F) << 2),
                (v[6] >> 6) | (v[7] << 1),
            ),
            dim=-1,
        )
        return packed.reshape(*values.shape[:-1], value_count * 7 // 8).to(
            dtype=torch.uint8
        )

    return None


def _can_use_triton_bucketize_pack(
    normalized: Any,
    boundaries: Any,
    *,
    bits: int,
    qjl_bits: int,
    code_format: str,
) -> bool:
    head_dim = int(normalized.shape[-1])
    return (
        triton is not None
        and _bucketize_pack_codes_kernel is not None
        and getattr(normalized, "is_cuda", False)
        and getattr(boundaries, "is_cuda", False)
        and qjl_bits == 0
        and code_format == "packed"
        and bits == 7
        and head_dim % _pack_group_size(bits) == 0
    )


def _triton_bucketize_pack_codes(
    normalized: Any,
    boundaries: Any,
    *,
    bits: int,
) -> Any:
    torch = _load_torch()
    if triton is None or _bucketize_pack_codes_kernel is None:
        raise RuntimeError("triton is required for fused bucketize/pack")
    if bits not in {6, 7}:
        raise ValueError("fused bucketize/pack currently supports K6 and K7")
    head_dim = int(normalized.shape[-1])
    group_size = _pack_group_size(bits)
    if head_dim % group_size != 0:
        raise ValueError("head_dim must be divisible by the fused pack group size")

    vector_count = int(normalized.numel() // head_dim)
    code_bytes = _ceil_div(head_dim * bits, 8)
    bytes_per_group = group_size * bits // 8
    output = torch.empty(
        (*normalized.shape[:-1], code_bytes),
        dtype=torch.uint8,
        device=normalized.device,
    )
    normalized_2d = normalized.reshape(vector_count, head_dim)
    output_2d = output.reshape(vector_count, code_bytes)
    groups_per_vector = head_dim // group_size
    _bucketize_pack_codes_kernel[(groups_per_vector, vector_count)](
        normalized_2d,
        boundaries,
        output_2d,
        HEAD_DIM=head_dim,
        BITS=bits,
        GROUP_SIZE=group_size,
        BYTES_PER_GROUP=bytes_per_group,
        CODE_BYTES=code_bytes,
        BOUNDARY_COUNT=(1 << bits) - 1,
        num_warps=1,
    )
    return output


def _pack_group_size(bits: int) -> int:
    if bits == 6:
        return 8
    if bits == 7:
        return 16
    raise ValueError("unsupported fused pack bit width")


def _unpack_bits(packed: Any, *, bits: int, value_count: int) -> Any:
    torch = _load_torch()
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in the range 1..8")
    values = torch.zeros(
        (*packed.shape[:-1], value_count),
        dtype=torch.int64,
        device=packed.device,
    )
    packed_int = packed.to(dtype=torch.int64)
    for index in range(value_count):
        bit_position = index * bits
        byte_index = bit_position // 8
        bit_offset = bit_position % 8
        produced = 0
        remaining = bits
        while remaining > 0:
            take = min(8 - bit_offset, remaining)
            mask = (1 << take) - 1
            chunk = (packed_int[..., byte_index] >> bit_offset) & mask
            values[..., index] = values[..., index] | (chunk << produced)
            produced += take
            remaining -= take
            byte_index += 1
            bit_offset = 0
    return values


def _timing_span(
    timing_recorder: Any | None,
    phase: str,
    tensor: Any,
    module: str | None,
    step_state: Any | None,
    metadata: dict[str, Any],
) -> Any:
    if timing_recorder is None:
        return _NoTimingSpan()
    return timing_recorder.span(
        phase,
        module=module,
        step_state=step_state,
        tensor=tensor,
        metadata=metadata,
    )


class _NoTimingSpan:
    def __enter__(self) -> "_NoTimingSpan":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        return None


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch is required for packed key blocks; install with "
            "`uv sync --extra dev --extra diffusers`"
        ) from exc
    return torch


if triton is not None and tl is not None:

    @triton.jit
    def _bucketize_pack_codes_kernel(
        normalized_ptr,
        boundaries_ptr,
        output_ptr,
        HEAD_DIM: tl.constexpr,
        BITS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BYTES_PER_GROUP: tl.constexpr,
        CODE_BYTES: tl.constexpr,
        BOUNDARY_COUNT: tl.constexpr,
    ):
        group_id = tl.program_id(0)
        vector_id = tl.program_id(1)
        lanes = tl.arange(0, GROUP_SIZE)
        dim_offsets = group_id * GROUP_SIZE + lanes
        values = tl.load(
            normalized_ptr + vector_id * HEAD_DIM + dim_offsets,
            mask=dim_offsets < HEAD_DIM,
            other=0.0,
        )

        low = tl.full((GROUP_SIZE,), 0, dtype=tl.int32)
        high = tl.full((GROUP_SIZE,), BOUNDARY_COUNT, dtype=tl.int32)
        for _ in range(BITS):
            mid = (low + high) // 2
            boundary = tl.load(
                boundaries_ptr + mid,
                mask=mid < BOUNDARY_COUNT,
                other=float("inf"),
            )
            go_right = values > boundary
            low = tl.where(go_right, mid + 1, low)
            high = tl.where(go_right, high, mid)

        c0 = tl.sum(tl.where(lanes == 0, low, 0), axis=0)
        c1 = tl.sum(tl.where(lanes == 1, low, 0), axis=0)
        c2 = tl.sum(tl.where(lanes == 2, low, 0), axis=0)
        c3 = tl.sum(tl.where(lanes == 3, low, 0), axis=0)
        c4 = tl.sum(tl.where(lanes == 4, low, 0), axis=0)
        c5 = tl.sum(tl.where(lanes == 5, low, 0), axis=0)
        c6 = tl.sum(tl.where(lanes == 6, low, 0), axis=0)
        c7 = tl.sum(tl.where(lanes == 7, low, 0), axis=0)
        out_base = vector_id * CODE_BYTES + group_id * BYTES_PER_GROUP

        if BITS == 6:
            b0 = c0 | ((c1 & 0x03) << 6)
            b1 = (c1 >> 2) | ((c2 & 0x0F) << 4)
            b2 = (c2 >> 4) | (c3 << 2)
            b3 = c4 | ((c5 & 0x03) << 6)
            b4 = (c5 >> 2) | ((c6 & 0x0F) << 4)
            b5 = (c6 >> 4) | (c7 << 2)
            tl.store(output_ptr + out_base + 0, b0.to(tl.uint8))
            tl.store(output_ptr + out_base + 1, b1.to(tl.uint8))
            tl.store(output_ptr + out_base + 2, b2.to(tl.uint8))
            tl.store(output_ptr + out_base + 3, b3.to(tl.uint8))
            tl.store(output_ptr + out_base + 4, b4.to(tl.uint8))
            tl.store(output_ptr + out_base + 5, b5.to(tl.uint8))
        else:
            c8 = tl.sum(tl.where(lanes == 8, low, 0), axis=0)
            c9 = tl.sum(tl.where(lanes == 9, low, 0), axis=0)
            c10 = tl.sum(tl.where(lanes == 10, low, 0), axis=0)
            c11 = tl.sum(tl.where(lanes == 11, low, 0), axis=0)
            c12 = tl.sum(tl.where(lanes == 12, low, 0), axis=0)
            c13 = tl.sum(tl.where(lanes == 13, low, 0), axis=0)
            c14 = tl.sum(tl.where(lanes == 14, low, 0), axis=0)
            c15 = tl.sum(tl.where(lanes == 15, low, 0), axis=0)
            b0 = c0 | ((c1 & 0x01) << 7)
            b1 = (c1 >> 1) | ((c2 & 0x03) << 6)
            b2 = (c2 >> 2) | ((c3 & 0x07) << 5)
            b3 = (c3 >> 3) | ((c4 & 0x0F) << 4)
            b4 = (c4 >> 4) | ((c5 & 0x1F) << 3)
            b5 = (c5 >> 5) | ((c6 & 0x3F) << 2)
            b6 = (c6 >> 6) | (c7 << 1)
            b7 = c8 | ((c9 & 0x01) << 7)
            b8 = (c9 >> 1) | ((c10 & 0x03) << 6)
            b9 = (c10 >> 2) | ((c11 & 0x07) << 5)
            b10 = (c11 >> 3) | ((c12 & 0x0F) << 4)
            b11 = (c12 >> 4) | ((c13 & 0x1F) << 3)
            b12 = (c13 >> 5) | ((c14 & 0x3F) << 2)
            b13 = (c14 >> 6) | (c15 << 1)
            tl.store(output_ptr + out_base + 0, b0.to(tl.uint8))
            tl.store(output_ptr + out_base + 1, b1.to(tl.uint8))
            tl.store(output_ptr + out_base + 2, b2.to(tl.uint8))
            tl.store(output_ptr + out_base + 3, b3.to(tl.uint8))
            tl.store(output_ptr + out_base + 4, b4.to(tl.uint8))
            tl.store(output_ptr + out_base + 5, b5.to(tl.uint8))
            tl.store(output_ptr + out_base + 6, b6.to(tl.uint8))
            tl.store(output_ptr + out_base + 7, b7.to(tl.uint8))
            tl.store(output_ptr + out_base + 8, b8.to(tl.uint8))
            tl.store(output_ptr + out_base + 9, b9.to(tl.uint8))
            tl.store(output_ptr + out_base + 10, b10.to(tl.uint8))
            tl.store(output_ptr + out_base + 11, b11.to(tl.uint8))
            tl.store(output_ptr + out_base + 12, b12.to(tl.uint8))
            tl.store(output_ptr + out_base + 13, b13.to(tl.uint8))

else:  # pragma: no cover
    _bucketize_pack_codes_kernel = None
