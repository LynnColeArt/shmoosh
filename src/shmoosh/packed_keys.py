from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np

from shmoosh.quantization import EncodedVectors, ShmooshCodec


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

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (*tuple(int(size) for size in self.norms.shape), self.head_dim)

    @property
    def vector_count(self) -> int:
        batch, heads, tokens, _dim = self.shape
        return batch * heads * tokens

    @property
    def code_bytes_per_vector(self) -> int:
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
        indices = _unpack_bits(self.codes, bits=self.bits, value_count=self.head_dim)
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
) -> PackedKeyBlock:
    torch = _load_torch()
    if keys.ndim != 4:
        raise ValueError("keys must have shape (batch, heads, tokens, head_dim)")
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in the range 1..8 for packed key blocks")
    if qjl_bits < 0:
        raise ValueError("qjl_bits must be non-negative")

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
        )

    encoded = codec.encode(
        keys.detach()
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    indices = torch.from_numpy(encoded.indices.astype(np.int64))
    codes = _pack_bits(indices, bits=bits).to(device=device)
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
) -> PackedKeyBlock:
    torch = _load_torch()
    device = keys.device
    _batch, _heads, _tokens, head_dim = (int(size) for size in keys.shape)
    rotation = resources.rotation.to(device=device, dtype=torch.float32)
    codebook = resources.codebook.to(device=device, dtype=torch.float32)
    if rotation.shape != (head_dim, head_dim):
        raise ValueError("score resources rotation does not match key head dimension")
    if codebook.numel() != (1 << bits):
        raise ValueError("score resources codebook does not match key bit depth")

    timing_metadata = {
        "bits": bits,
        "qjl_bits": qjl_bits,
        "head_dim": head_dim,
        "key_tokens": int(keys.shape[2]),
        "heads": int(keys.shape[1]),
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
        norms = torch.linalg.vector_norm(keys_f, dim=-1)
        safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        unit = keys_f / safe_norms.unsqueeze(-1)
    with _timing_span(
        timing_recorder,
        "encode_rotate_bucketize",
        keys,
        timing_module,
        step_state,
        timing_metadata,
    ):
        rotated = torch.matmul(unit, rotation.T)
        normalized = rotated * sqrt(head_dim)
        boundaries = ((codebook[:-1] + codebook[1:]) * 0.5).contiguous()
        indices = torch.bucketize(
            normalized.contiguous(),
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
        codes = _pack_bits(indices, bits=bits, validate=False)

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
            residual = keys_f - decoded_unit * norms.unsqueeze(-1)
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
    )


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
