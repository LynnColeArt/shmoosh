from __future__ import annotations

from dataclasses import dataclass
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
    codec = ShmooshCodec(
        dim=head_dim,
        bits=bits,
        qjl_bits=qjl_bits,
        seed=seed,
        codebook_samples=codebook_samples,
        lloyd_iters=lloyd_iters,
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


def _pack_bits(values: Any, *, bits: int) -> Any:
    torch = _load_torch()
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in the range 1..8")
    values = values.to(dtype=torch.int64)
    if values.numel() and (values.min() < 0 or values.max() >= (1 << bits)):
        raise ValueError("value is out of range for requested bit width")

    value_count = int(values.shape[-1])
    packed = torch.zeros(
        (*values.shape[:-1], _ceil_div(value_count * bits, 8)),
        dtype=torch.uint8,
        device=values.device,
    )
    for index in range(value_count):
        value = values[..., index]
        bit_position = index * bits
        byte_index = bit_position // 8
        bit_offset = bit_position % 8
        consumed = 0
        remaining = bits
        while remaining > 0:
            take = min(8 - bit_offset, remaining)
            mask = (1 << take) - 1
            chunk = ((value >> consumed) & mask) << bit_offset
            packed[..., byte_index] = packed[..., byte_index] | chunk.to(torch.uint8)
            consumed += take
            remaining -= take
            byte_index += 1
            bit_offset = 0
    return packed


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
