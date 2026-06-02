from __future__ import annotations

import pytest

from math import sqrt

from shmoosh.packed_keys import (
    _pack_bits,
    _triton_norm_rotate_bucketize_pack_keys,
    _triton_rotate_bucketize_pack_codes,
    _triton_bucketize_pack_codes,
    _unpack_bits,
    encode_packed_keys,
)
from shmoosh.packed_scores import score_resources_from_codec
from shmoosh.packed_scores import triton
from shmoosh.quantization import ShmooshCodec

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("bits", [1, 3, 5, 6, 8])
def test_pack_bits_round_trips(bits: int) -> None:
    values = torch.arange(0, 17, dtype=torch.int64) % (1 << bits)

    packed = _pack_bits(values, bits=bits)
    unpacked = _unpack_bits(packed, bits=bits, value_count=values.shape[-1])

    assert torch.equal(unpacked, values)


@pytest.mark.parametrize("bits", [1, 4, 5, 6, 7, 8])
def test_fast_pack_bits_round_trips_sdxl_width(bits: int) -> None:
    values = torch.arange(0, 64, dtype=torch.int64).reshape(2, 4, 8)
    values = values % (1 << bits)

    packed = _pack_bits(values, bits=bits)
    unpacked = _unpack_bits(packed, bits=bits, value_count=values.shape[-1])

    assert torch.equal(unpacked, values)


def test_encode_packed_keys_matches_reference_decode() -> None:
    generator = torch.Generator().manual_seed(0)
    keys = torch.randn(1, 2, 5, 8, generator=generator)
    block = encode_packed_keys(
        keys,
        bits=4,
        qjl_bits=16,
        seed=3,
        codebook_samples=2_000,
    )
    codec = ShmooshCodec(dim=8, bits=4, qjl_bits=16, seed=3, codebook_samples=2_000)
    reference = torch.from_numpy(codec.decode(codec.encode(keys.numpy())))

    assert block.codes.shape == (1, 2, 5, 4)
    assert block.residual_signs is not None
    assert block.residual_signs.shape == (1, 2, 5, 2)
    assert block.shape == (1, 2, 5, 8)
    assert block.packed_key_bytes() == 1 * 2 * 5 * (4 + 4 + 2 + 4)
    assert torch.allclose(block.decode(), reference)


def test_encode_packed_keys_without_qjl_has_no_residual_payload() -> None:
    keys = torch.zeros(1, 2, 5, 8)

    block = encode_packed_keys(keys, bits=4, qjl_bits=0, seed=3, codebook_samples=512)

    assert block.residual_signs is None
    assert block.residual_norms is None
    assert block.packed_key_bytes() == 1 * 2 * 5 * (4 + 4)


def test_encode_packed_keys_byte_format_matches_reference_decode() -> None:
    generator = torch.Generator().manual_seed(4)
    keys = torch.randn(1, 2, 5, 8, generator=generator)
    block = encode_packed_keys(
        keys,
        bits=4,
        qjl_bits=0,
        seed=3,
        codebook_samples=512,
        code_format="byte",
    )
    codec = ShmooshCodec(dim=8, bits=4, qjl_bits=0, seed=3, codebook_samples=512)
    reference = torch.from_numpy(codec.decode(codec.encode(keys.numpy())))

    assert block.code_format == "byte"
    assert block.codes.shape == (1, 2, 5, 8)
    assert block.code_bytes_per_vector == 8
    assert block.packed_bytes_per_vector == 8 + 4
    assert block.packed_key_bytes() == 1 * 2 * 5 * (8 + 4)
    assert torch.allclose(block.decode(), reference)


def test_encode_packed_keys_transposed_format_matches_reference_decode() -> None:
    generator = torch.Generator().manual_seed(11)
    keys = torch.randn(1, 2, 5, 8, generator=generator)
    block = encode_packed_keys(
        keys,
        bits=4,
        qjl_bits=0,
        seed=3,
        codebook_samples=512,
        code_format="packed_t",
    )
    codec = ShmooshCodec(dim=8, bits=4, qjl_bits=0, seed=3, codebook_samples=512)
    reference = torch.from_numpy(codec.decode(codec.encode(keys.numpy())))

    assert block.code_format == "packed_t"
    assert block.codes.shape == (1, 2, 4, 5)
    assert block.code_bytes_per_vector == 4
    assert block.packed_bytes_per_vector == 4 + 4
    assert block.packed_key_bytes() == 1 * 2 * 5 * (4 + 4)
    assert torch.allclose(block.decode(), reference)


def test_encode_packed_keys_accepts_fp16_norms() -> None:
    generator = torch.Generator().manual_seed(12)
    keys = torch.randn(1, 2, 5, 64, generator=generator)
    block = encode_packed_keys(
        keys,
        bits=7,
        qjl_bits=0,
        seed=3,
        codebook_samples=512,
        code_format="packed_t",
        norm_dtype="fp16",
    )
    codec = ShmooshCodec(dim=64, bits=7, qjl_bits=0, seed=3, codebook_samples=512)
    reference = torch.from_numpy(codec.decode(codec.encode(keys.numpy())))

    assert block.norm_dtype == "fp16"
    assert block.norms.dtype == torch.float16
    assert block.code_bytes_per_vector == 56
    assert block.packed_bytes_per_vector == 56 + 2
    assert torch.allclose(block.decode(), reference, atol=2e-3, rtol=2e-3)


def test_encode_packed_keys_rejects_mismatched_codec() -> None:
    keys = torch.zeros(1, 2, 5, 8)
    codec = ShmooshCodec(dim=8, bits=4, qjl_bits=0, seed=3, codebook_samples=256)

    with pytest.raises(ValueError, match="codec parameters"):
        encode_packed_keys(
            keys,
            bits=4,
            qjl_bits=0,
            seed=3,
            codebook_samples=512,
            codec=codec,
        )


def test_encode_packed_keys_rejects_invalid_norm_dtype() -> None:
    keys = torch.zeros(1, 2, 5, 8)

    with pytest.raises(ValueError, match="norm_dtype"):
        encode_packed_keys(
            keys,
            bits=4,
            qjl_bits=0,
            seed=3,
            codebook_samples=512,
            norm_dtype="quarterish",
        )


def test_encode_packed_keys_with_torch_resources_matches_reference() -> None:
    generator = torch.Generator().manual_seed(8)
    keys = torch.randn(1, 2, 5, 16, generator=generator)
    codec = ShmooshCodec(dim=16, bits=4, qjl_bits=16, seed=3, codebook_samples=2_000)
    resources = score_resources_from_codec(codec, device=keys.device)

    reference = encode_packed_keys(
        keys,
        bits=4,
        qjl_bits=16,
        seed=3,
        codebook_samples=2_000,
        codec=codec,
    )
    torch_block = encode_packed_keys(
        keys,
        bits=4,
        qjl_bits=16,
        seed=3,
        codebook_samples=2_000,
        codec=codec,
        resources=resources,
    )

    assert torch.equal(torch_block.codes, reference.codes)
    assert torch.allclose(torch_block.norms, reference.norms)
    assert torch.equal(torch_block.residual_signs, reference.residual_signs)
    assert torch.allclose(torch_block.residual_norms, reference.residual_norms)


@pytest.mark.skipif(
    triton is None or not torch.cuda.is_available(),
    reason="CUDA Triton is not available",
)
@pytest.mark.parametrize("bits", [6, 7])
def test_triton_bucketize_pack_codes_matches_torch(bits: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(9)
    keys = torch.randn(
        1,
        2,
        17,
        64,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    codec = ShmooshCodec(dim=64, bits=bits, qjl_bits=0, seed=3, codebook_samples=512)
    resources = score_resources_from_codec(codec, device=keys.device)
    keys_f = keys.detach().to(dtype=torch.float32)
    norms = torch.linalg.vector_norm(keys_f, dim=-1)
    safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    normalized = (
        torch.matmul(keys_f / safe_norms.unsqueeze(-1), resources.rotation.T)
        * sqrt(64)
    ).contiguous()
    indices = torch.bucketize(normalized, resources.boundaries).to(dtype=torch.int64)
    expected = _pack_bits(indices, bits=bits)

    packed = _triton_bucketize_pack_codes(
        normalized,
        resources.boundaries,
        bits=bits,
    )

    assert torch.equal(packed, expected)
    assert torch.equal(_unpack_bits(packed, bits=bits, value_count=64), indices)


@pytest.mark.skipif(
    triton is None or not torch.cuda.is_available(),
    reason="CUDA Triton is not available",
)
@pytest.mark.parametrize("code_format", ["packed", "packed_t"])
def test_triton_norm_rotate_bucketize_pack_keys_matches_torch(
    code_format: str,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(13)
    keys = torch.randn(
        1,
        2,
        17,
        64,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    codec = ShmooshCodec(dim=64, bits=7, qjl_bits=0, seed=3, codebook_samples=512)
    resources = score_resources_from_codec(codec, device=keys.device)
    keys_f = keys.detach().to(dtype=torch.float32)
    expected_norms = torch.linalg.vector_norm(keys_f, dim=-1)
    safe_norms = torch.where(
        expected_norms > 0,
        expected_norms,
        torch.ones_like(expected_norms),
    )
    unit = keys_f / safe_norms.unsqueeze(-1)
    normalized = (torch.matmul(unit, resources.rotation.T) * sqrt(64)).contiguous()
    indices = torch.bucketize(normalized, resources.boundaries).to(dtype=torch.int64)
    expected_codes = _pack_bits(indices, bits=7)
    if code_format == "packed_t":
        expected_codes = expected_codes.transpose(-1, -2).contiguous()

    codes, norms = _triton_norm_rotate_bucketize_pack_keys(
        keys,
        resources.rotation,
        resources.boundaries,
        bits=7,
        code_format=code_format,
        norm_dtype="fp32",
    )

    assert torch.equal(codes, expected_codes)
    assert torch.allclose(norms, expected_norms)


@pytest.mark.skipif(
    triton is None or not torch.cuda.is_available(),
    reason="CUDA Triton is not available",
)
@pytest.mark.parametrize("code_format", ["packed", "packed_t"])
def test_triton_rotate_bucketize_pack_codes_matches_torch(code_format: str) -> None:
    generator = torch.Generator(device="cuda").manual_seed(10)
    keys = torch.randn(
        1,
        2,
        17,
        64,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    codec = ShmooshCodec(dim=64, bits=7, qjl_bits=0, seed=3, codebook_samples=512)
    resources = score_resources_from_codec(codec, device=keys.device)
    keys_f = keys.detach().to(dtype=torch.float32)
    norms = torch.linalg.vector_norm(keys_f, dim=-1)
    safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    unit = (keys_f / safe_norms.unsqueeze(-1)).contiguous()
    normalized = (torch.matmul(unit, resources.rotation.T) * sqrt(64)).contiguous()
    indices = torch.bucketize(normalized, resources.boundaries).to(dtype=torch.int64)
    expected = _pack_bits(indices, bits=7)
    if code_format == "packed_t":
        expected = expected.transpose(-1, -2).contiguous()

    packed = _triton_rotate_bucketize_pack_codes(
        unit,
        resources.rotation,
        resources.boundaries,
        bits=7,
        code_format=code_format,
    )

    assert torch.equal(packed, expected)
    unpackable = packed.transpose(-1, -2).contiguous() if code_format == "packed_t" else packed
    assert torch.equal(_unpack_bits(unpackable, bits=7, value_count=64), indices)


def test_packed_key_bytes_match_sdxl_assumption() -> None:
    keys = torch.zeros(2, 20, 77, 64)
    block = encode_packed_keys(
        keys,
        bits=5,
        qjl_bits=128,
        seed=11,
        codebook_samples=512,
    )

    assert block.exact_key_bytes() == 394_240
    assert block.packed_key_bytes() == 197_120
    assert block.compression_ratio() == 2.0
