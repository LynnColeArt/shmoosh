from __future__ import annotations

import pytest

from shmoosh.packed_keys import _pack_bits, _unpack_bits, encode_packed_keys
from shmoosh.packed_scores import score_resources_from_codec
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
