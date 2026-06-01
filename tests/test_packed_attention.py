from __future__ import annotations

import numpy as np
import pytest

from shmoosh.packed_attention import (
    _select_streaming_key_tile,
    encode_and_attention_output,
    packed_key_attention_output,
    torch_packed_key_attention_output,
    triton_packed_key_attention_output,
)
from shmoosh.packed_keys import encode_packed_keys
from shmoosh.packed_scores import score_resources_from_codec, triton
from shmoosh.quantization import ShmooshCodec
from shmoosh.rotated_attention import (
    rotated_key_attention_output,
    triton_rotated_key_attention_output,
)
from shmoosh.rotated_keys import encode_rotated_keys
from shmoosh.runtime_attention import shmoosh_attention_output

torch = pytest.importorskip("torch")


def test_packed_key_attention_matches_runtime_reference() -> None:
    generator = torch.Generator().manual_seed(0)
    query = torch.randn(1, 2, 4, 8, generator=generator)
    key = torch.randn(1, 2, 5, 8, generator=generator)
    value = torch.randn(1, 2, 5, 8, generator=generator)

    output = encode_and_attention_output(
        query,
        key,
        value,
        bits=4,
        qjl_bits=16,
        seed=3,
        backend="torch",
        codebook_samples=2_000,
    )
    reference = _reference_output(
        query,
        key,
        value,
        bits=4,
        qjl_bits=16,
        seed=3,
        codebook_samples=2_000,
    )

    assert output.shape == query.shape
    assert torch.allclose(output, reference, atol=2e-5, rtol=1e-5)


def test_packed_key_attention_accepts_preencoded_block() -> None:
    query = torch.zeros(1, 2, 3, 8)
    key = torch.zeros(1, 2, 5, 8)
    value = torch.ones(1, 2, 5, 8)
    block = encode_packed_keys(key, bits=4, qjl_bits=0, seed=3, codebook_samples=512)

    output = packed_key_attention_output(query, block, value, backend="torch")

    assert output.shape == query.shape
    assert torch.allclose(output, torch.ones_like(output))


def test_packed_key_attention_accepts_byte_code_block() -> None:
    generator = torch.Generator().manual_seed(6)
    query = torch.randn(1, 2, 4, 8, generator=generator)
    key = torch.randn(1, 2, 5, 8, generator=generator)
    value = torch.randn(1, 2, 5, 8, generator=generator)
    byte_block = encode_packed_keys(
        key,
        bits=4,
        qjl_bits=0,
        seed=3,
        codebook_samples=512,
        code_format="byte",
    )
    packed_block = encode_packed_keys(
        key,
        bits=4,
        qjl_bits=0,
        seed=3,
        codebook_samples=512,
    )

    byte_output = packed_key_attention_output(query, byte_block, value, backend="torch")
    packed_output = packed_key_attention_output(
        query,
        packed_block,
        value,
        backend="torch",
    )

    assert byte_output.shape == query.shape
    assert torch.allclose(byte_output, packed_output, atol=2e-5, rtol=1e-5)


def test_rotated_key_attention_matches_exact_reference() -> None:
    generator = torch.Generator().manual_seed(8)
    query = torch.randn(1, 2, 4, 8, generator=generator)
    key = torch.randn(1, 2, 5, 8, generator=generator)
    value = torch.randn(1, 2, 5, 8, generator=generator)
    codec = ShmooshCodec(dim=8, bits=4, qjl_bits=0, seed=3, codebook_samples=512)
    resources = score_resources_from_codec(codec, device=query.device)
    block = encode_rotated_keys(
        key,
        resources=resources,
        seed=3,
        storage_dtype=torch.float32,
    )

    output = rotated_key_attention_output(
        query,
        block,
        value,
        resources=resources,
        backend="torch",
    )
    reference = _exact_torch_attention(query, key, value)

    assert output.shape == query.shape
    assert torch.allclose(output, reference, atol=2e-5, rtol=1e-5)


def test_auto_streaming_key_tile_uses_wider_no_qjl_default() -> None:
    assert _select_streaming_key_tile(128, 0) == 32
    assert _select_streaming_key_tile(128, 64) == 16
    assert _select_streaming_key_tile(64, 0) == 64


@pytest.mark.skipif(
    triton is None or not torch.cuda.is_available(),
    reason="CUDA Triton is not available",
)
def test_triton_packed_key_attention_matches_torch() -> None:
    generator = torch.Generator(device="cuda").manual_seed(1)
    query = torch.randn(
        1, 2, 4, 16, generator=generator, device="cuda", dtype=torch.float16
    )
    key = torch.randn(
        1, 2, 5, 16, generator=generator, device="cuda", dtype=torch.float16
    )
    value = torch.randn(
        1, 2, 5, 16, generator=generator, device="cuda", dtype=torch.float16
    )
    block = encode_packed_keys(
        key,
        bits=4,
        qjl_bits=16,
        seed=5,
        codebook_samples=2_000,
    )

    triton_output = packed_key_attention_output(
        query,
        block,
        value,
        backend="triton",
    )
    torch_output = packed_key_attention_output(query, block, value, backend="torch")

    assert triton_output.shape == torch_output.shape
    assert torch.allclose(triton_output, torch_output, atol=5e-4, rtol=5e-4)


@pytest.mark.skipif(
    triton is None or not torch.cuda.is_available(),
    reason="CUDA Triton is not available",
)
def test_fused_triton_attention_matches_torch_across_key_tiles() -> None:
    generator = torch.Generator(device="cuda").manual_seed(2)
    query = torch.randn(
        1, 2, 7, 16, generator=generator, device="cuda", dtype=torch.float16
    )
    key = torch.randn(
        1, 2, 33, 16, generator=generator, device="cuda", dtype=torch.float16
    )
    value = torch.randn(
        1, 2, 33, 16, generator=generator, device="cuda", dtype=torch.float16
    )
    block = encode_packed_keys(
        key,
        bits=4,
        qjl_bits=16,
        seed=5,
        codebook_samples=2_000,
    )

    triton_output = triton_packed_key_attention_output(
        query,
        block,
        value,
        block_k=16,
    )
    torch_output = packed_key_attention_output(query, block, value, backend="torch")

    assert triton_output.shape == torch_output.shape
    assert torch.allclose(triton_output, torch_output, atol=5e-4, rtol=5e-4)


@pytest.mark.skipif(
    triton is None or not torch.cuda.is_available(),
    reason="CUDA Triton is not available",
)
def test_fused_triton_attention_matches_torch_with_byte_codes() -> None:
    generator = torch.Generator(device="cuda").manual_seed(7)
    query = torch.randn(
        1, 2, 7, 16, generator=generator, device="cuda", dtype=torch.float16
    )
    key = torch.randn(
        1, 2, 33, 16, generator=generator, device="cuda", dtype=torch.float16
    )
    value = torch.randn(
        1, 2, 33, 16, generator=generator, device="cuda", dtype=torch.float16
    )
    block = encode_packed_keys(
        key,
        bits=4,
        qjl_bits=0,
        seed=5,
        codebook_samples=512,
        code_format="byte",
    )

    triton_output = triton_packed_key_attention_output(
        query,
        block,
        value,
        block_k=16,
    )
    torch_output = packed_key_attention_output(query, block, value, backend="torch")

    assert triton_output.shape == torch_output.shape
    assert torch.allclose(triton_output, torch_output, atol=5e-4, rtol=5e-4)


@pytest.mark.skipif(
    triton is None or not torch.cuda.is_available(),
    reason="CUDA Triton is not available",
)
def test_triton_rotated_key_attention_matches_torch() -> None:
    generator = torch.Generator(device="cuda").manual_seed(9)
    query = torch.randn(
        1, 2, 7, 16, generator=generator, device="cuda", dtype=torch.float16
    )
    key = torch.randn(
        1, 2, 33, 16, generator=generator, device="cuda", dtype=torch.float16
    )
    value = torch.randn(
        1, 2, 33, 16, generator=generator, device="cuda", dtype=torch.float16
    )
    codec = ShmooshCodec(dim=16, bits=4, qjl_bits=0, seed=5, codebook_samples=512)
    resources = score_resources_from_codec(codec, device=query.device)
    block = encode_rotated_keys(key, resources=resources, seed=5)

    triton_output = triton_rotated_key_attention_output(
        query,
        block,
        value,
        resources=resources,
        block_k=16,
    )
    torch_output = rotated_key_attention_output(
        query,
        block,
        value,
        resources=resources,
        backend="torch",
    )

    assert triton_output.shape == torch_output.shape
    assert torch.allclose(triton_output, torch_output, atol=5e-4, rtol=5e-4)


@pytest.mark.skipif(
    triton is None or not torch.cuda.is_available(),
    reason="CUDA Triton is not available",
)
def test_auto_uses_fused_triton_attention_across_key_tiles(monkeypatch) -> None:
    generator = torch.Generator(device="cuda").manual_seed(4)
    query = torch.randn(
        1, 2, 7, 16, generator=generator, device="cuda", dtype=torch.float16
    )
    key = torch.randn(
        1, 2, 129, 16, generator=generator, device="cuda", dtype=torch.float16
    )
    value = torch.randn(
        1, 2, 129, 16, generator=generator, device="cuda", dtype=torch.float16
    )
    block = encode_packed_keys(
        key,
        bits=4,
        qjl_bits=16,
        seed=5,
        codebook_samples=2_000,
    )
    reference = torch_packed_key_attention_output(query, block, value)

    def _fail_materialized(*_args, **_kwargs):
        raise AssertionError("materialized fallback should not be used")

    monkeypatch.setattr(
        "shmoosh.packed_attention.torch_packed_key_attention_output",
        _fail_materialized,
    )

    auto_output = packed_key_attention_output(query, block, value, backend="auto")

    assert auto_output.shape == reference.shape
    assert torch.allclose(auto_output, reference, atol=5e-4, rtol=5e-4)


@pytest.mark.skipif(
    triton is None or not torch.cuda.is_available(),
    reason="CUDA Triton is not available",
)
def test_fused_triton_attention_rejects_invalid_tile_size() -> None:
    query = torch.zeros(1, 2, 4, 16, device="cuda")
    key = torch.zeros(1, 2, 6, 16, device="cuda")
    value = torch.zeros(1, 2, 6, 16, device="cuda")
    block = encode_packed_keys(key, bits=4, qjl_bits=0, seed=5, codebook_samples=512)

    with pytest.raises(ValueError, match="tile size"):
        triton_packed_key_attention_output(query, block, value, block_k=4)


def _reference_output(query, key, value, *, bits, qjl_bits, seed, codebook_samples):
    batch, heads, query_tokens, dim = query.shape
    key_tokens = int(key.shape[2])
    reference = shmoosh_attention_output(
        query.detach()
        .to(device="cpu", dtype=torch.float32)
        .reshape(batch * heads, query_tokens, dim)
        .numpy(),
        key.detach()
        .to(device="cpu", dtype=torch.float32)
        .reshape(batch * heads, key_tokens, dim)
        .numpy(),
        value.detach()
        .to(device="cpu", dtype=torch.float32)
        .reshape(batch * heads, key_tokens, dim)
        .numpy(),
        bits=bits,
        qjl_bits=qjl_bits,
        seed=seed,
        quantize_values=False,
        codebook_samples=codebook_samples,
    )
    return torch.from_numpy(np.asarray(reference)).reshape(
        batch,
        heads,
        query_tokens,
        dim,
    )


def _exact_torch_attention(query, key, value):
    scores = torch.matmul(
        query.to(dtype=torch.float32),
        key.to(dtype=torch.float32).transpose(-2, -1),
    ) / np.sqrt(int(query.shape[-1]))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value.to(dtype=torch.float32)).to(dtype=query.dtype)
