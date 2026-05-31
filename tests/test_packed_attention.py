from __future__ import annotations

import numpy as np
import pytest

from shmoosh.packed_attention import (
    encode_and_attention_output,
    packed_key_attention_output,
)
from shmoosh.packed_keys import encode_packed_keys
from shmoosh.packed_scores import triton
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


@pytest.mark.skipif(
    triton is None or not torch.cuda.is_available(),
    reason="CUDA Triton is not available",
)
def test_triton_packed_key_attention_matches_torch() -> None:
    generator = torch.Generator(device="cuda").manual_seed(1)
    query = torch.randn(1, 2, 4, 8, generator=generator, device="cuda")
    key = torch.randn(1, 2, 5, 8, generator=generator, device="cuda")
    value = torch.randn(1, 2, 5, 8, generator=generator, device="cuda")
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
    assert torch.allclose(triton_output, torch_output, atol=2e-5, rtol=1e-5)


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
