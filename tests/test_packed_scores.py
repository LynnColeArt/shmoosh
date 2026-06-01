from __future__ import annotations

import numpy as np
import pytest

from shmoosh.packed_keys import encode_packed_keys
from shmoosh.packed_scores import (
    build_score_resources,
    packed_key_scores,
    score_resources_from_codec,
    torch_packed_key_scores,
    triton,
    triton_packed_key_scores,
)
from shmoosh.quantization import EncodedVectors, ShmooshCodec

torch = pytest.importorskip("torch")


def test_torch_packed_key_scores_match_codec_estimate_dot() -> None:
    generator = torch.Generator().manual_seed(0)
    query = torch.randn(1, 2, 4, 8, generator=generator)
    key = torch.randn(1, 2, 5, 8, generator=generator)
    block = encode_packed_keys(
        key,
        bits=4,
        qjl_bits=16,
        seed=3,
        codebook_samples=2_000,
    )

    scores = torch_packed_key_scores(query, block)
    reference = _reference_scores(query, block)

    assert scores.shape == (1, 2, 4, 5)
    assert torch.allclose(scores, reference, atol=2e-5, rtol=1e-5)


def test_packed_key_scores_auto_uses_torch_on_cpu() -> None:
    query = torch.zeros(1, 2, 3, 8)
    key = torch.zeros(1, 2, 5, 8)
    block = encode_packed_keys(key, bits=4, qjl_bits=0, seed=3, codebook_samples=512)

    scores = packed_key_scores(query, block, backend="auto")

    assert scores.shape == (1, 2, 3, 5)
    assert torch.count_nonzero(scores) == 0


def test_build_score_resources_rejects_mismatched_codec() -> None:
    key = torch.zeros(1, 2, 5, 8)
    block = encode_packed_keys(key, bits=4, qjl_bits=0, seed=3, codebook_samples=512)
    codec = ShmooshCodec(dim=8, bits=4, qjl_bits=0, seed=3, codebook_samples=256)

    with pytest.raises(ValueError, match="codec parameters"):
        build_score_resources(block, codec=codec)


def test_score_resources_include_codebook_boundaries() -> None:
    codec = ShmooshCodec(dim=8, bits=4, qjl_bits=0, seed=3, codebook_samples=512)

    resources = score_resources_from_codec(codec, device="cpu")

    assert resources.boundaries.shape == (15,)
    assert torch.allclose(
        resources.boundaries,
        (resources.codebook[:-1] + resources.codebook[1:]) * 0.5,
    )


@pytest.mark.skipif(
    triton is None or not torch.cuda.is_available(),
    reason="CUDA Triton is not available",
)
def test_triton_packed_key_scores_match_torch() -> None:
    generator = torch.Generator(device="cuda").manual_seed(1)
    query = torch.randn(1, 2, 4, 8, generator=generator, device="cuda")
    key = torch.randn(1, 2, 5, 8, generator=generator, device="cuda")
    block = encode_packed_keys(
        key,
        bits=4,
        qjl_bits=16,
        seed=5,
        codebook_samples=2_000,
    )

    triton_scores = triton_packed_key_scores(query, block)
    torch_scores = torch_packed_key_scores(query, block)

    assert triton_scores.shape == torch_scores.shape
    assert torch.allclose(triton_scores, torch_scores, atol=2e-5, rtol=1e-5)


def _reference_scores(query, block):
    codec = ShmooshCodec(
        dim=block.head_dim,
        bits=block.bits,
        qjl_bits=block.qjl_bits,
        seed=block.seed,
        codebook_samples=block.codebook_samples,
        lloyd_iters=block.lloyd_iters,
    )
    encoded = block.to_encoded_vectors()
    query_np = query.detach().to(device="cpu", dtype=torch.float32).numpy()
    output = np.empty(query.shape[:-1] + (block.shape[2],), dtype=np.float32)
    for batch in range(query.shape[0]):
        for head in range(query.shape[1]):
            residual_signs = None
            if encoded.residual_signs is not None:
                residual_signs = encoded.residual_signs.reshape(
                    block.shape[:-1] + (block.qjl_bits,)
                )[batch, head]
            residual_norms = None
            if encoded.residual_norms is not None:
                residual_norms = encoded.residual_norms.reshape(block.shape[:-1])[
                    batch, head
                ]
            key_slice = EncodedVectors(
                indices=encoded.indices[batch, head],
                norms=encoded.norms[batch, head],
                original_shape=encoded.original_shape[2:],
                residual_signs=residual_signs,
                residual_norms=residual_norms,
            )
            output[batch, head] = codec.estimate_dot(query_np[batch, head], key_slice)
    return torch.from_numpy(output).to(device=query.device, dtype=torch.float32)
