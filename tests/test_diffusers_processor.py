from __future__ import annotations

import pytest

from shmoosh.diffusers_processor import (
    ShmooshAttnProcessor,
    warm_packed_attention_processor,
)

torch = pytest.importorskip("torch")


class _Identity:
    out_features = 16

    def __call__(self, value):
        return value


class _FakeAttention:
    def __init__(self, *, heads: int) -> None:
        self.heads = heads
        self.spatial_norm = None
        self.group_norm = None
        self.norm_cross = False
        self.norm_q = None
        self.norm_k = None
        self.residual_connection = False
        self.rescale_output_factor = 1.0
        self.to_q = _Identity()
        self.to_k = _Identity()
        self.to_v = _Identity()
        self.to_out = [_Identity(), _Identity()]


def test_packed_processor_matches_reference_for_k_only_policy() -> None:
    generator = torch.Generator().manual_seed(0)
    hidden_states = torch.randn(1, 5, 16, generator=generator)
    attn = _FakeAttention(heads=2)
    reference = ShmooshAttnProcessor(
        bits=4,
        qjl_bits=16,
        seed=3,
        quantize_values=False,
        codebook_samples=2_000,
    )
    packed = ShmooshAttnProcessor(
        bits=4,
        qjl_bits=16,
        seed=3,
        quantize_values=False,
        codebook_samples=2_000,
        attention_backend="packed",
        packed_backend="torch",
    )

    reference_output = reference(attn, hidden_states)
    packed_output = packed(attn, hidden_states)

    assert packed._use_packed_attention() is True
    assert packed_output.shape == hidden_states.shape
    assert torch.allclose(packed_output, reference_output, atol=2e-5, rtol=1e-5)
    assert len(packed._packed_codec_cache) == 1
    assert len(packed._packed_resource_cache) == 1


def test_packed_processor_falls_back_when_values_are_quantized() -> None:
    hidden_states = torch.zeros(1, 5, 16)
    attn = _FakeAttention(heads=2)
    processor = ShmooshAttnProcessor(
        bits=4,
        qjl_bits=16,
        seed=3,
        quantize_values=True,
        codebook_samples=512,
        attention_backend="packed",
        packed_backend="triton",
    )

    output = processor(attn, hidden_states)

    assert processor._use_packed_attention() is False
    assert output.shape == hidden_states.shape
    assert processor._packed_codec_cache == {}


def test_warm_packed_attention_processor_populates_cache() -> None:
    attn = _FakeAttention(heads=2)
    processor = ShmooshAttnProcessor(
        bits=4,
        qjl_bits=16,
        seed=3,
        quantize_values=False,
        codebook_samples=512,
        attention_backend="packed",
        packed_backend="torch",
    )

    warmed = warm_packed_attention_processor(
        attn,
        processor,
        device="cpu",
        dtype=torch.float32,
    )

    assert warmed is True
    assert len(processor._packed_codec_cache) == 1
    assert len(processor._packed_resource_cache) == 1


def test_processor_validates_attention_backend() -> None:
    with pytest.raises(ValueError, match="attention_backend"):
        ShmooshAttnProcessor(attention_backend="turbo")


def test_processor_validates_packed_backend() -> None:
    with pytest.raises(ValueError, match="packed_backend"):
        ShmooshAttnProcessor(packed_backend="cuda-but-magic")
