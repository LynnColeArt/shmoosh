from __future__ import annotations

import pytest

from shmoosh.diffusers_processor import (
    DenoisingStepState,
    ScheduledShmooshAttnProcessor,
    ShmooshAttnProcessor,
    ShmooshTimingRecorder,
    warm_packed_attention_processor,
)

torch = pytest.importorskip("torch")


class _Identity:
    out_features = 16

    def __call__(self, value):
        return value


class _FakeProcessor:
    def __init__(self, label: str) -> None:
        self.label = label

    def __call__(self, *_args, **_kwargs) -> str:
        return self.label


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


def test_byte_code_packed_processor_matches_reference_for_k_only_policy() -> None:
    generator = torch.Generator().manual_seed(10)
    hidden_states = torch.randn(1, 5, 16, generator=generator)
    attn = _FakeAttention(heads=2)
    reference = ShmooshAttnProcessor(
        bits=4,
        qjl_bits=0,
        seed=3,
        quantize_values=False,
        codebook_samples=512,
    )
    packed = ShmooshAttnProcessor(
        bits=4,
        qjl_bits=0,
        seed=3,
        quantize_values=False,
        codebook_samples=512,
        attention_backend="packed",
        packed_backend="torch",
        code_format="byte",
    )

    reference_output = reference(attn, hidden_states)
    packed_output = packed(attn, hidden_states)

    assert packed_output.shape == hidden_states.shape
    assert torch.allclose(packed_output, reference_output, atol=2e-5, rtol=1e-5)


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


def test_packed_processor_records_timing_phases() -> None:
    generator = torch.Generator().manual_seed(1)
    hidden_states = torch.randn(1, 5, 16, generator=generator)
    attn = _FakeAttention(heads=2)
    recorder = ShmooshTimingRecorder()
    processor = ShmooshAttnProcessor(
        bits=4,
        qjl_bits=16,
        seed=3,
        quantize_values=False,
        codebook_samples=2_000,
        attention_backend="packed",
        packed_backend="torch",
        timing_recorder=recorder,
        timing_module="fake.attn",
        step_state=DenoisingStepState(current_step=4, total_steps=20),
    )

    output = processor(attn, hidden_states)

    assert output.shape == hidden_states.shape
    phases = [record["phase"] for record in recorder.records]
    assert "packed_encode" in phases
    assert "packed_attention" in phases
    assert "encode_rotate_bucketize" in phases
    assert "encode_residual_project" in phases
    assert {record["module"] for record in recorder.records} == {"fake.attn"}
    assert {record["step"] for record in recorder.records} == {4}
    assert all(record["seconds"] >= 0 for record in recorder.records)
    payload = recorder.payload()
    assert payload["record_count"] == len(recorder.records)
    assert {row["phase"] for row in payload["summary"]["by_phase"]} == set(phases)


def test_packed_processor_caches_cross_attention_keys() -> None:
    generator = torch.Generator().manual_seed(2)
    hidden_states = torch.randn(1, 5, 16, generator=generator)
    encoder_hidden_states = torch.randn(1, 7, 16, generator=generator)
    attn = _FakeAttention(heads=2)
    recorder = ShmooshTimingRecorder()
    processor = ShmooshAttnProcessor(
        bits=4,
        qjl_bits=16,
        seed=3,
        quantize_values=False,
        codebook_samples=2_000,
        attention_backend="packed",
        packed_backend="torch",
        cache_cross_attention=True,
        timing_recorder=recorder,
        timing_module="fake.attn2",
        step_state=DenoisingStepState(current_step=6, total_steps=20),
    )

    first = processor(attn, hidden_states, encoder_hidden_states=encoder_hidden_states)
    second = processor(attn, hidden_states, encoder_hidden_states=encoder_hidden_states)

    assert first.shape == hidden_states.shape
    assert second.shape == hidden_states.shape
    assert len(processor._cross_attention_cache) == 1
    cache_records = [
        record for record in recorder.records if record["phase"] == "cross_kv_cache"
    ]
    assert [record["hit"] for record in cache_records] == [False, True]
    assert sum(record["phase"] == "packed_encode" for record in recorder.records) == 1


def test_scheduled_processor_records_dispatch_and_branch_timing() -> None:
    recorder = ShmooshTimingRecorder()
    step_state = DenoisingStepState(current_step=0, total_steps=20)
    processor = ScheduledShmooshAttnProcessor(
        original_processor=_FakeProcessor("exact"),
        shmoosh_processor=_FakeProcessor("shmoosh"),
        step_state=step_state,
        quantize_start_step=4,
        timing_recorder=recorder,
        timing_module="fake.attn",
    )

    assert processor(None, None) == "exact"
    step_state.current_step = 4
    assert processor(None, None) == "shmoosh"

    phases = [record["phase"] for record in recorder.records]
    assert phases == [
        "policy_dispatch",
        "scheduled_exact",
        "policy_dispatch",
        "scheduled_quantized",
    ]
    assert [record["quantized"] for record in recorder.records] == [
        False,
        False,
        True,
        True,
    ]


def test_processor_validates_attention_backend() -> None:
    with pytest.raises(ValueError, match="attention_backend"):
        ShmooshAttnProcessor(attention_backend="turbo")


def test_processor_validates_packed_backend() -> None:
    with pytest.raises(ValueError, match="packed_backend"):
        ShmooshAttnProcessor(packed_backend="cuda-but-magic")


def test_processor_validates_packed_block_tiles() -> None:
    ShmooshAttnProcessor(packed_block_q=64, packed_block_k=32)
    with pytest.raises(ValueError, match="packed_block_q"):
        ShmooshAttnProcessor(packed_block_q=12)
    with pytest.raises(ValueError, match="packed_block_k"):
        ShmooshAttnProcessor(packed_block_k=24)


def test_processor_validates_code_format() -> None:
    with pytest.raises(ValueError, match="code_format"):
        ShmooshAttnProcessor(code_format="wide-open")


def test_processor_validates_norm_dtype() -> None:
    with pytest.raises(ValueError, match="norm_dtype"):
        ShmooshAttnProcessor(norm_dtype="neon")


def test_processor_validates_key_encode_backend() -> None:
    with pytest.raises(ValueError, match="key_encode_backend"):
        ShmooshAttnProcessor(key_encode_backend="wishful")


def test_processor_validates_dot_precision() -> None:
    with pytest.raises(ValueError, match="dot_precision"):
        ShmooshAttnProcessor(dot_precision="squint")


def test_processor_validates_split_dot_precision() -> None:
    with pytest.raises(ValueError, match="score_dot_precision"):
        ShmooshAttnProcessor(score_dot_precision="sparkly")
