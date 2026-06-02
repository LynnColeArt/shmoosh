from __future__ import annotations

from argparse import Namespace

from shmoosh.cli.image_ab_smoke import (
    _policy_processor_metadata,
    _processor_config,
    _select_policy_module_entries,
    _select_policy_modules,
)
from shmoosh.cli.image_policy_suite import _aggregate_rows, _cases_from_payload
from shmoosh.cli.image_policy_compare import (
    _aggregate_candidate_rows,
    _parse_candidate_spec,
)
from shmoosh.diffusers_processor import DenoisingStepState, ScheduledShmooshAttnProcessor


class _FakeProcessor:
    def __init__(self, label: str) -> None:
        self.label = label

    def __call__(self, *_args, **_kwargs) -> str:
        return self.label


def test_select_policy_modules_prefers_names() -> None:
    modules = [
        ("a.attn1", object()),
        ("b.attn2", object()),
    ]
    policy = {
        "quantized_modules": [
            {
                "index": 0,
                "name": "b.attn2",
            }
        ]
    }

    assert _select_policy_modules(modules, policy=policy) == [modules[1]]


def test_policy_processor_config_overrides_cli_defaults() -> None:
    args = Namespace(
        bits=4,
        qjl_bits=0,
        processor_seed=3,
        exact_keys=True,
        quantize_values=True,
        key_bits=None,
        value_bits=5,
        codebook_samples=100,
        attention_backend="reference",
        packed_backend="auto",
        packed_block_q=None,
        packed_block_k=None,
        code_format="packed",
        norm_dtype="fp32",
        key_encode_backend="split",
        dot_precision="ieee",
        rotation_dot_precision=None,
        score_dot_precision=None,
        value_dot_precision=None,
        qjl_dot_precision=None,
        steps=20,
    )
    policy = {
        "shmoosh_policy": {
            "bits": 3,
            "qjl_bits": 128,
            "processor_seed": 11,
            "quantize_keys": True,
            "quantize_values": False,
            "key_bits": None,
            "value_bits": None,
            "codebook_samples": 20000,
            "attention_backend": "packed",
            "packed_backend": "triton",
            "packed_block_q": 64,
            "packed_block_k": 32,
            "static_head_topk_budgets": [2, 3],
            "code_format": "byte",
            "norm_dtype": "fp16",
            "key_encode_backend": "fused",
            "dot_precision": "tf32",
            "score_dot_precision": "ieee",
            "cache_cross_attention": True,
        }
    }

    assert _processor_config(args, policy=policy) == {
        "bits": 3,
        "qjl_bits": 128,
        "seed": 11,
        "quantize_keys": True,
        "quantize_values": False,
        "key_bits": None,
        "value_bits": None,
        "codebook_samples": 20000,
        "attention_backend": "packed",
        "packed_backend": "triton",
        "packed_block_q": 64,
        "packed_block_k": 32,
        "static_head_topk_budgets": [2, 3],
        "code_format": "byte",
        "norm_dtype": "fp16",
        "key_encode_backend": "fused",
        "dot_precision": "tf32",
        "rotation_dot_precision": None,
        "score_dot_precision": "ieee",
        "value_dot_precision": None,
        "qjl_dot_precision": None,
        "cache_cross_attention": True,
    }


def test_module_policy_can_override_processor_config() -> None:
    args = Namespace(
        bits=4,
        qjl_bits=0,
        processor_seed=3,
        exact_keys=True,
        quantize_values=True,
        key_bits=None,
        value_bits=5,
        codebook_samples=100,
        attention_backend="reference",
        packed_backend="auto",
        packed_block_q=None,
        packed_block_k=None,
        code_format="packed",
        norm_dtype="fp32",
        key_encode_backend="split",
        dot_precision="ieee",
        rotation_dot_precision=None,
        score_dot_precision=None,
        value_dot_precision=None,
        qjl_dot_precision=None,
        steps=20,
    )
    policy = {
        "shmoosh_policy": {
            "bits": 5,
            "qjl_bits": 128,
            "processor_seed": 11,
            "quantize_keys": True,
            "quantize_values": False,
            "key_bits": None,
            "value_bits": None,
            "codebook_samples": 20000,
            "packed_block_q": 64,
            "packed_block_k": 16,
            "static_head_topk_budgets": [8, 9],
        }
    }
    module_entry = {
        "bits": 6,
        "shmoosh_policy": {
            "qjl_bits": 256,
            "packed_block_k": 32,
        },
    }

    assert _processor_config(args, policy=policy, module_entry=module_entry) == {
        "bits": 6,
        "qjl_bits": 256,
        "seed": 11,
        "quantize_keys": True,
        "quantize_values": False,
        "key_bits": None,
        "value_bits": None,
        "codebook_samples": 20000,
        "attention_backend": "reference",
        "packed_backend": "auto",
        "packed_block_q": 64,
        "packed_block_k": 32,
        "static_head_topk_budgets": [8, 9],
        "code_format": "packed",
        "norm_dtype": "fp32",
        "key_encode_backend": "split",
        "dot_precision": "ieee",
        "rotation_dot_precision": None,
        "score_dot_precision": None,
        "value_dot_precision": None,
        "qjl_dot_precision": None,
        "cache_cross_attention": False,
    }


def test_policy_processor_metadata_reports_mixed_modules() -> None:
    args = Namespace(
        bits=4,
        qjl_bits=0,
        processor_seed=3,
        exact_keys=True,
        quantize_values=True,
        key_bits=None,
        value_bits=5,
        codebook_samples=100,
        attention_backend="reference",
        packed_backend="auto",
        packed_block_q=None,
        packed_block_k=None,
        code_format="packed",
        norm_dtype="fp32",
        key_encode_backend="split",
        dot_precision="ieee",
        rotation_dot_precision=None,
        score_dot_precision=None,
        value_dot_precision=None,
        qjl_dot_precision=None,
        steps=20,
    )
    first = object()
    second = object()
    modules = [
        ("a.attn2", first),
        ("b.attn2", second),
    ]
    policy = {
        "shmoosh_policy": {
            "bits": 5,
            "qjl_bits": 128,
            "processor_seed": 11,
            "quantize_keys": True,
            "quantize_values": False,
            "key_bits": None,
            "value_bits": None,
            "codebook_samples": 20000,
            "attention_backend": "packed",
            "packed_backend": "auto",
            "packed_block_q": 64,
            "packed_block_k": 32,
            "static_head_topk_budgets": [8, 9],
            "code_format": "byte",
            "norm_dtype": "fp16",
            "key_encode_backend": "auto",
            "dot_precision": "tf32",
            "score_dot_precision": "ieee",
            "cache_cross_attention": True,
        },
        "quantized_modules": [
            {
                "name": "a.attn2",
            },
            {
                "name": "b.attn2",
                "bits": 6,
                "quantize_start_percent": 0.2,
                "packed_backend": "torch",
                "packed_block_k": 16,
                "static_head_topk_budgets": [4, 5],
                "code_format": "packed",
                "norm_dtype": "fp32",
                "key_encode_backend": "fused",
                "dot_precision": "ieee",
                "value_dot_precision": "tf32",
            },
        ],
    }
    selection = _select_policy_module_entries(modules, policy=policy)

    metadata = _policy_processor_metadata(modules, selection, args=args, policy=policy)

    assert metadata["mixed"] is True
    assert [entry["bits"] for entry in metadata["modules"]] == [5, 6]
    assert [entry["attention_backend"] for entry in metadata["modules"]] == ["packed", "packed"]
    assert [entry["packed_backend"] for entry in metadata["modules"]] == ["auto", "torch"]
    assert [entry["packed_block_q"] for entry in metadata["modules"]] == [64, 64]
    assert [entry["packed_block_k"] for entry in metadata["modules"]] == [32, 16]
    assert [entry["static_head_topk_budgets"] for entry in metadata["modules"]] == [
        [8, 9],
        [4, 5],
    ]
    assert [entry["code_format"] for entry in metadata["modules"]] == ["byte", "packed"]
    assert [entry["norm_dtype"] for entry in metadata["modules"]] == ["fp16", "fp32"]
    assert [entry["key_encode_backend"] for entry in metadata["modules"]] == ["auto", "fused"]
    assert [entry["dot_precision"] for entry in metadata["modules"]] == ["tf32", "ieee"]
    assert [entry["score_dot_precision"] for entry in metadata["modules"]] == ["ieee", "ieee"]
    assert [entry["value_dot_precision"] for entry in metadata["modules"]] == ["tf32", "tf32"]
    assert [entry["cache_cross_attention"] for entry in metadata["modules"]] == [True, True]
    assert [entry["index"] for entry in metadata["modules"]] == [0, 1]
    assert [entry["quantize_start_percent"] for entry in metadata["modules"]] == [None, 0.2]
    assert [entry["resolved_quantize_start_step"] for entry in metadata["modules"]] == [0, 4]


def test_scheduled_processor_dispatches_by_step_window() -> None:
    step_state = DenoisingStepState(current_step=0, total_steps=20)
    processor = ScheduledShmooshAttnProcessor(
        original_processor=_FakeProcessor("exact"),
        shmoosh_processor=_FakeProcessor("shmoosh"),
        step_state=step_state,
        quantize_start_step=4,
        quantize_end_step=10,
    )

    assert processor(None, None) == "exact"
    step_state.current_step = 4
    assert processor(None, None) == "shmoosh"
    step_state.current_step = 9
    assert processor(None, None) == "shmoosh"
    step_state.current_step = 10
    assert processor(None, None) == "exact"


def test_scheduled_processor_resolves_percent_window() -> None:
    step_state = DenoisingStepState(current_step=5, total_steps=30)
    processor = ScheduledShmooshAttnProcessor(
        original_processor=_FakeProcessor("exact"),
        shmoosh_processor=_FakeProcessor("shmoosh"),
        step_state=step_state,
        quantize_start_percent=0.2,
        quantize_end_percent=0.5,
    )

    assert processor(None, None) == "exact"
    step_state.current_step = 6
    assert processor(None, None) == "shmoosh"
    step_state.current_step = 14
    assert processor(None, None) == "shmoosh"
    step_state.current_step = 15
    assert processor(None, None) == "exact"


def test_policy_suite_cases_use_file_defaults() -> None:
    args = Namespace(
        steps=8,
        height=256,
        width=256,
        guidance_scale=3.0,
        device="cpu",
    )
    payload = {
        "defaults": {
            "steps": 20,
            "height": 512,
            "width": 512,
            "guidance_scale": 5.0,
        },
        "cases": [
            {
                "id": "case-a",
                "prompt": "a small brass compass",
                "seed": 7,
            }
        ],
    }

    [case] = _cases_from_payload(payload, args=args)

    assert case.case_id == "case-a"
    assert case.seed == 7
    assert case.steps == 20
    assert case.height == 512
    assert case.width == 512
    assert case.guidance_scale == 5.0


def test_policy_suite_aggregate_reports_timing_speedup() -> None:
    rows = [
        {
            "mse": 0.1,
            "mae": 0.2,
            "psnr_db": 40.0,
            "baseline_seconds": 10.0,
            "shmoosh_seconds": 5.0,
        },
        {
            "mse": 0.3,
            "mae": 0.4,
            "psnr_db": 44.0,
            "baseline_seconds": 8.0,
            "shmoosh_seconds": 7.0,
        },
    ]

    aggregate = _aggregate_rows(rows)

    assert aggregate["mean_baseline_seconds"] == 9.0
    assert aggregate["mean_shmoosh_seconds"] == 6.0
    assert aggregate["mean_speedup"] == 1.5


def test_policy_compare_candidate_spec_accepts_label_path() -> None:
    label, path = _parse_candidate_spec("score-value=configs/policy.json")

    assert label == "score_value"
    assert path == "configs/policy.json"


def test_policy_compare_candidate_spec_uses_file_stem() -> None:
    label, path = _parse_candidate_spec("configs/packed-tf32-policy.json")

    assert label == "packed_tf32_policy"
    assert path == "configs/packed-tf32-policy.json"


def test_policy_compare_aggregates_by_candidate() -> None:
    rows = [
        {
            "candidate_label": "ieee",
            "policy_file": "ieee.json",
            "mse": 0.1,
            "mae": 0.2,
            "psnr_db": 40.0,
            "baseline_seconds": 10.0,
            "shmoosh_seconds": 5.0,
            "processor_timing_seconds": 0.3,
            "processor_timing_records": 3,
            "packed_attention_seconds": 0.02,
            "mean_packed_attention_ms": 2.0,
        },
        {
            "candidate_label": "ieee",
            "policy_file": "ieee.json",
            "mse": 0.3,
            "mae": 0.4,
            "psnr_db": 44.0,
            "baseline_seconds": 8.0,
            "shmoosh_seconds": 7.0,
            "processor_timing_seconds": 0.5,
            "processor_timing_records": 5,
            "packed_attention_seconds": 0.04,
            "mean_packed_attention_ms": 4.0,
        },
        {
            "candidate_label": "tf32",
            "policy_file": "tf32.json",
            "mse": 0.2,
            "mae": 0.3,
            "psnr_db": 42.0,
            "baseline_seconds": 10.0,
            "shmoosh_seconds": 4.0,
            "processor_timing_seconds": 0.2,
            "processor_timing_records": 2,
            "packed_attention_seconds": 0.01,
            "mean_packed_attention_ms": 1.0,
        },
    ]

    aggregates = _aggregate_candidate_rows(rows)

    assert [row["candidate_label"] for row in aggregates] == ["ieee", "tf32"]
    assert aggregates[0]["mean_speedup"] == 1.5
    assert aggregates[0]["mean_processor_timing_seconds"] == 0.4
    assert aggregates[0]["mean_packed_attention_seconds"] == 0.03
    assert aggregates[0]["mean_packed_attention_ms"] == 3.0
    assert aggregates[1]["mean_speedup"] == 2.5
