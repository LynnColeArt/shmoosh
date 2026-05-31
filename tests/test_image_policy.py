from __future__ import annotations

from argparse import Namespace

from shmoosh.cli.image_ab_smoke import (
    _policy_processor_metadata,
    _processor_config,
    _select_policy_module_entries,
    _select_policy_modules,
)
from shmoosh.cli.image_policy_suite import _cases_from_payload
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
        }
    }
    module_entry = {
        "bits": 6,
        "shmoosh_policy": {
            "qjl_bits": 256,
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
        },
        "quantized_modules": [
            {
                "name": "a.attn2",
            },
            {
                "name": "b.attn2",
                "bits": 6,
                "quantize_start_percent": 0.2,
            },
        ],
    }
    selection = _select_policy_module_entries(modules, policy=policy)

    metadata = _policy_processor_metadata(modules, selection, args=args, policy=policy)

    assert metadata["mixed"] is True
    assert [entry["bits"] for entry in metadata["modules"]] == [5, 6]
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
