from __future__ import annotations

from argparse import Namespace

from turbo_d.cli.image_ab_smoke import _processor_config, _select_policy_modules
from turbo_d.cli.image_policy_suite import _cases_from_payload


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
    )
    policy = {
        "turbo_policy": {
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
