from __future__ import annotations

from argparse import Namespace

from turbo_d.cli.image_ab_smoke import _processor_config, _select_policy_modules


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
