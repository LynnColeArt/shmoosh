from __future__ import annotations

from argparse import Namespace

import pytest

from shmoosh.cli.image_module_sweep import (
    _resolved_sweep_window_config,
    _suggest_policy,
    _validate_window_args,
)


def _args(**overrides):
    values = {
        "bits": 6,
        "key_bits": None,
        "value_bits": None,
        "qjl_bits": 128,
        "codebook_samples": 80_000,
        "processor_seed": 11,
        "exact_keys": False,
        "quantize_values": False,
        "attention_backend": "packed",
        "packed_backend": "auto",
        "candidate_psnr_db": 45.0,
        "steps": 20,
        "quantize_start_step": 0,
        "quantize_end_step": None,
        "quantize_start_percent": None,
        "quantize_end_percent": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_resolved_sweep_window_rounds_percent_to_steps() -> None:
    args = _args(
        steps=21,
        quantize_start_percent=0.5,
        quantize_end_percent=0.75,
    )

    assert _resolved_sweep_window_config(args) == {
        "resolved_quantize_start_step": 11,
        "resolved_quantize_end_step": 16,
    }


def test_validate_window_rejects_inverted_resolved_window() -> None:
    args = _args(
        steps=20,
        quantize_start_percent=0.5,
        quantize_end_step=10,
    )

    with pytest.raises(SystemExit, match="start before"):
        _validate_window_args(args)


def test_suggested_policy_carries_sweep_window_on_candidates() -> None:
    args = _args(quantize_start_percent=0.5)
    rows = [
        {
            "policy": "shmoosh",
            "module_index": 48,
            "module_name": "up_blocks.0.attentions.0.transformer_blocks.0.attn1",
            "mse": 0.0,
            "mae": 0.0,
            "psnr_db": 50.0,
            "quantize_start_step": 0,
            "quantize_end_step": None,
            "quantize_start_percent": 0.5,
            "quantize_end_percent": None,
        }
    ]

    policy = _suggest_policy(args, rows)

    assert policy["quantize_window"]["resolved_quantize_start_step"] == 10
    assert policy["quantized_modules"] == [
        {
            "index": 48,
            "name": "up_blocks.0.attentions.0.transformer_blocks.0.attn1",
            "mse": 0.0,
            "mae": 0.0,
            "psnr_db": 50.0,
            "quantize_start_step": 0,
            "quantize_end_step": None,
            "quantize_start_percent": 0.5,
            "quantize_end_percent": None,
        }
    ]
