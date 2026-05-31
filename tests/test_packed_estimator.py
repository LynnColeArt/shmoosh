from __future__ import annotations

from turbo_d.packed_estimator import (
    PackedKeyAssumptions,
    PackedKeyFormat,
    estimate_policy_storage,
)


def test_packed_key_format_matches_expected_vector_sizes() -> None:
    assert (
        PackedKeyFormat(key_bits=5, qjl_bits=128, head_dim=64).bytes_per_vector
        == 64
    )
    assert (
        PackedKeyFormat(key_bits=6, qjl_bits=128, head_dim=64).bytes_per_vector
        == 72
    )


def test_estimate_policy_storage_for_mixed_gated_policy() -> None:
    policy = {
        "turbo_policy": {
            "bits": 5,
            "qjl_bits": 128,
            "quantize_keys": True,
        },
        "quantized_modules": [
            {"index": 49, "name": "a", "quantize_start_percent": 0.3},
            {"index": 59, "name": "b", "quantize_start_percent": 0.3},
            {"index": 61, "name": "c", "quantize_start_percent": 0.3},
            {"index": 65, "name": "d", "quantize_start_percent": 0.3},
            {"index": 67, "name": "e", "quantize_start_percent": 0.3},
            {"index": 79, "name": "f", "bits": 6, "quantize_start_percent": 0.3},
            {"index": 87, "name": "g", "bits": 6, "quantize_start_percent": 0.3},
        ],
    }

    estimate = estimate_policy_storage(
        policy,
        steps=[20, 30],
        assumptions=PackedKeyAssumptions(),
    )

    assert estimate["per_quantized_step"]["exact_bytes"] == 2_759_680
    assert estimate["per_quantized_step"]["packed_bytes"] == 1_429_120
    assert estimate["steps"][0]["saved_key_bytes"] == 18_627_840
    assert estimate["steps"][0]["modules"][0]["resolved_quantize_start_step"] == 6
    assert estimate["steps"][0]["modules"][0]["quantized_steps"] == 14
    assert estimate["steps"][1]["saved_key_bytes"] == 27_941_760
    assert estimate["steps"][1]["modules"][0]["resolved_quantize_start_step"] == 9
    assert estimate["steps"][1]["modules"][0]["quantized_steps"] == 21
