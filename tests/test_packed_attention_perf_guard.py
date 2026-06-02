from __future__ import annotations

from argparse import Namespace

import pytest

from shmoosh.cli.packed_attention_perf_guard import (
    _summarize_rows,
    _threshold_failures,
    _validate_args,
)


def test_summarize_rows_reports_latency_and_quality_stats() -> None:
    rows = [
        {
            "attention_ms_per_iter": 0.30,
            "encode_ms_per_iter": 0.20,
            "total_ms_per_iter": 0.50,
            "relative_rmse": 0.024,
            "cosine": 0.9998,
            "packed_bytes_per_vector": 60,
        },
        {
            "attention_ms_per_iter": 0.36,
            "encode_ms_per_iter": 0.18,
            "total_ms_per_iter": 0.54,
            "relative_rmse": 0.025,
            "cosine": 0.9997,
            "packed_bytes_per_vector": 60,
        },
        {
            "attention_ms_per_iter": 0.33,
            "encode_ms_per_iter": 0.19,
            "total_ms_per_iter": 0.52,
            "relative_rmse": 0.023,
            "cosine": 0.9999,
            "packed_bytes_per_vector": 60,
        },
    ]

    summary = _summarize_rows(rows)

    assert summary["attention_ms_per_iter"]["median"] == pytest.approx(0.33)
    assert summary["total_ms_per_iter"]["max"] == pytest.approx(0.54)
    assert summary["relative_rmse"]["max"] == pytest.approx(0.025)
    assert summary["cosine"]["min"] == pytest.approx(0.9997)
    assert summary["packed_bytes_per_vector"] == [60]


def test_threshold_failures_reports_latency_and_quality_regressions() -> None:
    args = _args()
    payload = {
        "aggregate": {
            "attention_ms_per_iter": {"median": 0.50},
            "total_ms_per_iter": {"median": 0.80},
            "relative_rmse": {"max": 0.040},
            "cosine": {"min": 0.990},
        }
    }

    failures = _threshold_failures(payload, args)

    assert len(failures) == 4
    assert "median attention" in failures[0]
    assert "median total" in failures[1]
    assert "relative_rmse" in failures[2]
    assert "min cosine" in failures[3]


def test_threshold_failures_passes_current_shape() -> None:
    args = _args()
    payload = {
        "aggregate": {
            "attention_ms_per_iter": {"median": 0.31},
            "total_ms_per_iter": {"median": 0.52},
            "relative_rmse": {"max": 0.024},
            "cosine": {"min": 0.9997},
        }
    }

    assert _threshold_failures(payload, args) == []


def test_validate_args_rejects_nonpositive_samples() -> None:
    args = _args(samples=0)

    with pytest.raises(SystemExit, match="samples"):
        _validate_args(args)


def _args(**overrides) -> Namespace:
    values = {
        "samples": 3,
        "iters": 200,
        "warmup_iters": 12,
        "max_attention_ms": 0.45,
        "max_total_ms": 0.75,
        "max_relative_rmse": 0.03,
        "min_cosine": 0.999,
    }
    values.update(overrides)
    return Namespace(**values)
