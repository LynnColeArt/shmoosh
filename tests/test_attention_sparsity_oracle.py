from __future__ import annotations

import argparse

import pytest

from shmoosh.cli.attention_sparsity_oracle import (
    _local_window_mask,
    _mask_stats,
    _parse_local_spatial_block_specs,
    _parse_floats,
    _parse_ints,
    _parse_spatial_block_specs,
    _quality_metrics,
    _run_tensor_oracle,
    _spatial_block_topk_mask,
    _square_side,
    _supports_local_window,
    _topk_mask,
    _topp_mask,
)

torch = pytest.importorskip("torch")


def test_parse_ints_allows_empty_list() -> None:
    assert _parse_ints("") == []
    assert _parse_ints("64, 128") == [64, 128]


def test_parse_floats_allows_empty_list() -> None:
    assert _parse_floats("") == []
    assert _parse_floats("0.95, 0.98") == [0.95, 0.98]


def test_parse_spatial_block_specs() -> None:
    assert _parse_spatial_block_specs("") == []
    assert _parse_spatial_block_specs("4:2, 8:1") == [(4, 2), (8, 1)]


def test_parse_local_spatial_block_specs() -> None:
    assert _parse_local_spatial_block_specs("") == []
    assert _parse_local_spatial_block_specs("4:2:9") == [(4, 2, 9)]
    with pytest.raises(ValueError, match="windows must be odd"):
        _parse_local_spatial_block_specs("4:2:8")


def test_square_side_accepts_square_tokens() -> None:
    assert _square_side(1024) == 32


def test_square_side_rejects_non_square_tokens() -> None:
    with pytest.raises(ValueError, match="square self-attention"):
        _square_side(77)


def test_supports_local_window_requires_square_self_attention() -> None:
    assert _supports_local_window(torch.zeros(1, 2, 16, 16))
    assert not _supports_local_window(torch.zeros(1, 2, 16, 8))
    assert not _supports_local_window(torch.zeros(1, 2, 77, 77))


def test_topk_mask_keeps_k_entries_per_query() -> None:
    logits = torch.tensor([[[[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]]]])

    mask = _topk_mask(logits, 2, torch=torch)

    assert mask.sum(dim=-1).tolist() == [[[2, 2]]]
    assert mask.tolist() == [[[[False, True, True], [True, False, True]]]]


def test_topp_mask_keeps_minimal_mass_prefix() -> None:
    weights = torch.tensor([[[[0.50, 0.25, 0.15, 0.10]]]])

    mask = _topp_mask(weights, 0.80, torch=torch)

    assert mask.tolist() == [[[[True, True, True, False]]]]


def test_local_window_mask_uses_square_grid_neighbors() -> None:
    mask = _local_window_mask(torch, 9, 3, device="cpu")

    assert int(mask[4].sum().item()) == 9
    assert int(mask[0].sum().item()) == 4


def test_spatial_block_topk_mask_keeps_top_spatial_tile_per_query() -> None:
    logits = torch.zeros(1, 1, 16, 16)
    logits[..., 0, 10] = 7.0
    logits[..., 1, 0] = 6.0

    mask = _spatial_block_topk_mask(
        logits,
        block_side=2,
        block_count=1,
        torch=torch,
    )

    assert mask.shape == logits.shape
    assert mask[0, 0, 0].nonzero().squeeze(-1).tolist() == [10, 11, 14, 15]
    assert mask[0, 0, 1].nonzero().squeeze(-1).tolist() == [0, 1, 4, 5]


def test_mask_stats_reports_kept_fraction_and_attention_mass() -> None:
    weights = torch.tensor([[[[0.50, 0.30, 0.20]]]])
    mask = torch.tensor([[[[True, False, True]]]])

    stats = _mask_stats(mask, weights, torch=torch)

    assert stats["kept_keys_mean"] == 2.0
    assert stats["kept_key_fraction"] == pytest.approx(2.0 / 3.0)
    assert stats["attention_mass_mean"] == pytest.approx(0.70)


def test_quality_metrics_are_zero_for_identical_outputs() -> None:
    output = torch.randn(1, 2, 3, 4)

    metrics = _quality_metrics(output, output, torch=torch)

    assert metrics["mse"] == 0.0
    assert metrics["relative_rmse"] == 0.0
    assert metrics["cosine_error"] == pytest.approx(0.0)


def test_run_tensor_oracle_emits_dense_and_sparse_rows() -> None:
    generator = torch.Generator(device="cpu").manual_seed(3)
    query = torch.randn(1, 2, 4, 8, generator=generator)
    key = torch.randn(1, 2, 4, 8, generator=generator)
    value = torch.randn(1, 2, 4, 8, generator=generator)
    args = argparse.Namespace(dtype="fp32")

    rows = _run_tensor_oracle(
        query,
        key,
        value,
        torch=torch,
        specs=[("top_k", 2), ("top_p", 0.90), ("local_window", 3)],
        base_row={
            "capture": "synthetic",
            "module": "synthetic.attn1",
            "capture_index": "",
            "dtype": args.dtype,
            "device": "cpu",
        },
    )

    assert [row["mode"] for row in rows] == [
        "dense",
        "top_k",
        "top_p",
        "local_window",
    ]
    assert rows[0]["relative_rmse"] == 0.0


def test_run_tensor_oracle_emits_spatial_block_rows() -> None:
    generator = torch.Generator(device="cpu").manual_seed(4)
    query = torch.randn(1, 2, 16, 8, generator=generator)
    key = torch.randn(1, 2, 16, 8, generator=generator)
    value = torch.randn(1, 2, 16, 8, generator=generator)

    rows = _run_tensor_oracle(
        query,
        key,
        value,
        torch=torch,
        specs=[
            ("spatial_block_top_k", (2, 1)),
            ("local_spatial_block_top_k", (2, 1, 3)),
        ],
        base_row={
            "capture": "synthetic",
            "module": "synthetic.attn1",
            "capture_index": "",
            "dtype": "fp32",
            "device": "cpu",
        },
    )

    assert [row["mode"] for row in rows] == [
        "dense",
        "spatial_block_top_k",
        "local_spatial_block_top_k",
    ]
    assert rows[1]["kept_keys_min"] == 4
    assert rows[1]["kept_keys_max"] == 4
    assert rows[2]["kept_keys_min"] >= rows[1]["kept_keys_min"]
