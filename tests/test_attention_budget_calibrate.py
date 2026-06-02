from __future__ import annotations

import argparse

import numpy as np
import pytest

from shmoosh.cli.attention_budget_calibrate import (
    _build_policies,
    _per_head_topk_mask,
    _policy_matches_shape,
    _quantile_budget,
    _top_p_kept_counts,
    run_calibration,
)

torch = pytest.importorskip("torch")


def test_top_p_kept_counts_returns_first_prefix_crossing_threshold() -> None:
    weights = torch.tensor([[[[0.50, 0.25, 0.15, 0.10]]]])

    counts = _top_p_kept_counts(weights, 0.80, torch=torch)

    assert counts.tolist() == [[[3]]]


def test_quantile_budget_rounds_up_and_clamps() -> None:
    counts = np.array([1, 2, 3, 4])

    assert _quantile_budget(counts, 0.50, 1024) == 3
    assert _quantile_budget(counts, 1.00, 3) == 3


def test_per_head_topk_mask_uses_independent_head_budgets() -> None:
    logits = torch.tensor(
        [
            [
                [[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]],
                [[0.0, 9.0, 8.0], [7.0, 6.0, 5.0]],
            ]
        ]
    )

    mask = _per_head_topk_mask(logits, [1, 2], torch=torch)

    assert mask.sum(dim=-1).tolist() == [[[1, 1], [2, 2]]]
    assert mask[0, 0].tolist() == [[False, True, False], [False, False, True]]
    assert mask[0, 1].tolist() == [[False, True, True], [True, True, False]]


def test_build_policies_aggregates_budgets_per_module_and_head() -> None:
    signature_h0 = ("module.attn1", 4, 4, 8, 2, 0.95, 0)
    signature_h1 = ("module.attn1", 4, 4, 8, 2, 0.95, 1)
    policies = _build_policies(
        {
            signature_h0: [np.array([1, 2, 2, 3])],
            signature_h1: [np.array([2, 3, 3, 4])],
        },
        budget_quantiles=[0.50, 1.0],
        key_tokens_by_signature={("module.attn1", 4, 4, 8, 2, 0.95): 4},
    )

    assert policies[0]["budgets"] == [2, 3]
    assert policies[1]["budgets"] == [3, 4]
    assert policies[0]["budget_mean"] == 2.5


def test_policy_matches_shape_requires_attention_geometry() -> None:
    policy = {"heads": 2, "query_tokens": 4, "key_tokens": 4, "head_dim": 8}

    assert _policy_matches_shape(
        policy,
        {"heads": 2, "query_tokens": 4, "key_tokens": 4, "head_dim": 8},
    )
    assert not _policy_matches_shape(
        policy,
        {"heads": 2, "query_tokens": 4, "key_tokens": 3, "head_dim": 8},
    )


def test_run_calibration_evaluates_synthetic_policy_rows() -> None:
    args = argparse.Namespace(
        captures=[],
        synthetic=True,
        batch_size=1,
        heads=2,
        tokens=4,
        query_tokens=None,
        key_tokens=None,
        dim=8,
        seed=3,
        device="cpu",
        dtype="fp32",
        top_p="0.90",
        budget_quantiles="0.50,1.0",
        limit=None,
        min_key_tokens=0,
        module_filter=None,
        self_attn_only=False,
    )

    result = run_calibration(args)

    assert len(result["policies"]) == 2
    assert len(result["rows"]) == 2
    assert {row["mode"] for row in result["rows"]} == {"static_head_topk"}
