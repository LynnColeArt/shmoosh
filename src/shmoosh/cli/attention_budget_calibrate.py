from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from shmoosh.cli.attention_sparsity_oracle import (
    _dtype,
    _load_torch,
    _mask_stats,
    _parse_floats,
    _quality_metrics,
    _select_device,
    _sparse_attention,
    _synthetic_tensors,
    _topk_mask,
)
from shmoosh.cli.packed_encode_parity import (
    _attention_tensor,
    _expand_captures,
    _load_metadata,
)


@dataclass
class TensorSource:
    query: Any
    key: Any
    value: Any
    base_row: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate static per-head top-k budgets from top-p attention mass "
            "on captured or synthetic Q/K/V tensors."
        )
    )
    parser.add_argument(
        "captures",
        nargs="*",
        help="Capture .npz files or directories. Omit with --synthetic.",
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--query-tokens", type=int)
    parser.add_argument("--key-tokens", type=int)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--top-p", default="0.95,0.98")
    parser.add_argument(
        "--budget-quantiles",
        default="0.50,0.90,0.95,0.99,1.0",
        help=(
            "Comma-separated quantiles of top-p kept counts to convert into "
            "static per-head top-k budgets."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-key-tokens", type=int, default=0)
    parser.add_argument("--module-filter")
    parser.add_argument("--self-attn-only", action="store_true")
    parser.add_argument("--csv", default="captures/attention_budget_calibration.csv")
    parser.add_argument("--json", default="captures/attention_budget_calibration.json")
    parser.add_argument("--head-csv")
    parser.add_argument("--head-json")
    parser.add_argument("--budget-json")
    args = parser.parse_args()

    result = run_calibration(args)
    _write_csv(Path(args.csv), result["rows"], fieldnames=_row_fieldnames())
    _write_json(Path(args.json), args, result)
    if args.head_csv:
        _write_csv(Path(args.head_csv), result["head_rows"], fieldnames=_head_fieldnames())
    if args.head_json:
        _write_payload(Path(args.head_json), args, "rows", result["head_rows"])
    if args.budget_json:
        _write_payload(Path(args.budget_json), args, "policies", result["policies"])
    _print_summary(result["rows"])


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    torch = _load_torch()
    device = _select_device(torch, args.device)
    dtype = _dtype(torch, args.dtype)
    top_p_values = _parse_top_p_values(args.top_p)
    budget_quantiles = _parse_budget_quantiles(args.budget_quantiles)

    head_rows: list[dict[str, Any]] = []
    count_accumulator: dict[tuple[Any, ...], list[np.ndarray]] = defaultdict(list)
    for source in _iter_sources(args, torch=torch, device=device, dtype=dtype):
        _, dense_weights, _ = _attention_state(
            source,
            torch=torch,
            include_output=False,
        )
        shape_row = _shape_row(source.query, source.key)
        for top_p in top_p_values:
            counts = _top_p_kept_counts(dense_weights, top_p, torch=torch)
            head_rows.extend(
                _head_rows(
                    source.base_row,
                    shape_row,
                    top_p,
                    counts,
                    dense_weights,
                    torch=torch,
                )
            )
            _accumulate_counts(
                count_accumulator,
                source.base_row,
                shape_row,
                top_p,
                counts,
            )
        del dense_weights

    if not head_rows:
        raise SystemExit("No captures matched the requested calibration filters.")

    policies = _build_policies(
        count_accumulator,
        budget_quantiles=budget_quantiles,
        key_tokens_by_signature=_key_tokens_by_signature(head_rows),
    )
    rows = _evaluate_policies(args, policies, torch=torch, device=device, dtype=dtype)
    return {
        "rows": rows,
        "head_rows": head_rows,
        "policies": policies,
        "summary": _summary(rows),
    }


def _iter_sources(
    args: argparse.Namespace,
    *,
    torch: Any,
    device: str,
    dtype: Any,
) -> list[TensorSource]:
    if args.synthetic or not args.captures:
        query, key, value = _synthetic_tensors(torch, args, device=device, dtype=dtype)
        return [
            TensorSource(
                query=query,
                key=key,
                value=value,
                base_row={
                    "capture": "synthetic",
                    "module": "synthetic.attn1",
                    "capture_index": "",
                    "dtype": args.dtype,
                    "device": str(device),
                },
            )
        ]

    paths = _expand_captures(args.captures)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("No capture files found.")

    sources: list[TensorSource] = []
    for path in paths:
        loaded = np.load(path)
        metadata = _load_metadata(loaded)
        module = str(metadata.get("module", ""))
        if args.module_filter and args.module_filter not in module:
            continue
        if args.self_attn_only and not module.endswith("attn1"):
            continue
        query = _attention_tensor(torch, loaded["q"], device=device, dtype=dtype)
        key = _attention_tensor(torch, loaded["k"], device=device, dtype=dtype)
        value = _attention_tensor(torch, loaded["v"], device=device, dtype=dtype)
        if int(key.shape[2]) < args.min_key_tokens:
            continue
        sources.append(
            TensorSource(
                query=query,
                key=key,
                value=value,
                base_row={
                    "capture": str(path),
                    "module": module,
                    "capture_index": metadata.get("capture_index", ""),
                    "dtype": args.dtype,
                    "device": str(device),
                },
            )
        )
    return sources


def _attention_state(
    source: TensorSource,
    *,
    torch: Any,
    include_output: bool = True,
) -> tuple[Any, Any, Any | None]:
    query_f = source.query.to(dtype=torch.float32)
    key_f = source.key.to(dtype=torch.float32)
    value_f = source.value.to(dtype=torch.float32)
    logits = torch.matmul(query_f, key_f.transpose(-2, -1)) / sqrt(
        int(source.query.shape[-1])
    )
    dense_weights = torch.softmax(logits, dim=-1)
    dense_output = torch.matmul(dense_weights, value_f) if include_output else None
    return logits, dense_weights, dense_output


def _top_p_kept_counts(weights: Any, threshold: float, *, torch: Any) -> Any:
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("top-p thresholds must be greater than 0 and less than 1")
    sorted_weights = torch.sort(weights, dim=-1, descending=True).values
    cumulative = sorted_weights.cumsum(dim=-1)
    first_cross = (cumulative >= threshold).to(dtype=torch.int64).argmax(dim=-1)
    return first_cross + 1


def _head_rows(
    base_row: dict[str, Any],
    shape_row: dict[str, Any],
    top_p: float,
    counts: Any,
    dense_weights: Any,
    *,
    torch: Any,
) -> list[dict[str, Any]]:
    rows = []
    for head in range(int(counts.shape[1])):
        head_counts = counts[:, head, :].reshape(-1).to(dtype=torch.float32)
        rows.append(
            {
                **base_row,
                **shape_row,
                "top_p": top_p,
                "head": head,
                "kept_keys_mean": float(head_counts.mean().item()),
                "kept_keys_min": int(head_counts.min().item()),
                "kept_keys_max": int(head_counts.max().item()),
                "kept_keys_p50": _torch_quantile(head_counts, 0.50, torch=torch),
                "kept_keys_p90": _torch_quantile(head_counts, 0.90, torch=torch),
                "kept_keys_p95": _torch_quantile(head_counts, 0.95, torch=torch),
                "kept_keys_p99": _torch_quantile(head_counts, 0.99, torch=torch),
                "kept_key_fraction_mean": float(
                    (head_counts / dense_weights.shape[-1]).mean().item()
                ),
            }
        )
    return rows


def _accumulate_counts(
    accumulator: dict[tuple[Any, ...], list[np.ndarray]],
    base_row: dict[str, Any],
    shape_row: dict[str, Any],
    top_p: float,
    counts: Any,
) -> None:
    counts_np = counts.detach().cpu().numpy()
    for head in range(int(counts_np.shape[1])):
        signature = _policy_signature(base_row, shape_row, top_p, head=head)
        accumulator[signature].append(counts_np[:, head, :].reshape(-1))


def _build_policies(
    accumulator: dict[tuple[Any, ...], list[np.ndarray]],
    *,
    budget_quantiles: list[float],
    key_tokens_by_signature: dict[tuple[Any, ...], int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[int, np.ndarray]] = defaultdict(dict)
    for signature, chunks in accumulator.items():
        module, query_tokens, key_tokens, head_dim, heads, top_p, head = signature
        grouped[(module, query_tokens, key_tokens, head_dim, heads, top_p)][int(head)] = (
            np.concatenate(chunks)
        )

    policies: list[dict[str, Any]] = []
    for group_signature, head_counts in sorted(grouped.items()):
        module, query_tokens, key_tokens, head_dim, heads, top_p = group_signature
        if len(head_counts) != int(heads):
            continue
        for quantile in budget_quantiles:
            budgets = [
                _quantile_budget(head_counts[head], quantile, int(key_tokens))
                for head in range(int(heads))
            ]
            policies.append(
                {
                    "module": module,
                    "query_tokens": query_tokens,
                    "key_tokens": key_tokens_by_signature.get(
                        group_signature,
                        int(key_tokens),
                    ),
                    "head_dim": head_dim,
                    "heads": heads,
                    "top_p": top_p,
                    "budget_quantile": quantile,
                    "budgets": budgets,
                    "budget_min": min(budgets),
                    "budget_max": max(budgets),
                    "budget_mean": float(np.mean(budgets)),
                    "kept_key_fraction": float(np.mean(budgets) / int(key_tokens)),
                }
            )
    return policies


def _evaluate_policies(
    args: argparse.Namespace,
    policies: list[dict[str, Any]],
    *,
    torch: Any,
    device: str,
    dtype: Any,
) -> list[dict[str, Any]]:
    policies_by_module = defaultdict(list)
    for policy in policies:
        policies_by_module[policy["module"]].append(policy)

    rows: list[dict[str, Any]] = []
    for source in _iter_sources(args, torch=torch, device=device, dtype=dtype):
        logits, dense_weights, dense_output = _attention_state(source, torch=torch)
        if dense_output is None:
            raise AssertionError("policy evaluation requires dense attention output")
        shape_row = _shape_row(source.query, source.key)
        value_f = source.value.to(dtype=torch.float32)
        for policy in policies_by_module[source.base_row["module"]]:
            if not _policy_matches_shape(policy, shape_row):
                continue
            mask = _per_head_topk_mask(logits, policy["budgets"], torch=torch)
            sparse_output = _sparse_attention(
                logits,
                value_f,
                mask,
                torch=torch,
            )
            rows.append(
                {
                    **source.base_row,
                    **shape_row,
                    "mode": "static_head_topk",
                    "setting": _policy_setting(policy),
                    "top_p": policy["top_p"],
                    "budget_quantile": policy["budget_quantile"],
                    "budget_mean": policy["budget_mean"],
                    "budget_min": policy["budget_min"],
                    "budget_max": policy["budget_max"],
                    **_mask_stats(mask, dense_weights, torch=torch),
                    **_quality_metrics(sparse_output, dense_output, torch=torch),
                }
            )
        del logits, dense_weights, dense_output
    return rows


def _per_head_topk_mask(logits: Any, budgets: list[int], *, torch: Any) -> Any:
    if len(budgets) != int(logits.shape[1]):
        raise ValueError("budget count must match attention head count")
    head_masks = [
        _topk_mask(logits[:, head : head + 1, :, :], int(budget), torch=torch)
        for head, budget in enumerate(budgets)
    ]
    return torch.cat(head_masks, dim=1)


def _shape_row(query: Any, key: Any) -> dict[str, Any]:
    return {
        "batch_size": int(query.shape[0]),
        "heads": int(query.shape[1]),
        "query_tokens": int(query.shape[2]),
        "key_tokens": int(key.shape[2]),
        "head_dim": int(query.shape[-1]),
    }


def _policy_signature(
    base_row: dict[str, Any],
    shape_row: dict[str, Any],
    top_p: float,
    *,
    head: int,
) -> tuple[Any, ...]:
    return (
        base_row["module"],
        shape_row["query_tokens"],
        shape_row["key_tokens"],
        shape_row["head_dim"],
        shape_row["heads"],
        top_p,
        head,
    )


def _key_tokens_by_signature(head_rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], int]:
    values = {}
    for row in head_rows:
        signature = (
            row["module"],
            row["query_tokens"],
            row["key_tokens"],
            row["head_dim"],
            row["heads"],
            row["top_p"],
        )
        values[signature] = int(row["key_tokens"])
    return values


def _quantile_budget(counts: np.ndarray, quantile: float, key_tokens: int) -> int:
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError("budget quantiles must be in the range 0..1")
    budget = int(ceil(float(np.quantile(counts, quantile))))
    return max(1, min(key_tokens, budget))


def _torch_quantile(values: Any, quantile: float, *, torch: Any) -> float:
    return float(torch.quantile(values, quantile).item())


def _policy_matches_shape(policy: dict[str, Any], shape_row: dict[str, Any]) -> bool:
    return (
        int(policy["heads"]) == int(shape_row["heads"])
        and int(policy["query_tokens"]) == int(shape_row["query_tokens"])
        and int(policy["key_tokens"]) == int(shape_row["key_tokens"])
        and int(policy["head_dim"]) == int(shape_row["head_dim"])
    )


def _policy_setting(policy: dict[str, Any]) -> str:
    return f"top_p={policy['top_p']}:q={policy['budget_quantile']}"


def _parse_top_p_values(raw: str) -> list[float]:
    values = _parse_floats(raw)
    if not values:
        raise SystemExit("--top-p must contain at least one threshold")
    for value in values:
        if value <= 0.0 or value >= 1.0:
            raise SystemExit("--top-p values must be greater than 0 and less than 1")
    return values


def _parse_budget_quantiles(raw: str) -> list[float]:
    values = _parse_floats(raw)
    if not values:
        raise SystemExit("--budget-quantiles must contain at least one quantile")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise SystemExit("--budget-quantiles values must be in the range 0..1")
    return values


def _row_fieldnames() -> list[str]:
    return [
        "capture",
        "module",
        "capture_index",
        "dtype",
        "device",
        "batch_size",
        "heads",
        "query_tokens",
        "key_tokens",
        "head_dim",
        "mode",
        "setting",
        "top_p",
        "budget_quantile",
        "budget_mean",
        "budget_min",
        "budget_max",
        "kept_keys_mean",
        "kept_keys_min",
        "kept_keys_max",
        "kept_key_fraction",
        "attention_mass_mean",
        "attention_mass_min",
        "mse",
        "rmse",
        "relative_rmse",
        "mae",
        "max_abs",
        "cosine",
        "cosine_error",
    ]


def _head_fieldnames() -> list[str]:
    return [
        "capture",
        "module",
        "capture_index",
        "dtype",
        "device",
        "batch_size",
        "heads",
        "query_tokens",
        "key_tokens",
        "head_dim",
        "top_p",
        "head",
        "kept_keys_mean",
        "kept_keys_min",
        "kept_keys_max",
        "kept_keys_p50",
        "kept_keys_p90",
        "kept_keys_p95",
        "kept_keys_p99",
        "kept_key_fraction_mean",
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, args: argparse.Namespace, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "rows": result["rows"],
        "summary": result["summary"],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_payload(
    path: Path,
    args: argparse.Namespace,
    key: str,
    payload_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        key: payload_rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "captures": len({row["capture"] for row in rows}),
        "best_relative_rmse": min(
            (float(row["relative_rmse"]) for row in rows),
            default=None,
        ),
        "best_cosine_error": min(
            (float(row["cosine_error"]) for row in rows),
            default=None,
        ),
    }


def _print_summary(rows: list[dict[str, Any]]) -> None:
    summary = _summary(rows)
    print("attention budget calibration complete")
    print(
        f"captures={summary['captures']} rows={summary['rows']} "
        f"best_rel_rmse={summary['best_relative_rmse']} "
        f"best_cos_err={summary['best_cosine_error']}"
    )
    for row in rows:
        print(
            f"{Path(str(row['capture'])).name} "
            f"{row['setting']} "
            f"budget_mean={row['budget_mean']:.1f} "
            f"kept={row['kept_key_fraction']:.4f} "
            f"mass={row['attention_mass_mean']:.4f} "
            f"rel_rmse={row['relative_rmse']:.6f} "
            f"cos_err={row['cosine_error']:.6f} "
            f"module={row['module']}"
        )


if __name__ == "__main__":
    main()
