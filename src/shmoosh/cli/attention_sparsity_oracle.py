from __future__ import annotations

import argparse
import csv
import json
from math import isqrt, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from shmoosh.cli.packed_encode_parity import (
    _attention_tensor,
    _expand_captures,
    _load_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure exact-attention quality loss for candidate sparse attention "
            "masks on captured or synthetic Q/K/V tensors."
        )
    )
    parser.add_argument(
        "captures",
        nargs="*",
        help="Capture .npz files or directories. Omit with --synthetic.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run one synthetic self-attention tensor instead of loading captures.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--query-tokens", type=int)
    parser.add_argument("--key-tokens", type=int)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument(
        "--top-k",
        default="64,128,256",
        help="Comma-separated top-k budgets. Use an empty string to skip.",
    )
    parser.add_argument(
        "--top-p",
        default="0.95,0.98",
        help="Comma-separated attention-mass budgets. Use an empty string to skip.",
    )
    parser.add_argument(
        "--local-windows",
        default="9,17,33",
        help="Comma-separated square local windows. Use an empty string to skip.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-key-tokens", type=int, default=0)
    parser.add_argument(
        "--module-filter",
        help="Substring filter applied to capture metadata['module'].",
    )
    parser.add_argument(
        "--self-attn-only",
        action="store_true",
        help="Only include captures whose module metadata ends in attn1.",
    )
    parser.add_argument("--csv", default="captures/attention_sparsity_oracle.csv")
    parser.add_argument("--json", default="captures/attention_sparsity_oracle.json")
    args = parser.parse_args()

    rows = run_oracle(args)
    _write_csv(Path(args.csv), rows)
    _write_json(Path(args.json), args, rows)
    _print_summary(rows)


def run_oracle(args: argparse.Namespace) -> list[dict[str, Any]]:
    torch = _load_torch()
    device = _select_device(torch, args.device)
    dtype = _dtype(torch, args.dtype)
    specs = _mask_specs(args)
    if not specs:
        raise SystemExit("At least one sparse mask family must be enabled.")

    rows: list[dict[str, Any]] = []
    if args.synthetic or not args.captures:
        query, key, value = _synthetic_tensors(torch, args, device=device, dtype=dtype)
        rows.extend(
            _run_tensor_oracle(
                query,
                key,
                value,
                torch=torch,
                specs=specs,
                base_row={
                    "capture": "synthetic",
                    "module": "synthetic.attn1",
                    "capture_index": "",
                    "dtype": args.dtype,
                    "device": str(device),
                },
            )
        )
        return rows

    paths = _expand_captures(args.captures)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("No capture files found.")

    for path in paths:
        rows.extend(
            _run_capture_oracle(
                path,
                args=args,
                torch=torch,
                device=device,
                dtype=dtype,
                specs=specs,
            )
        )
    if not rows:
        raise SystemExit("No captures matched the requested filters.")
    return rows


def _run_capture_oracle(
    path: Path,
    *,
    args: argparse.Namespace,
    torch: Any,
    device: str,
    dtype: Any,
    specs: list[tuple[str, float | int]],
) -> list[dict[str, Any]]:
    loaded = np.load(path)
    metadata = _load_metadata(loaded)
    module = str(metadata.get("module", ""))
    if args.module_filter and args.module_filter not in module:
        return []
    if args.self_attn_only and not module.endswith("attn1"):
        return []

    query = _attention_tensor(torch, loaded["q"], device=device, dtype=dtype)
    key = _attention_tensor(torch, loaded["k"], device=device, dtype=dtype)
    value = _attention_tensor(torch, loaded["v"], device=device, dtype=dtype)
    if int(key.shape[2]) < args.min_key_tokens:
        return []

    return _run_tensor_oracle(
        query,
        key,
        value,
        torch=torch,
        specs=specs,
        base_row={
            "capture": str(path),
            "module": module,
            "capture_index": metadata.get("capture_index", ""),
            "dtype": args.dtype,
            "device": str(device),
        },
    )


def _run_tensor_oracle(
    query: Any,
    key: Any,
    value: Any,
    *,
    torch: Any,
    specs: list[tuple[str, float | int]],
    base_row: dict[str, Any],
) -> list[dict[str, Any]]:
    query_f = query.to(dtype=torch.float32)
    key_f = key.to(dtype=torch.float32)
    value_f = value.to(dtype=torch.float32)
    logits = torch.matmul(query_f, key_f.transpose(-2, -1)) / sqrt(int(query.shape[-1]))
    dense_weights = torch.softmax(logits, dim=-1)
    dense_output = torch.matmul(dense_weights, value_f)
    shape_row = {
        "batch_size": int(query.shape[0]),
        "heads": int(query.shape[1]),
        "query_tokens": int(query.shape[2]),
        "key_tokens": int(key.shape[2]),
        "head_dim": int(query.shape[-1]),
    }

    rows = [
        {
            **base_row,
            **shape_row,
            "mode": "dense",
            "setting": "all",
            "kept_keys_mean": float(key.shape[2]),
            "kept_keys_min": int(key.shape[2]),
            "kept_keys_max": int(key.shape[2]),
            "kept_key_fraction": 1.0,
            "attention_mass_mean": 1.0,
            "attention_mass_min": 1.0,
            **_quality_metrics(dense_output, dense_output, torch=torch),
        }
    ]
    for mode, setting in specs:
        if mode == "local_window" and not _supports_local_window(logits):
            continue
        mask = _build_mask(mode, setting, logits, dense_weights, torch=torch)
        sparse_output = _sparse_attention(logits, value_f, mask, torch=torch)
        rows.append(
            {
                **base_row,
                **shape_row,
                "mode": mode,
                "setting": setting,
                **_mask_stats(mask, dense_weights, torch=torch),
                **_quality_metrics(sparse_output, dense_output, torch=torch),
            }
        )
    return rows


def _build_mask(
    mode: str,
    setting: float | int,
    logits: Any,
    dense_weights: Any,
    *,
    torch: Any,
) -> Any:
    if mode == "top_k":
        return _topk_mask(logits, int(setting), torch=torch)
    if mode == "top_p":
        return _topp_mask(dense_weights, float(setting), torch=torch)
    if mode == "local_window":
        return _local_window_mask(
            torch,
            int(logits.shape[-2]),
            int(setting),
            device=logits.device,
        )
    raise ValueError(f"unknown mask mode: {mode}")


def _supports_local_window(logits: Any) -> bool:
    query_tokens = int(logits.shape[-2])
    key_tokens = int(logits.shape[-1])
    if query_tokens != key_tokens:
        return False
    side = isqrt(query_tokens)
    return side * side == query_tokens


def _topk_mask(logits: Any, k: int, *, torch: Any) -> Any:
    if k <= 0:
        raise ValueError("top-k values must be positive")
    k = min(k, int(logits.shape[-1]))
    indices = torch.topk(logits, k=k, dim=-1).indices
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask


def _topp_mask(weights: Any, threshold: float, *, torch: Any) -> Any:
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("top-p thresholds must be greater than 0 and less than 1")
    sorted_weights, sorted_indices = torch.sort(weights, dim=-1, descending=True)
    cumulative = sorted_weights.cumsum(dim=-1)
    first_cross = (cumulative >= threshold).to(dtype=torch.int64).argmax(dim=-1)
    positions = torch.arange(weights.shape[-1], device=weights.device)
    keep_sorted = positions <= first_cross.unsqueeze(-1)
    mask = torch.zeros_like(weights, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, keep_sorted)
    return mask


def _local_window_mask(torch: Any, tokens: int, window: int, *, device: Any) -> Any:
    if window <= 0 or window % 2 == 0:
        raise ValueError("local windows must be positive odd integers")
    side = _square_side(tokens)
    radius = window // 2
    positions = torch.arange(tokens, device=device)
    rows = positions // side
    cols = positions % side
    row_delta = torch.abs(rows[:, None] - rows[None, :])
    col_delta = torch.abs(cols[:, None] - cols[None, :])
    return (row_delta <= radius) & (col_delta <= radius)


def _square_side(tokens: int) -> int:
    side = isqrt(tokens)
    if side * side != tokens:
        raise ValueError("local-window masks require square self-attention tokens")
    return side


def _sparse_attention(logits: Any, value: Any, mask: Any, *, torch: Any) -> Any:
    expanded_mask = _expand_mask(mask, logits)
    masked_logits = logits.masked_fill(~expanded_mask, float("-inf"))
    weights = torch.softmax(masked_logits, dim=-1)
    return torch.matmul(weights, value)


def _mask_stats(mask: Any, dense_weights: Any, *, torch: Any) -> dict[str, float | int]:
    expanded_mask = _expand_mask(mask, dense_weights)
    kept = expanded_mask.to(dtype=torch.float32).sum(dim=-1)
    attention_mass = dense_weights.masked_fill(~expanded_mask, 0.0).sum(dim=-1)
    return {
        "kept_keys_mean": float(kept.mean().item()),
        "kept_keys_min": int(kept.min().item()),
        "kept_keys_max": int(kept.max().item()),
        "kept_key_fraction": float((kept / dense_weights.shape[-1]).mean().item()),
        "attention_mass_mean": float(attention_mass.mean().item()),
        "attention_mass_min": float(attention_mass.min().item()),
    }


def _expand_mask(mask: Any, reference: Any) -> Any:
    if mask.ndim == 2:
        return mask.reshape(1, 1, mask.shape[0], mask.shape[1]).expand_as(reference)
    return mask


def _quality_metrics(output: Any, exact: Any, *, torch: Any) -> dict[str, float]:
    output_f = output.to(dtype=torch.float32)
    exact_f = exact.to(dtype=torch.float32)
    delta = output_f - exact_f
    mse = torch.mean(delta.square()).item()
    rmse = mse**0.5
    exact_rms = torch.mean(exact_f.square()).sqrt().item()
    cosine = torch.nn.functional.cosine_similarity(
        output_f.reshape(1, -1),
        exact_f.reshape(1, -1),
    ).item()
    cosine = max(min(float(cosine), 1.0), -1.0)
    cosine_error = 0.0 if mse == 0.0 else 1.0 - cosine
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "relative_rmse": float(rmse / exact_rms) if exact_rms else 0.0,
        "mae": float(torch.mean(torch.abs(delta)).item()),
        "max_abs": float(torch.max(torch.abs(delta)).item()),
        "cosine": cosine,
        "cosine_error": float(cosine_error),
    }


def _synthetic_tensors(
    torch: Any,
    args: argparse.Namespace,
    *,
    device: str,
    dtype: Any,
) -> tuple[Any, Any, Any]:
    query_tokens = args.query_tokens or args.tokens
    key_tokens = args.key_tokens or args.tokens
    generator = torch.Generator(device=_generator_device(torch, device)).manual_seed(
        args.seed
    )
    query = torch.randn(
        args.batch_size,
        args.heads,
        query_tokens,
        args.dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    key = torch.randn(
        args.batch_size,
        args.heads,
        key_tokens,
        args.dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    value = torch.randn(
        args.batch_size,
        args.heads,
        key_tokens,
        args.dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return query, key, value


def _mask_specs(args: argparse.Namespace) -> list[tuple[str, float | int]]:
    specs: list[tuple[str, float | int]] = []
    specs.extend(("top_k", value) for value in _parse_ints(args.top_k))
    specs.extend(("top_p", value) for value in _parse_floats(args.top_p))
    specs.extend(("local_window", value) for value in _parse_ints(args.local_windows))
    return specs


def _parse_ints(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    for value in values:
        if value <= 0:
            raise ValueError("integer lists must contain positive values")
    return values


def _parse_floats(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    for value in values:
        if value <= 0.0:
            raise ValueError("float lists must contain positive values")
    return values


def _dtype(torch: Any, raw: str) -> Any:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[raw]


def _select_device(torch: Any, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _generator_device(torch: Any, device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return device
    return "cpu"


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "torch is required for attention sparsity oracle; "
            "install optional dependencies first"
        ) from exc
    return torch


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "rows": rows,
        "summary": _summary(rows),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sparse_rows = [row for row in rows if row["mode"] != "dense"]
    return {
        "rows": len(rows),
        "sparse_rows": len(sparse_rows),
        "captures": len({row["capture"] for row in rows}),
        "best_relative_rmse": min(
            (float(row["relative_rmse"]) for row in sparse_rows),
            default=None,
        ),
        "best_cosine_error": min(
            (float(row["cosine_error"]) for row in sparse_rows),
            default=None,
        ),
    }


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("attention sparsity oracle complete")
    summary = _summary(rows)
    print(
        f"captures={summary['captures']} sparse_rows={summary['sparse_rows']} "
        f"best_rel_rmse={summary['best_relative_rmse']} "
        f"best_cos_err={summary['best_cosine_error']}"
    )
    for row in rows:
        if row["mode"] == "dense":
            continue
        print(
            f"{Path(str(row['capture'])).name} "
            f"{row['mode']}={row['setting']} "
            f"kept={row['kept_key_fraction']:.4f} "
            f"mass={row['attention_mass_mean']:.4f} "
            f"rel_rmse={row['relative_rmse']:.6f} "
            f"cos_err={row['cosine_error']:.6f} "
            f"module={row['module']}"
        )


if __name__ == "__main__":
    main()
