from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from shmoosh.cli.self_attention_variant_bench import (
    _exact_attention,
    _generator_device,
    _load_torch,
    _run_variant,
    _select_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical CUDA perf guard for the 1024-token "
            "K7/packed_t/no-QJL self-attention path."
        )
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--query-tokens", type=int, default=1024)
    parser.add_argument("--key-tokens", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--codebook-samples", type=int, default=80_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmup-iters", type=int, default=12)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--max-attention-ms", type=float, default=0.45)
    parser.add_argument("--max-total-ms", type=float, default=0.75)
    parser.add_argument("--max-relative-rmse", type=float, default=0.03)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument(
        "--allow-cpu-skip",
        action="store_true",
        help="Exit successfully with a skipped payload when CUDA is unavailable.",
    )
    parser.add_argument(
        "--output-json",
        default="captures/packed-attention-perf-guard/k7-packedt-1024.json",
    )
    args = parser.parse_args()
    _validate_args(args)

    torch = _load_torch()
    device = _select_device(torch, args.device)
    output_path = Path(args.output_json)
    if not device.startswith("cuda") or not torch.cuda.is_available():
        payload = _skipped_payload(args, device=device)
        _write_payload(output_path, payload)
        print(f"packed attention perf guard skipped: CUDA unavailable ({device})")
        if args.allow_cpu_skip:
            return
        raise SystemExit("CUDA is required for packed attention perf guard")

    payload = _run_guard(args, torch=torch, device=device)
    failures = _threshold_failures(payload, args)
    payload["status"] = "failed" if failures else "passed"
    payload["failures"] = failures
    _write_payload(output_path, payload)
    _print_guard_summary(payload, output_path=output_path)
    if failures:
        raise SystemExit(1)


def _validate_args(args: argparse.Namespace) -> None:
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if args.warmup_iters < 0:
        raise SystemExit("--warmup-iters must be non-negative")
    for name in (
        "max_attention_ms",
        "max_total_ms",
        "max_relative_rmse",
        "min_cosine",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")


def _run_guard(args: argparse.Namespace, *, torch: Any, device: str) -> dict[str, Any]:
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    generator = torch.Generator(device=_generator_device(torch, device)).manual_seed(
        args.seed
    )
    query = torch.randn(
        args.batch_size,
        args.heads,
        args.query_tokens,
        args.dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    key = torch.randn(
        args.batch_size,
        args.heads,
        args.key_tokens,
        args.dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    value = torch.randn(
        args.batch_size,
        args.heads,
        args.key_tokens,
        args.dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    exact = _exact_attention(query, key, value)
    bench_args = _bench_args(args)
    rows = [
        _run_variant(
            query,
            key,
            value,
            exact,
            torch=torch,
            args=bench_args,
            device=device,
            bits=7,
            qjl_bits=0,
        )
        for _index in range(args.samples)
    ]
    return {
        "status": "running",
        "profile": "k7-packedt-noqjl-head64-1024",
        "shape": {
            "batch_size": args.batch_size,
            "heads": args.heads,
            "query_tokens": args.query_tokens,
            "key_tokens": args.key_tokens,
            "dim": args.dim,
        },
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "samples": args.samples,
        "warmup_iters": args.warmup_iters,
        "iters": args.iters,
        "thresholds": _threshold_payload(args),
        "config": {
            "bits": 7,
            "qjl_bits": 0,
            "code_format": "packed_t",
            "norm_dtype": "fp32",
            "key_encode_backend": "split",
            "backend": "auto",
            "dot_precision": "ieee",
            "rotation_dot_precision": "ieee",
            "score_dot_precision": "tf32",
            "value_dot_precision": "tf32",
            "qjl_dot_precision": "tf32",
            "block_q": None,
            "block_k": None,
        },
        "aggregate": _summarize_rows(rows),
        "rows": rows,
    }


def _bench_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dim=args.dim,
        seed=args.seed,
        codebook_samples=args.codebook_samples,
        backend="auto",
        code_format="packed_t",
        norm_dtype="fp32",
        key_encode_backend="split",
        dot_precision="ieee",
        rotation_dot_precision=None,
        score_dot_precision="tf32",
        value_dot_precision="tf32",
        qjl_dot_precision=None,
        block_q=None,
        block_k=None,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
        cuda_graph=False,
    )


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if any(row.get("error") for row in rows):
        return {"errors": [row["error"] for row in rows if row.get("error")]}
    return {
        "attention_ms_per_iter": _stats(
            [float(row["attention_ms_per_iter"]) for row in rows]
        ),
        "encode_ms_per_iter": _stats(
            [float(row["encode_ms_per_iter"]) for row in rows]
        ),
        "total_ms_per_iter": _stats(
            [float(row["total_ms_per_iter"]) for row in rows]
        ),
        "relative_rmse": _stats([float(row["relative_rmse"]) for row in rows]),
        "cosine": _stats([float(row["cosine"]) for row in rows]),
        "packed_bytes_per_vector": sorted(
            {int(row["packed_bytes_per_vector"]) for row in rows}
        ),
    }


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": float(median(values)),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _threshold_failures(payload: dict[str, Any], args: argparse.Namespace) -> list[str]:
    aggregate = payload.get("aggregate", {})
    if aggregate.get("errors"):
        return [f"sample error: {error}" for error in aggregate["errors"]]
    failures = []
    attention_median = aggregate["attention_ms_per_iter"]["median"]
    total_median = aggregate["total_ms_per_iter"]["median"]
    relative_rmse_max = aggregate["relative_rmse"]["max"]
    cosine_min = aggregate["cosine"]["min"]
    if attention_median > args.max_attention_ms:
        failures.append(
            f"median attention {attention_median:.4f}ms > {args.max_attention_ms:.4f}ms"
        )
    if total_median > args.max_total_ms:
        failures.append(
            f"median total {total_median:.4f}ms > {args.max_total_ms:.4f}ms"
        )
    if relative_rmse_max > args.max_relative_rmse:
        failures.append(
            f"max relative_rmse {relative_rmse_max:.6f} > {args.max_relative_rmse:.6f}"
        )
    if cosine_min < args.min_cosine:
        failures.append(f"min cosine {cosine_min:.6f} < {args.min_cosine:.6f}")
    return failures


def _threshold_payload(args: argparse.Namespace) -> dict[str, float]:
    return {
        "max_attention_ms_median": args.max_attention_ms,
        "max_total_ms_median": args.max_total_ms,
        "max_relative_rmse": args.max_relative_rmse,
        "min_cosine": args.min_cosine,
    }


def _skipped_payload(args: argparse.Namespace, *, device: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": "CUDA unavailable",
        "device": str(device),
        "thresholds": _threshold_payload(args),
    }


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _print_guard_summary(payload: dict[str, Any], *, output_path: Path) -> None:
    aggregate = payload["aggregate"]
    if payload["status"] == "failed":
        print("packed attention perf guard failed")
        for failure in payload["failures"]:
            print(f"  {failure}")
    else:
        print("packed attention perf guard passed")
    print(
        "attention="
        f"{aggregate['attention_ms_per_iter']['median']:.4f}ms median "
        f"(max {aggregate['attention_ms_per_iter']['max']:.4f}ms), "
        "total="
        f"{aggregate['total_ms_per_iter']['median']:.4f}ms median, "
        "rel_rmse="
        f"{aggregate['relative_rmse']['max']:.6f} max"
    )
    print(f"wrote guard payload: {output_path}")


if __name__ == "__main__":
    main()
