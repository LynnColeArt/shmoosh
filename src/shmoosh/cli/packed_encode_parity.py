from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from shmoosh.packed_attention import packed_key_attention_output
from shmoosh.packed_keys import PackedKeyBlock, encode_packed_keys
from shmoosh.packed_scores import _code_indices, score_resources_from_codec
from shmoosh.quantization import ShmooshCodec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare split and fused packed-key encode parity on Q/K/V captures."
    )
    parser.add_argument("captures", nargs="+", help="Capture .npz files or directories.")
    parser.add_argument("--bits", type=int, default=7)
    parser.add_argument("--qjl-bits", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--codebook-samples", type=int, default=80_000)
    parser.add_argument("--lloyd-iters", type=int, default=80)
    parser.add_argument(
        "--code-format",
        choices=["packed", "packed_t"],
        default="packed_t",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Tensor dtype used for the CUDA encode probe.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--attention-backend",
        choices=["torch", "triton", "auto"],
        default="torch",
        help="Backend used only for split-vs-fused attention-output comparison.",
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
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Skip split-vs-fused attention-output comparison.",
    )
    parser.add_argument("--csv", default="captures/packed_encode_parity.csv")
    parser.add_argument("--json", default="captures/packed_encode_parity.json")
    parser.add_argument("--max-code-diff-rate", type=float)
    parser.add_argument("--max-output-mse", type=float)
    args = parser.parse_args()

    rows = run_parity(args)
    _write_csv(Path(args.csv), rows)
    _write_json(Path(args.json), args, rows)
    _print_summary(rows)
    _enforce_thresholds(args, rows)


def run_parity(args: argparse.Namespace) -> list[dict[str, Any]]:
    torch = _load_torch()
    device = _select_device(torch, args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    paths = _expand_captures(args.captures)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("No capture files found.")

    resources_cache: dict[tuple[int, str], tuple[ShmooshCodec, Any]] = {}
    rows: list[dict[str, Any]] = []
    for path in paths:
        row = _compare_capture(
            path,
            args=args,
            torch=torch,
            device=device,
            dtype=dtype,
            resources_cache=resources_cache,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise SystemExit("No captures matched the requested filters.")
    return rows


def _compare_capture(
    path: Path,
    *,
    args: argparse.Namespace,
    torch: Any,
    device: Any,
    dtype: Any,
    resources_cache: dict[tuple[int, str], tuple[ShmooshCodec, Any]],
) -> dict[str, Any] | None:
    loaded = np.load(path)
    metadata = _load_metadata(loaded)
    module = str(metadata.get("module", ""))
    if args.module_filter and args.module_filter not in module:
        return None
    if args.self_attn_only and not module.endswith("attn1"):
        return None

    q = _attention_tensor(torch, loaded["q"], device=device, dtype=dtype)
    key = _attention_tensor(torch, loaded["k"], device=device, dtype=dtype)
    value = _attention_tensor(torch, loaded["v"], device=device, dtype=dtype)
    if int(key.shape[2]) < args.min_key_tokens:
        return None
    dim = int(key.shape[-1])
    cache_key = (dim, str(device))
    if cache_key not in resources_cache:
        codec = ShmooshCodec(
            dim=dim,
            bits=args.bits,
            qjl_bits=args.qjl_bits,
            seed=args.seed,
            codebook_samples=args.codebook_samples,
            lloyd_iters=args.lloyd_iters,
        )
        resources_cache[cache_key] = (
            codec,
            score_resources_from_codec(codec, device=device),
        )
    codec, resources = resources_cache[cache_key]

    split = encode_packed_keys(
        key,
        bits=args.bits,
        qjl_bits=args.qjl_bits,
        seed=args.seed,
        codebook_samples=args.codebook_samples,
        lloyd_iters=args.lloyd_iters,
        codec=codec,
        resources=resources,
        key_encode_backend="split",
        code_format=args.code_format,
    )
    fused = encode_packed_keys(
        key,
        bits=args.bits,
        qjl_bits=args.qjl_bits,
        seed=args.seed,
        codebook_samples=args.codebook_samples,
        lloyd_iters=args.lloyd_iters,
        codec=codec,
        resources=resources,
        key_encode_backend="fused",
        code_format=args.code_format,
    )

    split_indices = _code_indices(split, device=device)
    fused_indices = _code_indices(fused, device=device)
    code_delta = split_indices != fused_indices
    norm_delta = (
        split.norms.to(dtype=torch.float32) - fused.norms.to(dtype=torch.float32)
    ).abs()

    output_metrics = {
        "output_max_abs": None,
        "output_mean_abs": None,
        "output_mse": None,
        "codes_only_output_mse": None,
        "norms_only_output_mse": None,
    }
    if not args.no_output:
        split_output = packed_key_attention_output(
            q,
            split,
            value,
            resources=resources,
            backend=args.attention_backend,
        )
        fused_output = packed_key_attention_output(
            q,
            fused,
            value,
            resources=resources,
            backend=args.attention_backend,
        )
        fused_codes_split_norms = _replace_block_norms(fused, split)
        split_codes_fused_norms = _replace_block_norms(split, fused)
        codes_only_output = packed_key_attention_output(
            q,
            fused_codes_split_norms,
            value,
            resources=resources,
            backend=args.attention_backend,
        )
        norms_only_output = packed_key_attention_output(
            q,
            split_codes_fused_norms,
            value,
            resources=resources,
            backend=args.attention_backend,
        )
        output_metrics = {
            **_delta_metrics(torch, split_output, fused_output, prefix="output"),
            "codes_only_output_mse": _mse(torch, split_output, codes_only_output),
            "norms_only_output_mse": _mse(torch, split_output, norms_only_output),
        }

    if getattr(torch, "cuda", None) is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)

    code_diff_count = int(code_delta.sum().item())
    code_element_count = int(code_delta.numel())
    return {
        "capture": str(path),
        "module": module,
        "capture_index": metadata.get("capture_index", ""),
        "dtype": "fp16" if dtype == torch.float16 else "fp32",
        "bits": args.bits,
        "qjl_bits": args.qjl_bits,
        "code_format": args.code_format,
        "heads": int(key.shape[1]),
        "query_tokens": int(q.shape[2]),
        "key_tokens": int(key.shape[2]),
        "head_dim": dim,
        "code_diff_count": code_diff_count,
        "code_element_count": code_element_count,
        "code_diff_rate": (
            code_diff_count / code_element_count if code_element_count else 0.0
        ),
        "norm_max_abs": float(norm_delta.max().item()),
        "norm_mean_abs": float(norm_delta.mean().item()),
        **output_metrics,
    }


def _replace_block_norms(
    codes_block: PackedKeyBlock,
    norms_block: PackedKeyBlock,
) -> PackedKeyBlock:
    return PackedKeyBlock(
        codes=codes_block.codes,
        norms=norms_block.norms,
        residual_signs=codes_block.residual_signs,
        residual_norms=codes_block.residual_norms,
        bits=codes_block.bits,
        qjl_bits=codes_block.qjl_bits,
        head_dim=codes_block.head_dim,
        seed=codes_block.seed,
        codebook_samples=codes_block.codebook_samples,
        lloyd_iters=codes_block.lloyd_iters,
        code_format=codes_block.code_format,
        norm_dtype=codes_block.norm_dtype,
    )


def _delta_metrics(torch: Any, reference: Any, candidate: Any, *, prefix: str) -> dict[str, float]:
    delta = (reference.to(dtype=torch.float32) - candidate.to(dtype=torch.float32)).abs()
    return {
        f"{prefix}_max_abs": float(delta.max().item()),
        f"{prefix}_mean_abs": float(delta.mean().item()),
        f"{prefix}_mse": _mse(torch, reference, candidate),
    }


def _mse(torch: Any, reference: Any, candidate: Any) -> float:
    reference_f = reference.to(dtype=torch.float32)
    candidate_f = candidate.to(dtype=torch.float32)
    return float(torch.mean((reference_f - candidate_f) ** 2).item())


def _attention_tensor(torch: Any, array: np.ndarray, *, device: Any, dtype: Any) -> Any:
    tensor = torch.from_numpy(array)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(
            "capture tensors must have shape (heads,tokens,dim) "
            "or (batch,heads,tokens,dim)"
        )
    return tensor.to(device=device, dtype=dtype)


def _load_metadata(loaded: Any) -> dict[str, Any]:
    if "metadata" not in loaded:
        return {}
    return json.loads(str(loaded["metadata"]))


def _expand_captures(paths: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.npz")))
        elif path.suffix == ".npz":
            expanded.append(path)
    return expanded


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "capture",
        "module",
        "capture_index",
        "dtype",
        "bits",
        "qjl_bits",
        "code_format",
        "heads",
        "query_tokens",
        "key_tokens",
        "head_dim",
        "code_diff_count",
        "code_element_count",
        "code_diff_rate",
        "norm_max_abs",
        "norm_mean_abs",
        "output_max_abs",
        "output_mean_abs",
        "output_mse",
        "codes_only_output_mse",
        "norms_only_output_mse",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
    code_diffs = [int(row["code_diff_count"]) for row in rows]
    code_rates = [float(row["code_diff_rate"]) for row in rows]
    output_mses = [
        float(row["output_mse"])
        for row in rows
        if row.get("output_mse") is not None
    ]
    return {
        "captures": len(rows),
        "total_code_diff_count": sum(code_diffs),
        "max_code_diff_count": max(code_diffs) if code_diffs else 0,
        "max_code_diff_rate": max(code_rates) if code_rates else 0.0,
        "max_output_mse": max(output_mses) if output_mses else None,
    }


def _print_summary(rows: list[dict[str, Any]]) -> None:
    summary = _summary(rows)
    print("Shmoosh packed encode parity")
    print(
        f"captures={summary['captures']} "
        f"total_code_diffs={summary['total_code_diff_count']} "
        f"max_code_diff_rate={summary['max_code_diff_rate']:.8g} "
        f"max_output_mse={summary['max_output_mse']}"
    )
    for row in rows:
        print(
            f"{Path(str(row['capture'])).name} "
            f"tokens={row['key_tokens']} "
            f"code_diffs={row['code_diff_count']} "
            f"rate={row['code_diff_rate']:.8g} "
            f"output_mse={row['output_mse']} "
            f"module={row['module']}"
        )


def _enforce_thresholds(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    failures: list[str] = []
    if args.max_code_diff_rate is not None:
        for row in rows:
            if float(row["code_diff_rate"]) > args.max_code_diff_rate:
                failures.append(
                    f"{row['capture']} code_diff_rate={row['code_diff_rate']:.8g}"
                )
    if args.max_output_mse is not None:
        for row in rows:
            output_mse = row.get("output_mse")
            if output_mse is not None and float(output_mse) > args.max_output_mse:
                failures.append(f"{row['capture']} output_mse={output_mse:.8g}")
    if failures:
        raise SystemExit("Parity thresholds exceeded:\n" + "\n".join(failures))


def _select_device(torch: Any, device: str) -> Any:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise SystemExit("CUDA is required for fused key encode parity checks.")
    selected = torch.device(device)
    if selected.type != "cuda":
        raise SystemExit("CUDA is required for fused key encode parity checks.")
    return selected


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "torch is required for packed encode parity; install optional dependencies first"
        ) from exc
    return torch


if __name__ == "__main__":
    main()
