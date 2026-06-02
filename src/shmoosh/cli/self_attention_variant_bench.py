from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path
import time
from typing import Any, Callable

from shmoosh.packed_attention import (
    packed_key_attention_output,
    triton_packed_key_attention_output,
)
from shmoosh.packed_keys import encode_packed_keys
from shmoosh.packed_scores import score_resources_from_codec
from shmoosh.quantization import ShmooshCodec
from shmoosh.rotated_attention import (
    rotated_key_attention_output,
    triton_rotated_key_attention_output,
)
from shmoosh.rotated_keys import encode_rotated_keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark packed-K exact-V self-attention variants on synthetic "
            "Diffusers-shaped Q/K/V tensors."
        )
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--query-tokens", type=int, default=1024)
    parser.add_argument("--key-tokens", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument(
        "--variants",
        default="6:128,6:64,6:0,7:0",
        help="Comma-separated bits:qjl_bits variants, e.g. 6:128,6:64,6:0,7:0.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--codebook-samples", type=int, default=80_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument(
        "--backend",
        choices=["auto", "torch", "triton"],
        default="auto",
    )
    parser.add_argument(
        "--code-format",
        choices=["packed", "packed_t", "byte", "rotated"],
        default="packed",
        help="Runtime K representation to benchmark.",
    )
    parser.add_argument(
        "--norm-dtype",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Stored dtype for packed-key norms.",
    )
    parser.add_argument(
        "--key-encode-backend",
        choices=["split", "fused", "auto"],
        default="split",
        help="Packed-key encode path for K7/head64/no-QJL experiments.",
    )
    parser.add_argument(
        "--dot-precision",
        choices=["ieee", "tf32", "tf32x3"],
        default="ieee",
        help="Triton tl.dot input precision for packed attention.",
    )
    parser.add_argument(
        "--rotation-dot-precision",
        choices=["ieee", "tf32", "tf32x3"],
        help="Override Triton tl.dot input precision for Q rotation.",
    )
    parser.add_argument(
        "--score-dot-precision",
        choices=["ieee", "tf32", "tf32x3"],
        help="Override Triton tl.dot input precision for codebook score dots.",
    )
    parser.add_argument(
        "--value-dot-precision",
        choices=["ieee", "tf32", "tf32x3"],
        help="Override Triton tl.dot input precision for attention-weight/V dots.",
    )
    parser.add_argument(
        "--qjl-dot-precision",
        choices=["ieee", "tf32", "tf32x3"],
        help="Override Triton tl.dot input precision for QJL residual dots.",
    )
    parser.add_argument(
        "--block-k",
        type=int,
        help="Optional explicit Triton streaming key tile for attention timing.",
    )
    parser.add_argument(
        "--block-q",
        type=int,
        help="Optional explicit Triton query tile for attention timing.",
    )
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help=(
            "Also time attention and encode+attention through CUDA graph replay. "
            "This is a fixed-shape ceiling probe, not a production runtime path."
        ),
    )
    parser.add_argument("--output-dir", default="captures/self-attention-variant-bench")
    args = parser.parse_args()

    torch = _load_torch()
    device = _select_device(torch, args.device)
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    variants = _parse_variants(args.variants)
    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if args.warmup_iters < 0:
        raise SystemExit("--warmup-iters must be non-negative")

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
    exact_seconds = _time_call(
        lambda: _exact_attention(query, key, value),
        torch=torch,
        device=device,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
    )

    rows = []
    for bits, qjl_bits in variants:
        try:
            rows.append(
                _run_variant(
                    query,
                    key,
                    value,
                    exact,
                    torch=torch,
                    args=args,
                    device=device,
                    bits=bits,
                    qjl_bits=qjl_bits,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "bits": bits,
                    "qjl_bits": qjl_bits,
                    "backend": args.backend,
                    "code_format": args.code_format,
                    "norm_dtype": args.norm_dtype,
                    "key_encode_backend": args.key_encode_backend,
                    "dot_precision": args.dot_precision,
                    **_dot_precision_payload(args),
                    "block_q": args.block_q,
                    "block_k": args.block_k,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "shape": {
            "batch_size": args.batch_size,
            "heads": args.heads,
            "query_tokens": args.query_tokens,
            "key_tokens": args.key_tokens,
            "dim": args.dim,
        },
        "seed": args.seed,
        "dtype": args.dtype,
        "device": str(device),
        "backend": args.backend,
        "code_format": args.code_format,
        "norm_dtype": args.norm_dtype,
        "key_encode_backend": args.key_encode_backend,
        "dot_precision": args.dot_precision,
        **_dot_precision_payload(args),
        "block_q": args.block_q,
        "block_k": args.block_k,
        "cuda_graph": args.cuda_graph,
        "warmup_iters": args.warmup_iters,
        "iters": args.iters,
        "exact_attention_seconds": exact_seconds,
        "exact_attention_ms_per_iter": exact_seconds * 1000 / args.iters,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "summary.csv", rows)
    _print_summary(rows, exact_ms=payload["exact_attention_ms_per_iter"])


def _run_variant(
    query: Any,
    key: Any,
    value: Any,
    exact: Any,
    *,
    torch: Any,
    args: argparse.Namespace,
    device: str,
    bits: int,
    qjl_bits: int,
) -> dict[str, Any]:
    codec = ShmooshCodec(
        dim=args.dim,
        bits=bits,
        qjl_bits=qjl_bits,
        seed=args.seed,
        codebook_samples=args.codebook_samples,
    )
    resources = score_resources_from_codec(codec, device=device)

    def encode_once():
        if args.code_format == "rotated":
            return encode_rotated_keys(
                key,
                resources=resources,
                seed=args.seed,
            )
        return encode_packed_keys(
            key,
            bits=bits,
            qjl_bits=qjl_bits,
            seed=args.seed,
            codebook_samples=args.codebook_samples,
            codec=codec,
            resources=resources,
            key_encode_backend=args.key_encode_backend,
            code_format=args.code_format,
            norm_dtype=args.norm_dtype,
        )

    block = encode_once()
    attention = _attention_call(query, block, value, resources=resources, args=args)
    packed_output = attention()
    metrics = _quality_metrics(packed_output, exact, torch=torch)

    encode_seconds = _time_call(
        encode_once,
        torch=torch,
        device=device,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
    )
    attention_seconds = _time_call(
        attention,
        torch=torch,
        device=device,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
    )

    def encode_and_attention_once():
        active_block = encode_once()
        return _attention_call(
            query,
            active_block,
            value,
            resources=resources,
            args=args,
        )()

    total_seconds = _time_call(
        encode_and_attention_once,
        torch=torch,
        device=device,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
    )

    row = {
        "bits": bits,
        "qjl_bits": qjl_bits,
        "code_format": block.code_format,
        "norm_dtype": getattr(block, "norm_dtype", "fp32"),
        "key_encode_backend": args.key_encode_backend,
        "dot_precision": args.dot_precision,
        **_dot_precision_payload(args),
        "packed_bytes_per_vector": block.packed_bytes_per_vector,
        "compression_ratio_fp16": block.compression_ratio(dtype_bytes=2),
        "encode_ms_per_iter": encode_seconds * 1000 / args.iters,
        "attention_ms_per_iter": attention_seconds * 1000 / args.iters,
        "total_ms_per_iter": total_seconds * 1000 / args.iters,
        "backend": args.backend,
        "block_q": args.block_q,
        "block_k": args.block_k,
    }
    if args.cuda_graph:
        try:
            graph_attention_seconds = _time_cuda_graph_call(
                attention,
                torch=torch,
                device=device,
                warmup_iters=args.warmup_iters,
                iters=args.iters,
            )
            graph_total_seconds = _time_cuda_graph_call(
                encode_and_attention_once,
                torch=torch,
                device=device,
                warmup_iters=args.warmup_iters,
                iters=args.iters,
            )
            row["graph_attention_ms_per_iter"] = (
                graph_attention_seconds * 1000 / args.iters
            )
            row["graph_total_ms_per_iter"] = graph_total_seconds * 1000 / args.iters
        except Exception as exc:
            row["cuda_graph_error"] = f"{type(exc).__name__}: {exc}"
    row.update(metrics)
    return row


def _attention_call(
    query: Any,
    block: Any,
    value: Any,
    *,
    resources: Any,
    args: argparse.Namespace,
) -> Callable[[], Any]:
    if args.block_q is not None or args.block_k is not None:
        kwargs = {}
        if args.block_q is not None:
            kwargs["block_q"] = args.block_q
        if args.block_k is not None:
            kwargs["block_k"] = args.block_k
        if args.code_format == "rotated":
            return lambda: triton_rotated_key_attention_output(
                query,
                block,
                value,
                resources=resources,
                **kwargs,
            )
        return lambda: triton_packed_key_attention_output(
            query,
            block,
            value,
            resources=resources,
            dot_precision=args.dot_precision,
            **_dot_precision_kwargs(args),
            **kwargs,
        )
    if args.code_format == "rotated":
        return lambda: rotated_key_attention_output(
            query,
            block,
            value,
            resources=resources,
            backend=args.backend,
        )
    return lambda: packed_key_attention_output(
        query,
        block,
        value,
        resources=resources,
        backend=args.backend,
        dot_precision=args.dot_precision,
        **_dot_precision_kwargs(args),
    )


def _exact_attention(query: Any, key: Any, value: Any) -> Any:
    torch = _load_torch()
    query_f = query.to(dtype=torch.float32)
    key_f = key.to(dtype=torch.float32)
    value_f = value.to(dtype=torch.float32)
    scores = torch.matmul(query_f, key_f.transpose(-2, -1)) / sqrt(int(query.shape[-1]))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value_f)


def _quality_metrics(output: Any, exact: Any, *, torch: Any) -> dict[str, float]:
    output_f = output.to(dtype=torch.float32)
    exact_f = exact.to(dtype=torch.float32)
    delta = output_f - exact_f
    mse = torch.mean(delta.square()).item()
    rmse = mse ** 0.5
    exact_rms = torch.mean(exact_f.square()).sqrt().item()
    cosine = torch.nn.functional.cosine_similarity(
        output_f.reshape(1, -1),
        exact_f.reshape(1, -1),
    ).item()
    return {
        "mse": mse,
        "rmse": rmse,
        "relative_rmse": rmse / exact_rms if exact_rms else 0.0,
        "mae": torch.mean(torch.abs(delta)).item(),
        "max_abs": torch.max(torch.abs(delta)).item(),
        "cosine": cosine,
        "cosine_error": 1.0 - cosine,
    }


def _time_call(
    fn: Callable[[], Any],
    *,
    torch: Any,
    device: str,
    warmup_iters: int,
    iters: int,
) -> float:
    for _ in range(warmup_iters):
        fn()
    _synchronize(torch, device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _synchronize(torch, device)
    return time.perf_counter() - start


def _time_cuda_graph_call(
    fn: Callable[[], Any],
    *,
    torch: Any,
    device: str,
    warmup_iters: int,
    iters: int,
) -> float:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("CUDA graph timing requires a CUDA device")

    cuda_device = torch.device(device)
    with torch.cuda.device(cuda_device):
        side_stream = torch.cuda.Stream(device=cuda_device)
        side_stream.wait_stream(torch.cuda.current_stream(cuda_device))
        with torch.cuda.stream(side_stream):
            for _ in range(max(warmup_iters, 1)):
                fn()
        torch.cuda.current_stream(cuda_device).wait_stream(side_stream)
        torch.cuda.synchronize(cuda_device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()

        for _ in range(warmup_iters):
            graph.replay()
        torch.cuda.synchronize(cuda_device)

        start = time.perf_counter()
        for _ in range(iters):
            graph.replay()
        torch.cuda.synchronize(cuda_device)
        return time.perf_counter() - start


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "bits",
        "qjl_bits",
        "code_format",
        "norm_dtype",
        "key_encode_backend",
        "dot_precision",
        "rotation_dot_precision",
        "score_dot_precision",
        "value_dot_precision",
        "qjl_dot_precision",
        "packed_bytes_per_vector",
        "compression_ratio_fp16",
        "encode_ms_per_iter",
        "attention_ms_per_iter",
        "total_ms_per_iter",
        "graph_attention_ms_per_iter",
        "graph_total_ms_per_iter",
        "mse",
        "rmse",
        "relative_rmse",
        "mae",
        "max_abs",
        "cosine",
        "cosine_error",
        "backend",
        "block_q",
        "block_k",
        "cuda_graph_error",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict[str, Any]], *, exact_ms: float) -> None:
    print("self-attention variant bench complete")
    print(f"exact_attention_ms_per_iter={exact_ms:.4f}")
    for row in rows:
        if row.get("error"):
            print(
                f"K{row['bits']} QJL{row['qjl_bits']} "
                f"failed: {row['error']}"
            )
            continue
        print(
            f"K{row['bits']} QJL{row['qjl_bits']} "
            f"{row['code_format']} norms={row['norm_dtype']} "
            f"encode_backend={row['key_encode_backend']} "
            f"dot={_dot_precision_label(row)} "
            f"total={row['total_ms_per_iter']:.4f}ms "
            f"encode={row['encode_ms_per_iter']:.4f}ms "
            f"attention={row['attention_ms_per_iter']:.4f}ms "
            f"rel_rmse={row['relative_rmse']:.6f} "
            f"cos_err={row['cosine_error']:.6f} "
            f"bytes={row['packed_bytes_per_vector']}"
        )
        if row.get("graph_attention_ms_per_iter") is not None:
            print(
                f"  cuda_graph attention={row['graph_attention_ms_per_iter']:.4f}ms "
                f"total={row['graph_total_ms_per_iter']:.4f}ms"
            )
        elif row.get("cuda_graph_error"):
            print(f"  cuda_graph failed: {row['cuda_graph_error']}")


def _parse_variants(raw: str) -> list[tuple[int, int]]:
    variants = []
    for entry in raw.split(","):
        if not entry.strip():
            continue
        try:
            bits_raw, qjl_raw = entry.split(":", maxsplit=1)
            bits = int(bits_raw.strip())
            qjl_bits = int(qjl_raw.strip())
        except ValueError as exc:
            raise SystemExit(
                "--variants must be comma-separated bits:qjl_bits entries"
            ) from exc
        if bits <= 0 or bits > 8:
            raise SystemExit("variant bits must be in the range 1..8")
        if qjl_bits < 0:
            raise SystemExit("variant qjl_bits must be non-negative")
        variants.append((bits, qjl_bits))
    if not variants:
        raise SystemExit("--variants must contain at least one variant")
    return variants


def _dot_precision_kwargs(args: argparse.Namespace) -> dict[str, str]:
    return {
        key: value
        for key, value in {
            "rotation_dot_precision": args.rotation_dot_precision,
            "score_dot_precision": args.score_dot_precision,
            "value_dot_precision": args.value_dot_precision,
            "qjl_dot_precision": args.qjl_dot_precision,
        }.items()
        if value is not None
    }


def _dot_precision_payload(args: argparse.Namespace) -> dict[str, str]:
    return {
        "rotation_dot_precision": args.rotation_dot_precision or args.dot_precision,
        "score_dot_precision": args.score_dot_precision or args.dot_precision,
        "value_dot_precision": args.value_dot_precision or args.dot_precision,
        "qjl_dot_precision": (
            args.qjl_dot_precision
            or args.score_dot_precision
            or args.dot_precision
        ),
    }


def _dot_precision_label(row: dict[str, Any]) -> str:
    base = row.get("dot_precision", "ieee")
    split = {
        "rot": row.get("rotation_dot_precision", base),
        "score": row.get("score_dot_precision", base),
        "value": row.get("value_dot_precision", base),
        "qjl": row.get("qjl_dot_precision", row.get("score_dot_precision", base)),
    }
    if all(value == base for value in split.values()):
        return str(base)
    return (
        f"{base}/rot={split['rot']},score={split['score']},"
        f"value={split['value']},qjl={split['qjl']}"
    )


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "torch is required for self-attention variant benchmarking; "
            "install optional dependencies first"
        ) from exc
    return torch


def _select_device(torch: Any, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _generator_device(torch: Any, device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return device
    return "cpu"


def _synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


if __name__ == "__main__":
    main()
