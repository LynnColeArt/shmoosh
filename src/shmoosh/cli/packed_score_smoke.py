from __future__ import annotations

import argparse
import time

from shmoosh.packed_keys import encode_packed_keys
from shmoosh.packed_scores import packed_key_scores, torch_packed_key_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test packed-key attention score kernels."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--query-tokens", type=int, default=8)
    parser.add_argument("--key-tokens", type=int, default=16)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--qjl-bits", type=int, default=32)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--codebook-samples", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--backend",
        choices=["auto", "torch", "triton"],
        default="auto",
    )
    args = parser.parse_args()

    torch = _load_torch()
    device = _select_device(torch, args.device)
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
        dtype=torch.float32,
    )
    key = torch.randn(
        args.batch_size,
        args.heads,
        args.key_tokens,
        args.dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    block = encode_packed_keys(
        key,
        bits=args.bits,
        qjl_bits=args.qjl_bits,
        seed=args.seed,
        codebook_samples=args.codebook_samples,
    )

    start = time.perf_counter()
    scores = packed_key_scores(query, block, backend=args.backend)
    score_seconds = time.perf_counter() - start
    reference = torch_packed_key_scores(query, block)
    max_abs = torch.max(torch.abs(scores - reference)).item()
    mean_abs = torch.mean(torch.abs(scores - reference)).item()

    print("Shmoosh packed-score smoke")
    print(
        "shape: "
        f"batch={args.batch_size} heads={args.heads} "
        f"query_tokens={args.query_tokens} key_tokens={args.key_tokens} "
        f"dim={args.dim}"
    )
    print(f"bits={args.bits} qjl_bits={args.qjl_bits} seed={args.seed}")
    print(f"device={device} backend={args.backend}")
    print(f"score_shape={tuple(scores.shape)}")
    print(f"max_abs_delta_vs_torch={max_abs:.8g}")
    print(f"mean_abs_delta_vs_torch={mean_abs:.8g}")
    print(f"score_seconds={score_seconds:.4f}")


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "torch is required for packed score smoke; install optional dependencies first"
        ) from exc
    return torch


def _select_device(torch, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _generator_device(torch, device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return device
    return "cpu"


if __name__ == "__main__":
    main()
