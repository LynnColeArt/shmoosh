from __future__ import annotations

import argparse
import time

from shmoosh.packed_keys import encode_packed_keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test Shmoosh packed-key encode/decode metadata."
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--qjl-bits", type=int, default=32)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--codebook-samples", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch = _load_torch()
    generator = torch.Generator(device=_generator_device(torch, args.device)).manual_seed(
        args.seed
    )
    keys = torch.randn(
        args.batch_size,
        args.heads,
        args.tokens,
        args.dim,
        generator=generator,
        device=args.device,
        dtype=torch.float32,
    )

    start = time.perf_counter()
    block = encode_packed_keys(
        keys,
        bits=args.bits,
        qjl_bits=args.qjl_bits,
        seed=args.seed,
        codebook_samples=args.codebook_samples,
    )
    encoded_seconds = time.perf_counter() - start
    decoded = block.decode(dtype=keys.dtype, device=keys.device)
    mse = torch.mean((decoded - keys) ** 2).item()

    print("Shmoosh packed-key smoke")
    print(
        "shape: "
        f"batch={args.batch_size} heads={args.heads} "
        f"tokens={args.tokens} dim={args.dim}"
    )
    print(f"bits={args.bits} qjl_bits={args.qjl_bits} seed={args.seed}")
    print(
        "bytes: "
        f"exact={block.exact_key_bytes()} "
        f"packed={block.packed_key_bytes()} "
        f"ratio={block.compression_ratio():.2f}x"
    )
    print(f"debug_decode_mse={mse:.8g}")
    print(f"encode_seconds={encoded_seconds:.4f}")


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "torch is required for packed key smoke; install optional dependencies first"
        ) from exc
    return torch


def _generator_device(torch, device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return device
    return "cpu"


if __name__ == "__main__":
    main()
