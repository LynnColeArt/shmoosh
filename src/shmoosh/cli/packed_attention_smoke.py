from __future__ import annotations

import argparse
import time

from shmoosh.packed_attention import encode_and_attention_output
from shmoosh.runtime_attention import shmoosh_attention_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test packed-key exact-value attention output."
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
    value = torch.randn(
        args.batch_size,
        args.heads,
        args.key_tokens,
        args.dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    start = time.perf_counter()
    output = encode_and_attention_output(
        query,
        key,
        value,
        bits=args.bits,
        qjl_bits=args.qjl_bits,
        seed=args.seed,
        backend=args.backend,
        codebook_samples=args.codebook_samples,
    )
    output_seconds = time.perf_counter() - start
    reference = _reference_output(
        query,
        key,
        value,
        bits=args.bits,
        qjl_bits=args.qjl_bits,
        seed=args.seed,
        codebook_samples=args.codebook_samples,
    ).to(device=device)
    max_abs = torch.max(torch.abs(output - reference)).item()
    mean_abs = torch.mean(torch.abs(output - reference)).item()

    print("Shmoosh packed-attention smoke")
    print(
        "shape: "
        f"batch={args.batch_size} heads={args.heads} "
        f"query_tokens={args.query_tokens} key_tokens={args.key_tokens} "
        f"dim={args.dim}"
    )
    print(f"bits={args.bits} qjl_bits={args.qjl_bits} seed={args.seed}")
    print(f"device={device} backend={args.backend}")
    print(f"output_shape={tuple(output.shape)}")
    print(f"max_abs_delta_vs_reference={max_abs:.8g}")
    print(f"mean_abs_delta_vs_reference={mean_abs:.8g}")
    print(f"output_seconds={output_seconds:.4f}")


def _reference_output(query, key, value, *, bits, qjl_bits, seed, codebook_samples):
    torch = _load_torch()
    batch, heads, query_tokens, dim = query.shape
    key_tokens = int(key.shape[2])
    reference = shmoosh_attention_output(
        query.detach()
        .to(device="cpu", dtype=torch.float32)
        .reshape(batch * heads, query_tokens, dim)
        .numpy(),
        key.detach()
        .to(device="cpu", dtype=torch.float32)
        .reshape(batch * heads, key_tokens, dim)
        .numpy(),
        value.detach()
        .to(device="cpu", dtype=torch.float32)
        .reshape(batch * heads, key_tokens, dim)
        .numpy(),
        bits=bits,
        qjl_bits=qjl_bits,
        seed=seed,
        quantize_values=False,
        codebook_samples=codebook_samples,
    )
    return torch.from_numpy(reference).reshape(batch, heads, query_tokens, dim)


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "torch is required for packed attention smoke; install optional dependencies first"
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
