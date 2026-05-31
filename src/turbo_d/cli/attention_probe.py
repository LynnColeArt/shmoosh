from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np

from turbo_d.probe import load_npz_tensors, run_attention_probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Turbo-D attention preservation on synthetic or captured tensors."
    )
    parser.add_argument("--npz", help="Optional .npz file containing q, k, and v arrays.")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--qjl-bits", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--codebook-samples", type=int, default=200_000)
    parser.add_argument("--outlier-scale", type=float, default=3.0)
    parser.add_argument("--timestep-skew", type=float, default=1.0)
    args = parser.parse_args()

    q, k, v = load_tensors(args)
    result = run_attention_probe(
        q,
        k,
        v,
        bits=args.bits,
        qjl_bits=args.qjl_bits,
        seed=args.seed,
        codebook_samples=args.codebook_samples,
    )

    print("Turbo-D attention probe")
    print(
        f"shape: heads={q.shape[0]} q_tokens={q.shape[1]} "
        f"k_tokens={k.shape[1]} dim={q.shape[2]}"
    )
    print(f"bits={args.bits} qjl_bits={args.qjl_bits} seed={args.seed}")
    print("")
    _print_metrics("turbo_d", result.turbo_d)
    _print_metrics("scalar", result.scalar)


def load_tensors(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.npz:
        return load_npz_tensors(args.npz)
    return synthetic_attention(
        heads=args.heads,
        tokens=args.tokens,
        dim=args.dim,
        seed=args.seed,
        outlier_scale=args.outlier_scale,
        timestep_skew=args.timestep_skew,
    )


def synthetic_attention(
    *,
    heads: int,
    tokens: int,
    dim: int,
    seed: int,
    outlier_scale: float,
    timestep_skew: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(heads, tokens, dim)).astype(np.float32)
    k = rng.normal(size=(heads, tokens, dim)).astype(np.float32)
    v = rng.normal(size=(heads, tokens, dim)).astype(np.float32)

    channel_scale = np.ones((1, 1, dim), dtype=np.float32)
    salient = max(1, dim // 32)
    channel_scale[..., :salient] = outlier_scale
    drift = np.linspace(1.0, timestep_skew, tokens, dtype=np.float32).reshape(1, tokens, 1)
    return q * drift, k * channel_scale * drift, v * channel_scale


def _print_metrics(name: str, metrics) -> None:
    values = asdict(metrics)
    formatted = " ".join(f"{key}={value:.6g}" for key, value in values.items())
    print(f"{name}: {formatted}")


if __name__ == "__main__":
    main()
