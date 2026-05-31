from __future__ import annotations

import argparse

from turbo_d.metrics import cosine_error, mse
from turbo_d.probe import load_npz_tensors
from turbo_d.runtime_attention import exact_attention_output, turbo_d_attention_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare exact attention output to runtime-style Turbo-D attention."
    )
    parser.add_argument("capture", help="Capture .npz containing q, k, and v.")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--key-bits", type=int)
    parser.add_argument("--value-bits", type=int)
    parser.add_argument("--qjl-bits", type=int, default=128)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--exact-keys",
        action="store_true",
        help="Use exact K tensors and only quantize V if values are enabled.",
    )
    parser.add_argument(
        "--exact-values",
        action="store_true",
        help="Use exact V tensors while Turbo-D quantizes K and estimates scores.",
    )
    parser.add_argument("--codebook-samples", type=int, default=80_000)
    args = parser.parse_args()

    q, k, v = load_npz_tensors(args.capture)
    reference = exact_attention_output(q, k, v)
    turbo = turbo_d_attention_output(
        q,
        k,
        v,
        bits=args.bits,
        qjl_bits=args.qjl_bits,
        seed=args.seed,
        quantize_keys=not args.exact_keys,
        quantize_values=not args.exact_values,
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        codebook_samples=args.codebook_samples,
    )

    print("Turbo-D runtime attention smoke")
    print(f"capture={args.capture}")
    print(f"shape q={q.shape} k={k.shape} v={v.shape}")
    print(
        f"bits={args.bits} key_bits={args.key_bits or args.bits} "
        f"value_bits={args.value_bits or args.bits} qjl_bits={args.qjl_bits} seed={args.seed}"
    )
    print(f"quantize_keys={not args.exact_keys}")
    print(f"quantize_values={not args.exact_values}")
    print(f"output_mse={mse(reference, turbo):.8g}")
    print(f"output_cosine_error={cosine_error(reference, turbo):.8g}")
    active_count, total_count, active_cosine = _active_cosine_error(reference, turbo)
    print(
        f"active_output_cosine_error={active_cosine:.8g} "
        f"active_rows={active_count}/{total_count}"
    )


def _active_cosine_error(reference, candidate, min_norm: float = 1e-6):
    import numpy as np

    reference = reference.reshape(-1, reference.shape[-1])
    candidate = candidate.reshape(-1, candidate.shape[-1])
    reference_norm = np.linalg.norm(reference, axis=-1)
    candidate_norm = np.linalg.norm(candidate, axis=-1)
    active = reference_norm > min_norm
    if not np.any(active):
        return 0, reference.shape[0], float("nan")

    dot = np.sum(reference[active] * candidate[active], axis=-1)
    cosine = dot / np.maximum(reference_norm[active] * candidate_norm[active], 1e-8)
    return int(active.sum()), reference.shape[0], float(np.mean(1.0 - cosine))


if __name__ == "__main__":
    main()
