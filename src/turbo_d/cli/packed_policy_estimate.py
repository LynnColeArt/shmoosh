from __future__ import annotations

import argparse
import json
from pathlib import Path

from turbo_d.packed_estimator import PackedKeyAssumptions, estimate_policy_storage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate packed-key storage savings for a Turbo-D image policy."
    )
    parser.add_argument("--policy-file", required=True)
    parser.add_argument(
        "--steps",
        type=int,
        action="append",
        default=[],
        help="Denoising step count to estimate. Can be passed more than once.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--key-tokens", type=int, default=77)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--norm-bytes", type=int, default=4)
    parser.add_argument("--residual-norm-bytes", type=int, default=4)
    parser.add_argument("--json-output", help="Optional path for the full JSON estimate.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON estimate.")
    args = parser.parse_args()

    policy = json.loads(Path(args.policy_file).read_text(encoding="utf-8"))
    assumptions = PackedKeyAssumptions(
        batch_size=args.batch_size,
        heads=args.heads,
        key_tokens=args.key_tokens,
        head_dim=args.head_dim,
        dtype_bytes=args.dtype_bytes,
        norm_bytes=args.norm_bytes,
        residual_norm_bytes=args.residual_norm_bytes,
    )
    steps = args.steps or [20, 30]
    estimate = estimate_policy_storage(policy, steps=steps, assumptions=assumptions)

    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(estimate, indent=2) + "\n", encoding="utf-8"
        )
    if args.json:
        print(json.dumps(estimate, indent=2))
    else:
        _print_summary(args.policy_file, estimate)


def _print_summary(policy_file: str, estimate: dict) -> None:
    assumptions = estimate["assumptions"]
    per_step = estimate["per_quantized_step"]
    print(f"policy: {policy_file}")
    print(
        "assumptions: "
        f"batch={assumptions['batch_size']} "
        f"heads={assumptions['heads']} "
        f"key_tokens={assumptions['key_tokens']} "
        f"head_dim={assumptions['head_dim']} "
        f"dtype_bytes={assumptions['dtype_bytes']}"
    )
    print(
        "per quantized step: "
        f"exact={per_step['exact_mib']:.2f} MiB "
        f"packed={per_step['packed_mib']:.2f} MiB "
        f"saved={per_step['saved_mib']:.2f} MiB "
        f"ratio={per_step['compression_ratio']:.2f}x"
    )
    print("steps:")
    for step_estimate in estimate["steps"]:
        horizon = step_estimate["scheduled_horizon"]
        window = step_estimate["quantized_window"]
        print(
            f"  {step_estimate['total_steps']:>3}: "
            f"horizon_saved={horizon['saved_mib']:.2f} MiB "
            f"horizon_ratio={horizon['compression_ratio']:.2f}x "
            f"window_saved={window['saved_mib']:.2f} MiB "
            f"window_ratio={window['compression_ratio']:.2f}x"
        )


if __name__ == "__main__":
    main()
