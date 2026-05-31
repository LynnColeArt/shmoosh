from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from turbo_d.metrics import mse
from turbo_d.probe import load_npz_metadata, load_npz_tensors
from turbo_d.runtime_attention import exact_attention_output, turbo_d_attention_output


POLICIES = {
    "k_only": {"quantize_keys": True, "quantize_values": False},
    "v_only": {"quantize_keys": False, "quantize_values": True},
    "kv": {"quantize_keys": True, "quantize_values": True},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep runtime-style Turbo-D K/V quantization policies."
    )
    parser.add_argument("captures", nargs="+", help="Capture .npz files or directories.")
    parser.add_argument("--policies", default="k_only,v_only,kv")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--key-bits", type=int)
    parser.add_argument("--value-bits", type=int)
    parser.add_argument("--qjl-bits", type=int, default=128)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--codebook-samples", type=int, default=80_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--csv", default="captures/policy_sweep.csv")
    parser.add_argument("--json", default="captures/policy_sweep.json")
    args = parser.parse_args()

    capture_paths = _expand_captures(args.captures)
    if args.limit is not None:
        capture_paths = capture_paths[: args.limit]
    if not capture_paths:
        raise SystemExit("No capture files found.")

    policies = _parse_policies(args.policies)
    rows = []
    for path in capture_paths:
        q, k, v = load_npz_tensors(path)
        metadata = load_npz_metadata(path)
        reference = exact_attention_output(q, k, v)
        for policy_name in policies:
            policy = POLICIES[policy_name]
            output = turbo_d_attention_output(
                q,
                k,
                v,
                bits=args.bits,
                key_bits=args.key_bits,
                value_bits=args.value_bits,
                qjl_bits=args.qjl_bits,
                seed=args.seed,
                codebook_samples=args.codebook_samples,
                **policy,
            )
            row = _row(path, metadata, reference, output, policy_name, args)
            rows.append(row)
            print(_format_row(row))

    _write_csv(Path(args.csv), rows)
    _write_json(Path(args.json), rows)
    _print_summary(rows)


def _expand_captures(paths: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.npz")))
        elif path.suffix == ".npz":
            expanded.append(path)
    return expanded


def _parse_policies(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(values) - set(POLICIES))
    if unknown:
        raise SystemExit(f"unknown policies: {', '.join(unknown)}")
    return values


def _row(path, metadata, reference, output, policy_name, args) -> dict[str, object]:
    active_count, total_count, active_cosine = _active_cosine_error(reference, output)
    raw_cosine = _cosine_error(reference, output)
    return {
        "capture": str(path),
        "module": metadata.get("module", ""),
        "capture_index": metadata.get("capture_index", ""),
        "policy": policy_name,
        "bits": args.bits,
        "key_bits": args.key_bits or args.bits,
        "value_bits": args.value_bits or args.bits,
        "qjl_bits": args.qjl_bits,
        "output_mse": mse(reference, output),
        "output_cosine_error": raw_cosine,
        "active_output_cosine_error": active_cosine,
        "active_rows": active_count,
        "total_rows": total_count,
    }


def _format_row(row: dict[str, object]) -> str:
    return (
        f"{Path(str(row['capture'])).name} policy={row['policy']} "
        f"module={row['module']} "
        f"mse={float(row['output_mse']):.6g} "
        f"active_cos={float(row['active_output_cosine_error']):.6g} "
        f"active={row['active_rows']}/{row['total_rows']}"
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _print_summary(rows: list[dict[str, object]]) -> None:
    print("")
    print("Summary")
    for policy in sorted({str(row["policy"]) for row in rows}):
        matching = [row for row in rows if row["policy"] == policy]
        print(
            f"{policy}: n={len(matching)} "
            f"mean_mse={_mean(matching, 'output_mse'):.6g} "
            f"mean_active_cos={_mean(matching, 'active_output_cosine_error'):.6g}"
        )


def _active_cosine_error(reference, candidate, min_norm: float = 1e-6):
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


def _cosine_error(reference, candidate):
    active_count, total_count, value = _active_cosine_error(
        reference, candidate, min_norm=0.0
    )
    return value if active_count == total_count else float("nan")


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.nanmean([float(row[key]) for row in rows]))


if __name__ == "__main__":
    main()
