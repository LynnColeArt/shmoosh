from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from turbo_d.probe import load_npz_metadata, load_npz_tensors, run_attention_probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Turbo-D vs scalar probes across captured Q/K/V tensors."
    )
    parser.add_argument("captures", nargs="+", help="Capture .npz files or directories.")
    parser.add_argument("--bits", default="3,4")
    parser.add_argument("--qjl-bits", default="0,128")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--codebook-samples", type=int, default=80_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--csv", default="captures/sweep_results.csv")
    parser.add_argument("--json", default="captures/sweep_results.json")
    args = parser.parse_args()

    capture_paths = _expand_captures(args.captures)
    if args.limit is not None:
        capture_paths = capture_paths[: args.limit]
    if not capture_paths:
        raise SystemExit("No capture files found.")

    bit_values = _parse_ints(args.bits)
    qjl_values = _parse_ints(args.qjl_bits)

    rows = []
    for path in capture_paths:
        q, k, v = load_npz_tensors(path)
        metadata = load_npz_metadata(path)
        for bits in bit_values:
            for qjl_bits in qjl_values:
                result = run_attention_probe(
                    q,
                    k,
                    v,
                    bits=bits,
                    qjl_bits=qjl_bits,
                    seed=args.seed,
                    codebook_samples=args.codebook_samples,
                )
                row = _row(path, metadata, q, k, bits, qjl_bits, result)
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


def _parse_ints(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _row(path, metadata, q, k, bits, qjl_bits, result) -> dict[str, object]:
    turbo = asdict(result.turbo_d)
    scalar = asdict(result.scalar)
    row: dict[str, object] = {
        "capture": str(path),
        "module": metadata.get("module", ""),
        "capture_index": metadata.get("capture_index", ""),
        "bits": bits,
        "qjl_bits": qjl_bits,
        "heads": q.shape[0],
        "q_tokens": q.shape[1],
        "k_tokens": k.shape[1],
        "dim": q.shape[2],
    }
    for key, value in turbo.items():
        row[f"turbo_d_{key}"] = value
        row[f"scalar_{key}"] = scalar[key]
        row[f"delta_{key}"] = value - scalar[key]
        row[f"ratio_{key}"] = value / scalar[key] if scalar[key] else np.nan
    return row


def _format_row(row: dict[str, object]) -> str:
    return (
        f"{Path(str(row['capture'])).name} bits={row['bits']} qjl={row['qjl_bits']} "
        f"module={row['module']} "
        f"score_ratio={float(row['ratio_score_mse']):.4f} "
        f"kl_ratio={float(row['ratio_softmax_kl']):.4f} "
        f"out_cos_ratio={float(row['ratio_output_cosine_error']):.4f}"
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
    for bits in sorted({row["bits"] for row in rows}):
        for qjl_bits in sorted({row["qjl_bits"] for row in rows}):
            matching = [
                row for row in rows if row["bits"] == bits and row["qjl_bits"] == qjl_bits
            ]
            if not matching:
                continue
            score_wins = sum(row["delta_score_mse"] < 0 for row in matching)
            kl_wins = sum(row["delta_softmax_kl"] < 0 for row in matching)
            cos_wins = sum(row["delta_output_cosine_error"] < 0 for row in matching)
            print(
                f"bits={bits} qjl={qjl_bits}: "
                f"score_wins={score_wins}/{len(matching)} "
                f"kl_wins={kl_wins}/{len(matching)} "
                f"out_cos_wins={cos_wins}/{len(matching)} "
                f"mean_score_ratio={_mean(matching, 'ratio_score_mse'):.4f} "
                f"mean_kl_ratio={_mean(matching, 'ratio_softmax_kl'):.4f}"
            )


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


if __name__ == "__main__":
    main()
