from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from shmoosh.cli.image_ab_smoke import (
    _image_metrics,
    _install_policy_processors,
    _list_attention_modules,
    _load_pipeline,
    _load_policy,
    _load_torch_and_diffusers,
    _module_metadata,
    _move_pipeline,
    _pipeline_kwargs,
    _policy_processor_metadata,
    _print_modules,
    _processor_timing_payload,
    _processor_timing_recorder,
    _run_image,
    _select_component,
    _select_policy_module_entries,
    _set_progress_bar,
    _warm_packed_processors,
    _write_diff_heatmap,
)
from shmoosh.cli.image_policy_suite import (
    _aggregate_rows,
    _case_metadata,
    _case_policy_args,
    _cases_from_payload,
    _load_case_file,
    _mean,
    _ratio,
    _restore_processors,
    _safe_case_id,
)
from shmoosh.diffusers_processor import DenoisingStepState


@dataclass(frozen=True)
class _PolicyCandidate:
    label: str
    policy_file: str
    policy: dict[str, Any]
    selection: list[tuple[str, Any, dict[str, Any]]]
    modules: list[tuple[str, Any]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple Shmoosh image policies in one loaded pipeline, "
            "using one exact baseline render per case."
        )
    )
    _add_arguments(parser)
    args = parser.parse_args()
    if not args.list_modules and not args.candidate:
        raise SystemExit("Provide at least one --candidate LABEL=POLICY.json")

    torch = _load_torch_and_diffusers()
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]

    pipe = _load_pipeline(args, torch_dtype=dtype)
    component = _select_component(pipe, args.component)
    all_modules = _list_attention_modules(component, module_filter=args.module_filter)
    if args.list_modules:
        _print_modules(all_modules)
        return

    cases_payload = _load_case_file(args.case_file)
    cases = _cases_from_payload(cases_payload, args=args)
    if not cases:
        raise RuntimeError("case file did not contain any cases")

    candidates = _load_candidates(args.candidate, all_modules=all_modules)
    union_modules = _candidate_module_union(candidates)
    original_processors = {
        id(module): getattr(module, "processor", None) for _name, module in union_modules
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _move_pipeline(pipe, args)
    _set_progress_bar(pipe)

    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            _restore_processors(union_modules, original_processors)
            rows.extend(
                _run_case(
                    pipe,
                    torch=torch,
                    args=args,
                    case=case,
                    output_dir=output_dir,
                    all_modules=all_modules,
                    candidates=candidates,
                    union_modules=union_modules,
                    original_processors=original_processors,
                )
            )
    finally:
        _restore_processors(union_modules, original_processors)

    first_case_args = _case_policy_args(args, cases[0])
    summary = {
        "model_id": args.model_id,
        "single_file": args.single_file,
        "pipeline_class": args.pipeline_class,
        "case_file": args.case_file,
        "case_file_metadata": {
            key: value for key, value in cases_payload.items() if key != "cases"
        },
        "device": args.device,
        "dtype": args.dtype,
        "model_cpu_offload": args.model_cpu_offload,
        "selected_modules": _module_metadata(all_modules, union_modules),
        "candidates": [
            {
                "label": candidate.label,
                "policy_file": candidate.policy_file,
                "selected_modules": _module_metadata(all_modules, candidate.modules),
                "processor": _policy_processor_metadata(
                    all_modules,
                    candidate.selection,
                    args=first_case_args,
                    policy=candidate.policy,
                ),
                "policy": candidate.policy,
            }
            for candidate in candidates
        ],
        "rows": rows,
        "aggregate_by_candidate": _aggregate_candidate_rows(rows),
    }
    _write_summary(output_dir, summary, rows)
    _print_summary(summary["aggregate_by_candidate"])


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id")
    parser.add_argument("--single-file")
    parser.add_argument(
        "--pipeline-class",
        choices=["auto", "stable-diffusion", "sdxl"],
        default="auto",
    )
    parser.add_argument(
        "--config",
        help="Optional local Diffusers config directory or Hub repo id for single-file checkpoints.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Policy candidate as LABEL=path.json. If LABEL= is omitted, the file stem is used.",
    )
    parser.add_argument("--case-file", required=True)
    parser.add_argument("--output-dir", default="captures/image-policy-compare")
    parser.add_argument("--component", choices=["auto", "transformer", "unet"], default="auto")
    parser.add_argument("--module-filter", default="")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--model-cpu-offload", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--list-modules", action="store_true")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--key-bits", type=int)
    parser.add_argument("--value-bits", type=int)
    parser.add_argument("--qjl-bits", type=int, default=128)
    parser.add_argument("--codebook-samples", type=int, default=80_000)
    parser.add_argument("--processor-seed", type=int, default=11)
    parser.add_argument(
        "--attention-backend",
        choices=["reference", "packed"],
        default="reference",
    )
    parser.add_argument(
        "--packed-backend",
        choices=["auto", "torch", "triton"],
        default="auto",
    )
    parser.add_argument("--packed-block-q", type=int)
    parser.add_argument("--packed-block-k", type=int)
    parser.add_argument(
        "--code-format",
        choices=["packed", "packed_t", "byte"],
        default="packed",
    )
    parser.add_argument(
        "--norm-dtype",
        choices=["fp32", "fp16"],
        default="fp32",
    )
    parser.add_argument(
        "--key-encode-backend",
        choices=["split", "fused", "auto"],
        default="split",
    )
    parser.add_argument(
        "--dot-precision",
        choices=["ieee", "tf32", "tf32x3"],
        default="ieee",
    )
    parser.add_argument(
        "--rotation-dot-precision",
        choices=["ieee", "tf32", "tf32x3"],
    )
    parser.add_argument(
        "--score-dot-precision",
        choices=["ieee", "tf32", "tf32x3"],
    )
    parser.add_argument(
        "--value-dot-precision",
        choices=["ieee", "tf32", "tf32x3"],
    )
    parser.add_argument(
        "--qjl-dot-precision",
        choices=["ieee", "tf32", "tf32x3"],
    )
    parser.add_argument("--exact-keys", action="store_true")
    parser.add_argument("--quantize-values", action="store_true")
    parser.add_argument(
        "--trace-processor-timing",
        action="store_true",
        help="Record per-processor timing spans in each candidate metrics JSON.",
    )
    parser.add_argument(
        "--cache-cross-attention",
        action="store_true",
        help="Cache packed cross-attention K/V across denoising steps.",
    )


def _load_candidates(
    raw_candidates: list[str], *, all_modules: list[tuple[str, Any]]
) -> list[_PolicyCandidate]:
    candidates = []
    labels = set()
    for raw in raw_candidates:
        label, policy_file = _parse_candidate_spec(raw)
        if label in labels:
            raise SystemExit(f"duplicate candidate label: {label}")
        labels.add(label)

        policy = _load_policy(policy_file)
        if policy is None:
            raise SystemExit(f"candidate policy is required: {raw}")
        selection = _select_policy_module_entries(all_modules, policy=policy)
        modules = [(name, module) for name, module, _entry in selection]
        if not modules:
            raise RuntimeError(f"candidate {label} selected no attention modules")
        candidates.append(
            _PolicyCandidate(
                label=label,
                policy_file=policy_file,
                policy=policy,
                selection=selection,
                modules=modules,
            )
        )
    return candidates


def _parse_candidate_spec(raw: str) -> tuple[str, str]:
    if "=" in raw:
        label, path = raw.split("=", 1)
    else:
        path = raw
        label = Path(path).stem

    label = _safe_label(label)
    if not label:
        raise SystemExit(f"candidate label is empty: {raw}")
    if not path:
        raise SystemExit(f"candidate path is empty: {raw}")
    return label, path


def _safe_label(label: str) -> str:
    return _safe_case_id(label.strip())


def _candidate_module_union(candidates: list[_PolicyCandidate]) -> list[tuple[str, Any]]:
    modules: list[tuple[str, Any]] = []
    seen = set()
    for candidate in candidates:
        for name, module in candidate.modules:
            if id(module) in seen:
                continue
            seen.add(id(module))
            modules.append((name, module))
    return modules


def _run_case(
    pipe: Any,
    *,
    torch: Any,
    args: argparse.Namespace,
    case: argparse.Namespace,
    output_dir: Path,
    all_modules: list[tuple[str, Any]],
    candidates: list[_PolicyCandidate],
    union_modules: list[tuple[str, Any]],
    original_processors: dict[int, Any],
) -> list[dict[str, Any]]:
    case_dir = output_dir / _safe_case_id(case.case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    common_kwargs = _pipeline_kwargs(case)

    print(f"{case.case_id}: baseline")
    baseline_image, baseline_stats = _run_image(
        pipe,
        torch=torch,
        args=case,
        common_kwargs=common_kwargs,
        label=f"{case.case_id}:baseline",
    )
    baseline_path = case_dir / "baseline.png"
    baseline_image.save(baseline_path)

    rows = []
    for candidate in candidates:
        _restore_processors(union_modules, original_processors)
        rows.append(
            _run_candidate(
                pipe,
                torch=torch,
                args=args,
                case=case,
                case_dir=case_dir,
                baseline_image=baseline_image,
                baseline_stats=baseline_stats,
                baseline_path=baseline_path,
                common_kwargs=common_kwargs,
                all_modules=all_modules,
                candidate=candidate,
            )
        )
    _restore_processors(union_modules, original_processors)
    return rows


def _run_candidate(
    pipe: Any,
    *,
    torch: Any,
    args: argparse.Namespace,
    case: argparse.Namespace,
    case_dir: Path,
    baseline_image: Any,
    baseline_stats: dict[str, Any],
    baseline_path: Path,
    common_kwargs: dict[str, Any],
    all_modules: list[tuple[str, Any]],
    candidate: _PolicyCandidate,
) -> dict[str, Any]:
    print(f"{case.case_id}: {candidate.label}")
    candidate_dir = case_dir / candidate.label
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_args = _case_policy_args(args, case)
    step_state = DenoisingStepState(total_steps=case.steps)
    timing_recorder = _processor_timing_recorder(torch, args)
    _install_policy_processors(
        candidate.selection,
        args=candidate_args,
        policy=candidate.policy,
        step_state=step_state,
        timing_recorder=timing_recorder,
    )
    _warm_packed_processors(candidate.modules, torch=torch, args=args)
    shmoosh_image, shmoosh_stats = _run_image(
        pipe,
        torch=torch,
        args=case,
        common_kwargs=common_kwargs,
        label=f"{case.case_id}:{candidate.label}",
        step_state=step_state,
    )

    shmoosh_path = candidate_dir / "shmoosh.png"
    diff_path = candidate_dir / "diff_heatmap.png"
    metrics_path = candidate_dir / "metrics.json"
    shmoosh_image.save(shmoosh_path)
    image_metrics = _image_metrics(baseline_image, shmoosh_image)
    _write_diff_heatmap(baseline_image, shmoosh_image, diff_path)

    processor_timing = _processor_timing_payload(timing_recorder)
    row = {
        "candidate_label": candidate.label,
        "policy_file": candidate.policy_file,
        "case_id": case.case_id,
        "prompt": case.prompt,
        "negative_prompt": case.negative_prompt,
        "seed": case.seed,
        "steps": case.steps,
        "height": case.height,
        "width": case.width,
        "guidance_scale": case.guidance_scale,
        "mse": image_metrics["mse"],
        "mae": image_metrics["mae"],
        "psnr_db": image_metrics["psnr_db"],
        "max_abs": image_metrics["max_abs"],
        "baseline_seconds": baseline_stats["seconds"],
        "shmoosh_seconds": shmoosh_stats["seconds"],
        "speedup": _ratio(baseline_stats["seconds"], shmoosh_stats["seconds"]),
        "baseline_image": str(baseline_path),
        "shmoosh_image": str(shmoosh_path),
        "diff_heatmap": str(diff_path),
        "metrics": str(metrics_path),
    }
    if timing_recorder is not None:
        row["processor_timing_seconds"] = sum(
            float(record["seconds"]) for record in timing_recorder.records
        )
        row["processor_timing_records"] = len(timing_recorder.records)
        row["processor_timing_summary"] = processor_timing["summary"]
        _add_phase_timing_columns(row, processor_timing)
    if "cuda_max_memory_allocated_mib" in shmoosh_stats:
        row["shmoosh_cuda_max_memory_allocated_mib"] = shmoosh_stats[
            "cuda_max_memory_allocated_mib"
        ]
        row["shmoosh_cuda_max_memory_reserved_mib"] = shmoosh_stats[
            "cuda_max_memory_reserved_mib"
        ]

    payload = {
        "candidate": {
            "label": candidate.label,
            "policy_file": candidate.policy_file,
        },
        "case": _case_metadata(case),
        "selected_modules": _module_metadata(all_modules, candidate.modules),
        "processor": _policy_processor_metadata(
            all_modules,
            candidate.selection,
            args=candidate_args,
            policy=candidate.policy,
        ),
        "policy": candidate.policy,
        "baseline": baseline_stats,
        "shmoosh": shmoosh_stats,
        "processor_timing": processor_timing,
        "image_metrics": image_metrics,
        "outputs": {
            "baseline": str(baseline_path),
            "shmoosh": str(shmoosh_path),
            "diff_heatmap": str(diff_path),
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return row


def _add_phase_timing_columns(
    row: dict[str, Any], processor_timing: dict[str, Any]
) -> None:
    summary = processor_timing.get("summary", {})
    by_phase = summary.get("by_phase", [])
    if not isinstance(by_phase, list):
        return
    for phase_row in by_phase:
        if not isinstance(phase_row, dict):
            continue
        phase = _safe_timing_key(str(phase_row.get("phase", "unknown")))
        row[f"{phase}_seconds"] = float(phase_row["seconds"])
        row[f"{phase}_records"] = int(phase_row["count"])
        row[f"mean_{phase}_ms"] = float(phase_row["mean_seconds"]) * 1000.0


def _safe_timing_key(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value)


def _aggregate_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate_label"]), []).append(row)

    aggregates = []
    for label, candidate_rows in grouped.items():
        aggregate = {
            "candidate_label": label,
            "policy_file": candidate_rows[0]["policy_file"],
            **_aggregate_rows(candidate_rows),
        }
        if all("processor_timing_seconds" in row for row in candidate_rows):
            aggregate["mean_processor_timing_seconds"] = _mean(
                candidate_rows, "processor_timing_seconds"
            )
            aggregate["mean_processor_timing_records"] = _mean(
                candidate_rows, "processor_timing_records"
            )
        aggregate.update(_aggregate_phase_timing(candidate_rows))
        aggregates.append(aggregate)
    return sorted(aggregates, key=lambda row: str(row["candidate_label"]))


def _aggregate_phase_timing(rows: list[dict[str, Any]]) -> dict[str, float]:
    phase_keys = {
        key
        for row in rows
        for key in row
        if (
            key.startswith("mean_")
            and key.endswith("_ms")
            and key != "mean_processor_timing_records"
        )
        or key.endswith("_seconds")
    }
    aggregate = {}
    for key in sorted(phase_keys):
        present_rows = [row for row in rows if key in row]
        if not present_rows:
            continue
        aggregate_key = key if key.startswith("mean_") else f"mean_{key}"
        aggregate[aggregate_key] = _mean(present_rows, key)
    return aggregate


def _write_summary(
    output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    base_fieldnames = [
        "candidate_label",
        "policy_file",
        "case_id",
        "prompt",
        "negative_prompt",
        "seed",
        "steps",
        "height",
        "width",
        "guidance_scale",
        "mse",
        "mae",
        "psnr_db",
        "max_abs",
        "baseline_seconds",
        "shmoosh_seconds",
        "speedup",
        "processor_timing_seconds",
        "processor_timing_records",
        "shmoosh_cuda_max_memory_allocated_mib",
        "shmoosh_cuda_max_memory_reserved_mib",
        "baseline_image",
        "shmoosh_image",
        "diff_heatmap",
        "metrics",
    ]
    dynamic_fieldnames = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in base_fieldnames and key != "processor_timing_summary"
        }
    )
    fieldnames = base_fieldnames + dynamic_fieldnames
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(aggregates: list[dict[str, Any]]) -> None:
    print("policy comparison complete")
    for row in sorted(aggregates, key=lambda item: float(item["min_psnr_db"])):
        speedup = row.get("mean_speedup")
        speedup_text = "n/a" if speedup is None else f"{float(speedup):.3f}x"
        attention_ms = row.get("mean_packed_attention_ms")
        attention_text = (
            ""
            if attention_ms is None
            else f" packed_attention={float(attention_ms):.4f}ms"
        )
        print(
            f"{row['candidate_label']}: min_psnr={row['min_psnr_db']:.2f}dB "
            f"mean_psnr={row['mean_psnr_db']:.2f}dB "
            f"mean_speedup={speedup_text}{attention_text}"
        )


if __name__ == "__main__":
    main()
