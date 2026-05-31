from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from shmoosh.cli.image_ab_smoke import (
    _image_metrics,
    _install_processor,
    _install_policy_processors,
    _list_attention_modules,
    _load_pipeline,
    _load_policy,
    _load_torch_and_diffusers,
    _module_metadata,
    _move_pipeline,
    _pipeline_kwargs,
    _print_modules,
    _policy_processor_metadata,
    _run_image,
    _select_component,
    _select_policy_module_entries,
    _set_progress_bar,
    _write_diff_heatmap,
)
from shmoosh.diffusers_processor import DenoisingStepState


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a Shmoosh image policy across prompt/seed cases."
    )
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
    parser.add_argument("--policy-file", required=True)
    parser.add_argument("--case-file", required=True)
    parser.add_argument("--output-dir", default="captures/image-policy-suite")
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
    parser.add_argument("--exact-keys", action="store_true")
    parser.add_argument("--quantize-values", action="store_true")
    args = parser.parse_args()

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

    policy = _load_policy(args.policy_file)
    if policy is None:
        raise SystemExit("--policy-file is required")
    policy_selection = _select_policy_module_entries(all_modules, policy=policy)
    modules = [(name, module) for name, module, _entry in policy_selection]
    if not modules:
        raise RuntimeError("policy selected no attention modules")

    cases_payload = _load_case_file(args.case_file)
    cases = _cases_from_payload(cases_payload, args=args)
    if not cases:
        raise RuntimeError("case file did not contain any cases")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _move_pipeline(pipe, args)
    _set_progress_bar(pipe)

    summary_processor_metadata = _policy_processor_metadata(
        all_modules,
        policy_selection,
        args=_case_policy_args(args, cases[0]),
        policy=policy,
    )
    step_state = DenoisingStepState(total_steps=args.steps)
    original_processors = {
        id(module): getattr(module, "processor", None) for _name, module in modules
    }

    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            _restore_processors(modules, original_processors)
            rows.append(
                _run_case(
                    pipe,
                    torch=torch,
                    args=args,
                    case=case,
                    output_dir=output_dir,
                    all_modules=all_modules,
                    modules=modules,
                    policy_selection=policy_selection,
                    policy=policy,
                    step_state=step_state,
                )
            )
    finally:
        _restore_processors(modules, original_processors)

    summary = {
        "model_id": args.model_id,
        "single_file": args.single_file,
        "pipeline_class": args.pipeline_class,
        "policy_file": args.policy_file,
        "case_file": args.case_file,
        "case_file_metadata": {
            key: value for key, value in cases_payload.items() if key != "cases"
        },
        "device": args.device,
        "dtype": args.dtype,
        "model_cpu_offload": args.model_cpu_offload,
        "selected_modules": _module_metadata(all_modules, modules),
        "processor": summary_processor_metadata,
        "policy": policy,
        "rows": rows,
        "aggregate": _aggregate_rows(rows),
    }
    _write_summary(output_dir, summary, rows)
    _print_summary(rows)


def _run_case(
    pipe: Any,
    *,
    torch: Any,
    args: argparse.Namespace,
    case: argparse.Namespace,
    output_dir: Path,
    all_modules: list[tuple[str, Any]],
    modules: list[tuple[str, Any]],
    policy_selection: list[tuple[str, Any, dict[str, Any]]],
    policy: dict[str, Any],
    step_state: DenoisingStepState,
) -> dict[str, Any]:
    case_dir = output_dir / _safe_case_id(case.case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    common_kwargs = _pipeline_kwargs(case)
    step_state.total_steps = case.steps
    processor_metadata = _policy_processor_metadata(
        all_modules,
        policy_selection,
        args=_case_policy_args(args, case),
        policy=policy,
    )

    baseline_image, baseline_stats = _run_image(
        pipe,
        torch=torch,
        args=case,
        common_kwargs=common_kwargs,
        label=f"{case.case_id}:baseline",
    )
    _install_policy_processors(
        policy_selection, args=args, policy=policy, step_state=step_state
    )
    shmoosh_image, shmoosh_stats = _run_image(
        pipe,
        torch=torch,
        args=case,
        common_kwargs=common_kwargs,
        label=f"{case.case_id}:shmoosh",
        step_state=step_state,
    )

    baseline_path = case_dir / "baseline.png"
    shmoosh_path = case_dir / "shmoosh.png"
    diff_path = case_dir / "diff_heatmap.png"
    metrics_path = case_dir / "metrics.json"
    baseline_image.save(baseline_path)
    shmoosh_image.save(shmoosh_path)
    image_metrics = _image_metrics(baseline_image, shmoosh_image)
    _write_diff_heatmap(baseline_image, shmoosh_image, diff_path)

    row = {
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
        "baseline_image": str(baseline_path),
        "shmoosh_image": str(shmoosh_path),
        "diff_heatmap": str(diff_path),
        "metrics": str(metrics_path),
    }
    if "cuda_max_memory_allocated_mib" in shmoosh_stats:
        row["shmoosh_cuda_max_memory_allocated_mib"] = shmoosh_stats[
            "cuda_max_memory_allocated_mib"
        ]
        row["shmoosh_cuda_max_memory_reserved_mib"] = shmoosh_stats[
            "cuda_max_memory_reserved_mib"
        ]

    payload = {
        "case": _case_metadata(case),
        "selected_modules": _module_metadata(all_modules, modules),
        "processor": processor_metadata,
        "policy": policy,
        "baseline": baseline_stats,
        "shmoosh": shmoosh_stats,
        "image_metrics": image_metrics,
        "outputs": {
            "baseline": str(baseline_path),
            "shmoosh": str(shmoosh_path),
            "diff_heatmap": str(diff_path),
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return row


def _load_case_file(path: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"could not read --case-file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"could not parse --case-file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--case-file must contain a JSON object")
    return payload


def _cases_from_payload(
    payload: dict[str, Any], *, args: argparse.Namespace
) -> list[argparse.Namespace]:
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise SystemExit("--case-file must contain a cases array")
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, dict):
        raise SystemExit("--case-file defaults must be an object when present")
    return [
        _case_from_entry(index, entry, defaults=defaults, args=args)
        for index, entry in enumerate(raw_cases)
    ]


def _case_from_entry(
    index: int,
    entry: dict[str, Any],
    *,
    defaults: dict[str, Any],
    args: argparse.Namespace,
) -> argparse.Namespace:
    if not isinstance(entry, dict):
        raise SystemExit(f"case {index} must be an object")
    prompt = entry.get("prompt")
    if not prompt:
        raise SystemExit(f"case {index} is missing prompt")

    return argparse.Namespace(
        case_id=entry.get("id") or f"case_{index:03d}",
        prompt=prompt,
        negative_prompt=entry.get("negative_prompt"),
        seed=int(entry.get("seed", index)),
        steps=int(entry.get("steps", defaults.get("steps", args.steps))),
        height=int(entry.get("height", defaults.get("height", args.height))),
        width=int(entry.get("width", defaults.get("width", args.width))),
        guidance_scale=float(
            entry.get("guidance_scale", defaults.get("guidance_scale", args.guidance_scale))
        ),
        device=args.device,
    )


def _case_metadata(case: argparse.Namespace) -> dict[str, Any]:
    return {
        "id": case.case_id,
        "prompt": case.prompt,
        "negative_prompt": case.negative_prompt,
        "seed": case.seed,
        "steps": case.steps,
        "height": case.height,
        "width": case.width,
        "guidance_scale": case.guidance_scale,
    }


def _case_policy_args(
    args: argparse.Namespace, case: argparse.Namespace
) -> argparse.Namespace:
    values = vars(args).copy()
    values["steps"] = case.steps
    return argparse.Namespace(**values)


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(rows),
        "mean_mse": _mean(rows, "mse"),
        "mean_mae": _mean(rows, "mae"),
        "mean_psnr_db": _mean(rows, "psnr_db"),
        "min_psnr_db": min(float(row["psnr_db"]) for row in rows),
        "max_psnr_db": max(float(row["psnr_db"]) for row in rows),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _write_summary(
    output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    fieldnames = [
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
        "shmoosh_cuda_max_memory_allocated_mib",
        "shmoosh_cuda_max_memory_reserved_mib",
        "baseline_image",
        "shmoosh_image",
        "diff_heatmap",
        "metrics",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _restore_processors(
    modules: list[tuple[str, Any]], original_processors: dict[int, Any]
) -> None:
    for name, module in modules:
        _install_processor([(name, module)], original_processors[id(module)])


def _safe_case_id(case_id: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in case_id)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("policy suite complete")
    for row in sorted(rows, key=lambda item: float(item["psnr_db"])):
        print(
            f"{row['case_id']} seed={row['seed']} "
            f"mse={row['mse']:.8f} mae={row['mae']:.8f} "
            f"psnr={row['psnr_db']:.2f}dB"
        )


if __name__ == "__main__":
    main()
