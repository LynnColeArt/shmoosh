from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from shmoosh.cli.image_ab_smoke import (
    _image_metrics,
    _install_processor,
    _list_attention_modules,
    _load_pipeline,
    _load_torch_and_diffusers,
    _module_metadata,
    _move_pipeline,
    _pipeline_kwargs,
    _print_modules,
    _run_image,
    _select_attention_modules,
    _select_component,
    _set_progress_bar,
    _write_diff_heatmap,
)
from shmoosh.diffusers_processor import ShmooshAttnProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep same-seed Shmoosh image A/B results across attention modules."
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
    parser.add_argument("--prompt")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--output-dir", default="captures/image-module-sweep")
    parser.add_argument("--component", choices=["auto", "transformer", "unet"], default="auto")
    parser.add_argument("--module-filter", default="")
    parser.add_argument(
        "--module-indices",
        help="Comma-separated indices from --list-modules output, after filtering.",
    )
    parser.add_argument(
        "--module-names",
        help="Comma-separated exact module names from --list-modules output.",
    )
    parser.add_argument("--max-modules", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
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
        "--exact-keys",
        action="store_true",
        help="Leave K exact and only quantize values if --quantize-values is set.",
    )
    parser.add_argument(
        "--quantize-values",
        action="store_true",
        help="Quantize V as well as K. By default, values stay exact.",
    )
    parser.add_argument(
        "--include-exact-calibration",
        action="store_true",
        help="Also run each module through the custom processor with exact K and exact V.",
    )
    parser.add_argument(
        "--candidate-psnr-db",
        type=float,
        default=30.0,
        help="Minimum image PSNR for a module to enter suggested_policy.json.",
    )
    args = parser.parse_args()
    if not args.list_modules and not args.prompt:
        raise SystemExit("--prompt is required unless --list-modules is used.")

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

    modules = _select_attention_modules(all_modules, args=args)
    if not modules:
        raise RuntimeError("no attention-like modules with to_q/to_k/to_v were found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _move_pipeline(pipe, args)
    _set_progress_bar(pipe)

    common_kwargs = _pipeline_kwargs(args)
    baseline_image, baseline_stats = _run_image(
        pipe,
        torch=torch,
        args=args,
        common_kwargs=common_kwargs,
        label="baseline",
    )
    baseline_path = output_dir / "baseline.png"
    baseline_image.save(baseline_path)

    module_indices = {
        id(module): index for index, (_name, module) in enumerate(all_modules)
    }
    original_processors = {
        id(module): getattr(module, "processor", None) for _name, module in modules
    }
    rows: list[dict[str, Any]] = []

    try:
        for module_name, module in modules:
            _restore_processors(modules, original_processors)
            module_index = module_indices[id(module)]

            if args.include_exact_calibration:
                rows.append(
                    _run_module_policy(
                        pipe,
                        torch=torch,
                        args=args,
                        common_kwargs=common_kwargs,
                        output_dir=output_dir,
                        baseline_image=baseline_image,
                        baseline_stats=baseline_stats,
                        all_modules=all_modules,
                        module_name=module_name,
                        module=module,
                        module_index=module_index,
                        policy_name="exact_processor",
                        processor=ShmooshAttnProcessor(
                            bits=args.bits,
                            qjl_bits=args.qjl_bits,
                            seed=args.processor_seed,
                            quantize_keys=False,
                            quantize_values=False,
                            key_bits=args.key_bits,
                            value_bits=args.value_bits,
                            codebook_samples=args.codebook_samples,
                        ),
                    )
                )
                _restore_processors(modules, original_processors)

            rows.append(
                _run_module_policy(
                    pipe,
                    torch=torch,
                    args=args,
                    common_kwargs=common_kwargs,
                    output_dir=output_dir,
                    baseline_image=baseline_image,
                    baseline_stats=baseline_stats,
                    all_modules=all_modules,
                    module_name=module_name,
                    module=module,
                    module_index=module_index,
                    policy_name="shmoosh",
                    processor=ShmooshAttnProcessor(
                        bits=args.bits,
                        qjl_bits=args.qjl_bits,
                        seed=args.processor_seed,
                        quantize_keys=not args.exact_keys,
                        quantize_values=args.quantize_values,
                        key_bits=args.key_bits,
                        value_bits=args.value_bits,
                        codebook_samples=args.codebook_samples,
                    ),
                )
            )
    finally:
        _restore_processors(modules, original_processors)

    suggested_policy = _suggest_policy(args, rows)
    summary = {
        "model_id": args.model_id,
        "single_file": args.single_file,
        "pipeline_class": args.pipeline_class,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "steps": args.steps,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "model_cpu_offload": args.model_cpu_offload,
        "selected_modules": _module_metadata(all_modules, modules),
        "processor": {
            "bits": args.bits,
            "key_bits": args.key_bits,
            "value_bits": args.value_bits,
            "qjl_bits": args.qjl_bits,
            "codebook_samples": args.codebook_samples,
            "processor_seed": args.processor_seed,
            "quantize_keys": not args.exact_keys,
            "quantize_values": args.quantize_values,
        },
        "baseline": baseline_stats | {"image": str(baseline_path)},
        "suggested_policy": suggested_policy,
        "rows": rows,
    }
    _write_summary(output_dir, summary, rows, suggested_policy)
    _print_summary(rows)


def _run_module_policy(
    pipe: Any,
    *,
    torch: Any,
    args: argparse.Namespace,
    common_kwargs: dict[str, Any],
    output_dir: Path,
    baseline_image: Any,
    baseline_stats: dict[str, Any],
    all_modules: list[tuple[str, Any]],
    module_name: str,
    module: Any,
    module_index: int,
    policy_name: str,
    processor: ShmooshAttnProcessor,
) -> dict[str, Any]:
    module_dir = output_dir / _module_dirname(module_index, module_name, policy_name)
    module_dir.mkdir(parents=True, exist_ok=True)
    _install_processor([(module_name, module)], processor)

    shmoosh_image, shmoosh_stats = _run_image(
        pipe,
        torch=torch,
        args=args,
        common_kwargs=common_kwargs,
        label=f"{policy_name}:{module_index:03d}",
    )

    shmoosh_path = module_dir / "shmoosh.png"
    diff_path = module_dir / "diff_heatmap.png"
    metrics_path = module_dir / "metrics.json"
    shmoosh_image.save(shmoosh_path)
    image_metrics = _image_metrics(baseline_image, shmoosh_image)
    _write_diff_heatmap(baseline_image, shmoosh_image, diff_path)

    module_meta = _module_metadata(all_modules, [(module_name, module)])[0]
    row = {
        "policy": policy_name,
        "module_index": module_index,
        "module_name": module_name,
        "heads": module_meta["heads"],
        "cross_attention_dim": module_meta["cross_attention_dim"],
        "quantize_keys": processor.quantize_keys,
        "quantize_values": processor.quantize_values,
        "bits": processor.bits,
        "key_bits": processor.key_bits,
        "value_bits": processor.value_bits,
        "qjl_bits": processor.qjl_bits,
        "codebook_samples": processor.codebook_samples,
        "mse": image_metrics["mse"],
        "mae": image_metrics["mae"],
        "psnr_db": image_metrics["psnr_db"],
        "max_abs": image_metrics["max_abs"],
        "baseline_seconds": baseline_stats["seconds"],
        "shmoosh_seconds": shmoosh_stats["seconds"],
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
        "module": module_meta,
        "policy": policy_name,
        "processor": {
            "bits": processor.bits,
            "key_bits": processor.key_bits,
            "value_bits": processor.value_bits,
            "qjl_bits": processor.qjl_bits,
            "codebook_samples": processor.codebook_samples,
            "processor_seed": processor.seed,
            "quantize_keys": processor.quantize_keys,
            "quantize_values": processor.quantize_values,
        },
        "baseline": baseline_stats,
        "shmoosh": shmoosh_stats,
        "image_metrics": image_metrics,
        "outputs": {
            "shmoosh": str(shmoosh_path),
            "diff_heatmap": str(diff_path),
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return row


def _restore_processors(
    modules: list[tuple[str, Any]], original_processors: dict[int, Any]
) -> None:
    for name, module in modules:
        _install_processor([(name, module)], original_processors[id(module)])


def _module_dirname(index: int, name: str, policy: str) -> str:
    safe_name = "".join(char if char.isalnum() else "_" for char in name)
    return f"{index:03d}_{policy}_{safe_name}"


def _suggest_policy(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    shmoosh_rows = [row for row in rows if row["policy"] == "shmoosh"]
    candidates = [
        row
        for row in shmoosh_rows
        if float(row["psnr_db"]) >= args.candidate_psnr_db
    ]
    exact = [
        row
        for row in shmoosh_rows
        if float(row["psnr_db"]) < args.candidate_psnr_db
    ]
    return {
        "schema_version": 1,
        "selection_metric": {
            "candidate_psnr_db": args.candidate_psnr_db,
            "note": "Image PSNR is a coarse first-pass gate; visually inspect candidates before broadening policy.",
        },
        "default": {
            "quantize_keys": False,
            "quantize_values": False,
        },
        "shmoosh_policy": {
            "bits": args.bits,
            "key_bits": args.key_bits,
            "value_bits": args.value_bits,
            "qjl_bits": args.qjl_bits,
            "codebook_samples": args.codebook_samples,
            "processor_seed": args.processor_seed,
            "quantize_keys": not args.exact_keys,
            "quantize_values": args.quantize_values,
        },
        "quantized_modules": [_policy_module(row) for row in candidates],
        "exact_modules": [_policy_module(row) for row in exact],
    }


def _policy_module(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": row["module_index"],
        "name": row["module_name"],
        "mse": row["mse"],
        "mae": row["mae"],
        "psnr_db": row["psnr_db"],
    }


def _write_summary(
    output_dir: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    suggested_policy: dict[str, Any],
) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "suggested_policy.json").write_text(
        json.dumps(suggested_policy, indent=2) + "\n", encoding="utf-8"
    )

    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "policy",
        "module_index",
        "module_name",
        "heads",
        "cross_attention_dim",
        "quantize_keys",
        "quantize_values",
        "bits",
        "key_bits",
        "value_bits",
        "qjl_bits",
        "codebook_samples",
        "mse",
        "mae",
        "psnr_db",
        "max_abs",
        "baseline_seconds",
        "shmoosh_seconds",
        "shmoosh_cuda_max_memory_allocated_mib",
        "shmoosh_cuda_max_memory_reserved_mib",
        "shmoosh_image",
        "diff_heatmap",
        "metrics",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("module sweep complete")
    for row in sorted(rows, key=lambda item: float(item["mse"])):
        print(
            f"{row['policy']} module={row['module_index']:03d} "
            f"mse={row['mse']:.8f} mae={row['mae']:.8f} "
            f"psnr={row['psnr_db']:.2f}dB {row['module_name']}"
        )


if __name__ == "__main__":
    main()
