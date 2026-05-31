from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from turbo_d.diffusers_processor import TurboDAttnProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a same-seed baseline vs Turbo-D Diffusers image smoke test."
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
    parser.add_argument("--output-dir", default="captures/image-ab-smoke")
    parser.add_argument(
        "--policy-file",
        help="Optional JSON policy file with quantized_modules and turbo_policy.",
    )
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
    parser.add_argument("--max-modules", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
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

    policy = _load_policy(args.policy_file)
    if policy is not None and (args.module_indices or args.module_names):
        raise SystemExit("Use either --policy-file or explicit module selection, not both.")
    modules = (
        _select_policy_modules(all_modules, policy=policy)
        if policy is not None
        else _select_attention_modules(all_modules, args=args)
    )
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

    processor_config = _processor_config(args, policy=policy)
    processor = TurboDAttnProcessor(**processor_config)
    _install_processor(modules, processor)

    turbo_image, turbo_stats = _run_image(
        pipe,
        torch=torch,
        args=args,
        common_kwargs=common_kwargs,
        label="turbo",
    )

    baseline_path = output_dir / "baseline.png"
    turbo_path = output_dir / "turbo.png"
    diff_path = output_dir / "diff_heatmap.png"
    metrics_path = output_dir / "metrics.json"

    baseline_image.save(baseline_path)
    turbo_image.save(turbo_path)
    image_metrics = _image_metrics(baseline_image, turbo_image)
    _write_diff_heatmap(baseline_image, turbo_image, diff_path)

    metrics = {
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
        "policy_file": args.policy_file,
        "selected_modules": _module_metadata(all_modules, modules),
        "processor": _processor_metadata(processor_config),
        "policy": policy,
        "baseline": baseline_stats,
        "turbo": turbo_stats,
        "image_metrics": image_metrics,
        "outputs": {
            "baseline": str(baseline_path),
            "turbo": str(turbo_path),
            "diff_heatmap": str(diff_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"wrote baseline image: {baseline_path}")
    print(f"wrote turbo image: {turbo_path}")
    print(f"wrote diff heatmap: {diff_path}")
    print(f"wrote metrics: {metrics_path}")
    print(
        "image metrics: "
        f"mse={image_metrics['mse']:.8f} "
        f"mae={image_metrics['mae']:.8f} "
        f"psnr={image_metrics['psnr_db']:.2f}dB"
    )


def _load_torch_and_diffusers():
    try:
        import torch
        import diffusers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Install optional dependencies first: uv sync --extra dev --extra diffusers"
        ) from exc
    return torch


def _load_pipeline(args: argparse.Namespace, *, torch_dtype: Any) -> Any:
    if bool(args.model_id) == bool(args.single_file):
        raise SystemExit("Provide exactly one of --model-id or --single-file.")

    if args.single_file:
        return _load_single_file_pipeline(args, torch_dtype=torch_dtype)

    from diffusers import DiffusionPipeline

    return DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        local_files_only=args.local_files_only,
    )


def _load_single_file_pipeline(args: argparse.Namespace, *, torch_dtype: Any) -> Any:
    pipeline_class = args.pipeline_class
    if pipeline_class == "auto":
        pipeline_class = "sdxl"

    if pipeline_class == "stable-diffusion":
        from diffusers import StableDiffusionPipeline

        cls = StableDiffusionPipeline
    elif pipeline_class == "sdxl":
        from diffusers import StableDiffusionXLPipeline

        cls = StableDiffusionXLPipeline
    else:  # pragma: no cover - argparse constrains this today.
        raise SystemExit("--single-file currently supports stable-diffusion or sdxl.")

    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "local_files_only": args.local_files_only,
    }
    if args.config:
        kwargs["config"] = args.config

    return cls.from_single_file(args.single_file, **kwargs)


def _move_pipeline(pipe: Any, args: argparse.Namespace) -> None:
    if args.model_cpu_offload:
        offload = getattr(pipe, "enable_model_cpu_offload", None)
        if callable(offload):
            offload(device=args.device)
            return

    pipe.to(args.device)


def _set_progress_bar(pipe: Any) -> None:
    setter = getattr(pipe, "set_progress_bar_config", None)
    if callable(setter):
        setter(disable=False)


def _select_component(pipe: Any, component: str) -> Any:
    if component == "transformer":
        return pipe.transformer
    if component == "unet":
        return pipe.unet
    if hasattr(pipe, "transformer"):
        return pipe.transformer
    if hasattr(pipe, "unet"):
        return pipe.unet
    raise RuntimeError("pipeline has neither a transformer nor unet component")


def _list_attention_modules(component: Any, *, module_filter: str) -> list[tuple[str, Any]]:
    modules = []
    for name, module in component.named_modules():
        if module_filter and module_filter not in name:
            continue
        if all(hasattr(module, attr) for attr in ("to_q", "to_k", "to_v")):
            modules.append((name, module))
    return modules


def _select_attention_modules(
    modules: list[tuple[str, Any]], *, args: argparse.Namespace
) -> list[tuple[str, Any]]:
    if getattr(args, "policy_file", None) and (args.module_indices or args.module_names):
        raise SystemExit("Use either --policy-file or explicit module selection, not both.")

    if args.module_indices and args.module_names:
        raise SystemExit("Use either --module-indices or --module-names, not both.")

    if args.module_indices:
        selected = []
        for index in _parse_indices(args.module_indices):
            if index < 0 or index >= len(modules):
                raise SystemExit(f"module index {index} is out of range")
            selected.append(modules[index])
        return selected

    if args.module_names:
        by_name = {name: module for name, module in modules}
        selected = []
        for name in _parse_names(args.module_names):
            if name not in by_name:
                raise SystemExit(f"module name not found: {name}")
            selected.append((name, by_name[name]))
        return selected

    return modules[: args.max_modules]


def _load_policy(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    policy_path = Path(path)
    try:
        return json.loads(policy_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"could not read --policy-file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"could not parse --policy-file {path}: {exc}") from exc


def _select_policy_modules(
    modules: list[tuple[str, Any]], *, policy: dict[str, Any]
) -> list[tuple[str, Any]]:
    by_name = {name: module for name, module in modules}
    selected = []
    for entry in policy.get("quantized_modules", []):
        name = entry.get("name")
        if name is not None:
            if name not in by_name:
                raise SystemExit(f"policy module name not found: {name}")
            selected.append((name, by_name[name]))
            continue

        index = entry.get("index")
        if index is None:
            raise SystemExit("policy entries need either name or index")
        if index < 0 or index >= len(modules):
            raise SystemExit(f"policy module index {index} is out of range")
        selected.append(modules[index])

    return selected


def _processor_config(
    args: argparse.Namespace, *, policy: dict[str, Any] | None
) -> dict[str, Any]:
    config = {
        "bits": args.bits,
        "qjl_bits": args.qjl_bits,
        "seed": args.processor_seed,
        "quantize_keys": not args.exact_keys,
        "quantize_values": args.quantize_values,
        "key_bits": args.key_bits,
        "value_bits": args.value_bits,
        "codebook_samples": args.codebook_samples,
    }
    if policy is None:
        return config

    policy_config = policy.get("turbo_policy", {})
    for source_key, target_key in (
        ("bits", "bits"),
        ("qjl_bits", "qjl_bits"),
        ("processor_seed", "seed"),
        ("quantize_keys", "quantize_keys"),
        ("quantize_values", "quantize_values"),
        ("key_bits", "key_bits"),
        ("value_bits", "value_bits"),
        ("codebook_samples", "codebook_samples"),
    ):
        if source_key in policy_config:
            config[target_key] = policy_config[source_key]
    return config


def _processor_metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "bits": config["bits"],
        "key_bits": config["key_bits"],
        "value_bits": config["value_bits"],
        "qjl_bits": config["qjl_bits"],
        "codebook_samples": config["codebook_samples"],
        "processor_seed": config["seed"],
        "quantize_keys": config["quantize_keys"],
        "quantize_values": config["quantize_values"],
    }


def _parse_indices(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_names(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def _print_modules(modules: list[tuple[str, Any]]) -> None:
    for index, (name, module) in enumerate(modules):
        heads = getattr(module, "heads", "?")
        cross_dim = getattr(module, "cross_attention_dim", None)
        print(f"{index:03d} heads={heads} cross_dim={cross_dim} {name}")


def _module_metadata(
    all_modules: list[tuple[str, Any]], selected_modules: list[tuple[str, Any]]
) -> list[dict[str, Any]]:
    indices_by_module = {id(module): index for index, (_name, module) in enumerate(all_modules)}
    return [
        {
            "index": indices_by_module[id(module)],
            "name": name,
            "heads": getattr(module, "heads", None),
            "cross_attention_dim": getattr(module, "cross_attention_dim", None),
        }
        for name, module in selected_modules
    ]


def _pipeline_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
    }
    if args.negative_prompt is not None:
        kwargs["negative_prompt"] = args.negative_prompt
    if args.height is not None:
        kwargs["height"] = args.height
    if args.width is not None:
        kwargs["width"] = args.width
    return kwargs


def _run_image(
    pipe: Any,
    *,
    torch: Any,
    args: argparse.Namespace,
    common_kwargs: dict[str, Any],
    label: str,
) -> tuple[Any, dict[str, Any]]:
    if _is_cuda_device(torch, args.device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    generator = torch.Generator(device=_generator_device(torch, args.device)).manual_seed(
        args.seed
    )
    start = time.perf_counter()
    with torch.inference_mode():
        result = pipe(**common_kwargs, generator=generator)
    if _is_cuda_device(torch, args.device):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    stats: dict[str, Any] = {
        "label": label,
        "seconds": elapsed,
    }
    if _is_cuda_device(torch, args.device):
        stats.update(
            {
                "cuda_max_memory_allocated_mib": _mib(torch.cuda.max_memory_allocated()),
                "cuda_max_memory_reserved_mib": _mib(torch.cuda.max_memory_reserved()),
            }
        )

    return result.images[0], stats


def _generator_device(torch: Any, device: str) -> str:
    if _is_cuda_device(torch, device):
        return device
    return "cpu"


def _is_cuda_device(torch: Any, device: str) -> bool:
    return device.startswith("cuda") and torch.cuda.is_available()


def _mib(value: int) -> float:
    return value / (1024 * 1024)


def _install_processor(
    modules: list[tuple[str, Any]], processor: TurboDAttnProcessor
) -> None:
    for _name, module in modules:
        setter = getattr(module, "set_processor", None)
        if callable(setter):
            setter(processor)
        else:
            module.processor = processor


def _image_metrics(baseline_image: Any, turbo_image: Any) -> dict[str, float]:
    baseline = _image_array(baseline_image)
    turbo = _image_array(turbo_image)
    diff = turbo - baseline
    mse = float(np.mean(diff**2))
    rmse = math.sqrt(mse)
    max_abs = float(np.max(np.abs(diff)))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": max_abs,
        "psnr_db": float("inf") if mse == 0.0 else 20.0 * math.log10(1.0 / rmse),
        "mean_abs_red": float(np.mean(np.abs(diff[..., 0]))),
        "mean_abs_green": float(np.mean(np.abs(diff[..., 1]))),
        "mean_abs_blue": float(np.mean(np.abs(diff[..., 2]))),
    }


def _image_array(image: Any) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _write_diff_heatmap(baseline_image: Any, turbo_image: Any, path: Path) -> None:
    from PIL import Image, ImageOps

    baseline = _image_array(baseline_image)
    turbo = _image_array(turbo_image)
    diff = np.mean(np.abs(turbo - baseline), axis=-1)
    scale = float(np.percentile(diff, 99.0))
    if scale <= 0.0:
        scale = 1.0
    normalized = np.clip(diff / scale, 0.0, 1.0)
    gray = Image.fromarray((normalized * 255).astype(np.uint8))
    heatmap = ImageOps.colorize(gray, black="black", white="red")
    heatmap.save(path)


if __name__ == "__main__":
    main()
