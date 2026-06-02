from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from shmoosh.diffusers_processor import (
    DenoisingStepState,
    ScheduledShmooshAttnProcessor,
    ShmooshAttnProcessor,
    ShmooshTimingRecorder,
    warm_packed_attention_processor,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a same-seed baseline vs Shmoosh Diffusers image smoke test."
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
        help="Optional JSON policy file with quantized_modules and shmoosh_policy.",
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
        "--attention-backend",
        choices=["reference", "packed"],
        default="reference",
        help="Use the NumPy reference attention path or the packed-K exact-V path.",
    )
    parser.add_argument(
        "--packed-backend",
        choices=["auto", "torch", "triton"],
        default="auto",
        help="Packed score backend when --attention-backend=packed.",
    )
    parser.add_argument(
        "--code-format",
        choices=["packed", "packed_t", "byte"],
        default="packed",
        help="Runtime K-code layout for packed attention.",
    )
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
        "--trace-processor-timing",
        action="store_true",
        help="Record per-processor timing spans in metrics JSON.",
    )
    parser.add_argument(
        "--cache-cross-attention",
        action="store_true",
        help="Cache packed cross-attention K/V across denoising steps.",
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
    policy_selection = (
        _select_policy_module_entries(all_modules, policy=policy)
        if policy is not None
        else []
    )
    modules = (
        [(name, module) for name, module, _entry in policy_selection]
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

    processor_metadata: dict[str, Any]
    timing_recorder = _processor_timing_recorder(torch, args)
    step_state = DenoisingStepState(total_steps=args.steps) if policy_selection else None
    if policy_selection:
        _install_policy_processors(
            policy_selection,
            args=args,
            policy=policy,
            step_state=step_state,
            timing_recorder=timing_recorder,
        )
        processor_metadata = _policy_processor_metadata(
            all_modules, policy_selection, args=args, policy=policy
        )
    else:
        processor_config = _processor_config(args, policy=policy)
        processor = ShmooshAttnProcessor(
            **processor_config,
            timing_recorder=timing_recorder,
            timing_module=_single_timing_module(modules),
        )
        _install_processor(modules, processor)
        processor_metadata = _processor_metadata(processor_config)

    _warm_packed_processors(modules, torch=torch, args=args)

    shmoosh_image, shmoosh_stats = _run_image(
        pipe,
        torch=torch,
        args=args,
        common_kwargs=common_kwargs,
        label="shmoosh",
        step_state=step_state,
    )

    baseline_path = output_dir / "baseline.png"
    shmoosh_path = output_dir / "shmoosh.png"
    diff_path = output_dir / "diff_heatmap.png"
    metrics_path = output_dir / "metrics.json"

    baseline_image.save(baseline_path)
    shmoosh_image.save(shmoosh_path)
    image_metrics = _image_metrics(baseline_image, shmoosh_image)
    _write_diff_heatmap(baseline_image, shmoosh_image, diff_path)

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
        "processor": processor_metadata,
        "policy": policy,
        "baseline": baseline_stats,
        "shmoosh": shmoosh_stats,
        "image_metrics": image_metrics,
        "processor_timing": _processor_timing_payload(timing_recorder),
        "outputs": {
            "baseline": str(baseline_path),
            "shmoosh": str(shmoosh_path),
            "diff_heatmap": str(diff_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"wrote baseline image: {baseline_path}")
    print(f"wrote shmoosh image: {shmoosh_path}")
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
    return [
        (name, module)
        for name, module, _entry in _select_policy_module_entries(modules, policy=policy)
    ]


def _select_policy_module_entries(
    modules: list[tuple[str, Any]], *, policy: dict[str, Any]
) -> list[tuple[str, Any, dict[str, Any]]]:
    by_name = {name: module for name, module in modules}
    selected = []
    for entry in policy.get("quantized_modules", []):
        if not isinstance(entry, dict):
            raise SystemExit("policy quantized_modules entries must be objects")
        name = entry.get("name")
        if name is not None:
            if name not in by_name:
                raise SystemExit(f"policy module name not found: {name}")
            selected.append((name, by_name[name], entry))
            continue

        index = entry.get("index")
        if index is None:
            raise SystemExit("policy entries need either name or index")
        if index < 0 or index >= len(modules):
            raise SystemExit(f"policy module index {index} is out of range")
        name, module = modules[index]
        selected.append((name, module, entry))

    return selected


def _processor_config(
    args: argparse.Namespace,
    *,
    policy: dict[str, Any] | None,
    module_entry: dict[str, Any] | None = None,
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
        "attention_backend": args.attention_backend,
        "packed_backend": args.packed_backend,
        "code_format": args.code_format,
        "cache_cross_attention": getattr(args, "cache_cross_attention", False),
    }
    if policy is None:
        return config

    _apply_processor_overrides(config, policy.get("shmoosh_policy", {}))
    if module_entry is not None:
        _apply_processor_overrides(config, module_entry)
        _apply_processor_overrides(config, module_entry.get("shmoosh_policy", {}))
    return config


def _apply_processor_overrides(config: dict[str, Any], overrides: Any) -> None:
    if not isinstance(overrides, dict):
        return
    for source_key, target_key in (
        ("bits", "bits"),
        ("qjl_bits", "qjl_bits"),
        ("processor_seed", "seed"),
        ("quantize_keys", "quantize_keys"),
        ("quantize_values", "quantize_values"),
        ("key_bits", "key_bits"),
        ("value_bits", "value_bits"),
        ("codebook_samples", "codebook_samples"),
        ("attention_backend", "attention_backend"),
        ("packed_backend", "packed_backend"),
        ("code_format", "code_format"),
        ("cache_cross_attention", "cache_cross_attention"),
    ):
        if source_key in overrides:
            config[target_key] = overrides[source_key]


def _install_policy_processors(
    selection: list[tuple[str, Any, dict[str, Any]]],
    *,
    args: argparse.Namespace,
    policy: dict[str, Any],
    step_state: DenoisingStepState | None = None,
    timing_recorder: ShmooshTimingRecorder | None = None,
) -> None:
    for name, module, entry in selection:
        config = _processor_config(args, policy=policy, module_entry=entry)
        processor = ShmooshAttnProcessor(
            **config,
            timing_recorder=timing_recorder,
            timing_module=name,
            step_state=step_state,
        )
        window = _module_window_config(entry)
        if _uses_scheduled_window(window):
            if step_state is None:
                raise RuntimeError("scheduled policy processors require step_state")
            processor = ScheduledShmooshAttnProcessor(
                original_processor=getattr(module, "processor", None),
                shmoosh_processor=processor,
                step_state=step_state,
                quantize_start_step=window["quantize_start_step"],
                quantize_end_step=window["quantize_end_step"],
                quantize_start_percent=window["quantize_start_percent"],
                quantize_end_percent=window["quantize_end_percent"],
                timing_recorder=timing_recorder,
                timing_module=name,
            )
        _install_processor([(name, module)], processor)


def _policy_processor_metadata(
    all_modules: list[tuple[str, Any]],
    selection: list[tuple[str, Any, dict[str, Any]]],
    *,
    args: argparse.Namespace,
    policy: dict[str, Any],
) -> dict[str, Any]:
    default_config = _processor_config(args, policy=policy)
    module_indices = {
        id(module): index for index, (_name, module) in enumerate(all_modules)
    }
    module_configs = []
    for name, module, entry in selection:
        config = _processor_config(args, policy=policy, module_entry=entry)
        module_configs.append(
            {
                "index": module_indices[id(module)],
                "name": name,
                **_processor_metadata(config),
                **_module_window_config(entry),
                **_resolved_module_window_config(entry, total_steps=args.steps),
            }
        )

    default_metadata = _processor_metadata(default_config)
    default_module_metadata = {
        **default_metadata,
        "quantize_start_step": 0,
        "quantize_end_step": None,
        "quantize_start_percent": None,
        "quantize_end_percent": None,
        "resolved_quantize_start_step": 0,
        "resolved_quantize_end_step": None,
    }
    return {
        **default_metadata,
        "mixed": any(
            {
                key: value
                for key, value in module_config.items()
                if key not in {"index", "name"}
            }
            != default_module_metadata
            for module_config in module_configs
        ),
        "modules": module_configs,
    }


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
        "attention_backend": config["attention_backend"],
        "packed_backend": config["packed_backend"],
        "code_format": config["code_format"],
        "cache_cross_attention": config["cache_cross_attention"],
    }


def _module_window_config(entry: dict[str, Any]) -> dict[str, int | float | None]:
    return {
        "quantize_start_step": int(entry.get("quantize_start_step", 0)),
        "quantize_end_step": (
            None
            if entry.get("quantize_end_step") is None
            else int(entry["quantize_end_step"])
        ),
        "quantize_start_percent": (
            None
            if entry.get("quantize_start_percent") is None
            else float(entry["quantize_start_percent"])
        ),
        "quantize_end_percent": (
            None
            if entry.get("quantize_end_percent") is None
            else float(entry["quantize_end_percent"])
        ),
    }


def _resolved_module_window_config(
    entry: dict[str, Any], *, total_steps: int
) -> dict[str, int | None]:
    window = _module_window_config(entry)
    return {
        "resolved_quantize_start_step": _resolve_window_step(
            absolute_step=window["quantize_start_step"],
            percent=window["quantize_start_percent"],
            total_steps=total_steps,
            default=0,
        ),
        "resolved_quantize_end_step": _resolve_window_step(
            absolute_step=window["quantize_end_step"],
            percent=window["quantize_end_percent"],
            total_steps=total_steps,
            default=None,
        ),
    }


def _resolve_window_step(
    *,
    absolute_step: int | float | None,
    percent: int | float | None,
    total_steps: int,
    default: int | None,
) -> int | None:
    if percent is not None:
        return math.ceil(total_steps * float(percent))
    if absolute_step is None:
        return default
    return int(absolute_step)


def _uses_scheduled_window(window: dict[str, int | float | None]) -> bool:
    return (
        int(window["quantize_start_step"] or 0) > 0
        or window["quantize_end_step"] is not None
        or window["quantize_start_percent"] is not None
        or window["quantize_end_percent"] is not None
    )


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
    step_state: DenoisingStepState | None = None,
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
        if step_state is not None:
            step_state.current_step = 0
            result = pipe(
                **common_kwargs,
                generator=generator,
                callback_on_step_end=_step_end_callback(step_state),
                callback_on_step_end_tensor_inputs=[],
            )
        else:
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


def _step_end_callback(step_state: DenoisingStepState):
    def callback(_pipeline, step_index: int, _timestep, callback_kwargs: dict[str, Any]):
        step_state.current_step = step_index + 1
        return callback_kwargs

    return callback


def _generator_device(torch: Any, device: str) -> str:
    if _is_cuda_device(torch, device):
        return device
    return "cpu"


def _is_cuda_device(torch: Any, device: str) -> bool:
    return device.startswith("cuda") and torch.cuda.is_available()


def _mib(value: int) -> float:
    return value / (1024 * 1024)


def _processor_timing_recorder(
    torch: Any,
    args: argparse.Namespace,
) -> ShmooshTimingRecorder | None:
    if not getattr(args, "trace_processor_timing", False):
        return None
    return ShmooshTimingRecorder(synchronize_cuda=_is_cuda_device(torch, args.device))


def _processor_timing_payload(
    recorder: ShmooshTimingRecorder | None,
) -> dict[str, Any]:
    if recorder is None:
        return {"enabled": False}
    return recorder.payload()


def _single_timing_module(modules: list[tuple[str, Any]]) -> str | None:
    if len(modules) == 1:
        return modules[0][0]
    return None


def _install_processor(modules: list[tuple[str, Any]], processor: Any) -> None:
    for _name, module in modules:
        setter = getattr(module, "set_processor", None)
        if callable(setter):
            setter(processor)
        else:
            module.processor = processor


def _warm_packed_processors(
    modules: list[tuple[str, Any]],
    *,
    torch: Any,
    args: argparse.Namespace,
) -> list[str]:
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(getattr(args, "dtype", "fp32"), torch.float32)
    warmed = []
    for name, module in modules:
        if warm_packed_attention_processor(
            module,
            getattr(module, "processor", None),
            device=args.device,
            dtype=dtype,
        ):
            warmed.append(name)
    if warmed and _is_cuda_device(torch, args.device):
        torch.cuda.empty_cache()
    return warmed


def _image_metrics(baseline_image: Any, shmoosh_image: Any) -> dict[str, float]:
    baseline = _image_array(baseline_image)
    shmoosh = _image_array(shmoosh_image)
    diff = shmoosh - baseline
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


def _write_diff_heatmap(baseline_image: Any, shmoosh_image: Any, path: Path) -> None:
    from PIL import Image, ImageOps

    baseline = _image_array(baseline_image)
    shmoosh = _image_array(shmoosh_image)
    diff = np.mean(np.abs(shmoosh - baseline), axis=-1)
    scale = float(np.percentile(diff, 99.0))
    if scale <= 0.0:
        scale = 1.0
    normalized = np.clip(diff / scale, 0.0, 1.0)
    gray = Image.fromarray((normalized * 255).astype(np.uint8))
    heatmap = ImageOps.colorize(gray, black="black", white="red")
    heatmap.save(path)


if __name__ == "__main__":
    main()
