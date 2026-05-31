from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture Q/K/V tensors from Diffusers attention modules."
    )
    parser.add_argument("--model-id")
    parser.add_argument("--single-file")
    parser.add_argument(
        "--pipeline-class",
        choices=["auto", "stable-diffusion", "sdxl", "sana"],
        default="auto",
    )
    parser.add_argument(
        "--config",
        help="Optional local Diffusers config directory or Hub repo id for single-file checkpoints.",
    )
    parser.add_argument("--prompt")
    parser.add_argument("--output-dir", default="captures")
    parser.add_argument("--component", choices=["auto", "transformer", "unet"], default="auto")
    parser.add_argument("--module-filter", default="")
    parser.add_argument(
        "--module-indices",
        help="Comma-separated indices from --list-modules output, after filtering.",
    )
    parser.add_argument("--max-modules", type=int, default=4)
    parser.add_argument("--max-captures-per-module", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--model-cpu-offload", action="store_true")
    parser.add_argument("--list-modules", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if not args.list_modules and not args.prompt:
        raise SystemExit("--prompt is required unless --list-modules is used.")

    torch, DiffusionPipeline = _load_diffusers()
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

    _move_pipeline(pipe, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    captures: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"q": [], "k": [], "v": []} for name, _ in modules
    }
    handles = []

    for name, module in modules:
        handles.extend(
            [
                module.to_q.register_forward_hook(
                    _make_hook(args, captures, name, module, "q")
                ),
                module.to_k.register_forward_hook(
                    _make_hook(args, captures, name, module, "k")
                ),
                module.to_v.register_forward_hook(
                    _make_hook(args, captures, name, module, "v")
                ),
            ]
        )

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    pipe_kwargs: dict[str, Any] = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "generator": generator,
    }
    if args.height is not None:
        pipe_kwargs["height"] = args.height
    if args.width is not None:
        pipe_kwargs["width"] = args.width

    with torch.inference_mode():
        pipe(**pipe_kwargs)

    for handle in handles:
        handle.remove()

    written = _write_captures(output_dir, args, captures)
    print(f"wrote {written} capture file(s) to {output_dir}")


def _load_diffusers():
    try:
        import torch
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        raise SystemExit(
            "Install optional dependencies first: uv sync --extra dev --extra diffusers"
        ) from exc
    return torch, DiffusionPipeline


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
    else:
        raise SystemExit("--single-file currently supports stable-diffusion or sdxl.")

    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "local_files_only": args.local_files_only,
    }
    if args.config:
        kwargs["config"] = args.config

    return cls.from_single_file(
        args.single_file,
        **kwargs,
    )


def _move_pipeline(pipe: Any, args: argparse.Namespace) -> None:
    if args.model_cpu_offload:
        offload = getattr(pipe, "enable_model_cpu_offload", None)
        if callable(offload):
            offload(device=args.device)
            return

    pipe.to(args.device)


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
    if args.module_indices:
        selected = []
        for index in _parse_indices(args.module_indices):
            if index < 0 or index >= len(modules):
                raise SystemExit(f"module index {index} is out of range")
            selected.append(modules[index])
        return selected

    return modules[: args.max_modules]


def _parse_indices(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _print_modules(modules: list[tuple[str, Any]]) -> None:
    for index, (name, module) in enumerate(modules):
        heads = getattr(module, "heads", "?")
        cross_dim = getattr(module, "cross_attention_dim", None)
        print(f"{index:03d} heads={heads} cross_dim={cross_dim} {name}")


def _make_hook(
    args: argparse.Namespace,
    captures: dict[str, dict[str, list[np.ndarray]]],
    name: str,
    parent_module: Any,
    kind: str,
):
    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        bucket = captures[name][kind]
        if len(bucket) >= args.max_captures_per_module:
            return
        tensor = output.detach()
        if tensor.ndim != 3:
            return
        if tensor.shape[1] > args.max_tokens:
            return
        tensor = _reshape_heads(tensor, parent_module)
        bucket.append(tensor.float().cpu().numpy().astype(np.float32))

    return hook


def _reshape_heads(tensor: Any, parent_module: Any) -> Any:
    heads = getattr(parent_module, "heads", None)
    if not heads or tensor.shape[-1] % heads != 0:
        return tensor
    batch, tokens, inner_dim = tensor.shape
    head_dim = inner_dim // heads
    tensor = tensor.reshape(batch, tokens, heads, head_dim)
    return tensor.permute(0, 2, 1, 3).reshape(batch * heads, tokens, head_dim)


def _write_captures(
    output_dir: Path,
    args: argparse.Namespace,
    captures: dict[str, dict[str, list[np.ndarray]]],
) -> int:
    written = 0
    for module_name, values in captures.items():
        complete = min(len(values["q"]), len(values["k"]), len(values["v"]))
        for index in range(complete):
            metadata = {
                "model_id": args.model_id,
                "single_file": args.single_file,
                "pipeline_class": args.pipeline_class,
                "prompt": args.prompt,
                "steps": args.steps,
                "seed": args.seed,
                "module": module_name,
                "capture_index": index,
            }
            path = output_dir / f"capture_{written:03d}.npz"
            np.savez_compressed(
                path,
                q=values["q"][index],
                k=values["k"][index],
                v=values["v"][index],
                metadata=json.dumps(metadata),
            )
            written += 1
    return written


if __name__ == "__main__":
    main()
