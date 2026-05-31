from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class PackedKeyAssumptions:
    batch_size: int = 2
    heads: int = 20
    key_tokens: int = 77
    head_dim: int = 64
    dtype_bytes: int = 2
    norm_bytes: int = 4
    residual_norm_bytes: int = 4

    @property
    def vectors_per_module(self) -> int:
        return self.batch_size * self.heads * self.key_tokens

    @property
    def exact_key_bytes_per_vector(self) -> int:
        return self.head_dim * self.dtype_bytes

    def exact_key_bytes_per_module(self) -> int:
        return self.vectors_per_module * self.exact_key_bytes_per_vector


@dataclass(frozen=True)
class PackedKeyFormat:
    key_bits: int
    qjl_bits: int
    head_dim: int = 64
    norm_bytes: int = 4
    residual_norm_bytes: int = 4

    @property
    def code_bytes_per_vector(self) -> int:
        return math.ceil(self.head_dim * self.key_bits / 8)

    @property
    def qjl_sign_bytes_per_vector(self) -> int:
        return math.ceil(self.qjl_bits / 8)

    @property
    def bytes_per_vector(self) -> int:
        residual_bytes = (
            self.qjl_sign_bytes_per_vector + self.residual_norm_bytes
            if self.qjl_bits > 0
            else 0
        )
        return self.code_bytes_per_vector + self.norm_bytes + residual_bytes


def estimate_policy_storage(
    policy: dict[str, Any],
    *,
    steps: list[int],
    assumptions: PackedKeyAssumptions,
) -> dict[str, Any]:
    modules = [
        _estimate_module(entry, policy=policy, assumptions=assumptions)
        for entry in policy.get("quantized_modules", [])
    ]
    module_count = len(modules)
    if module_count == 0:
        raise ValueError("policy does not contain quantized_modules")

    step_estimates = [
        _estimate_steps(total_steps, modules=modules) for total_steps in steps
    ]
    exact_per_step = sum(int(module["exact_key_bytes_per_step"]) for module in modules)
    packed_per_step = sum(int(module["packed_key_bytes_per_step"]) for module in modules)
    return {
        "assumptions": {
            "batch_size": assumptions.batch_size,
            "heads": assumptions.heads,
            "key_tokens": assumptions.key_tokens,
            "head_dim": assumptions.head_dim,
            "dtype_bytes": assumptions.dtype_bytes,
            "norm_bytes": assumptions.norm_bytes,
            "residual_norm_bytes": assumptions.residual_norm_bytes,
            "vectors_per_module": assumptions.vectors_per_module,
            "exact_key_bytes_per_vector": assumptions.exact_key_bytes_per_vector,
        },
        "module_count": module_count,
        "modules": modules,
        "per_quantized_step": _bytes_summary(
            exact_bytes=exact_per_step,
            packed_bytes=packed_per_step,
        ),
        "steps": step_estimates,
    }


def _estimate_module(
    entry: dict[str, Any],
    *,
    policy: dict[str, Any],
    assumptions: PackedKeyAssumptions,
) -> dict[str, Any]:
    config = _module_config(policy, entry)
    key_bits = int(config["key_bits"] if config["key_bits"] is not None else config["bits"])
    qjl_bits = int(config["qjl_bits"])
    packed_format = PackedKeyFormat(
        key_bits=key_bits,
        qjl_bits=qjl_bits,
        head_dim=assumptions.head_dim,
        norm_bytes=assumptions.norm_bytes,
        residual_norm_bytes=assumptions.residual_norm_bytes,
    )
    exact_bytes = assumptions.exact_key_bytes_per_module()
    packed_bytes = assumptions.vectors_per_module * packed_format.bytes_per_vector
    return {
        "index": entry.get("index"),
        "name": entry.get("name"),
        "key_bits": key_bits,
        "qjl_bits": qjl_bits,
        "quantize_keys": bool(config["quantize_keys"]),
        "quantize_start_step": _optional_int(entry.get("quantize_start_step"), default=0),
        "quantize_end_step": _optional_int(entry.get("quantize_end_step")),
        "quantize_start_percent": _optional_float(entry.get("quantize_start_percent")),
        "quantize_end_percent": _optional_float(entry.get("quantize_end_percent")),
        "packed_bytes_per_vector": packed_format.bytes_per_vector,
        "code_bytes_per_vector": packed_format.code_bytes_per_vector,
        "qjl_sign_bytes_per_vector": packed_format.qjl_sign_bytes_per_vector,
        "exact_key_bytes_per_step": exact_bytes,
        "packed_key_bytes_per_step": packed_bytes if config["quantize_keys"] else exact_bytes,
        "quantized_window": _bytes_summary(
            exact_bytes=exact_bytes,
            packed_bytes=packed_bytes if config["quantize_keys"] else exact_bytes,
        ),
    }


def _estimate_steps(total_steps: int, *, modules: list[dict[str, Any]]) -> dict[str, Any]:
    module_estimates = []
    baseline_exact_bytes = 0
    scheduled_bytes = 0
    quantized_window_exact_bytes = 0
    quantized_window_packed_bytes = 0
    for module in modules:
        quantized_steps = _quantized_steps(module, total_steps=total_steps)
        exact_steps = total_steps - quantized_steps
        exact_per_step = int(module["exact_key_bytes_per_step"])
        packed_per_step = int(module["packed_key_bytes_per_step"])
        module_baseline = exact_per_step * total_steps
        module_scheduled = exact_per_step * exact_steps + packed_per_step * quantized_steps
        baseline_exact_bytes += module_baseline
        scheduled_bytes += module_scheduled
        quantized_window_exact_bytes += exact_per_step * quantized_steps
        quantized_window_packed_bytes += packed_per_step * quantized_steps
        module_estimates.append(
            {
                "index": module["index"],
                "name": module["name"],
                "quantized_steps": quantized_steps,
                "exact_steps": exact_steps,
                "resolved_quantize_start_step": _resolve_window_step(
                    absolute_step=module["quantize_start_step"],
                    percent=module["quantize_start_percent"],
                    total_steps=total_steps,
                    default=0,
                ),
                "resolved_quantize_end_step": _resolve_window_step(
                    absolute_step=module["quantize_end_step"],
                    percent=module["quantize_end_percent"],
                    total_steps=total_steps,
                    default=total_steps,
                ),
                "baseline_exact_key_bytes": module_baseline,
                "scheduled_key_bytes": module_scheduled,
                "saved_key_bytes": module_baseline - module_scheduled,
            }
        )

    scheduled_summary = _bytes_summary(
        exact_bytes=baseline_exact_bytes,
        packed_bytes=scheduled_bytes,
    )
    quantized_window_summary = _bytes_summary(
        exact_bytes=quantized_window_exact_bytes,
        packed_bytes=quantized_window_packed_bytes,
    )
    return {
        "total_steps": total_steps,
        "baseline_exact_key_bytes": baseline_exact_bytes,
        "scheduled_key_bytes": scheduled_bytes,
        "saved_key_bytes": baseline_exact_bytes - scheduled_bytes,
        "saved_key_mib": _mib(baseline_exact_bytes - scheduled_bytes),
        "scheduled_horizon": scheduled_summary,
        "quantized_window": quantized_window_summary,
        "modules": module_estimates,
    }


def _module_config(policy: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    config = {
        "bits": 3,
        "key_bits": None,
        "qjl_bits": 128,
        "quantize_keys": True,
    }
    _apply_overrides(config, policy.get("turbo_policy", {}))
    _apply_overrides(config, entry)
    _apply_overrides(config, entry.get("turbo_policy", {}))
    return config


def _apply_overrides(config: dict[str, Any], overrides: Any) -> None:
    if not isinstance(overrides, dict):
        return
    for key in ("bits", "key_bits", "qjl_bits", "quantize_keys"):
        if key in overrides:
            config[key] = overrides[key]


def _quantized_steps(module: dict[str, Any], *, total_steps: int) -> int:
    start_step = _resolve_window_step(
        absolute_step=module["quantize_start_step"],
        percent=module["quantize_start_percent"],
        total_steps=total_steps,
        default=0,
    )
    end_step = _resolve_window_step(
        absolute_step=module["quantize_end_step"],
        percent=module["quantize_end_percent"],
        total_steps=total_steps,
        default=total_steps,
    )
    start_step = min(max(start_step, 0), total_steps)
    end_step = min(max(end_step, 0), total_steps)
    return max(0, end_step - start_step)


def _resolve_window_step(
    *,
    absolute_step: int | None,
    percent: float | None,
    total_steps: int,
    default: int,
) -> int:
    if percent is not None:
        return math.ceil(total_steps * percent)
    if absolute_step is None:
        return default
    return absolute_step


def _bytes_summary(*, exact_bytes: int, packed_bytes: int) -> dict[str, float | int]:
    saved = exact_bytes - packed_bytes
    return {
        "exact_bytes": exact_bytes,
        "packed_bytes": packed_bytes,
        "saved_bytes": saved,
        "exact_mib": _mib(exact_bytes),
        "packed_mib": _mib(packed_bytes),
        "saved_mib": _mib(saved),
        "compression_ratio": exact_bytes / packed_bytes if packed_bytes else 0.0,
        "saved_percent": (saved / exact_bytes * 100.0) if exact_bytes else 0.0,
    }


def _mib(byte_count: int) -> float:
    return byte_count / 1024.0 / 1024.0


def _optional_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
