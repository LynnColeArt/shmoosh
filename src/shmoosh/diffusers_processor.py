from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from shmoosh.packed_attention import packed_key_attention_output
from shmoosh.packed_keys import encode_packed_keys
from shmoosh.packed_scores import score_resources_from_codec
from shmoosh.quantization import ShmooshCodec
from shmoosh.runtime_attention import torch_shmoosh_attention


@dataclass
class DenoisingStepState:
    current_step: int = 0
    total_steps: int | None = None


@dataclass(frozen=True)
class ScheduledShmooshAttnProcessor:
    original_processor: Any
    shmoosh_processor: Any
    step_state: DenoisingStepState
    quantize_start_step: int = 0
    quantize_end_step: int | None = None
    quantize_start_percent: float | None = None
    quantize_end_percent: float | None = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        processor = (
            self.shmoosh_processor
            if self._quantize_current_step()
            else self.original_processor or _sdpa_processor()
        )
        return processor(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            *args,
            **kwargs,
        )

    def _quantize_current_step(self) -> bool:
        step = self.step_state.current_step
        if step < self._start_step():
            return False
        end_step = self._end_step()
        if end_step is not None and step >= end_step:
            return False
        return True

    def _start_step(self) -> int:
        if self.quantize_start_percent is None:
            return self.quantize_start_step
        return self._percent_to_step(self.quantize_start_percent)

    def _end_step(self) -> int | None:
        if self.quantize_end_percent is None:
            return self.quantize_end_step
        return self._percent_to_step(self.quantize_end_percent)

    def _percent_to_step(self, percent: float) -> int:
        if self.step_state.total_steps is None:
            raise RuntimeError("percentage timestep windows require total_steps")
        return math.ceil(self.step_state.total_steps * percent)


@dataclass(frozen=True)
class ShmooshAttnProcessor:
    """Diffusers attention processor backed by Shmoosh attention paths.

    The default backend is the NumPy reference codec used for behavioral
    experiments. `attention_backend="packed"` routes K-only/exact-V policies
    through the Torch/Triton packed-key attention primitive.
    """

    bits: int = 3
    qjl_bits: int = 128
    seed: int = 11
    quantize_keys: bool = True
    quantize_values: bool = True
    key_bits: int | None = None
    value_bits: int | None = None
    codebook_samples: int = 80_000
    fallback_on_mask: bool = True
    attention_backend: str = "reference"
    packed_backend: str = "auto"
    _packed_codec_cache: dict[tuple[int, int, int, int, int, int], ShmooshCodec] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _packed_resource_cache: dict[tuple[int, int, int, int, int, int, str], Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.bits <= 0:
            raise ValueError("bits must be positive")
        if self.qjl_bits < 0:
            raise ValueError("qjl_bits must be non-negative")
        if self.attention_backend not in {"reference", "packed"}:
            raise ValueError("attention_backend must be one of: reference, packed")
        if self.packed_backend not in {"auto", "torch", "triton"}:
            raise ValueError("packed_backend must be one of: auto, torch, triton")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        if attention_mask is not None and self.fallback_on_mask:
            return _sdpa_processor()(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if self._use_packed_attention():
            key_bits = self.key_bits or self.bits
            codec = self._packed_codec(head_dim=head_dim, bits=key_bits)
            block = encode_packed_keys(
                key,
                bits=key_bits,
                qjl_bits=self.qjl_bits,
                seed=self.seed,
                codebook_samples=self.codebook_samples,
                codec=codec,
            )
            resources = self._packed_score_resources(
                query_device=query.device,
                head_dim=head_dim,
                bits=key_bits,
                codec=codec,
            )
            hidden_states = packed_key_attention_output(
                query,
                block,
                value,
                resources=resources,
                backend=self.packed_backend,
            )
        else:
            hidden_states = torch_shmoosh_attention(
                query,
                key,
                value,
                bits=self.bits,
                qjl_bits=self.qjl_bits,
                seed=self.seed,
                quantize_keys=self.quantize_keys,
                quantize_values=self.quantize_values,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                codebook_samples=self.codebook_samples,
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def _use_packed_attention(self) -> bool:
        return (
            self.attention_backend == "packed"
            and self.quantize_keys
            and not self.quantize_values
        )

    def warm_packed_attention(
        self,
        *,
        head_dim: int,
        device: Any,
        dtype: Any | None = None,
    ) -> bool:
        if not self._use_packed_attention():
            return False

        torch = _load_torch()
        target_device = torch.device(device)
        target_dtype = torch.float32 if dtype is None else dtype
        key_bits = self.key_bits or self.bits
        codec = self._packed_codec(head_dim=head_dim, bits=key_bits)
        resources = self._packed_score_resources(
            query_device=target_device,
            head_dim=head_dim,
            bits=key_bits,
            codec=codec,
        )
        query = torch.zeros(
            (1, 1, 1, head_dim),
            device=target_device,
            dtype=target_dtype,
        )
        key = torch.zeros_like(query)
        value = torch.zeros_like(query)
        block = encode_packed_keys(
            key,
            bits=key_bits,
            qjl_bits=self.qjl_bits,
            seed=self.seed,
            codebook_samples=self.codebook_samples,
            codec=codec,
        )
        packed_key_attention_output(
            query,
            block,
            value,
            resources=resources,
            backend=self.packed_backend,
        )
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        return True

    def _packed_codec(self, *, head_dim: int, bits: int) -> ShmooshCodec:
        cache_key = self._packed_codec_key(head_dim=head_dim, bits=bits)
        codec = self._packed_codec_cache.get(cache_key)
        if codec is None:
            codec = ShmooshCodec(
                dim=head_dim,
                bits=bits,
                qjl_bits=self.qjl_bits,
                seed=self.seed,
                codebook_samples=self.codebook_samples,
            )
            self._packed_codec_cache[cache_key] = codec
        return codec

    def _packed_score_resources(
        self,
        *,
        query_device: Any,
        head_dim: int,
        bits: int,
        codec: ShmooshCodec,
    ) -> Any:
        torch = _load_torch()
        device = torch.device(query_device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        device_key = str(device)
        cache_key = (*self._packed_codec_key(head_dim=head_dim, bits=bits), device_key)
        resources = self._packed_resource_cache.get(cache_key)
        if resources is None:
            resources = score_resources_from_codec(codec, device=device)
            self._packed_resource_cache[cache_key] = resources
        return resources

    def _packed_codec_key(
        self,
        *,
        head_dim: int,
        bits: int,
    ) -> tuple[int, int, int, int, int, int]:
        return (
            head_dim,
            bits,
            self.qjl_bits,
            self.seed,
            self.codebook_samples,
            80,
        )


def warm_packed_attention_processor(
    attn: Any,
    processor: Any,
    *,
    device: Any,
    dtype: Any | None = None,
) -> bool:
    shmoosh_processor = (
        processor.shmoosh_processor
        if isinstance(processor, ScheduledShmooshAttnProcessor)
        else processor
    )
    if not isinstance(shmoosh_processor, ShmooshAttnProcessor):
        return False
    if not shmoosh_processor._use_packed_attention():
        return False

    head_dim = _attention_head_dim(attn)
    return shmoosh_processor.warm_packed_attention(
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )


def _attention_head_dim(attn: Any) -> int:
    heads = int(getattr(attn, "heads"))
    to_q = getattr(attn, "to_q")
    inner_dim = getattr(to_q, "out_features", None)
    if inner_dim is None:
        weight = getattr(to_q, "weight", None)
        if weight is not None:
            inner_dim = int(weight.shape[0])
    if inner_dim is None:
        inner_dim = getattr(attn, "inner_dim", None)
    if inner_dim is None:
        raise RuntimeError("could not infer attention head dimension for warmup")
    return int(inner_dim) // heads


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch is required for packed attention warmup") from exc
    return torch


def _sdpa_processor():
    try:
        from diffusers.models.attention_processor import AttnProcessor2_0
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("diffusers is required for ShmooshAttnProcessor fallback") from exc

    return AttnProcessor2_0()
