from __future__ import annotations

from dataclasses import dataclass

from turbo_d.runtime_attention import torch_turbo_d_attention


@dataclass(frozen=True)
class TurboDAttnProcessor:
    """Slow Diffusers attention processor backed by the Turbo-D reference codec.

    This processor is for behavioral experiments only. It leaves projections and
    output layers in Torch/Diffusers, but computes attention itself through the
    NumPy reference codec. A production path would need fused Torch/Triton/CUDA
    kernels and packed codes.
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

    def __post_init__(self) -> None:
        if self.bits <= 0:
            raise ValueError("bits must be positive")
        if self.qjl_bits < 0:
            raise ValueError("qjl_bits must be non-negative")

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

        hidden_states = torch_turbo_d_attention(
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


def _sdpa_processor():
    try:
        from diffusers.models.attention_processor import AttnProcessor2_0
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("diffusers is required for TurboDAttnProcessor fallback") from exc

    return AttnProcessor2_0()
