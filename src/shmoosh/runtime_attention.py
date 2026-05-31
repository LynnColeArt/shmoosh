from __future__ import annotations

from math import sqrt

import numpy as np

from shmoosh.quantization import EncodedVectors, ShmooshCodec


def shmoosh_attention_output(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    bits: int,
    qjl_bits: int,
    seed: int,
    quantize_keys: bool = True,
    quantize_values: bool = True,
    key_bits: int | None = None,
    value_bits: int | None = None,
    codebook_samples: int = 80_000,
    lloyd_iters: int = 80,
) -> np.ndarray:
    """Compute attention output with Shmoosh encoded keys and values.

    Inputs are post-projection attention tensors with shape
    `(head_like, tokens, dim)`, where `head_like` can be heads or batch*heads.
    This reference path is intentionally NumPy-based and slow.
    """

    q = _as_attention_tensor(q, "q")
    k = _as_attention_tensor(k, "k")
    v = _as_attention_tensor(v, "v")
    if q.shape[0] != k.shape[0] or k.shape != v.shape:
        raise ValueError(
            "expected q/k/v to share head-like axis and k/v to share token shape"
        )
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must share the same head dimension")

    key_bits = bits if key_bits is None else key_bits
    value_bits = bits if value_bits is None else value_bits

    key_codec = None
    encoded_k = None
    if quantize_keys:
        key_codec = ShmooshCodec(
            dim=q.shape[-1],
            bits=key_bits,
            qjl_bits=qjl_bits,
            seed=seed,
            codebook_samples=codebook_samples,
            lloyd_iters=lloyd_iters,
        )
        encoded_k = key_codec.encode(k)

    if quantize_values:
        value_codec = ShmooshCodec(
            dim=v.shape[-1],
            bits=value_bits,
            qjl_bits=0,
            seed=seed + 10_000,
            codebook_samples=codebook_samples,
            lloyd_iters=lloyd_iters,
        )
        decoded_v = value_codec.decode(value_codec.encode(v))
    else:
        decoded_v = v

    output = np.empty_like(q, dtype=np.float32)

    for head in range(q.shape[0]):
        if encoded_k is not None and key_codec is not None:
            scores = key_codec.estimate_dot(q[head], _slice_encoded(encoded_k, head))
        else:
            scores = q[head] @ k[head].T
        weights = _softmax(scores / sqrt(q.shape[-1]))
        output[head] = weights @ decoded_v[head]

    return output


def exact_attention_output(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _as_attention_tensor(q, "q")
    k = _as_attention_tensor(k, "k")
    v = _as_attention_tensor(v, "v")
    scores = q @ np.swapaxes(k, -1, -2)
    weights = _softmax(scores / sqrt(q.shape[-1]))
    return (weights @ v).astype(np.float32)


def torch_shmoosh_attention(
    query,
    key,
    value,
    *,
    bits: int,
    qjl_bits: int,
    seed: int,
    quantize_keys: bool = True,
    quantize_values: bool = True,
    key_bits: int | None = None,
    value_bits: int | None = None,
    codebook_samples: int = 80_000,
):
    """Torch wrapper for the NumPy reference attention path.

    Accepts Diffusers/PyTorch attention tensors shaped
    `(batch, heads, tokens, head_dim)` and returns a tensor with the same shape.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("expected query/key/value to have shape (batch, heads, tokens, dim)")

    batch, heads, q_tokens, dim = query.shape
    k_tokens = key.shape[2]
    q_np = (
        query.detach()
        .to(device="cpu", dtype=_torch_float32(query))
        .reshape(batch * heads, q_tokens, dim)
        .numpy()
    )
    k_np = (
        key.detach()
        .to(device="cpu", dtype=_torch_float32(key))
        .reshape(batch * heads, k_tokens, dim)
        .numpy()
    )
    v_np = (
        value.detach()
        .to(device="cpu", dtype=_torch_float32(value))
        .reshape(batch * heads, k_tokens, dim)
        .numpy()
    )
    out_np = shmoosh_attention_output(
        q_np,
        k_np,
        v_np,
        bits=bits,
        qjl_bits=qjl_bits,
        seed=seed,
        quantize_keys=quantize_keys,
        quantize_values=quantize_values,
        key_bits=key_bits,
        value_bits=value_bits,
        codebook_samples=codebook_samples,
    )

    import torch

    return torch.from_numpy(out_np).to(device=query.device, dtype=query.dtype).reshape(
        batch, heads, q_tokens, dim
    )


def _torch_float32(tensor):
    import torch

    return torch.float32 if tensor.dtype != torch.float64 else torch.float64


def _as_attention_tensor(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (head_like, tokens, dim)")
    return array


def _slice_encoded(encoded: EncodedVectors, head: int) -> EncodedVectors:
    return EncodedVectors(
        indices=encoded.indices[head],
        norms=encoded.norms[head],
        original_shape=encoded.original_shape[1:],
        residual_signs=(
            None
            if encoded.residual_signs is None
            else encoded.residual_signs.reshape(encoded.original_shape[:-1] + (-1,))[head]
        ),
        residual_norms=(
            None
            if encoded.residual_norms is None
            else encoded.residual_norms.reshape(encoded.original_shape[:-1])[head]
        ),
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)
