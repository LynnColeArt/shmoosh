from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from shmoosh.packed_scores import PackedScoreResources


@dataclass(frozen=True)
class RotatedKeyBlock:
    """Direct rotated-K probe representation.

    This block stores normalized keys after the Shmoosh rotation plus exact key
    norms. It is intentionally not compressed; its job is to isolate whether
    packed-code decode and codebook lookup are now dominating the attention path.
    """

    rotated_keys: Any
    norms: Any
    head_dim: int
    seed: int
    code_format: Literal["rotated"] = "rotated"

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (*tuple(int(size) for size in self.norms.shape), self.head_dim)

    @property
    def vector_count(self) -> int:
        batch, heads, tokens, _dim = self.shape
        return batch * heads * tokens

    @property
    def packed_bytes_per_vector(self) -> int:
        return (
            self.head_dim * self.rotated_keys.element_size()
            + self.norms.element_size()
        )

    def exact_key_bytes(self, *, dtype_bytes: int = 2) -> int:
        return self.vector_count * self.head_dim * dtype_bytes

    def packed_key_bytes(self) -> int:
        tensors = [self.rotated_keys, self.norms]
        return sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors)

    def compression_ratio(self, *, dtype_bytes: int = 2) -> float:
        return self.exact_key_bytes(dtype_bytes=dtype_bytes) / self.packed_key_bytes()


def encode_rotated_keys(
    keys: Any,
    *,
    resources: PackedScoreResources,
    seed: int,
    storage_dtype: Any | None = None,
) -> RotatedKeyBlock:
    """Normalize and rotate keys without scalar quantization or bit packing."""

    torch = _load_torch()
    if keys.ndim != 4:
        raise ValueError("keys must have shape (batch, heads, tokens, head_dim)")

    device = keys.device
    _batch, _heads, _tokens, head_dim = (int(size) for size in keys.shape)
    rotation = resources.rotation.to(device=device, dtype=torch.float32)
    if rotation.shape != (head_dim, head_dim):
        raise ValueError("score resources rotation does not match key head dimension")

    keys_f = keys.detach().to(dtype=torch.float32)
    if keys_f.data_ptr() == keys.data_ptr():
        keys_f = keys_f.clone()
    norms = torch.linalg.vector_norm(keys_f, dim=-1)
    safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    keys_f.div_(safe_norms.unsqueeze(-1))
    rotated_keys = torch.matmul(keys_f, rotation.T).contiguous()
    target_dtype = keys.dtype if storage_dtype is None else storage_dtype
    return RotatedKeyBlock(
        rotated_keys=rotated_keys.to(dtype=target_dtype),
        norms=norms,
        head_dim=head_dim,
        seed=seed,
    )


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch is required for rotated key blocks; install with "
            "`uv sync --extra dev --extra diffusers`"
        ) from exc
    return torch
