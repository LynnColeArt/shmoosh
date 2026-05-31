from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from turbo_d.metrics import AttentionMetrics, attention_metrics
from turbo_d.quantization import EncodedVectors, TurboDCodec, scalar_quantize


@dataclass(frozen=True)
class ProbeResult:
    turbo_d: AttentionMetrics
    scalar: AttentionMetrics


def load_npz_tensors(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loaded = np.load(path)
    return (
        loaded["q"].astype(np.float32),
        loaded["k"].astype(np.float32),
        loaded["v"].astype(np.float32),
    )


def load_npz_metadata(path: str | Path) -> dict[str, Any]:
    loaded = np.load(path)
    if "metadata" not in loaded:
        return {}

    import json

    return json.loads(str(loaded["metadata"]))


def run_attention_probe(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    bits: int,
    qjl_bits: int,
    seed: int,
    codebook_samples: int = 200_000,
    lloyd_iters: int = 80,
) -> ProbeResult:
    codec = TurboDCodec(
        dim=q.shape[-1],
        bits=bits,
        qjl_bits=qjl_bits,
        seed=seed,
        codebook_samples=codebook_samples,
        lloyd_iters=lloyd_iters,
    )

    encoded_k = codec.encode(k)
    encoded_v = codec.encode(v)
    decoded_v = codec.decode(encoded_v)
    turbo_scores = np.empty(q.shape[:-1] + (k.shape[-2],), dtype=np.float32)

    for head in range(q.shape[0]):
        turbo_scores[head] = codec.estimate_dot(q[head], slice_encoded(encoded_k, head))

    turbo = attention_metrics(q, k, v, turbo_scores, decoded_v)

    scalar_k = scalar_quantize(k, bits)
    scalar_v = scalar_quantize(v, bits)
    scalar_scores = q @ np.swapaxes(scalar_k, -1, -2)
    scalar = attention_metrics(q, k, v, scalar_scores, scalar_v)

    return ProbeResult(turbo_d=turbo, scalar=scalar)


def slice_encoded(encoded: EncodedVectors, head: int) -> EncodedVectors:
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
