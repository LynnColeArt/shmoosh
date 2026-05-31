from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class EncodedVectors:
    """Compact representation produced by `ShmooshCodec.encode`."""

    indices: np.ndarray
    norms: np.ndarray
    original_shape: tuple[int, ...]
    residual_signs: np.ndarray | None = None
    residual_norms: np.ndarray | None = None

    @property
    def vector_count(self) -> int:
        return int(np.prod(self.original_shape[:-1], dtype=np.int64))

    @property
    def dim(self) -> int:
        return self.original_shape[-1]


class ShmooshCodec:
    """Reference TurboQuant-inspired vector codec.

    The implementation is intentionally simple and CPU-friendly:

    - vectors are normalized and randomly rotated;
    - rotated coordinates are scaled by sqrt(dim), then quantized with an
      empirical Lloyd-Max codebook for a standard normal source;
    - optional QJL-style residual signs can correct dot-product estimates.

    This is a research baseline, not a production diffusion kernel.
    """

    def __init__(
        self,
        dim: int,
        bits: int,
        *,
        qjl_bits: int = 0,
        seed: int = 0,
        codebook_samples: int = 200_000,
        lloyd_iters: int = 80,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if bits <= 0:
            raise ValueError("bits must be positive")
        if qjl_bits < 0:
            raise ValueError("qjl_bits must be non-negative")

        self.dim = dim
        self.bits = bits
        self.qjl_bits = qjl_bits
        self.seed = seed
        self.rotation = _orthogonal_matrix(dim, seed)
        self.codebook = _normal_lloyd_codebook(
            bits=bits,
            seed=seed + 1,
            samples=codebook_samples,
            iterations=lloyd_iters,
        )
        self.qjl_matrix = (
            _qjl_matrix(qjl_bits, dim, seed + 2) if qjl_bits > 0 else None
        )

    def encode(self, vectors: np.ndarray) -> EncodedVectors:
        vectors = _as_vectors(vectors, self.dim)
        original_shape = vectors.shape
        flat = vectors.reshape(-1, self.dim).astype(np.float32, copy=False)

        norms = np.linalg.norm(flat, axis=-1).astype(np.float32)
        safe_norms = np.where(norms > 0, norms, 1.0).astype(np.float32)
        unit = flat / safe_norms[:, None]
        rotated = unit @ self.rotation.T
        normalized = rotated * sqrt(self.dim)
        indices = _nearest_codebook_indices(normalized, self.codebook)

        residual_signs = None
        residual_norms = None
        if self.qjl_matrix is not None:
            decoded_unit = self._decode_unit(indices)
            reconstructed = decoded_unit * norms[:, None]
            residual = flat - reconstructed
            residual_norms = np.linalg.norm(residual, axis=-1).astype(np.float32)
            signs = residual @ self.qjl_matrix.T
            residual_signs = np.where(signs >= 0, 1, -1).astype(np.int8)

        return EncodedVectors(
            indices=indices.reshape(original_shape),
            norms=norms.reshape(original_shape[:-1]),
            original_shape=original_shape,
            residual_signs=residual_signs,
            residual_norms=residual_norms,
        )

    def decode(self, encoded: EncodedVectors) -> np.ndarray:
        flat_indices = encoded.indices.reshape(-1, self.dim)
        unit = self._decode_unit(flat_indices)
        vectors = unit * encoded.norms.reshape(-1, 1)
        return vectors.reshape(encoded.original_shape).astype(np.float32)

    def estimate_dot(self, queries: np.ndarray, encoded: EncodedVectors) -> np.ndarray:
        """Estimate dot products between queries and encoded vectors.

        Returns an array with shape `queries.shape[:-1] + encoded.shape[:-1]`.
        If QJL residuals are present, a sign-sketch correction is added to the
        reconstructed-vector dot product.
        """

        queries = _as_vectors(queries, self.dim)
        query_shape = queries.shape[:-1]
        flat_q = queries.reshape(-1, self.dim).astype(np.float32, copy=False)
        decoded = self.decode(encoded).reshape(-1, self.dim)

        estimate = flat_q @ decoded.T
        if (
            self.qjl_matrix is not None
            and encoded.residual_signs is not None
            and encoded.residual_norms is not None
        ):
            projected_q = flat_q @ self.qjl_matrix.T
            correction = projected_q @ encoded.residual_signs.T
            correction *= sqrt(np.pi / 2.0) / float(self.qjl_bits)
            correction *= encoded.residual_norms.reshape(1, -1)
            estimate = estimate + correction

        return estimate.reshape(query_shape + encoded.original_shape[:-1]).astype(
            np.float32
        )

    def _decode_unit(self, flat_indices: np.ndarray) -> np.ndarray:
        normalized = self.codebook[flat_indices] / sqrt(self.dim)
        return normalized @ self.rotation


def scalar_quantize(vectors: np.ndarray, bits: int) -> np.ndarray:
    """Symmetric per-vector min/max scalar quantization baseline."""

    if bits <= 0:
        raise ValueError("bits must be positive")
    levels = (1 << bits) - 1
    vectors = np.asarray(vectors, dtype=np.float32)
    scale = np.max(np.abs(vectors), axis=-1, keepdims=True)
    safe_scale = np.where(scale > 0, scale, 1.0)
    normalized = np.clip((vectors / safe_scale + 1.0) * 0.5, 0.0, 1.0)
    quantized = np.rint(normalized * levels) / levels
    return ((quantized * 2.0 - 1.0) * safe_scale).astype(np.float32)


def _as_vectors(vectors: np.ndarray, dim: int) -> np.ndarray:
    vectors = np.asarray(vectors)
    if vectors.shape[-1] != dim:
        raise ValueError(f"expected trailing dimension {dim}, got {vectors.shape[-1]}")
    return vectors


def _orthogonal_matrix(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dim, dim)).astype(np.float32)
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return (q * signs).astype(np.float32)


def _normal_lloyd_codebook(
    *, bits: int, seed: int, samples: int, iterations: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = np.sort(rng.normal(size=samples).astype(np.float32))
    levels = 1 << bits
    quantiles = (np.arange(levels, dtype=np.float32) + 0.5) / levels
    centroids = np.quantile(data, quantiles).astype(np.float32)

    for _ in range(iterations):
        boundaries = (centroids[:-1] + centroids[1:]) * 0.5
        bucket_ids = np.searchsorted(boundaries, data)
        updated = centroids.copy()
        for bucket in range(levels):
            values = data[bucket_ids == bucket]
            if values.size:
                updated[bucket] = values.mean(dtype=np.float64)
        if np.max(np.abs(updated - centroids)) < 1e-5:
            break
        centroids = updated

    return centroids.astype(np.float32)


def _nearest_codebook_indices(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    distances = np.abs(values[..., None] - codebook.reshape(1, 1, -1))
    indices = np.argmin(distances, axis=-1)
    if len(codebook) <= np.iinfo(np.uint8).max + 1:
        return indices.astype(np.uint8)
    return indices.astype(np.uint16)


def _qjl_matrix(bits: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(bits, dim)).astype(np.float32)
