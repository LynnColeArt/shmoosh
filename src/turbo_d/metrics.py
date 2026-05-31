from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AttentionMetrics:
    score_mse: float
    softmax_kl: float
    output_cosine_error: float
    output_mse: float


def mse(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.mean(delta * delta, dtype=np.float64))


def cosine_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1, a.shape[-1])
    b = np.asarray(b, dtype=np.float32).reshape(-1, b.shape[-1])
    dot = np.sum(a * b, axis=-1)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = dot / np.maximum(norms, eps)
    return float(np.mean(1.0 - cosine, dtype=np.float64))


def softmax_kl(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> float:
    reference = _softmax(reference_logits)
    candidate = _softmax(candidate_logits)
    kl = reference * (
        np.log(np.maximum(reference, 1e-12))
        - np.log(np.maximum(candidate, 1e-12))
    )
    return float(np.mean(np.sum(kl, axis=-1), dtype=np.float64))


def attention_metrics(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_v: np.ndarray,
) -> AttentionMetrics:
    dim = q.shape[-1]
    reference_scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(dim)
    scaled_candidate_scores = candidate_scores / np.sqrt(dim)

    reference_weights = _softmax(reference_scores)
    candidate_weights = _softmax(scaled_candidate_scores)
    reference_output = reference_weights @ v
    candidate_output = candidate_weights @ candidate_v

    return AttentionMetrics(
        score_mse=mse(reference_scores, scaled_candidate_scores),
        softmax_kl=softmax_kl(reference_scores, scaled_candidate_scores),
        output_cosine_error=cosine_error(reference_output, candidate_output),
        output_mse=mse(reference_output, candidate_output),
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)
