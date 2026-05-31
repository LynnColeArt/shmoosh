import numpy as np

from shmoosh.metrics import attention_metrics, cosine_error, mse, softmax_kl


def test_basic_metrics_are_zero_for_identical_inputs() -> None:
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    assert mse(x, x) == 0.0
    assert abs(cosine_error(x, x)) < 1e-6
    assert abs(softmax_kl(x, x)) < 1e-6


def test_attention_metrics_identical_candidate_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    q = rng.normal(size=(2, 8, 16)).astype(np.float32)
    k = rng.normal(size=(2, 8, 16)).astype(np.float32)
    v = rng.normal(size=(2, 8, 16)).astype(np.float32)
    scores = q @ np.swapaxes(k, -1, -2)

    metrics = attention_metrics(q, k, v, scores, v)

    assert metrics.score_mse < 1e-10
    assert metrics.softmax_kl < 1e-6
    assert metrics.output_mse < 1e-10
