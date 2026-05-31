import numpy as np

from shmoosh.metrics import cosine_error, mse
from shmoosh.runtime_attention import exact_attention_output, shmoosh_attention_output


def test_runtime_attention_shape() -> None:
    rng = np.random.default_rng(0)
    q = rng.normal(size=(3, 7, 16)).astype(np.float32)
    k = rng.normal(size=(3, 5, 16)).astype(np.float32)
    v = rng.normal(size=(3, 5, 16)).astype(np.float32)

    output = shmoosh_attention_output(
        q,
        k,
        v,
        bits=4,
        qjl_bits=32,
        seed=0,
        codebook_samples=10_000,
    )

    assert output.shape == q.shape
    assert np.isfinite(output).all()


def test_runtime_attention_more_bits_reduce_output_error() -> None:
    rng = np.random.default_rng(1)
    q = rng.normal(size=(2, 12, 32)).astype(np.float32)
    k = rng.normal(size=(2, 12, 32)).astype(np.float32)
    v = rng.normal(size=(2, 12, 32)).astype(np.float32)
    reference = exact_attention_output(q, k, v)

    output_3 = shmoosh_attention_output(
        q,
        k,
        v,
        bits=3,
        qjl_bits=0,
        seed=1,
        codebook_samples=10_000,
    )
    output_5 = shmoosh_attention_output(
        q,
        k,
        v,
        bits=5,
        qjl_bits=0,
        seed=1,
        codebook_samples=10_000,
    )

    assert mse(reference, output_5) < mse(reference, output_3)
    assert cosine_error(reference, output_5) < cosine_error(reference, output_3)


def test_runtime_attention_can_keep_values_exact() -> None:
    rng = np.random.default_rng(2)
    q = rng.normal(size=(2, 8, 16)).astype(np.float32)
    k = rng.normal(size=(2, 6, 16)).astype(np.float32)
    v = rng.normal(size=(2, 6, 16)).astype(np.float32)
    reference = exact_attention_output(q, k, v)

    quantized_values = shmoosh_attention_output(
        q,
        k,
        v,
        bits=3,
        qjl_bits=32,
        seed=2,
        quantize_values=True,
        codebook_samples=10_000,
    )
    exact_values = shmoosh_attention_output(
        q,
        k,
        v,
        bits=3,
        qjl_bits=32,
        seed=2,
        quantize_values=False,
        codebook_samples=10_000,
    )

    assert mse(reference, exact_values) < mse(reference, quantized_values)


def test_runtime_attention_can_keep_keys_exact() -> None:
    rng = np.random.default_rng(3)
    q = rng.normal(size=(2, 8, 16)).astype(np.float32)
    k = rng.normal(size=(2, 6, 16)).astype(np.float32)
    v = rng.normal(size=(2, 6, 16)).astype(np.float32)
    reference = exact_attention_output(q, k, v)

    output = shmoosh_attention_output(
        q,
        k,
        v,
        bits=3,
        qjl_bits=0,
        seed=3,
        quantize_keys=False,
        quantize_values=True,
        codebook_samples=10_000,
    )

    assert output.shape == reference.shape
    assert np.isfinite(output).all()


def test_runtime_attention_accepts_split_bit_widths() -> None:
    rng = np.random.default_rng(4)
    q = rng.normal(size=(2, 8, 16)).astype(np.float32)
    k = rng.normal(size=(2, 6, 16)).astype(np.float32)
    v = rng.normal(size=(2, 6, 16)).astype(np.float32)

    output = shmoosh_attention_output(
        q,
        k,
        v,
        bits=3,
        key_bits=3,
        value_bits=5,
        qjl_bits=32,
        seed=4,
        codebook_samples=10_000,
    )

    assert output.shape == q.shape
