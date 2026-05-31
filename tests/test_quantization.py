import numpy as np

from shmoosh.quantization import ShmooshCodec, scalar_quantize


def test_codec_round_trip_shape() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(2, 16, 32)).astype(np.float32)
    codec = ShmooshCodec(dim=32, bits=4, qjl_bits=32, seed=1, codebook_samples=20_000)

    encoded = codec.encode(vectors)
    decoded = codec.decode(encoded)

    assert decoded.shape == vectors.shape
    assert encoded.indices.shape == vectors.shape
    assert encoded.norms.shape == vectors.shape[:-1]
    assert encoded.residual_signs is not None


def test_more_bits_reduce_reference_error() -> None:
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(128, 64)).astype(np.float32)
    codec_3 = ShmooshCodec(dim=64, bits=3, seed=2, codebook_samples=20_000)
    codec_5 = ShmooshCodec(dim=64, bits=5, seed=2, codebook_samples=20_000)

    err_3 = np.mean((vectors - codec_3.decode(codec_3.encode(vectors))) ** 2)
    err_5 = np.mean((vectors - codec_5.decode(codec_5.encode(vectors))) ** 2)

    assert err_5 < err_3


def test_dot_estimate_shape() -> None:
    rng = np.random.default_rng(3)
    keys = rng.normal(size=(11, 32)).astype(np.float32)
    queries = rng.normal(size=(7, 32)).astype(np.float32)
    codec = ShmooshCodec(dim=32, bits=4, qjl_bits=64, seed=3, codebook_samples=20_000)

    estimate = codec.estimate_dot(queries, codec.encode(keys))

    assert estimate.shape == (7, 11)


def test_scalar_quantize_shape() -> None:
    vectors = np.array([[0.0, 1.0, -1.0], [2.0, 0.5, -0.5]], dtype=np.float32)
    quantized = scalar_quantize(vectors, bits=4)

    assert quantized.shape == vectors.shape
    assert quantized.dtype == np.float32
