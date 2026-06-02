from __future__ import annotations

import json

import numpy as np
import pytest

from shmoosh.cli.packed_encode_parity import (
    _attention_tensor,
    _expand_captures,
    _load_metadata,
    _summary,
)

torch = pytest.importorskip("torch")


def test_attention_tensor_adds_batch_dimension() -> None:
    array = np.zeros((2, 3, 4), dtype=np.float32)

    tensor = _attention_tensor(torch, array, device="cpu", dtype=torch.float32)

    assert tuple(tensor.shape) == (1, 2, 3, 4)


def test_attention_tensor_accepts_batched_tensor() -> None:
    array = np.zeros((1, 2, 3, 4), dtype=np.float32)

    tensor = _attention_tensor(torch, array, device="cpu", dtype=torch.float32)

    assert tuple(tensor.shape) == (1, 2, 3, 4)


def test_attention_tensor_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="capture tensors"):
        _attention_tensor(
            torch,
            np.zeros((3, 4), dtype=np.float32),
            device="cpu",
            dtype=torch.float32,
        )


def test_expand_captures_accepts_files_and_dirs(tmp_path) -> None:
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    ignored = tmp_path / "ignored.txt"
    first.write_bytes(b"")
    second.write_bytes(b"")
    ignored.write_text("nope", encoding="utf-8")

    expanded = _expand_captures([str(tmp_path), str(first), str(ignored)])

    assert expanded == [first, second, first]


def test_load_metadata_reads_json_payload(tmp_path) -> None:
    path = tmp_path / "capture.npz"
    np.savez_compressed(
        path,
        q=np.zeros((1, 1, 1)),
        metadata=json.dumps({"module": "attn1"}),
    )

    loaded = np.load(path)

    assert _load_metadata(loaded) == {"module": "attn1"}


def test_summary_reports_worst_case() -> None:
    rows = [
        {"code_diff_count": 2, "code_diff_rate": 0.1, "output_mse": 1e-5},
        {"code_diff_count": 3, "code_diff_rate": 0.2, "output_mse": 1e-6},
    ]

    assert _summary(rows) == {
        "captures": 2,
        "total_code_diff_count": 5,
        "max_code_diff_count": 3,
        "max_code_diff_rate": 0.2,
        "max_output_mse": 1e-5,
    }
