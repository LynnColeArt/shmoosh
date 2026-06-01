from __future__ import annotations

import pytest

from shmoosh.cli.self_attention_variant_bench import _parse_variants


def test_parse_variants_accepts_bits_and_qjl_bits() -> None:
    assert _parse_variants("6:128, 6:64,7:0") == [(6, 128), (6, 64), (7, 0)]


def test_parse_variants_rejects_empty_input() -> None:
    with pytest.raises(SystemExit, match="at least one"):
        _parse_variants("")


def test_parse_variants_rejects_malformed_entry() -> None:
    with pytest.raises(SystemExit, match="bits:qjl_bits"):
        _parse_variants("6")
