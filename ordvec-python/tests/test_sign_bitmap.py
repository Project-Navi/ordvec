"""Tests for SignBitmap — the pyo3 binding surface.

Exercise the 1-bit-per-coord sign-cosine retrieval substrate: add,
top_m_candidates (scalar), top_m_candidates_batched (AVX-512 VPOPCNTDQ
XOR-popcount kernel), and shape/dtype contracts. The Hamming-distance
algorithmic correctness and AVX-512/AVX2/scalar dispatch parity live in
the crate's Rust tests under `tests/index/`; these tests cover the pyo3
boundary only.
"""
from __future__ import annotations

import numpy as np
import pytest

from ordvec import SignBitmap


def unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def test_new_reports_dim_and_is_empty():
    idx = SignBitmap(dim=128)
    assert idx.dim == 128
    assert len(idx) == 0
    # 1 bit per coord ⇒ dim/8 bytes per doc.
    assert idx.bytes_per_vec == 16


def test_add_updates_length():
    idx = SignBitmap(dim=128)
    idx.add(unit_vectors(20, 128))
    assert len(idx) == 20
    assert idx.byte_size == 20 * idx.bytes_per_vec


def test_top_m_candidates_shape_and_dtype():
    idx = SignBitmap(dim=128)
    idx.add(unit_vectors(50, 128))
    cands = idx.top_m_candidates(unit_vectors(1, 128, seed=99)[0], m=20)
    assert cands.shape == (20,)
    assert cands.dtype == np.uint32


def test_top_m_candidates_caps_at_index_size():
    idx = SignBitmap(dim=128)
    idx.add(unit_vectors(10, 128))
    cands = idx.top_m_candidates(unit_vectors(1, 128, seed=99)[0], m=100)
    # m_eff = min(m, len(index)).
    assert cands.shape == (10,)


def test_self_query_top1_via_sign_agreement():
    # Self-query under sign-cosine: a vector's own sign-bitmap has zero
    # Hamming distance to itself, so it must be the top-1 candidate.
    vectors = unit_vectors(80, 128, seed=42)
    idx = SignBitmap(dim=128)
    idx.add(vectors)
    for i in range(20):
        cands = idx.top_m_candidates(vectors[i], m=1)
        assert int(cands[0]) == i, (
            f"row {i} did not self-rank at top-1 by sign Hamming"
        )


def test_top_m_candidates_batched_shape():
    idx = SignBitmap(dim=128)
    idx.add(unit_vectors(50, 128))
    queries = unit_vectors(8, 128, seed=99)
    batched = idx.top_m_candidates_batched(queries, m=10)
    assert batched.shape == (8, 10)
    assert batched.dtype == np.uint32


def test_batched_matches_scalar_for_each_row():
    # The batched AVX-512 VPOPCNTDQ kernel must agree with the scalar
    # path on the same query at top-1; we check the leading match for a
    # small batch (boundary ties at deeper ranks may diverge — see the
    # body-kernel tie-break follow-up, separate from sign-bitmap).
    idx = SignBitmap(dim=128)
    idx.add(unit_vectors(60, 128, seed=0))
    queries = unit_vectors(6, 128, seed=99)

    batched = idx.top_m_candidates_batched(queries, m=5)
    for i in range(6):
        scalar = idx.top_m_candidates(queries[i], m=5)
        # Top-1 must agree exactly across both code paths.
        assert int(batched[i, 0]) == int(scalar[0]), (
            f"batched vs scalar disagree on top-1 for query {i}: "
            f"batched={batched[i, 0]} scalar={scalar[0]}"
        )


def test_empty_batch_returns_consistent_column_count():
    # Regression for the empty-batch column-count invariant: batched on an
    # empty query array must return shape (0, m_eff), not (0, 0). Callers
    # that build a 2-D buffer expecting m_eff columns across all batches
    # break if this regresses.
    idx = SignBitmap(dim=64)
    idx.add(unit_vectors(20, 64))
    empty_q = np.empty((0, 64), dtype=np.float32)
    batched = idx.top_m_candidates_batched(empty_q, m=5)
    assert batched.shape == (0, 5), (
        f"empty-batch shape regressed: got {batched.shape}, expected (0, 5)"
    )
    assert batched.dtype == np.uint32


def test_empty_batch_against_empty_index_yields_zero_columns():
    # Companion to the empty-batch case: an empty *index* gives
    # m_eff = min(m, 0) = 0, so columns are zero too.
    idx = SignBitmap(dim=64)
    empty_q = np.empty((0, 64), dtype=np.float32)
    batched = idx.top_m_candidates_batched(empty_q, m=5)
    assert batched.shape == (0, 0)


def test_dim_not_multiple_of_64_rejected():
    # The binding validates that dim is a positive multiple of 64 -> ValueError.
    with pytest.raises(ValueError, match="multiple of 64"):
        SignBitmap(dim=65)
    with pytest.raises(ValueError, match="multiple of 64"):
        SignBitmap(dim=0)


def test_save_load_roundtrip(tmp_path):
    vectors = unit_vectors(30, 128, seed=7)
    idx = SignBitmap(dim=128)
    idx.add(vectors)

    path = str(tmp_path / "sign.tvsb")
    idx.write(path)
    loaded = SignBitmap.load(path)

    assert len(loaded) == 30
    assert loaded.dim == 128
    assert loaded.bytes_per_vec == idx.bytes_per_vec

    # Candidates from the loaded index must be byte-identical to
    # candidates from the original — the .tvsb encoding is exact.
    q = unit_vectors(1, 128, seed=8)[0]
    c_orig = idx.top_m_candidates(q, m=10)
    c_load = loaded.top_m_candidates(q, m=10)
    np.testing.assert_array_equal(c_orig, c_load)


def test_load_rejects_nonexistent_file():
    with pytest.raises(IOError):
        SignBitmap.load("/nonexistent/path/does-not-exist.tvsb")


def test_add_float64_is_rejected():
    idx = SignBitmap(dim=64)
    v64 = np.random.default_rng(0).standard_normal((4, 64)).astype(np.float64)
    with pytest.raises(TypeError):
        idx.add(v64)


@pytest.mark.parametrize("dim", [64, 128, 256, 1024])
def test_various_dims_supported(dim):
    # 1024 is the headline dim used in the paper's headline bench;
    # 64/128/256 cover smaller smoke configurations.
    idx = SignBitmap(dim=dim)
    idx.add(unit_vectors(10, dim))
    assert idx.dim == dim
    assert idx.bytes_per_vec == dim // 8


def test_is_empty():
    idx = SignBitmap(dim=128)
    assert idx.is_empty()
    idx.add(unit_vectors(2, 128))
    assert not idx.is_empty()


def test_build_query_bitmap_shape_and_semantics():
    # Parity with Rust SignBitmap::build_query_bitmap: bit j is set iff q[j] > 0.
    idx = SignBitmap(dim=128)
    q = np.arange(128, dtype=np.float32) - 64.0  # q[j] = j - 64
    qb = idx.build_query_bitmap(q)
    assert qb.dtype == np.uint64
    assert qb.shape == (128 // 64,)
    bits_set = sum(bin(int(w)).count("1") for w in qb)
    assert bits_set == int((q > 0.0).sum())
