"""Tests for Rank — the pyo3 binding surface.

Exercise the public Python API for the rank-cosine retrieval primitive
(add, search, search_asymmetric, save/load, swap_remove, len, dim,
bytes_per_vec, byte_size). The algorithmic correctness checks (kernel
parity, Spearman identity, recall bounds) live in the crate's Rust tests
under `tests/index/`; these tests cover the pyo3 boundary only.
"""
from __future__ import annotations

import numpy as np
import pytest

from ordvec import Rank


def unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def test_new_reports_dim_and_is_empty():
    idx = Rank(dim=64)
    assert idx.dim == 64
    assert len(idx) == 0
    # u16 rank storage → 2 bytes per coord.
    assert idx.bytes_per_vec == 128


def test_add_updates_length():
    idx = Rank(dim=64)
    idx.add(unit_vectors(20, 64))
    assert len(idx) == 20
    assert idx.byte_size == 20 * idx.bytes_per_vec


def test_is_empty():
    idx = Rank(dim=64)
    assert idx.is_empty()
    idx.add(unit_vectors(3, 64))
    assert not idx.is_empty()


def test_add_is_incremental():
    idx = Rank(dim=64)
    idx.add(unit_vectors(10, 64, seed=1))
    idx.add(unit_vectors(15, 64, seed=2))
    assert len(idx) == 25


def test_search_shape():
    idx = Rank(dim=128)
    idx.add(unit_vectors(50, 128))
    scores, indices = idx.search(unit_vectors(4, 128, seed=99), k=10)
    assert scores.shape == (4, 10)
    assert indices.shape == (4, 10)


def test_search_asymmetric_shape():
    idx = Rank(dim=128)
    idx.add(unit_vectors(50, 128))
    scores, indices = idx.search_asymmetric(
        unit_vectors(4, 128, seed=99), k=10
    )
    assert scores.shape == (4, 10)
    assert indices.shape == (4, 10)


def test_self_query_recall_at_1_symmetric():
    vectors = unit_vectors(50, 256, seed=42)
    idx = Rank(dim=256)
    idx.add(vectors)
    _, indices = idx.search(vectors, k=1)
    np.testing.assert_array_equal(indices[:, 0], np.arange(50))


def test_self_query_recall_at_1_asymmetric():
    # Asymmetric path: query stays FP32-L2-normalised, doc is rank
    # transform — self-query should still pick its own row at top-1.
    vectors = unit_vectors(50, 256, seed=42)
    idx = Rank(dim=256)
    idx.add(vectors)
    _, indices = idx.search_asymmetric(vectors, k=1)
    np.testing.assert_array_equal(indices[:, 0], np.arange(50))


def test_sym_and_asym_are_distinct_scoring_paths():
    # Both paths reachable from Python and produce non-NaN scores; they
    # are *not* expected to agree numerically — sym is Spearman on ranks,
    # asym is rank-cosine with fp32 queries.
    vectors = unit_vectors(50, 128, seed=0)
    queries = unit_vectors(3, 128, seed=99)
    idx = Rank(dim=128)
    idx.add(vectors)

    s_sym, _ = idx.search(queries, k=10)
    s_asym, _ = idx.search_asymmetric(queries, k=10)
    assert not np.isnan(s_sym).any()
    assert not np.isnan(s_asym).any()
    # They differ in at least one position — sanity check we're not
    # accidentally calling the same kernel via both names.
    assert not np.allclose(s_sym, s_asym)


def test_search_scores_descending():
    idx = Rank(dim=64)
    idx.add(unit_vectors(30, 64))
    scores, _ = idx.search(unit_vectors(2, 64, seed=99), k=10)
    for row in scores:
        assert all(row[i] >= row[i + 1] for i in range(len(row) - 1))


def test_save_load_roundtrip(tmp_path):
    vectors = unit_vectors(40, 128, seed=7)
    idx = Rank(dim=128)
    idx.add(vectors)

    path = str(tmp_path / "idx.tvr")
    idx.write(path)
    loaded = Rank.load(path)

    assert len(loaded) == 40
    assert loaded.dim == 128

    q = unit_vectors(3, 128, seed=8)
    s_orig, i_orig = idx.search(q, k=5)
    s_load, i_load = loaded.search(q, k=5)
    np.testing.assert_array_equal(i_orig, i_load)
    np.testing.assert_allclose(s_orig, s_load, rtol=1e-5)


def test_load_rejects_nonexistent_file():
    with pytest.raises(IOError):
        Rank.load("/nonexistent/path/does-not-exist.tvr")


def test_empty_index_search_does_not_panic():
    # An empty Rank index must accept a search call cleanly — the Rust
    # side has this test; verify it propagates through the pyo3 boundary
    # without raising.
    idx = Rank(dim=64)
    queries = unit_vectors(2, 64, seed=99)
    scores, indices = idx.search(queries, k=5)
    assert scores.shape[0] == 2
    assert indices.shape[0] == 2


def test_swap_remove_shrinks_length():
    idx = Rank(dim=64)
    idx.add(unit_vectors(10, 64))
    moved_from = idx.swap_remove(3)
    assert moved_from == 9
    assert len(idx) == 9


def test_add_float64_is_coerced():
    # ordvec normalizes real-valued input to float32 at the boundary: float64
    # (NumPy's default) is accepted and coerced, producing the same index as the
    # explicitly-f32 array. Rank discards magnitude, so coercion is lossless here.
    rng = np.random.default_rng(0)
    v32 = rng.standard_normal((9, 64)).astype(np.float32)
    a = Rank(dim=64)
    a.add(v32)
    b = Rank(dim=64)
    b.add(v32.astype(np.float64))
    assert len(a) == len(b) == 9
    q = rng.standard_normal((3, 64)).astype(np.float32)
    np.testing.assert_array_equal(a.search(q, k=5)[1], b.search(q, k=5)[1])
