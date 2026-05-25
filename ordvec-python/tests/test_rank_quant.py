"""Tests for RankQuant — the pyo3 binding surface.

Exercise the bucketed-rank Python API across the three supported bit
widths (1/2/4) — constructor, add, search (sym + asym),
search_asymmetric_subset, save/load roundtrip, len/dim/bits/bytes_per_vec
accessors. Algorithmic correctness (parity with the scalar reference,
bilinear identity, recall bounds) lives in the crate's Rust tests under
`tests/index/`; these tests cover the pyo3 boundary only.
"""
from __future__ import annotations

import numpy as np
import pytest

from ordvec import RankQuant


def unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_new_reports_dim_and_bits(bits):
    # dim must be a multiple of 2^bits; 128 is divisible by 2, 4, 16.
    idx = RankQuant(dim=128, bits=bits)
    assert idx.dim == 128
    assert idx.bits == bits
    assert len(idx) == 0
    # bytes_per_vec = dim * bits / 8.
    assert idx.bytes_per_vec == 128 * bits // 8


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_add_updates_length(bits):
    idx = RankQuant(dim=128, bits=bits)
    idx.add(unit_vectors(20, 128))
    assert len(idx) == 20
    assert idx.byte_size == 20 * idx.bytes_per_vec


def test_is_empty():
    idx = RankQuant(dim=128, bits=2)
    assert idx.is_empty()
    idx.add(unit_vectors(3, 128))
    assert not idx.is_empty()


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_search_shape(bits):
    idx = RankQuant(dim=128, bits=bits)
    idx.add(unit_vectors(50, 128))
    scores, indices = idx.search(unit_vectors(3, 128, seed=99), k=10)
    assert scores.shape == (3, 10)
    assert indices.shape == (3, 10)


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_search_asymmetric_shape(bits):
    idx = RankQuant(dim=128, bits=bits)
    idx.add(unit_vectors(50, 128))
    scores, indices = idx.search_asymmetric(
        unit_vectors(3, 128, seed=99), k=10
    )
    assert scores.shape == (3, 10)
    assert indices.shape == (3, 10)


@pytest.mark.parametrize("bits", [2, 4])
def test_self_query_recall_at_1(bits):
    # 1-bit is too lossy for a strict per-row self-query at this dim;
    # 2 and 4-bit are reliable. Keep the strict test where it's safe.
    vectors = unit_vectors(40, 128, seed=42)
    idx = RankQuant(dim=128, bits=bits)
    idx.add(vectors)
    _, indices = idx.search(vectors, k=1)
    np.testing.assert_array_equal(indices[:, 0], np.arange(40))


def test_invalid_bits_rejected():
    # The binding validates bits in {1, 2, 4} and raises a clean ValueError
    # (the core would otherwise panic and surface as a PanicException).
    with pytest.raises(ValueError, match="bits"):
        RankQuant(dim=64, bits=3)
    with pytest.raises(ValueError, match="bits"):
        RankQuant(dim=64, bits=8)


def test_dim_not_multiple_of_two_pow_bits_rejected():
    # dim must be a multiple of 8/bits and 2^bits — for bits=2 that's 4. 63 is not.
    with pytest.raises(ValueError, match="multiple"):
        RankQuant(dim=63, bits=2)


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_save_load_roundtrip(tmp_path, bits):
    vectors = unit_vectors(30, 128, seed=7)
    idx = RankQuant(dim=128, bits=bits)
    idx.add(vectors)

    path = str(tmp_path / f"idx_b{bits}.tvrq")
    idx.write(path)
    loaded = RankQuant.load(path)

    assert len(loaded) == 30
    assert loaded.dim == 128
    assert loaded.bits == bits
    assert loaded.bytes_per_vec == idx.bytes_per_vec

    q = unit_vectors(3, 128, seed=8)
    s_orig, i_orig = idx.search(q, k=5)
    s_load, i_load = loaded.search(q, k=5)
    np.testing.assert_array_equal(i_orig, i_load)
    np.testing.assert_allclose(s_orig, s_load, rtol=1e-5)


def test_load_rejects_nonexistent_file():
    with pytest.raises(IOError):
        RankQuant.load("/nonexistent/path/does-not-exist.tvrq")


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_add_float64_is_rejected(bits):
    # pyo3 numpy binding is strict on dtype — float64 is not silently
    # converted, the caller must convert.
    idx = RankQuant(dim=64, bits=bits)
    v64 = np.random.default_rng(0).standard_normal((4, 64)).astype(np.float64)
    with pytest.raises(TypeError):
        idx.add(v64)


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_swap_remove_shrinks_length(bits):
    idx = RankQuant(dim=64, bits=bits)
    idx.add(unit_vectors(8, 64))
    moved_from = idx.swap_remove(2)
    assert moved_from == 7
    assert len(idx) == 7


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_search_scores_descending(bits):
    idx = RankQuant(dim=128, bits=bits)
    idx.add(unit_vectors(30, 128))
    scores, _ = idx.search(unit_vectors(2, 128, seed=99), k=10)
    for row in scores:
        assert all(row[i] >= row[i + 1] for i in range(len(row) - 1))


def test_batch_query_matches_individual():
    idx = RankQuant(dim=128, bits=2)
    idx.add(unit_vectors(40, 128, seed=0))

    queries = unit_vectors(4, 128, seed=99)
    _, batch_indices = idx.search(queries, k=5)

    for i in range(4):
        _, single = idx.search(queries[i:i + 1], k=5)
        np.testing.assert_array_equal(batch_indices[i:i + 1], single)


def test_search_asymmetric_subset_returns_global_ids():
    # Subset rerank pins to the candidate IDs the caller supplied; the
    # returned `ids` are *global* doc indices, not local positions in
    # the candidate set.
    vectors = unit_vectors(50, 128, seed=0)
    idx = RankQuant(dim=128, bits=2)
    idx.add(vectors)

    candidates = np.array([0, 7, 13, 25, 41], dtype=np.uint32)
    scores, ids = idx.search_asymmetric_subset(vectors[0], candidates, k=3)

    assert scores.shape == (3,)
    assert ids.shape == (3,)
    assert scores.dtype == np.float32
    assert ids.dtype == np.int64
    # Self-query against a candidate set containing self → top-1 is self.
    assert int(ids[0]) == 0
    # All returned ids are from the candidate set (or sentinel -1).
    candidate_set = set(candidates.tolist()) | {-1}
    for i in ids:
        assert int(i) in candidate_set


def test_search_asymmetric_subset_matches_full_when_candidates_eq_all():
    # When the candidate set is every doc, the subset path must agree
    # with full `search_asymmetric` on the top-k. Both use the
    # asymmetric kernel; the subset path just iterates the candidate
    # list instead of all N docs. (Allow set equality — ties may
    # permute within the same scoring tier.)
    vectors = unit_vectors(40, 128, seed=0)
    idx = RankQuant(dim=128, bits=2)
    idx.add(vectors)

    query = unit_vectors(1, 128, seed=99)[0]
    candidates = np.arange(40, dtype=np.uint32)
    _, subset_ids = idx.search_asymmetric_subset(query, candidates, k=10)

    _, full_ids = idx.search_asymmetric(query[None, :], k=10)
    assert set(int(i) for i in subset_ids) == set(int(i) for i in full_ids[0])


def test_search_asymmetric_subset_k_caps_at_candidate_count():
    # k > len(candidates) should silently cap — no panic, no sentinel
    # padding beyond the candidate-set size.
    vectors = unit_vectors(40, 128, seed=0)
    idx = RankQuant(dim=128, bits=2)
    idx.add(vectors)

    candidates = np.array([3, 7, 11], dtype=np.uint32)
    scores, ids = idx.search_asymmetric_subset(
        vectors[0], candidates, k=20
    )
    # k_eff = min(k, len(candidates)) = 3.
    assert scores.shape == (3,)
    assert ids.shape == (3,)
