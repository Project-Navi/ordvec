"""FFI boundary guards for the ordvec bindings.

Three classes of malformed input are rejected with clean, typed Python
exceptions instead of an opaque pyo3 ``PanicException`` from the Rust core:

* **Non-finite f32 (NaN / ±Inf)** -> ``ValueError``. ordvec enforces a strict
  all-finite input policy in the core (``assert_all_finite``); the binding
  pre-checks every f32 input so the failure is a clean ``ValueError`` at the
  boundary rather than a panic.
* **Non-contiguous arrays** (transpose / strided / F-order) -> ``ValueError``
  telling the caller to ``np.ascontiguousarray()`` first.
* **Out-of-range subset candidate ids** -> ``IndexError`` (the core gathers
  ``packed[di * bpv ..]`` and only ``assert``s the bound, so an OOB id would
  otherwise panic and leak buffer geometry).

Algorithmic correctness lives in the crate's Rust tests under `tests/` and
`tests/index/`; these cover the pyo3 boundary only.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ordvec import Bitmap, Rank, RankQuant, SignBitmap

NONFINITE = [math.nan, math.inf, -math.inf]


def unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


# -------------------------------------------------------------------
# Non-finite input -> ValueError (ordvec strict-finite policy).
# -------------------------------------------------------------------


@pytest.mark.parametrize("bad", NONFINITE)
def test_rank_add_nonfinite_raises_value_error(bad):
    idx = Rank(dim=64)
    v = unit_vectors(4, 64)
    v[1, 5] = bad
    with pytest.raises(ValueError, match="finite"):
        idx.add(v)


@pytest.mark.parametrize("bad", NONFINITE)
def test_rank_search_nonfinite_raises_value_error(bad):
    idx = Rank(dim=64)
    idx.add(unit_vectors(10, 64))
    q = unit_vectors(2, 64, seed=1)
    q[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        idx.search(q, k=3)


@pytest.mark.parametrize("bad", NONFINITE)
def test_rankquant_add_nonfinite_raises_value_error(bad):
    idx = RankQuant(dim=64, bits=2)
    v = unit_vectors(4, 64)
    v[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        idx.add(v)


@pytest.mark.parametrize("bad", NONFINITE)
def test_rankquant_search_asymmetric_nonfinite_raises_value_error(bad):
    idx = RankQuant(dim=64, bits=2)
    idx.add(unit_vectors(10, 64))
    q = unit_vectors(2, 64, seed=1)
    q[1, 3] = bad
    with pytest.raises(ValueError, match="finite"):
        idx.search_asymmetric(q, k=3)


def test_rankquant_subset_nonfinite_query_raises_value_error():
    # The finite guard fires before the candidate bounds-check, so a
    # non-finite query is a ValueError even with valid candidates.
    idx = RankQuant(dim=64, bits=2)
    idx.add(unit_vectors(10, 64))
    q = unit_vectors(1, 64, seed=1)[0]
    q[7] = math.nan
    candidates = np.array([0, 1, 2], dtype=np.uint32)
    with pytest.raises(ValueError, match="finite"):
        idx.search_asymmetric_subset(q, candidates, k=2)


@pytest.mark.parametrize("bad", NONFINITE)
def test_bitmap_add_nonfinite_raises_value_error(bad):
    idx = Bitmap(dim=64, n_top=16)
    v = unit_vectors(4, 64)
    v[2, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        idx.add(v)


def test_bitmap_top_m_nonfinite_raises_value_error():
    idx = Bitmap(dim=64, n_top=16)
    idx.add(unit_vectors(10, 64))
    q = unit_vectors(1, 64, seed=1)[0]
    q[0] = math.inf
    with pytest.raises(ValueError, match="finite"):
        idx.top_m_candidates(q, m=5)


@pytest.mark.parametrize("bad", NONFINITE)
def test_signbitmap_add_nonfinite_raises_value_error(bad):
    idx = SignBitmap(dim=64)
    v = unit_vectors(4, 64)
    v[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        idx.add(v)


def test_signbitmap_batched_nonfinite_raises_value_error():
    idx = SignBitmap(dim=64)
    idx.add(unit_vectors(10, 64))
    q = unit_vectors(4, 64, seed=1)
    q[1, 1] = math.nan
    with pytest.raises(ValueError, match="finite"):
        idx.top_m_candidates_batched(q, m=5)


# -------------------------------------------------------------------
# Non-contiguous input -> ValueError (call np.ascontiguousarray first).
# -------------------------------------------------------------------


def test_rank_add_transpose_raises_value_error():
    # arr.T on a 2-D array is the canonical non-contiguous case.
    idx = Rank(dim=128)
    bad = unit_vectors(128, 8).T  # shape (8, 128), F-order
    assert not bad.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        idx.add(bad)


def test_rankquant_search_transpose_raises_value_error():
    idx = RankQuant(dim=128, bits=2)
    idx.add(unit_vectors(8, 128))
    bad = unit_vectors(128, 4).T  # shape (4, 128), F-order
    assert not bad.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        idx.search(bad, k=3)


def test_rankquant_subset_noncontiguous_query_raises_value_error():
    # A strided 1-D query (every other element) is non-contiguous.
    idx = RankQuant(dim=128, bits=2)
    idx.add(unit_vectors(10, 128))
    wide = unit_vectors(1, 256)[0]
    bad_query = wide[::2]  # length 128, stride 2 → non-contiguous
    assert not bad_query.flags["C_CONTIGUOUS"]
    candidates = np.array([0, 1, 2], dtype=np.uint32)
    with pytest.raises(ValueError, match="C-contiguous"):
        idx.search_asymmetric_subset(bad_query, candidates, k=2)


def test_bitmap_top_m_noncontiguous_query_raises_value_error():
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(10, 128))
    wide = unit_vectors(1, 256)[0]
    bad_query = wide[::2]
    assert not bad_query.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        idx.top_m_candidates(bad_query, m=5)


def test_signbitmap_batched_noncontiguous_raises_value_error():
    idx = SignBitmap(dim=128)
    idx.add(unit_vectors(10, 128))
    bad = unit_vectors(128, 4).T  # (4, 128) F-order
    assert not bad.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        idx.top_m_candidates_batched(bad, m=5)


def test_contiguous_copy_still_works():
    # The documented escape hatch: np.ascontiguousarray() makes it pass.
    idx = Rank(dim=128)
    bad = unit_vectors(128, 8).T
    idx.add(np.ascontiguousarray(bad))
    assert len(idx) == 8


# -------------------------------------------------------------------
# Out-of-range subset candidate id -> IndexError (not PanicException).
# -------------------------------------------------------------------


def test_subset_out_of_range_candidate_raises_index_error():
    # Index holds 50 vectors → valid ids are 0..49. Candidate id 999 is
    # out of range and must raise IndexError, NOT a PanicException.
    vectors = unit_vectors(50, 128, seed=0)
    idx = RankQuant(dim=128, bits=2)
    idx.add(vectors)
    candidates = np.array([0, 7, 999], dtype=np.uint32)
    with pytest.raises(IndexError, match="out of range"):
        idx.search_asymmetric_subset(vectors[0], candidates, k=3)


def test_subset_candidate_equal_to_len_raises_index_error():
    # Boundary: id == len(index) is one past the end → must reject.
    vectors = unit_vectors(10, 128, seed=1)
    idx = RankQuant(dim=128, bits=2)
    idx.add(vectors)
    candidates = np.array([0, 10], dtype=np.uint32)  # 10 == len(idx)
    with pytest.raises(IndexError, match="out of range"):
        idx.search_asymmetric_subset(vectors[0], candidates, k=2)


def test_subset_in_range_candidates_still_work():
    # Regression guard: the range check must not reject valid ids.
    vectors = unit_vectors(50, 128, seed=0)
    idx = RankQuant(dim=128, bits=2)
    idx.add(vectors)
    candidates = np.array([0, 7, 13, 25, 41], dtype=np.uint32)
    scores, ids = idx.search_asymmetric_subset(vectors[0], candidates, k=3)
    assert scores.shape == (3,)
    assert ids.shape == (3,)
    assert int(ids[0]) == 0  # self-query → self ranks first


# -------------------------------------------------------------------
# Candidate / doc-id dtype acceptance. The core takes u32 ids, but NumPy
# index arrays are int64 by default (np.arange, np.where()[0], fancy
# indexing, np.argpartition). The binding accepts any integer dtype and
# converts to u32 with checked bounds rather than rejecting non-uint32
# with an opaque "ndarray cannot be cast as ndarray" TypeError.
# -------------------------------------------------------------------


# Every integer dtype a candidate set might realistically arrive in.
INT_DTYPES = [
    np.uint32,  # ordvec's own top_m_candidates output (zero-copy fast path)
    np.int64,  # NumPy default — np.arange / np.array([...]) / np.where()[0]
    np.int32,
    np.uint64,
    np.int16,
    np.uint16,
    np.int8,
    np.uint8,
]


@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_subset_candidate_dtype_accepted_and_equivalent(dtype):
    # Any integer dtype is accepted and yields results identical to the
    # uint32 reference. (Friend's report: int64/int32/uint64 used to raise
    # TypeError.) ids stay small enough for int8 (max 127).
    vectors = unit_vectors(50, 128, seed=0)
    idx = RankQuant(dim=128, bits=2)
    idx.add(vectors)
    ref = np.array([0, 7, 13, 25, 41], dtype=np.uint32)
    s_ref, id_ref = idx.search_asymmetric_subset(vectors[0], ref, k=4)

    s, ids = idx.search_asymmetric_subset(vectors[0], ref.astype(dtype), k=4)
    np.testing.assert_array_equal(ids, id_ref)
    np.testing.assert_array_equal(s, s_ref)


def test_subset_candidate_natural_numpy_idioms_accepted():
    # The ways a user actually builds a candidate set — all int64.
    vectors = unit_vectors(50, 128, seed=0)
    idx = RankQuant(dim=128, bits=2)
    idx.add(vectors)
    for candidates in (
        np.arange(20),
        np.where(np.arange(50) % 5 == 0)[0],
        np.argpartition(np.arange(50)[::-1], 15)[:15],
    ):
        assert candidates.dtype == np.int64  # confirm the trap dtype
        scores, ids = idx.search_asymmetric_subset(vectors[0], candidates, k=3)
        assert scores.shape == (3,) and ids.shape == (3,)


def test_subset_noncontiguous_uint32_candidates_accepted():
    # A strided uint32 view (non-contiguous) is copied through the checked
    # path rather than rejected — the contiguous fast path is just an
    # optimisation, not a requirement, for candidate ids.
    vectors = unit_vectors(50, 128, seed=0)
    idx = RankQuant(dim=128, bits=2)
    idx.add(vectors)
    strided = np.arange(0, 48, 2, dtype=np.uint32)[::3]
    assert not strided.flags["C_CONTIGUOUS"]
    scores, ids = idx.search_asymmetric_subset(vectors[0], strided, k=3)
    assert scores.shape == (3,) and ids.shape == (3,)


def test_subset_negative_candidate_raises_value_error():
    # Fail-loud: a negative id must NOT silently wrap to a huge u32
    # (np.asarray(-1, uint32) -> 4294967295). Reject with a clear ValueError.
    vectors = unit_vectors(50, 128, seed=0)
    idx = RankQuant(dim=128, bits=2)
    idx.add(vectors)
    candidates = np.array([0, -1, 5], dtype=np.int64)
    with pytest.raises(ValueError, match="out of range for a u32"):
        idx.search_asymmetric_subset(vectors[0], candidates, k=2)


def test_subset_overflow_candidate_raises_value_error():
    # Fail-loud: an id >= 2**32 must NOT silently wrap (2**32 + 5 -> 5) and
    # score the wrong document. Reject with a clear ValueError.
    vectors = unit_vectors(50, 128, seed=0)
    idx = RankQuant(dim=128, bits=2)
    idx.add(vectors)
    candidates = np.array([0, 2**32 + 5], dtype=np.int64)
    with pytest.raises(ValueError, match="out of range for a u32"):
        idx.search_asymmetric_subset(vectors[0], candidates, k=2)


def test_subset_out_of_range_int64_candidate_raises_index_error():
    # The >= len(index) check applies regardless of input dtype.
    vectors = unit_vectors(50, 128, seed=0)
    idx = RankQuant(dim=128, bits=2)
    idx.add(vectors)
    candidates = np.array([0, 999], dtype=np.int64)
    with pytest.raises(IndexError, match="out of range"):
        idx.search_asymmetric_subset(vectors[0], candidates, k=2)


def test_subset_float_candidates_raise_type_error():
    # A non-integer dtype is a clear TypeError, not a silent truncation.
    vectors = unit_vectors(50, 128, seed=0)
    idx = RankQuant(dim=128, bits=2)
    idx.add(vectors)
    candidates = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    with pytest.raises(TypeError, match="integer dtype"):
        idx.search_asymmetric_subset(vectors[0], candidates, k=2)


def test_body_overlap_doc_ids_int64_accepted():
    # Bitmap.body_overlap_scores_subset shares the same coercion: int64
    # (sorted) doc_ids are accepted; the ascending-order policy still holds.
    vectors = unit_vectors(50, 128, seed=0)
    bm = Bitmap(dim=128, n_top=32)
    bm.add(vectors)
    qb = bm.build_query_bitmap_fp32(vectors[0])
    ids_sorted = np.array([2, 4, 8, 16, 32], dtype=np.int64)
    out = bm.body_overlap_scores_subset(qb, ids_sorted)
    assert out.shape == (5,)
    with pytest.raises(ValueError, match="sorted"):
        bm.body_overlap_scores_subset(qb, np.array([16, 2, 4], dtype=np.int64))


# -------------------------------------------------------------------
# Wrong array width (ncols/len != dim) -> ValueError, not silent
# misalignment or a reshape panic. The core derives n = len/dim and only
# checks divisibility, so a wrong-but-divisible width would slip through.
# -------------------------------------------------------------------


def test_rank_add_wrong_width_raises_value_error():
    # (4, 128) into a dim-64 index: total length divides by 64, so the core
    # would silently treat it as 8 vectors. The width guard rejects it.
    idx = Rank(dim=64)
    bad = unit_vectors(4, 128)  # contiguous, but ncols 128 != dim 64
    with pytest.raises(ValueError, match="dimension"):
        idx.add(bad)


def test_rank_search_wrong_width_raises_value_error():
    idx = Rank(dim=64)
    idx.add(unit_vectors(10, 64))
    with pytest.raises(ValueError, match="dimension"):
        idx.search(unit_vectors(2, 128), k=3)


def test_rankquant_add_wrong_width_raises_value_error():
    idx = RankQuant(dim=64, bits=2)
    with pytest.raises(ValueError, match="dimension"):
        idx.add(unit_vectors(4, 128))


def test_rankquant_subset_wrong_width_query_raises_value_error():
    idx = RankQuant(dim=64, bits=2)
    idx.add(unit_vectors(10, 64))
    bad_q = unit_vectors(1, 128)[0]  # length 128 != dim 64
    candidates = np.array([0, 1, 2], dtype=np.uint32)
    with pytest.raises(ValueError, match="dimension"):
        idx.search_asymmetric_subset(bad_q, candidates, k=2)


def test_bitmap_top_m_wrong_width_raises_value_error():
    idx = Bitmap(dim=64, n_top=16)
    idx.add(unit_vectors(10, 64))
    with pytest.raises(ValueError, match="dimension"):
        idx.top_m_candidates(unit_vectors(1, 128)[0], m=5)


def test_signbitmap_batched_wrong_width_raises_value_error():
    idx = SignBitmap(dim=64)
    idx.add(unit_vectors(10, 64))
    with pytest.raises(ValueError, match="dimension"):
        idx.top_m_candidates_batched(unit_vectors(4, 128), m=5)


# -------------------------------------------------------------------
# swap_remove out-of-range -> IndexError (not PanicException).
# -------------------------------------------------------------------


def test_rank_swap_remove_out_of_range_raises_index_error():
    idx = Rank(dim=64)
    idx.add(unit_vectors(5, 64))
    with pytest.raises(IndexError, match="out of range"):
        idx.swap_remove(5)  # valid indices are 0..4


def test_rankquant_swap_remove_out_of_range_raises_index_error():
    idx = RankQuant(dim=64, bits=2)
    idx.add(unit_vectors(5, 64))
    with pytest.raises(IndexError, match="out of range"):
        idx.swap_remove(99)


def test_swap_remove_in_range_still_works():
    idx = Rank(dim=64)
    idx.add(unit_vectors(5, 64))
    moved = idx.swap_remove(1)
    assert moved == 4
    assert len(idx) == 4
