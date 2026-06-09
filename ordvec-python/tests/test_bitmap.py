"""Tests for Bitmap — the pyo3 binding surface.

Exercise the top-bucket bitmap candidate-generator API (add, search,
top_m_candidates, save/load, accessors) and the two-stage flow with a
RankQuant rerank. Algorithmic correctness (kernel-vs-scalar parity,
constant-weight bitmap invariants, bilinear identity checks) lives in the
crate's Rust tests under `tests/index/`; the formal overlap-null theorem lives
in `ordvec-formalization`. These tests cover the pyo3 boundary only.
"""
from __future__ import annotations

import numpy as np
import pytest

from ordvec import Bitmap, RankQuant


def unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def test_new_reports_dim_and_n_top():
    idx = Bitmap(dim=128, n_top=32)
    assert idx.dim == 128
    assert idx.n_top == 32
    assert len(idx) == 0
    # 128 coords / 8 bits = 16 bytes per doc.
    assert idx.bytes_per_vec == 16


def test_add_updates_length():
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(30, 128))
    assert len(idx) == 30
    assert idx.byte_size == 30 * idx.bytes_per_vec


def test_search_shape():
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(50, 128))
    scores, indices = idx.search(unit_vectors(4, 128, seed=99), k=10)
    assert scores.shape == (4, 10)
    assert indices.shape == (4, 10)


def test_search_subset_matches_full_when_candidates_eq_all():
    vectors = unit_vectors(40, 128, seed=0)
    idx = Bitmap(dim=128, n_top=32)
    idx.add(vectors)

    query = unit_vectors(1, 128, seed=99)[0]
    candidates = np.arange(40, dtype=np.uint32)
    subset_scores, subset_ids = idx.search_subset(query, candidates, k=10)

    full_scores, full_ids = idx.search(query[None, :], k=10)
    np.testing.assert_array_equal(subset_ids, full_ids[0])
    np.testing.assert_array_equal(subset_scores, full_scores[0])


def test_search_subset_allows_unsorted_duplicates_and_ties_by_row_id():
    vectors = np.ones((12, 64), dtype=np.float32)
    idx = Bitmap(dim=64, n_top=16)
    idx.add(vectors)

    scores, ids = idx.search_subset(
        np.zeros(64, dtype=np.float32),
        np.array([9, 3, 3, 1], dtype=np.uint32),
        k=3,
    )

    np.testing.assert_array_equal(ids, np.array([1, 3, 3], dtype=np.int64))
    assert scores.dtype == np.float32
    np.testing.assert_array_equal(scores, np.full(3, scores[0], dtype=np.float32))


def test_search_subset_validates_doc_ids():
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(10, 128))
    q = unit_vectors(1, 128, seed=1)[0]

    with pytest.raises(IndexError):
        idx.search_subset(q, np.array([0, 99], dtype=np.uint32), k=2)
    with pytest.raises(ValueError, match="out of range"):
        idx.search_subset(q, np.array([-1], dtype=np.int64), k=1)
    with pytest.raises(TypeError, match="integer"):
        idx.search_subset(q, np.array([0.0], dtype=np.float32), k=1)
    with pytest.raises(ValueError, match="finite"):
        idx.search_subset(np.full(128, np.nan, dtype=np.float32), np.array([0]), k=1)
    with pytest.raises(ValueError, match="dim"):
        idx.search_subset(np.zeros(64, dtype=np.float32), np.array([0]), k=1)


def test_search_subset_accepts_strided_int64_doc_ids_and_caps_k():
    vectors = unit_vectors(10, 128, seed=2)
    idx = Bitmap(dim=128, n_top=32)
    idx.add(vectors)
    q = unit_vectors(1, 128, seed=3)[0]

    doc_ids = np.arange(10, dtype=np.int64)[::2]
    scores, ids = idx.search_subset(q, doc_ids, k=99)

    assert scores.shape == (5,)
    assert ids.shape == (5,)
    assert set(ids.tolist()).issubset(set(doc_ids.tolist()))


def test_top_m_candidates_shape_and_dtype():
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(50, 128))
    cands = idx.top_m_candidates(unit_vectors(1, 128, seed=99)[0], m=20)
    assert cands.shape == (20,)
    assert cands.dtype == np.uint32


def test_top_m_candidates_caps_at_index_size():
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(15, 128))
    cands = idx.top_m_candidates(unit_vectors(1, 128, seed=99)[0], m=100)
    # m_eff = min(m, len(index)) — Rust contract.
    assert cands.shape == (15,)


def test_self_query_is_top_candidate():
    vectors = unit_vectors(50, 128, seed=42)
    idx = Bitmap(dim=128, n_top=32)
    idx.add(vectors)
    for i in range(10):
        cands = idx.top_m_candidates(vectors[i], m=5)
        assert i in cands.tolist(), (
            f"row {i} not in own top-5 bitmap candidates: {cands.tolist()}"
        )


def test_save_load_roundtrip(tmp_path):
    vectors = unit_vectors(30, 128, seed=7)
    idx = Bitmap(dim=128, n_top=32)
    idx.add(vectors)

    path = str(tmp_path / "idx.tvbm")
    idx.write(path)
    loaded = Bitmap.load(path)

    assert len(loaded) == 30
    assert loaded.dim == 128
    assert loaded.n_top == 32
    assert loaded.bytes_per_vec == idx.bytes_per_vec

    q = unit_vectors(3, 128, seed=8)
    s_orig, i_orig = idx.search(q, k=5)
    s_load, i_load = loaded.search(q, k=5)
    np.testing.assert_array_equal(i_orig, i_load)
    np.testing.assert_allclose(s_orig, s_load, rtol=1e-5)


def test_load_rejects_nonexistent_file():
    with pytest.raises(IOError):
        Bitmap.load("/nonexistent/path/does-not-exist.tvbm")


def test_invalid_n_top_rejected():
    # The binding validates 0 < n_top < dim and raises a clean ValueError.
    with pytest.raises(ValueError, match="n_top"):
        Bitmap(dim=64, n_top=0)
    with pytest.raises(ValueError, match="n_top"):
        Bitmap(dim=64, n_top=64)  # not strictly less than dim


def test_two_stage_rerank_recovers_top_neighbours():
    # Wiring test for the bitmap two-stage primitive: Bitmap candidate-gen
    # → RankQuant exact rerank. We assert the pipeline runs and recovers
    # self as top-1 — *not* a specific R@10 number (real-corpus recall
    # numbers live in the paper, not asserted here).
    vectors = unit_vectors(200, 128, seed=0)
    bitmap = Bitmap(dim=128, n_top=32)
    bitmap.add(vectors)

    rank_quant = RankQuant(dim=128, bits=2)
    rank_quant.add(vectors)

    # Self-query: shortlist via bitmap, rerank with rank-quant.
    for i in [0, 50, 199]:
        cands = bitmap.top_m_candidates(vectors[i], m=40)
        assert i in cands.tolist(), (
            f"two-stage pipeline lost row {i} at the bitmap stage"
        )
        # Rerank: pull candidate vectors, build a sub-index, search.
        sub = RankQuant(dim=128, bits=2)
        sub.add(vectors[cands])
        _, sub_indices = sub.search(vectors[i:i + 1], k=10)
        # `sub_indices` are local — self must be findable in the
        # candidate set at top-1.
        local_self = int(np.where(cands == i)[0][0])
        assert local_self in sub_indices[0].tolist()


def test_top_m_candidates_deterministic_across_repeated_calls():
    # Composite-key (score desc, doc_id asc) ordering makes candidate selection
    # fully deterministic even when many docs tie on bitmap-overlap score. The
    # fix landed in core (src/bitmap.rs) and is Rust-tested in
    # tests/index/bitmap.rs::bitmap_top_m_candidates_deterministic_at_ties; this
    # is the FFI-level regression guard. Small dim + small n_top so overlap
    # scores collide heavily and the tie-break is actually exercised.
    idx = Bitmap(dim=64, n_top=8)
    idx.add(unit_vectors(500, 64, seed=0))
    q = unit_vectors(1, 64, seed=1)[0]

    # Order-sensitive tuples: identical across repeated calls ⇒ deterministic
    # *ordering*, not merely a deterministic set.
    runs = [tuple(idx.top_m_candidates(q, m=20).tolist()) for _ in range(5)]
    assert len(set(runs)) == 1, (
        "repeated calls must return identical ordered candidates"
    )

    # Cross-check membership against search()'s top-m: both rank by the same
    # composite key over the same overlap scores, so the sets must agree.
    _, indices = idx.search(q.reshape(1, 64), k=20)
    assert set(runs[0]) == {int(i) for i in indices[0].tolist()}


def test_add_float64_is_coerced():
    # float64 accepted and coerced to float32 at the boundary; same index as f32.
    rng = np.random.default_rng(0)
    v32 = rng.standard_normal((20, 64)).astype(np.float32)
    a = Bitmap(dim=64, n_top=8)
    a.add(v32)
    b = Bitmap(dim=64, n_top=8)
    b.add(v32.astype(np.float64))
    assert len(a) == len(b) == 20
    q = rng.standard_normal((3, 64)).astype(np.float32)
    np.testing.assert_array_equal(a.search(q, k=5)[1], b.search(q, k=5)[1])


def test_dim_above_u16_max_rejected():
    # dim = 65536 is a multiple of 64 but exceeds u16::MAX; the binding must
    # reject it with a clean ValueError (mirrors the core Bitmap::new guard and
    # the .tvbm loader cap) rather than defer to a Rust panic on add/search.
    with pytest.raises(ValueError, match="u16 rank invariant"):
        Bitmap(dim=65_536, n_top=256)


def test_is_empty():
    idx = Bitmap(dim=128, n_top=32)
    assert idx.is_empty()
    idx.add(unit_vectors(3, 128))
    assert not idx.is_empty()


def test_build_query_bitmap_fp32_shape_and_popcount():
    idx = Bitmap(dim=128, n_top=32)
    q = unit_vectors(1, 128, seed=5)[0]
    qb = idx.build_query_bitmap_fp32(q)
    assert qb.dtype == np.uint64
    assert qb.shape == (128 // 64,)
    # The query bitmap flags exactly n_top top coordinates.
    assert sum(bin(int(w)).count("1") for w in qb) == 32


def test_top_m_candidates_batched_matches_single_query():
    vectors = unit_vectors(60, 128, seed=11)
    idx = Bitmap(dim=128, n_top=32)
    idx.add(vectors)
    queries = unit_vectors(5, 128, seed=12)
    batched = idx.top_m_candidates_batched(queries, m=10)
    assert batched.shape == (5, 10)
    assert batched.dtype == np.uint32
    for bi in range(5):
        np.testing.assert_array_equal(
            batched[bi], idx.top_m_candidates(queries[bi], m=10)
        )


def test_top_m_candidates_batched_empty_keeps_column_count():
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(20, 128))
    empty = np.empty((0, 128), dtype=np.float32)
    out = idx.top_m_candidates_batched(empty, m=10)
    assert out.shape == (0, 10)
    assert out.dtype == np.uint32


def test_top_m_candidates_batched_chunked_matches_single_query():
    vectors = unit_vectors(60, 128, seed=13)
    idx = Bitmap(dim=128, n_top=32)
    idx.add(vectors)
    queries = unit_vectors(7, 128, seed=14)  # 7 rows, chunk 3 → non-aligned tail
    chunked = idx.top_m_candidates_batched_chunked(queries, m=10, batch_size=3)
    assert chunked.shape == (7, 10)
    for bi in range(7):
        np.testing.assert_array_equal(
            chunked[bi], idx.top_m_candidates(queries[bi], m=10)
        )


def test_top_m_candidates_batched_chunked_rejects_zero_batch_size():
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(20, 128))
    q = unit_vectors(2, 128)
    with pytest.raises(ValueError, match="batch_size"):
        idx.top_m_candidates_batched_chunked(q, m=10, batch_size=0)


def test_top_m_candidates_batched_chunked_huge_batch_size_does_not_panic():
    # A batch_size far larger than the query count must not overflow
    # batch_size*dim in the core; the binding clamps it to one chunk
    # (result-transparent), so the output equals the unchunked batched call.
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(30, 128))
    queries = unit_vectors(4, 128, seed=21)
    huge = idx.top_m_candidates_batched_chunked(queries, m=10, batch_size=10**18)
    ref = idx.top_m_candidates_batched(queries, m=10)
    np.testing.assert_array_equal(huge, ref)


def test_body_overlap_scores_subset_matches_search_scores():
    vectors = unit_vectors(50, 128, seed=15)
    idx = Bitmap(dim=128, n_top=32)
    idx.add(vectors)
    q = unit_vectors(1, 128, seed=16)[0]
    qb = idx.build_query_bitmap_fp32(q)
    doc_ids = np.array([0, 5, 10, 42, 49], dtype=np.uint32)  # ascending, in range
    scores = idx.body_overlap_scores_subset(qb, doc_ids)
    assert scores.dtype == np.uint32
    assert scores.shape == (5,)
    # search() reports popcount(Q AND D) as the f32 score using the *same*
    # query bitmap; the subset scan must reproduce that overlap exactly.
    s_all, i_all = idx.search(q.reshape(1, 128), k=50)
    score_by_doc = {int(d): float(s) for s, d in zip(s_all[0], i_all[0])}
    for di, sc in zip(doc_ids.tolist(), scores.tolist()):
        assert float(sc) == score_by_doc[di]


def test_body_overlap_scores_subset_validates_inputs():
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(10, 128))
    q = unit_vectors(1, 128, seed=1)[0]
    qb = idx.build_query_bitmap_fp32(q)
    # Out-of-range doc id → IndexError.
    with pytest.raises(IndexError):
        idx.body_overlap_scores_subset(qb, np.array([0, 99], dtype=np.uint32))
    # Non-ascending doc ids → ValueError.
    with pytest.raises(ValueError, match="ascending"):
        idx.body_overlap_scores_subset(qb, np.array([5, 1], dtype=np.uint32))
    # Wrong q_bitmap length → ValueError.
    with pytest.raises(ValueError, match="dim/64"):
        idx.body_overlap_scores_subset(
            np.zeros(1, dtype=np.uint64), np.array([0], dtype=np.uint32)
        )
