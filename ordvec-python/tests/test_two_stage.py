"""Cross-binding wiring test for the two-stage retrieval primitive.

`SignBitmap` → candidate shortlist → `RankQuant` exact rerank → final
top-K. The Python pipeline must compose the two bindings without
intermediate conversions: candidate IDs as `uint32`, vectors gathered
from the corpus, exact subset search executed.

This is a wiring test. Real-corpus R@K numbers are the paper's job and
are intentionally not asserted here.
"""
from __future__ import annotations

import numpy as np

from ordvec import RankQuant, SignBitmap


def unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def two_stage_rerank(
    corpus: np.ndarray,
    queries: np.ndarray,
    *,
    m_candidates: int,
    k_final: int,
    bits: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference two-stage flow used by the paper pipeline.

    Stage 1: SignBitmap picks top-M candidate doc IDs per query via
    XOR-popcount Hamming.
    Stage 2: For each query, `RankQuant.search_asymmetric_subset` scores
    the candidate set in-place against the original index — no per-query
    rebuild, no copy of candidate vectors, ids mapped back to global doc
    indices by the Rust side.

    Returns ``(global_ids, scores)`` of shape ``(n_queries, k_final)``.
    """
    dim = corpus.shape[1]
    n_q = queries.shape[0]

    sign_idx = SignBitmap(dim=dim)
    sign_idx.add(corpus)

    # Build the rerank index *once* for the whole corpus; the subset
    # method indexes into it per-query without copying vectors.
    rq = RankQuant(dim=dim, bits=bits)
    rq.add(corpus)

    # Stage 1 — batched candidate generation.
    cand_matrix = sign_idx.top_m_candidates_batched(queries, m=m_candidates)
    # m_eff caps at the index size; it equals m_candidates in these tests but
    # stays correct if this helper is reused as example code with m > len(index).
    m_eff = min(m_candidates, corpus.shape[0])
    assert cand_matrix.shape == (n_q, m_eff)

    # Stage 2 — per-query exact rerank against the global RankQuant.
    out_ids = np.empty((n_q, k_final), dtype=np.int64)
    out_scores = np.empty((n_q, k_final), dtype=np.float32)
    for i in range(n_q):
        scores, ids = rq.search_asymmetric_subset(
            queries[i], cand_matrix[i], k=k_final
        )
        out_ids[i] = ids
        out_scores[i] = scores
    return out_ids, out_scores


def test_two_stage_pipeline_runs_end_to_end():
    corpus = unit_vectors(300, 128, seed=0)
    queries = unit_vectors(8, 128, seed=99)

    ids, scores = two_stage_rerank(
        corpus, queries, m_candidates=40, k_final=10
    )

    assert ids.shape == (8, 10)
    assert scores.shape == (8, 10)
    assert ids.dtype == np.int64
    # Scores must be sorted descending per row.
    for row in scores:
        assert all(row[i] >= row[i + 1] for i in range(len(row) - 1))


def test_two_stage_self_query_recovers_own_id():
    # Self-query path: each corpus vector reranks to itself at top-1
    # through both stages. This is the load-bearing wiring property —
    # if either binding mis-handles the candidate IDs, this fails.
    corpus = unit_vectors(200, 128, seed=0)
    queries = corpus[:10]
    ids, _ = two_stage_rerank(
        corpus, queries, m_candidates=40, k_final=5
    )
    np.testing.assert_array_equal(ids[:, 0], np.arange(10))


def test_two_stage_batched_candidate_matrix_dtype_consumed_correctly():
    # The SignBitmap kernel emits uint32 IDs; numpy gather (corpus[ids])
    # must accept that dtype and produce a float32 sub-corpus that
    # RankQuant.add will accept without conversion.
    corpus = unit_vectors(100, 128, seed=0)
    sign_idx = SignBitmap(dim=128)
    sign_idx.add(corpus)
    cand_matrix = sign_idx.top_m_candidates_batched(corpus[:3], m=20)
    assert cand_matrix.dtype == np.uint32
    sub = RankQuant(dim=128, bits=2)
    sub.add(corpus[cand_matrix[0]])  # gather on uint32 IDs
    assert len(sub) == 20
