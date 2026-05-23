//! Red-team hardening regression suite (ported from turbovec).
//!
//! Each test pins a specific robustness fix in the `rank_index` family.
//! These cases were extracted from turbovec's `redteam_alpha.rs`; only
//! the rank-relevant cases (the substrate that lives in `ordvec`) are
//! kept. The `MultiBucketBitmap` cases are gated behind the
//! `experimental` feature because that type is not on the default
//! public surface.
//!
//! - RT-1: `Bitmap::body_overlap_scores_subset` must reject
//!   out-of-range `doc_ids` *before* dispatching into the AVX-512
//!   VPOPCNTDQ kernel, which would otherwise issue an unchecked
//!   `bitmaps.as_ptr().add(di * qpv)` load → heap-buffer-overflow read.
//! - P-F: `MultiBucketBitmap::bilinear_score` must guard a
//!   `doc_idx` out of range with a clear assert. (experimental)
//! - P-G: `MultiBucketBitmap::new` must reject `dim == 0` (which
//!   would defer a div-by-zero into `add`). (experimental)
//! - P-H: top-k methods must clamp a user `k`/`m` to `n_vectors` so a
//!   `usize::MAX` request does not trigger a `Vec` capacity overflow.
//! - R2: `TopK` (via `Bitmap::search`) must tie-break deterministically
//!   on `(score desc, doc_id asc)` so SIMD-vs-scalar summation-order
//!   differences cannot flip near-ties across CPUs.

use ordvec::rank::rank_transform;
use ordvec::Bitmap;
#[cfg(feature = "experimental")]
use ordvec::MultiBucketBitmap;

/// Reconstruct a single document's top-bucket bitmap exactly the way
/// `Bitmap::add` does (bit j set iff rank[j] >= dim - n_top), so
/// the subset-overlap correctness check is independent of the kernel
/// under test.
fn ref_doc_bitmap(doc: &[f32], dim: usize, n_top: usize) -> Vec<u64> {
    let qpv = dim / 64;
    let cutoff = (dim - n_top) as u16;
    let ranks = rank_transform(doc);
    let mut bm = vec![0u64; qpv];
    for j in 0..dim {
        if ranks[j] >= cutoff {
            bm[j / 64] |= 1u64 << (j % 64);
        }
    }
    bm
}

fn ref_overlap(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x & y).count_ones())
        .sum()
}

// --- RT-1 (CRITICAL: heap OOB read in the AVX-512 subset kernel) -----

#[test]
#[should_panic(expected = "out of range")]
fn rt1_subset_rejects_oob_doc_id_at_simd_dim() {
    // D=1024 → qpv=16 (16 % 8 == 0) selects the AVX-512 VPOPCNTDQ
    // dispatch on capable CPUs. Four docs → valid ids are 0..4. An id
    // of 4 is one past the end; forwarded unchecked it would issue a
    // 64-byte ZMM load at `bitmaps[4*16..]`, reading past the heap
    // allocation. The guard must panic before dispatch instead.
    const DIM: usize = 1024;
    const N_TOP: usize = 256;
    let mut idx = Bitmap::new(DIM, N_TOP);
    let corpus: Vec<f32> = (0..4 * DIM)
        .map(|i| ((i * 7) % 101) as f32 - 50.0)
        .collect();
    idx.add(&corpus);
    let q: Vec<f32> = (0..DIM).map(|i| ((i * 13) % 97) as f32 - 48.0).collect();
    let qb = idx.build_query_bitmap_fp32(&q);
    let doc_ids = [4u32]; // OOB: only ids 0..4 exist
    let mut out = vec![0u32; 1];
    idx.body_overlap_scores_subset(&qb, &doc_ids, &mut out);
}

#[test]
fn rt1_subset_in_range_matches_reference_popcount() {
    // In-range ids must return the exact AND-popcount overlap. Compare
    // the kernel (SIMD on this host) against a hand-rebuilt bitmap
    // overlap reconstructed from the corpus rank transform.
    const DIM: usize = 1024;
    const N_TOP: usize = 256;
    let n_docs = 6usize;
    let mut idx = Bitmap::new(DIM, N_TOP);
    let corpus: Vec<f32> = (0..n_docs * DIM)
        .map(|i| (((i * 31) % 211) as f32) - 105.0)
        .collect();
    idx.add(&corpus);
    let q: Vec<f32> = (0..DIM).map(|i| ((i * 17) % 89) as f32 - 44.0).collect();
    let qb = idx.build_query_bitmap_fp32(&q);

    // Ascending subset (the public contract requires sorted ids).
    let doc_ids = [0u32, 2, 3, 5];
    let mut out = vec![0u32; doc_ids.len()];
    idx.body_overlap_scores_subset(&qb, &doc_ids, &mut out);

    for (i, &di) in doc_ids.iter().enumerate() {
        let doc = &corpus[di as usize * DIM..(di as usize + 1) * DIM];
        let expected = ref_overlap(&qb, &ref_doc_bitmap(doc, DIM, N_TOP));
        assert_eq!(
            out[i], expected,
            "subset overlap mismatch at doc {di}: kernel {} vs reference {expected}",
            out[i],
        );
    }
}

// --- P-F (guarded OOB in bilinear_score) -----------------------------

#[cfg(feature = "experimental")]
#[test]
#[should_panic(expected = "out of range")]
fn pf_bilinear_score_rejects_oob_doc_idx() {
    const DIM: usize = 128;
    let n_docs = 4usize;
    let mut mb = MultiBucketBitmap::new(DIM, 2);
    let corpus: Vec<f32> = (0..n_docs * DIM)
        .map(|i| ((i * 5) % 71) as f32 - 35.0)
        .collect();
    mb.add(&corpus);
    let q: Vec<f32> = (0..DIM).map(|i| ((i * 9) % 67) as f32 - 33.0).collect();
    let qb = mb.query_bitmaps_from_ranks(&q);
    let w = mb.outer_product_weights();
    // Valid doc indices are 0..4; 4 is one past the end.
    let _ = mb.bilinear_score(&qb, &w, n_docs);
}

// --- P-G (degenerate dim == 0 constructor) ---------------------------

#[cfg(feature = "experimental")]
#[test]
#[should_panic(expected = "dim must be > 0")]
fn pg_multi_bucket_new_rejects_dim_zero() {
    // dim=0 satisfies `dim % 64 == 0` and `dim % n_buckets == 0`, so it
    // currently constructs and defers a div-by-zero to `add`. Reject it
    // at construction, mirroring SignBitmap::new.
    let _ = MultiBucketBitmap::new(0, 2);
}

// --- P-H (k-clamp: no capacity overflow on huge k/m) -----------------

#[test]
fn ph_bitmap_search_clamps_huge_k() {
    // `search` allocates `nq * k` f32/i64 slots. With k = usize::MAX
    // this overflows `Vec` capacity and aborts. The clamp must bound
    // the result (and the allocation) to n_vectors.
    const DIM: usize = 128;
    let n_docs = 16usize;
    let mut idx = Bitmap::new(DIM, DIM / 4);
    let corpus: Vec<f32> = (0..n_docs * DIM)
        .map(|i| ((i * 3) % 53) as f32 - 26.0)
        .collect();
    idx.add(&corpus);
    let q: Vec<f32> = (0..DIM).map(|i| ((i * 11) % 59) as f32 - 29.0).collect();

    let res = idx.search(&q, usize::MAX);
    // Exactly one query; at most n_vectors real results.
    assert_eq!(res.nq, 1);
    let valid = res.indices_for_query(0).iter().filter(|&&i| i >= 0).count();
    assert!(
        valid <= n_docs,
        "search returned {valid} results, exceeds n_vectors {n_docs}",
    );
    assert!(
        res.indices_for_query(0).len() <= n_docs,
        "result row length {} exceeds n_vectors {n_docs} (k not clamped)",
        res.indices_for_query(0).len(),
    );
}

#[cfg(feature = "experimental")]
#[test]
fn ph_multi_bucket_top_m_bilinear_clamps_huge_m() {
    const DIM: usize = 128;
    let n_docs = 16usize;
    let mut mb = MultiBucketBitmap::new(DIM, 2);
    let corpus: Vec<f32> = (0..n_docs * DIM)
        .map(|i| ((i * 7) % 61) as f32 - 30.0)
        .collect();
    mb.add(&corpus);
    let q: Vec<f32> = (0..DIM).map(|i| ((i * 13) % 73) as f32 - 36.0).collect();
    let qb = mb.query_bitmaps_from_ranks(&q);
    let w = mb.outer_product_weights();
    let head = mb.top_m_bilinear(&qb, &w, usize::MAX);
    assert!(
        head.len() <= n_docs,
        "top_m_bilinear returned {} ids, exceeds n_vectors {n_docs}",
        head.len(),
    );
}

// --- R2 (deterministic (score, doc_id) tie-break in TopK) ------------

#[test]
fn r2_topk_breaks_ties_by_lower_doc_id() {
    // Discriminating construction (see probe scenario C). The corpus is
    // laid out so that the documents `Bitmap::search` scans first
    // (low ids) have a LOWER overlap than a block of exact-duplicate
    // documents at HIGHER ids that all tie at the maximum overlap.
    //
    //   ids 0,1,2 : distinct docs, lower overlap  → fill the buffer
    //   ids 3,4,5,6 : exact duplicates of the query → max, tied overlap
    //
    // With k=4 the four tied duplicates are the winners. Under the old
    // score-only TopK the buffer-eviction history strands id 3 at the
    // end of the kept array, so a score-only finalize emits
    // [4, 5, 6, 3]. The composite `(score desc, doc_id asc)` key emits
    // the deterministic [3, 4, 5, 6] — lower doc_id first — independent
    // of the eviction order (and therefore of SIMD-vs-scalar summation
    // order on genuine near-ties).
    const DIM: usize = 128;
    let n_top = DIM / 4;
    let mut idx = Bitmap::new(DIM, n_top);

    // The query / duplicate vector: a clean ramp so its top-bucket is
    // well defined and an exact copy reproduces the maximum overlap.
    let dup: Vec<f32> = (0..DIM).map(|i| i as f32).collect();

    // Three low-overlap distinct docs (reversed ramp + offsets) so their
    // top-bucket only partially intersects the query's.
    let mut corpus: Vec<f32> = Vec::new();
    for d in 0..3usize {
        for i in 0..DIM {
            corpus.push((DIM - 1 - i) as f32 + d as f32 * 0.001);
        }
    }
    // Four exact duplicates of the query at ids 3,4,5,6.
    for _ in 0..4 {
        corpus.extend_from_slice(&dup);
    }
    idx.add(&corpus);

    let res = idx.search(&dup, 4);
    let top = res.indices_for_query(0);
    assert_eq!(
        top,
        &[3i64, 4, 5, 6],
        "tied winners must be ordered by ascending doc_id (composite key)",
    );

    // Sanity: the four winners genuinely tie at the maximum overlap, so
    // the ordering above is decided purely by the doc_id tie-break, not
    // by any score difference.
    let scores = res.scores_for_query(0);
    let max = scores[0];
    for (slot, &s) in scores.iter().enumerate() {
        assert_eq!(s, max, "winner at slot {slot} must share the max overlap");
    }

    // Determinism across repeated calls.
    let res2 = idx.search(&dup, 4);
    assert_eq!(
        res2.indices_for_query(0),
        top,
        "tie-break must be deterministic"
    );
}
