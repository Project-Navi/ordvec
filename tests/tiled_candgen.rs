//! Contract-pinning tests for sign candidate generation, written ahead of the
//! tiled internals swap of `top_m_candidates` /
//! `top_m_candidates_batched_serial_csr`. The oracle is independent of the
//! implementation under test: `score_all` (dense agreement counts) plus a
//! full lexicographic sort by `(hamming asc, doc_id asc)`. These tests pin
//! today's behavior exactly — including tie handling at the m-th position —
//! and must pass bit-identically before and after the swap.

use ordvec::SignBitmap;

/// Deterministic xorshift so corpora are reproducible without a rand dep.
struct XorShift(u64);

impl XorShift {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // Map to [-1, 1) with plenty of sign variety.
        ((self.0 >> 40) as f32 / 8_388_608.0) - 1.0
    }
}

fn random_corpus(dim: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut rng = XorShift(seed | 1);
    (0..n * dim).map(|_| rng.next_f32()).collect()
}

/// Tie-heavy corpus: every coordinate is +/-1 drawn from a tiny pattern set,
/// so hamming distances collide massively and the (hamming, doc_id)
/// tie-break does real work at the selection boundary.
fn tie_heavy_corpus(dim: usize, n: usize) -> Vec<f32> {
    (0..n)
        .flat_map(|doc| {
            let pattern = doc % 4;
            (0..dim).map(move |c| {
                if (c + pattern) % 3 == 0 {
                    -1.0
                } else {
                    1.0
                }
            })
        })
        .collect()
}

fn oracle_top_m(sign: &SignBitmap, q: &[f32], m: usize) -> Vec<u32> {
    let dim_u32 = u32::try_from(q.len()).unwrap();
    // score_all returns agreement (dim - hamming), higher is better.
    let agreements = sign.score_all(q);
    let mut ids: Vec<u32> = (0..agreements.len() as u32).collect();
    ids.sort_by_key(|&i| (dim_u32 - agreements[i as usize], i));
    ids.truncate(m.min(agreements.len()));
    ids
}

fn assert_contract(dim: usize, vectors: &[f32], queries: &[f32], m: usize, label: &str) {
    let mut sign = SignBitmap::new(dim);
    sign.add(vectors);
    let nq = queries.len() / dim;

    // Single-query path.
    for qi in 0..nq {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let got = sign.top_m_candidates(q, m);
        let want = oracle_top_m(&sign, q, m);
        assert_eq!(got, want, "{label}: single-query mismatch at query {qi}, m={m}");
    }

    // Batched serial CSR path: row qi must equal the single-query result.
    let cb = sign.top_m_candidates_batched_serial_csr(queries, m);
    assert_eq!(cb.offsets.len(), nq + 1, "{label}: CSR offsets length");
    for qi in 0..nq {
        let row = &cb.candidates[cb.offsets[qi]..cb.offsets[qi + 1]];
        let want = oracle_top_m(&sign, &queries[qi * dim..(qi + 1) * dim], m);
        assert_eq!(row, &want[..], "{label}: CSR row mismatch at query {qi}, m={m}");
    }
}

/// Random corpus large enough to span many doc blocks under any plausible
/// tile size, at a SIMD-friendly dim.
#[test]
fn random_corpus_matches_oracle_across_block_boundaries() {
    let dim = 128;
    let n = 10_240;
    let vectors = random_corpus(dim, n, 0xC0FFEE);
    let queries = random_corpus(dim, 33, 0xBEEF);
    for m in [1, 7, 256, 500] {
        assert_contract(dim, &vectors, &queries, m, "random");
    }
}

/// Massive hamming ties: selection at the boundary is decided purely by
/// doc_id ascending. This is the case a streaming collector most easily gets
/// subtly wrong.
#[test]
fn tie_heavy_corpus_selects_lowest_doc_ids_at_boundary() {
    let dim = 64;
    let n = 4_096;
    let vectors = tie_heavy_corpus(dim, n);
    let queries = random_corpus(dim, 9, 0xABCD);
    for m in [1, 3, 100, 1_000] {
        assert_contract(dim, &vectors, &queries, m, "tie-heavy");
    }
}

/// Exact duplicate documents: every duplicate group is one giant tie run,
/// longer than m, exercising equal-hamming runs that exceed the collector.
#[test]
fn duplicate_documents_tie_runs_longer_than_m() {
    let dim = 64;
    let base = random_corpus(dim, 8, 0x1234);
    // 8 distinct vectors, each repeated 512 times => tie runs of 512.
    let mut vectors = Vec::with_capacity(8 * 512 * dim);
    for rep in 0..512 {
        let _ = rep;
        vectors.extend_from_slice(&base);
    }
    let queries = random_corpus(dim, 5, 0x9999);
    for m in [10, 100, 513] {
        assert_contract(dim, &vectors, &queries, m, "duplicates");
    }
}

/// Edge geometry: m >= n, m == n, single doc, single query, nq == 0.
#[test]
fn edge_geometries_match_oracle() {
    let dim = 64;
    let vectors = random_corpus(dim, 17, 0x42);
    let queries = random_corpus(dim, 3, 0x43);
    for m in [17, 25, 1] {
        assert_contract(dim, &vectors, &queries, m, "edge");
    }

    let single_doc = random_corpus(dim, 1, 0x77);
    assert_contract(dim, &single_doc, &queries, 4, "single-doc");

    // Empty query batch: CSR must be a single zero offset and no candidates.
    let mut sign = SignBitmap::new(dim);
    sign.add(&vectors);
    let cb = sign.top_m_candidates_batched_serial_csr(&[], 8);
    assert_eq!(cb.offsets, vec![0]);
    assert!(cb.candidates.is_empty());
}

/// Large-dim smoke at the shape the arXiv corpus uses (1024 dims), enough
/// rows to cross several L2-sized doc blocks.
#[test]
fn dim_1024_shape_matches_oracle() {
    let dim = 1024;
    let n = 6_000;
    let vectors = random_corpus(dim, n, 0xA5A5);
    let queries = random_corpus(dim, 8, 0x5A5A);
    for m in [256, 320] {
        assert_contract(dim, &vectors, &queries, m, "dim1024");
    }
}
