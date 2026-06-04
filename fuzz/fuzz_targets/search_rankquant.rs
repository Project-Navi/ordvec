//! libFuzzer target for the RankQuant ingest + search hot paths:
//! `add` (rank_transform -> bucket -> pack), then both symmetric `search` and
//! asymmetric `search_asymmetric` (runtime-dispatched scalar / AVX2 / AVX-512
//! scan kernels -> TopK). The `load_*` targets cover deserialization; this
//! exercises the compute surface they feed, through the public API.
//!
//! The fuzzer varies valid `(dim, bits)`, document count, query count, tie-heavy
//! finite vector values, and `k` shapes including zero, one, `n`, `n + 1`, and a
//! huge value. Invalid dimensions, non-finite floats, and ragged vector lengths
//! are caller contract violations, so this target avoids them and treats any
//! panic as a compute-path bug. Assertions stay structural: shape, finite
//! scores, valid doc IDs, score-descending/doc-ID-ascending rows, and repeat
//! determinism in one process.
#![no_main]

use libfuzzer_sys::{
    arbitrary::{Arbitrary, Error, Unstructured},
    fuzz_target,
};
use ordvec::{RankQuant, SearchResults};

const DIMS_B1: &[usize] = &[8, 16, 32, 48, 64, 80, 128];
const DIMS_B2: &[usize] = &[8, 16, 20, 32, 36, 48, 64, 80, 128];
const DIMS_B4: &[usize] = &[16, 32, 48, 64, 80, 128];
const MAX_PAYLOAD_BYTES: usize = 4096;

#[derive(Debug)]
struct HotPathCase {
    bits: u8,
    dim: usize,
    n: usize,
    nq: usize,
    k: usize,
    value_mode: u8,
    payload: Vec<u8>,
}

impl<'a> Arbitrary<'a> for HotPathCase {
    fn arbitrary(raw: &mut Unstructured<'a>) -> Result<Self, Error> {
        let bits = *raw.choose(&[1u8, 2, 4])?;
        let dim = match bits {
            1 => *raw.choose(DIMS_B1)?,
            2 => *raw.choose(DIMS_B2)?,
            4 => *raw.choose(DIMS_B4)?,
            _ => unreachable!(),
        };
        let n: usize = raw.int_in_range(0..=32)?;
        let nq: usize = raw.int_in_range(0..=4)?;
        let k_seed: usize = raw.int_in_range(0..=64)?;
        let k_mode: u8 = raw.int_in_range(0..=5)?;
        let k = match k_mode {
            0 => 0,
            1 => 1,
            2 => n,
            3 => n.saturating_add(1),
            4 => usize::MAX,
            _ => k_seed,
        };
        let value_mode: u8 = raw.int_in_range(0..=3)?;
        let max_payload = ((n + nq) * dim).min(MAX_PAYLOAD_BYTES);
        let payload = raw.bytes(raw.len().min(max_payload))?.to_vec();

        Ok(Self {
            bits,
            dim,
            n,
            nq,
            k,
            value_mode,
            payload,
        })
    }
}

fn finite_values(case: &HotPathCase, total: usize) -> Vec<f32> {
    (0..total)
        .map(|i| {
            let byte = case.payload.get(i % case.payload.len().max(1)).copied().unwrap_or(0);
            match case.value_mode {
                0 => 0.0,
                1 => (byte % 5) as f32 - 2.0,
                2 => byte as f32 - 128.0,
                _ => (byte as f32 - 128.0) / 16.0,
            }
        })
        .collect()
}

fn assert_results(label: &str, res: &SearchResults, nq: usize, k_eff: usize, n: usize) {
    assert_eq!(res.nq, nq, "{label}: wrong query count");
    assert_eq!(res.k, k_eff, "{label}: wrong effective k");
    assert_eq!(res.scores.len(), nq * k_eff, "{label}: wrong score length");
    assert_eq!(res.indices.len(), nq * k_eff, "{label}: wrong id length");

    for qi in 0..nq {
        let scores = res.scores_for_query(qi);
        let ids = res.indices_for_query(qi);
        for slot in 0..k_eff {
            let score = scores[slot];
            let id = ids[slot];
            assert!(score.is_finite(), "{label}: non-finite score at query {qi} slot {slot}");
            assert!(id >= 0, "{label}: negative doc id at query {qi} slot {slot}");
            assert!(
                (id as usize) < n,
                "{label}: doc id {id} out of range for n={n} at query {qi} slot {slot}",
            );
        }
        assert_score_then_id_order(label, qi, scores, ids);
    }
}

fn assert_score_then_id_order(label: &str, qi: usize, scores: &[f32], ids: &[i64]) {
    for slot in 1..scores.len() {
        let prev = (scores[slot - 1], ids[slot - 1]);
        let cur = (scores[slot], ids[slot]);
        assert!(
            cur.0 < prev.0 || (cur.0 == prev.0 && cur.1 >= prev.1),
            "{label}: row {qi} violates score-desc/doc-id-asc order at slots {} and {slot}",
            slot - 1,
        );
    }
}

fuzz_target!(|case: HotPathCase| {
    let total = (case.n + case.nq) * case.dim;
    let floats = finite_values(&case, total);
    let (docs, queries) = floats.split_at(case.n * case.dim);

    let mut idx = RankQuant::new(case.dim, case.bits);
    idx.add(docs);

    assert_eq!(idx.len(), case.n);
    assert_eq!(idx.dim(), case.dim);
    assert_eq!(idx.bits(), case.bits);
    assert_eq!(idx.byte_size(), case.n * idx.bytes_per_vec());

    let k_eff = case.k.min(case.n);

    let symmetric = idx.search(queries, case.k);
    assert_results("search", &symmetric, case.nq, k_eff, case.n);
    let symmetric_again = idx.search(queries, case.k);
    assert_eq!(
        symmetric.indices, symmetric_again.indices,
        "search returned nondeterministic ids",
    );
    assert_eq!(
        symmetric.scores, symmetric_again.scores,
        "search returned nondeterministic scores",
    );

    let asymmetric = idx.search_asymmetric(queries, case.k);
    assert_results("search_asymmetric", &asymmetric, case.nq, k_eff, case.n);
    let asymmetric_again = idx.search_asymmetric(queries, case.k);
    assert_eq!(
        asymmetric.indices, asymmetric_again.indices,
        "search_asymmetric returned nondeterministic ids",
    );
    assert_eq!(
        asymmetric.scores, asymmetric_again.scores,
        "search_asymmetric returned nondeterministic scores",
    );
});
