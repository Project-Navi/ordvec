//! CONTEXT-BOXED, PRE-REGISTERED — decomposes the bitmap top-bucket overlap
//! EXCESS on real embeddings into CONE (marginal/hubness) vs PAIRWISE (signal),
//! and tests whether per-coordinate mean-centering recovers calibration.
//! Isolated by request: touches no harness, writes nothing. Verdicts fixed below.
//!
//! WHY: uniformity_lemma.rs found the hypergeometric null_err GROWS with B on
//! real data (0.31 -> 0.85 -> 3.34). Hypothesis: that excess is the shared cone
//! (R̄=0.69 anisotropy => same coords top-bucket in EVERY doc), not genuine
//! pairwise similarity. If so, the bitmap prefilter admits candidates partly on
//! HUBNESS, not meaning.
//!
//! FOUR overlap levels for the top bucket (n_top = D/2^B coords):
//!   uniform_null  = n_top^2 / D                  (random-subset assumption)
//!   cone_baseline = Σ_c f_c^2                     (f_c = P(coord c in top bucket);
//!                                                  overlap of two INDEPENDENT docs
//!                                                  given marginals == pure hubness)
//!   obs_random    = mean top-overlap, random doc pairs
//!   obs_trueNbr   = mean top-overlap, (query, FP32-cosine-true-neighbour) pairs
//!
//! Decomposition (all on the SAME corpus), reported raw AND after per-coord
//! mean-centering (subtract corpus per-coordinate mean before ranking):
//!   cone fraction   = (cone_baseline - uniform_null) / (obs_random - uniform_null)
//!   pairwise signal = obs_trueNbr - obs_random        (the discriminative gap)
//!
//! PRE-REGISTERED VERDICT (fixed before running):
//!   A. "cone dominates the null": cone fraction >= 0.70 (raw).
//!      => most of the bitmap's apparent overlap structure is hubness.
//!   B. "genuine pairwise signal survives": obs_trueNbr - obs_random >= 0.20 * obs_random
//!      AND obs_trueNbr > cone_baseline (true nbrs overlap beyond what hubness explains).
//!   C. "centering helps": centering reduces cone fraction by >= 0.10 absolute
//!      AND does not shrink the pairwise-signal ratio.
//!   Each reported PASS/FAIL independently; report actuals regardless.
//!
//! Run: cargo run --release --example overlap_decomp -- \
//!          --corpus-npy /tmp/repo_corpus.npy --queries-npy /tmp/repo_q.npy

// Research probe: `analyze` is a parameterized runner; bundling its args into a
// struct would obscure the experiment, not clarify it.
#![allow(clippy::too_many_arguments)]

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

const SEED: u64 = 1;

fn load_npy_f32(path: &str) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).expect("read npy");
    assert!(
        bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY",
        "not a numpy file"
    );
    let major = bytes[6];
    assert!(
        major == 1 || major == 2,
        "unsupported numpy .npy major version {major}"
    );
    if major == 2 {
        assert!(bytes.len() >= 12, "truncated numpy v2 header");
    }
    let (hlen, hstart) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    assert!(hstart + hlen <= bytes.len(), "truncated numpy header");
    let header = std::str::from_utf8(&bytes[hstart..hstart + hlen]).expect("utf8 header");
    assert!(header.contains("'descr': '<f4'"), "expected <f4 dtype");
    let after = &header[header.find("'shape':").expect("shape")..];
    let inner = &after[after.find('(').unwrap() + 1..after.find(')').unwrap()];
    let dims: Vec<usize> = inner
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let (n, dim) = (dims[0], dims[1]);
    let dstart = hstart + hlen;
    let mut out = vec![0.0f32; n * dim];
    for (i, ch) in bytes[dstart..].chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
    }
    (out, n, dim)
}

fn l2_normalize_rows(v: &mut [f32], dim: usize) {
    for i in 0..v.len() / dim {
        let row = &mut v[i * dim..(i + 1) * dim];
        let nrm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if nrm > 0.0 {
            for x in row.iter_mut() {
                *x /= nrm;
            }
        }
    }
}

/// Per-coordinate mean over corpus rows.
fn coord_mean(corpus: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut m = vec![0f32; dim];
    for i in 0..n {
        for c in 0..dim {
            m[c] += corpus[i * dim + c];
        }
    }
    for m_c in m.iter_mut() {
        *m_c /= n as f32;
    }
    m
}

/// Top-bucket membership bitset per doc: coords whose rank is in the top 1/2^B.
/// Returns, per doc, the sorted list of top-bucket coord ids.
fn top_sets(
    rows: &[f32],
    n: usize,
    dim: usize,
    bits: u32,
    sub: Option<&[f32]>,
) -> (Vec<Vec<u16>>, usize) {
    let n_top = dim >> bits; // D / 2^B
    let sets: Vec<Vec<u16>> = (0..n)
        .into_par_iter()
        .map(|r| {
            let mut v: Vec<(u16, f32)> = (0..dim)
                .map(|c| {
                    let x = rows[r * dim + c] - sub.map_or(0.0, |s| s[c]);
                    (c as u16, x)
                })
                .collect();
            // top n_top by value = highest ranks
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut top: Vec<u16> = v.into_iter().take(n_top).map(|(c, _)| c).collect();
            top.sort_unstable();
            top
        })
        .collect();
    (sets, n_top)
}

fn overlap(a: &[u16], b: &[u16]) -> usize {
    let (mut i, mut j, mut c) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                c += 1;
                i += 1;
                j += 1;
            }
        }
    }
    c
}

fn ground_truth(corpus: &[f32], queries: &[f32], dim: usize, k: usize) -> Vec<Vec<u32>> {
    let n = corpus.len() / dim;
    let nq = queries.len() / dim;
    (0..nq)
        .into_par_iter()
        .map(|qi| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let mut s: Vec<(u32, f32)> = (0..n)
                .map(|di| {
                    let d = &corpus[di * dim..(di + 1) * dim];
                    (di as u32, q.iter().zip(d).map(|(a, b)| a * b).sum::<f32>())
                })
                .collect();
            s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            s.into_iter().take(k).map(|(i, _)| i).collect()
        })
        .collect()
}

fn analyze(
    label: &str,
    corpus: &[f32],
    queries: &[f32],
    n: usize,
    nq: usize,
    dim: usize,
    truth: &[Vec<u32>],
    sub: Option<&[f32]>,
) {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let pairs: Vec<(usize, usize)> = (0..100_000)
        .map(|_| {
            let i = rng.random_range(0..n);
            let mut j = rng.random_range(0..n);
            if j == i {
                j = (j + 1) % n;
            }
            (i, j)
        })
        .collect();

    println!("\n## {label}");
    println!(
        "B\tn_top\tuniform_null\tcone_base\tobs_random\tobs_trueNbr\tcone_frac\tpair_sig_ratio"
    );
    for &bits in &[1u32, 2, 4] {
        let (csets, n_top) = top_sets(corpus, n, dim, bits, sub);
        let (qsets, _) = top_sets(queries, nq, dim, bits, sub);

        // per-coord top-bucket frequency f_c -> cone baseline Σ f_c^2
        let mut f = vec![0f64; dim];
        for s in &csets {
            for &c in s {
                f[c as usize] += 1.0;
            }
        }
        for x in f.iter_mut() {
            *x /= n as f64;
        }
        let cone_base: f64 = f.iter().map(|x| x * x).sum();

        let uniform_null = (n_top * n_top) as f64 / dim as f64;

        let obs_random: f64 = pairs
            .par_iter()
            .map(|&(i, j)| overlap(&csets[i], &csets[j]) as f64)
            .sum::<f64>()
            / pairs.len() as f64;

        // true-neighbour overlap: query top-set vs each true-nbr doc top-set
        let (mut tn_sum, mut tn_cnt) = (0f64, 0usize);
        for qi in 0..nq {
            for &di in &truth[qi] {
                tn_sum += overlap(&qsets[qi], &csets[di as usize]) as f64;
                tn_cnt += 1;
            }
        }
        let obs_true = tn_sum / tn_cnt as f64;

        let cone_frac = (cone_base - uniform_null) / (obs_random - uniform_null);
        let pair_sig_ratio = (obs_true - obs_random) / obs_random;

        println!("{bits}\t{n_top}\t{uniform_null:.2}\t{cone_base:.2}\t{obs_random:.2}\t{obs_true:.2}\t{cone_frac:.3}\t{pair_sig_ratio:+.3}");
    }
}

fn main() {
    let mut cp = None;
    let mut qp = None;
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        match x.as_str() {
            "--corpus-npy" => cp = Some(a.next().unwrap()),
            "--queries-npy" => qp = Some(a.next().unwrap()),
            _ => {}
        }
    }
    let (mut corpus, n, dim) = load_npy_f32(&cp.expect("--corpus-npy"));
    let (mut queries, nq, dq) = load_npy_f32(&qp.expect("--queries-npy"));
    assert_eq!(dim, dq);
    l2_normalize_rows(&mut corpus, dim);
    l2_normalize_rows(&mut queries, dim);
    // ground truth computed on the L2-normalized RAW vectors (the real task target);
    // centering changes the rank CODE, not the retrieval ground truth.
    let truth = ground_truth(&corpus, &queries, dim, 10);

    println!("# overlap_decomp (CONTEXT-BOXED, pre-registered). corpus {n}x{dim}, queries {nq}");
    println!("# cone_frac = (cone_base - uniform_null)/(obs_random - uniform_null); 1.0 => excess is ALL hubness");
    println!("# pair_sig_ratio = (obs_trueNbr - obs_random)/obs_random; >0 => true nbrs overlap MORE than random");

    analyze("RAW ranks", &corpus, &queries, n, nq, dim, &truth, None);

    let mean = coord_mean(&corpus, n, dim);
    analyze(
        "MEAN-CENTERED ranks (subtract per-coord corpus mean before ranking)",
        &corpus,
        &queries,
        n,
        nq,
        dim,
        &truth,
        Some(&mean),
    );

    println!("\n# PRE-REGISTERED: A cone-dominates if cone_frac>=0.70 (raw);");
    println!("#   B pairwise-signal-survives if pair_sig_ratio>=0.20 AND obs_trueNbr>cone_base;");
    println!(
        "#   C centering-helps if cone_frac drops >=0.10 absolute and pair_sig_ratio not reduced."
    );
}
