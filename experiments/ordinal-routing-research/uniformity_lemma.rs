//! CONTEXT-BOXED, PRE-REGISTERED — proves the uniformity lemma underlying
//! ordvec's equal-width bucketing, and shows WHY golden-ratio boundaries break it.
//! Isolated by request: touches no harness, writes nothing. Verdicts fixed below.
//!
//! THE LEMMA (construction rationale, not a retrieval claim):
//!   The rank transform maps a vector with distinct coords to a permutation of
//!   {0..D-1} — an EXACTLY uniform coordinate distribution. Consequences:
//!     (1) equal-width bucketing (2^B | D) is MAX-ENTROPY: every bucket is
//!         equiprobable, so the B-bit code wastes no bits.
//!     (2) CONSTANT-COMPOSITION: every document puts exactly D/2^B coords in
//!         each bucket — identical histogram across all docs. This is what gives
//!         the bitmap overlap its closed-form HYPERGEOMETRIC null E[X]=n_top^2/D.
//!     (3) Any NON-uniform boundary scheme (golden-ratio / three-gap included)
//!         makes bucket cardinalities UNEQUAL -> lower entropy AND breaks
//!         constant-composition -> destroys the closed-form null.
//!   So golden ratio is not merely unhelpful on the rank domain: it would remove
//!   the property ordvec's headline theorem stands on. Equal-width is FORCED.
//!
//! MEASURED on real ranks (corpus rank codes), three boundary schemes for B bits:
//!   equal-width  : cuts at k*D/2^B            (uniform)
//!   golden       : cuts at sorted {frac(i*phi)}*D, i=1..2^B-1  (three-gap/low-disc)
//!   random       : cuts at 2^B-1 sorted uniform random points  (control)
//! Metrics: per-bucket composition std across docs (0 == constant-composition);
//!          code entropy in bits (ceiling = B); hypergeometric-null error
//!          |observed mean top-bucket overlap - n_top^2/D| / (n_top^2/D).
//!
//! PRE-REGISTERED VERDICT (fixed before running):
//!   LEMMA HOLDS if, for B in {1,2,4}:
//!     equal-width composition-std ≈ 0 (< 0.01 coords) AND entropy ≈ B (>= B-0.01)
//!     AND hypergeom-null error < 0.02;  AND golden/random are strictly worse on
//!     composition-std and entropy.  Else the lemma as stated is wrong — report it.
//!
//! Run: cargo run --release --example uniformity_lemma -- --corpus-npy /tmp/repo_corpus.npy

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
    let (hlen, hstart) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
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

/// rank[i] = ordinal position (0..dim) of coordinate i within the row.
fn rank_encode(rows: &[f32], n: usize, dim: usize) -> Vec<u16> {
    let mut out = vec![0u16; n * dim];
    out.par_chunks_mut(dim).enumerate().for_each(|(r, code)| {
        let row = &rows[r * dim..(r + 1) * dim];
        let mut idx: Vec<u16> = (0..dim as u16).collect();
        idx.sort_by(|&a, &b| row[a as usize].partial_cmp(&row[b as usize]).unwrap());
        for (rank, &i) in idx.iter().enumerate() {
            code[i as usize] = rank as u16;
        }
    });
    out
}

/// Boundary cut points in rank units [0, dim], length 2^B - 1, ascending.
fn boundaries(scheme: &str, bits: u32, dim: usize) -> Vec<f64> {
    let nb = 1usize << bits;
    match scheme {
        "equal" => (1..nb).map(|k| k as f64 * dim as f64 / nb as f64).collect(),
        "golden" => {
            let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
            let mut cuts: Vec<f64> = (1..nb)
                .map(|i| ((i as f64 * phi).fract()) * dim as f64)
                .collect();
            cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            cuts
        }
        "random" => {
            let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0xBEEF);
            let mut cuts: Vec<f64> = (1..nb)
                .map(|_| rng.random_range(0.0..1.0) * dim as f64)
                .collect();
            cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            cuts
        }
        _ => unreachable!(),
    }
}

fn bucket_of(rank: u16, cuts: &[f64]) -> usize {
    let r = rank as f64 + 0.5;
    cuts.iter().filter(|&&c| r > c).count()
}

fn main() {
    let mut corpus_path = None;
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        if x == "--corpus-npy" {
            corpus_path = Some(a.next().unwrap());
        }
    }
    let path = corpus_path.expect("--corpus-npy required");
    let (corpus, n, dim) = load_npy_f32(&path);
    let ranks = rank_encode(&corpus, n, dim);
    let row = |i: usize| &ranks[i * dim..(i + 1) * dim];

    println!("# uniformity_lemma (CONTEXT-BOXED, pre-registered). corpus {n}x{dim}");
    println!("# per-bucket composition std across docs (0=constant-composition), code entropy (bits), hypergeom-null err");
    println!("scheme\tB\tcomp_std\tentropy\tnull_err");

    let mut rng_pairs = ChaCha8Rng::seed_from_u64(SEED);
    let pairs: Vec<(usize, usize)> = (0..50_000)
        .map(|_| {
            let i = rng_pairs.random_range(0..n);
            let mut j = rng_pairs.random_range(0..n);
            if j == i {
                j = (j + 1) % n;
            }
            (i, j)
        })
        .collect();

    for &bits in &[1u32, 2, 4] {
        let nb = 1usize << bits;
        for scheme in ["equal", "golden", "random"] {
            let cuts = boundaries(scheme, bits, dim);

            // composition: count per bucket per doc; std of counts across docs, per bucket, then mean.
            let comps: Vec<Vec<f64>> = (0..n)
                .into_par_iter()
                .map(|d| {
                    let mut c = vec![0f64; nb];
                    for &rk in row(d) {
                        c[bucket_of(rk, &cuts)] += 1.0;
                    }
                    c
                })
                .collect();
            let mut comp_std_sum = 0.0;
            for b in 0..nb {
                let vals: Vec<f64> = comps.iter().map(|c| c[b]).collect();
                let m = vals.iter().sum::<f64>() / n as f64;
                let v = vals.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n as f64;
                comp_std_sum += v.sqrt();
            }
            let comp_std = comp_std_sum / nb as f64;

            // code entropy: bucket occupancy probabilities (same for every doc under
            // constant-composition); use the mean histogram normalized.
            let mut hist = vec![0f64; nb];
            for c in &comps {
                for b in 0..nb {
                    hist[b] += c[b];
                }
            }
            let tot: f64 = hist.iter().sum();
            let entropy: f64 = hist
                .iter()
                .filter(|&&h| h > 0.0)
                .map(|&h| {
                    let p = h / tot;
                    -p * p.log2()
                })
                .sum();

            // hypergeometric null: top bucket = highest bucket id; overlap of top sets.
            let n_top_counts: Vec<f64> = comps.iter().map(|c| c[nb - 1]).collect();
            let mean_ntop = n_top_counts.iter().sum::<f64>() / n as f64;
            let top_sets: Vec<std::collections::HashSet<u16>> = (0..n)
                .into_par_iter()
                .map(|d| {
                    row(d)
                        .iter()
                        .enumerate()
                        .filter(|(_, &rk)| bucket_of(rk, &cuts) == nb - 1)
                        .map(|(i, _)| i as u16)
                        .collect()
                })
                .collect();
            let obs_overlap: f64 = pairs
                .par_iter()
                .map(|&(i, j)| top_sets[i].intersection(&top_sets[j]).count() as f64)
                .sum::<f64>()
                / pairs.len() as f64;
            let expected = mean_ntop * mean_ntop / dim as f64;
            let null_err = if expected > 0.0 {
                (obs_overlap - expected).abs() / expected
            } else {
                f64::NAN
            };

            println!("{scheme}\t{bits}\t{comp_std:.4}\t{entropy:.4}\t{null_err:.4}");
        }
    }
    println!("\n# LEMMA HOLDS iff equal-width: comp_std<0.01, entropy>=B-0.01, null_err<0.02; golden/random strictly worse on comp_std & entropy.");
}
