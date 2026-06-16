//! CONTEXT-BOXED, PRE-REGISTERED — the round-on-round thread: does cone-removal
//! (centering) make a BALANCED, PRUNABLE coarse partition key for a sharding/IVF
//! layer? This is the one place cone-removal might buy SCALING (sublinear pruning)
//! rather than recall. Isolated by request: touches no harness, writes nothing.
//!
//! WHY: overlap_decomp showed raw low-B top-buckets are ~100% cone (every doc
//! shares the same hub coords) -> a raw coarse key is maximally UNBALANCED: docs
//! pile into a few cells, so an IVF layer probing few cells can't prune without
//! losing neighbours. Centering removes the cone -> should SPREAD docs across
//! cells. The decisive question is NOT balance for its own sake but candidates-
//! scanned at matched recall under a real inverted-index probe.
//!
//! PARTITION KEY = the coarse cell id of a doc. We build cells by k-means-free,
//! training-free coarse coding: project (centered or raw) onto the top PCA axes,
//! take the SIGN pattern of the top `bits_key` axes -> 2^bits_key cells (a
//! data-oblivious-given-PCA hash; PCA basis is the only data-dependent part and is
//! shared identically across arms, so arms differ ONLY in centering).
//! Query probes the P nearest cells (by Hamming on the key); candidate = union of
//! those cells' docs; recall@10 vs FP32 cosine top-10.
//!
//! ARMS (PCA basis identical across all; only the centering differs):
//!   raw      — sign-key from raw (uncentered) projections.
//!   centered — sign-key from per-coord-mean-centered projections.
//! Controls reported: cell-occupancy Gini (0=perfectly balanced, 1=all in one
//! cell) and the largest-cell fraction.
//!
//! PRE-REGISTERED VERDICT (fixed before running):
//!   BALANCE: centered must cut largest-cell-fraction by >= 1.5x vs raw.
//!   PRUNING (the one that matters): at matched recall@10 (>=0.90), centered must
//!     scan <= 0.80x the candidates raw scans, in >= 2 of 3 corpora.
//!   => PASS = centered is a materially better coarse key (a SCALING win, distinct
//!      from the recall FAIL at b>=2). FAIL = centering doesn't help partitioning
//!      either; the cone is not the bottleneck for routing.
//!
//! Run: cargo run --release --example partition_balance -- \
//!          --corpus-npy /tmp/corpora/fiqa_corpus.npy --queries-npy /tmp/corpora/fiqa_q.npy

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

fn coord_mean(rows: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut m = vec![0f32; dim];
    for i in 0..n {
        for c in 0..dim {
            m[c] += rows[i * dim + c];
        }
    }
    for m_c in m.iter_mut() {
        *m_c /= n as f32;
    }
    m
}

/// top-k PCA via power iteration + deflation (implicit covariance action). Pure Rust.
fn pca_topk(centered: &[f32], n: usize, dim: usize, k: usize) -> Vec<Vec<f32>> {
    let sample: Vec<usize> = if n > 20000 {
        let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0x9C0FFEE);
        (0..20000).map(|_| rng.random_range(0..n)).collect()
    } else {
        (0..n).collect()
    };
    let cov_mul = |x: &[f32]| -> Vec<f32> {
        let acc: Vec<f32> = sample
            .par_iter()
            .map(|&i| {
                let row = &centered[i * dim..(i + 1) * dim];
                let d: f32 = row.iter().zip(x).map(|(a, b)| a * b).sum();
                row.iter().map(|&r| r * d).collect::<Vec<f32>>()
            })
            .reduce(
                || vec![0f32; dim],
                |mut a, b| {
                    for j in 0..dim {
                        a[j] += b[j];
                    }
                    a
                },
            );
        let s = sample.len() as f32;
        acc.iter().map(|v| v / s).collect()
    };
    let mut comps: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0xABCD);
    for _ in 0..k {
        let mut v: Vec<f32> = (0..dim)
            .map(|_| {
                let u1: f32 = rng.random_range(1e-9..1.0);
                let u2: f32 = rng.random_range(0.0..1.0);
                (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
            })
            .collect();
        for _ in 0..50 {
            for c in &comps {
                let p: f32 = v.iter().zip(c).map(|(a, b)| a * b).sum();
                for j in 0..dim {
                    v[j] -= p * c[j];
                }
            }
            let mut y = cov_mul(&v);
            for c in &comps {
                let p: f32 = y.iter().zip(c).map(|(a, b)| a * b).sum();
                for j in 0..dim {
                    y[j] -= p * c[j];
                }
            }
            let nrm: f32 = y.iter().map(|t| t * t).sum::<f32>().sqrt();
            if nrm < 1e-20 {
                break;
            }
            for j in 0..dim {
                v[j] = y[j] / nrm;
            }
        }
        comps.push(v);
    }
    comps
}

/// sign-pattern key over the top `bits_key` PCA axes -> cell id in [0, 2^bits_key).
fn cell_key(v: &[f32], mean: Option<&[f32]>, basis: &[Vec<f32>], bits_key: usize) -> u32 {
    let mut key = 0u32;
    for (a, axis) in basis.iter().take(bits_key).enumerate() {
        let proj: f32 = axis
            .iter()
            .enumerate()
            .map(|(c, &w)| (v[c] - mean.map_or(0.0, |m| m[c])) * w)
            .sum();
        if proj > 0.0 {
            key |= 1 << a;
        }
    }
    key
}

fn gini(counts: &[usize]) -> f32 {
    let n = counts.len();
    if n == 0 {
        return 0.0;
    }
    let mut s: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tot: f64 = s.iter().sum();
    if tot == 0.0 {
        return 0.0;
    }
    let mut cum = 0.0;
    let mut lorenz = 0.0;
    for &x in &s {
        cum += x;
        lorenz += cum;
    }
    let g = (n as f64 + 1.0 - 2.0 * lorenz / tot) / n as f64;
    g as f32
}

fn analyze(
    label: &str,
    corpus: &[f32],
    queries: &[f32],
    n: usize,
    nq: usize,
    dim: usize,
    truth: &[Vec<u32>],
    mean: Option<&[f32]>,
    basis: &[Vec<f32>],
    bits_key: usize,
) {
    let n_cells = 1usize << bits_key;
    // assign cells
    let ckeys: Vec<u32> = (0..n)
        .into_par_iter()
        .map(|i| cell_key(&corpus[i * dim..(i + 1) * dim], mean, basis, bits_key))
        .collect();
    let qkeys: Vec<u32> = (0..nq)
        .into_par_iter()
        .map(|i| cell_key(&queries[i * dim..(i + 1) * dim], mean, basis, bits_key))
        .collect();
    let mut cells: Vec<Vec<u32>> = vec![Vec::new(); n_cells];
    for (i, &k) in ckeys.iter().enumerate() {
        cells[k as usize].push(i as u32);
    }
    let counts: Vec<usize> = cells.iter().map(|c| c.len()).collect();
    let g = gini(&counts);
    let largest = *counts.iter().max().unwrap() as f64 / n as f64;
    let nonempty = counts.iter().filter(|&&c| c > 0).count();

    println!("\n## {label} (bits_key={bits_key}, {n_cells} cells, {nonempty} non-empty)");
    println!("  occupancy Gini={g:.4} (0=balanced), largest-cell-fraction={largest:.4}");
    println!("  P_probe\tcand_scanned\trecall@10");
    // probe P nearest cells by Hamming on the key
    for p_probe in [1usize, 2, 4, 8, 16, 32] {
        if p_probe > n_cells {
            break;
        }
        let stats: Vec<(usize, f32)> = (0..nq)
            .into_par_iter()
            .map(|qi| {
                let qk = qkeys[qi];
                // rank cells by Hamming distance to qk, take P nearest
                let mut order: Vec<u32> = (0..n_cells as u32).collect();
                order.sort_by_key(|&c| (c ^ qk).count_ones());
                let mut cand = 0usize;
                use std::collections::HashSet;
                let mut hit_set: HashSet<u32> = HashSet::new();
                for &c in order.iter().take(p_probe) {
                    cand += cells[c as usize].len();
                    for &d in &cells[c as usize] {
                        hit_set.insert(d);
                    }
                }
                let hits = truth[qi].iter().filter(|&&t| hit_set.contains(&t)).count();
                (cand, hits as f32 / truth[qi].len() as f32)
            })
            .collect();
        let mc = stats.iter().map(|s| s.0 as f64).sum::<f64>() / nq as f64;
        let mr = stats.iter().map(|s| s.1 as f64).sum::<f64>() / nq as f64;
        println!("  {p_probe}\t{mc:.0}\t{mr:.4}");
    }
}

fn main() {
    let mut cp = None;
    let mut qp = None;
    let mut bits_key = 10usize;
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        match x.as_str() {
            "--corpus-npy" => cp = Some(a.next().unwrap()),
            "--queries-npy" => qp = Some(a.next().unwrap()),
            "--bits-key" => bits_key = a.next().unwrap().parse().unwrap(),
            _ => {}
        }
    }
    let (mut corpus, n, dim) = load_npy_f32(&cp.expect("--corpus-npy"));
    let (mut queries, nq, dq) = load_npy_f32(&qp.expect("--queries-npy"));
    assert_eq!(dim, dq);
    l2_normalize_rows(&mut corpus, dim);
    l2_normalize_rows(&mut queries, dim);
    let truth: Vec<Vec<u32>> = (0..nq)
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
            s.into_iter().take(10).map(|(i, _)| i).collect()
        })
        .collect();

    let mean = coord_mean(&corpus, n, dim);
    // PCA basis from the CENTERED corpus, shared identically across arms.
    let mut cc = corpus.clone();
    for i in 0..n {
        for c in 0..dim {
            cc[i * dim + c] -= mean[c];
        }
    }
    eprintln!("# computing top-{bits_key} PCA basis...");
    let basis = pca_topk(&cc, n, dim, bits_key);

    println!("# partition_balance (CONTEXT-BOXED, pre-registered). corpus {n}x{dim}, queries {nq}");
    println!("# coarse cell key = sign pattern of top-{bits_key} PCA axes; PCA basis shared across arms.");
    analyze(
        "RAW key (uncentered projections)",
        &corpus,
        &queries,
        n,
        nq,
        dim,
        &truth,
        None,
        &basis,
        bits_key,
    );
    analyze(
        "CENTERED key (mean-subtracted projections)",
        &corpus,
        &queries,
        n,
        nq,
        dim,
        &truth,
        Some(&mean),
        &basis,
        bits_key,
    );
    println!("\n# PRE-REGISTERED: BALANCE = centered cuts largest-cell-fraction >=1.5x; PRUNING = centered scans <=0.80x");
    println!("#   candidates at matched recall>=0.90 in >=2/3 corpora => PASS (a scaling win). Else FAIL.");
}
