//! CONTEXT-BOXED, PRE-REGISTERED PROBE — isolated by request.
//!
//! This is a standalone curiosity gate. It does NOT touch the routing harness,
//! does NOT write to benchmarks/, and its result MUST NOT be used to reframe any
//! other experiment on this branch. It answers ONE question and stops.
//!
//! QUESTION (the user's hypothesis, sharpened):
//!   Is there a closed-form golden-angle ("sine spiral") ADDRESS on the data such
//!   that address-distance preserves cosine locality? If yes, retrieval could be
//!   COMPUTED (invert the address) instead of INDEXED (search). If no, the spiral
//!   address is inert — the same wall the ambient-probe experiment hit.
//!
//! METHOD (ceiling test, like the tau bake-off):
//!   Tile dim into d/3 triples. For a unit vector, each triple -> point on S^2;
//!   its ADDRESS on that triple is the nearest golden-spiral index (argmin over R
//!   spiral points — the EXACT nearest, an upper bound on what a closed-form
//!   inverse could achieve). Address(v) = the tuple of per-triple indices.
//!   Address-distance = sum of |Δindex| across triples.
//!   Primary metric: Spearman rho(address-distance, FP32 cosine-distance) over a
//!   random sample of corpus pairs.
//!
//! CONTROLS (golden must beat BOTH to mean anything):
//!   - random-pts : same machinery, spiral replaced by R random S^2 points
//!                  (isolates "does the GOLDEN structure help" vs "any quantizer").
//!   - sign       : ordvec's existing lever — sign bits, Hamming distance vs cosine
//!                  (isolates "does the spiral beat what ordvec already has").
//!
//! PRE-REGISTERED VERDICT (fixed BEFORE running — do not move):
//!   PASS         : golden rho >= 0.50  AND  golden beats both controls by >= 0.05.
//!                  => locality-preserving address; computed-retrieval worth pursuing.
//!   INCONCLUSIVE : 0.30 <= golden rho < 0.50, or golden within 0.05 of a control.
//!   FAIL         : golden rho < 0.30, or golden <= either control.
//!                  => spiral address is inert; stop, same wall as ambient probes.
//!
//! Run: cargo run --release --example fib_address_gate -- \
//!          --corpus-npy /tmp/repo_corpus.npy
//! Output is a single table; nothing is committed.

// Research probe: the header keeps aligned multi-line doc lists for readability.
#![allow(clippy::doc_overindented_list_items)]

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

fn gauss(rng: &mut ChaCha8Rng) -> f32 {
    let u1: f32 = rng.random_range(1e-9..1.0);
    let u2: f32 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Golden-angle spiral point i of R on S^2 (the "sine spiral" address grid).
fn spiral_point(i: usize, r: usize) -> [f32; 3] {
    let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    let z = if r == 1 {
        0.0
    } else {
        1.0 - 2.0 * (i as f32 + 0.5) / r as f32
    };
    let rad = (1.0 - z * z).max(0.0).sqrt();
    let theta = i as f32 * golden;
    [rad * theta.cos(), rad * theta.sin(), z]
}

/// R random points on S^2 (the structure-free control grid), seeded deterministically.
fn random_points(r: usize) -> Vec<[f32; 3]> {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0x5A5A_5A5A);
    (0..r)
        .map(|_| {
            let mut p = [gauss(&mut rng), gauss(&mut rng), gauss(&mut rng)];
            let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            for x in p.iter_mut() {
                *x /= n;
            }
            p
        })
        .collect()
}

/// Per-triple address: tuple of nearest-grid-point indices (argmin = exact nearest,
/// the CEILING for any closed-form inverse). grid is the R points on S^2.
fn address(v: &[f32], dim: usize, grid: &[[f32; 3]]) -> Vec<u16> {
    let n_tri = dim / 3;
    let mut addr = Vec::with_capacity(n_tri);
    for t in 0..n_tri {
        let s = &v[t * 3..t * 3 + 3];
        let nrm = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt().max(1e-12);
        let p = [s[0] / nrm, s[1] / nrm, s[2] / nrm];
        // nearest grid point by max dot (points are unit) == min distance
        let mut best = 0usize;
        let mut best_dot = f32::MIN;
        for (gi, g) in grid.iter().enumerate() {
            let d = p[0] * g[0] + p[1] * g[1] + p[2] * g[2];
            if d > best_dot {
                best_dot = d;
                best = gi;
            }
        }
        addr.push(best as u16);
    }
    addr
}

fn addr_dist(a: &[u16], b: &[u16]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as f32)
        .sum()
}

fn sign_bits(v: &[f32]) -> Vec<u64> {
    let mut bits = vec![0u64; v.len().div_ceil(64)];
    for (i, &x) in v.iter().enumerate() {
        if x >= 0.0 {
            bits[i / 64] |= 1u64 << (i % 64);
        }
    }
    bits
}

fn hamming(a: &[u64], b: &[u64]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x ^ y).count_ones())
        .sum::<u32>() as f32
}

/// Spearman rho = Pearson on ranks. Ties broken by stable index (adequate for a gate).
fn spearman(xs: &[f32], ys: &[f32]) -> f32 {
    let rank = |v: &[f32]| -> Vec<f32> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
        let mut r = vec![0.0f32; v.len()];
        for (rank, &i) in idx.iter().enumerate() {
            r[i] = rank as f32;
        }
        r
    };
    let rx = rank(xs);
    let ry = rank(ys);
    let n = rx.len() as f32;
    let mx = rx.iter().sum::<f32>() / n;
    let my = ry.iter().sum::<f32>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..rx.len() {
        let dx = rx[i] - mx;
        let dy = ry[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    cov / (vx.sqrt() * vy.sqrt()).max(1e-12)
}

fn main() {
    let mut corpus_path = None;
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        if x == "--corpus-npy" {
            corpus_path = Some(a.next().unwrap());
        }
    }
    let path = corpus_path.expect("--corpus-npy required (real embeddings)");
    let (mut corpus, n, dim) = load_npy_f32(&path);
    l2_normalize_rows(&mut corpus, dim);
    let n_pairs = 200_000usize;
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let pairs: Vec<(usize, usize)> = (0..n_pairs)
        .map(|_| {
            let i = rng.random_range(0..n);
            let mut j = rng.random_range(0..n);
            if j == i {
                j = (j + 1) % n;
            }
            (i, j)
        })
        .collect();

    let row = |i: usize| &corpus[i * dim..(i + 1) * dim];
    let cos_dist: Vec<f32> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let d: f32 = row(i).iter().zip(row(j)).map(|(a, b)| a * b).sum();
            1.0 - d
        })
        .collect();

    println!(
        "# fib_address_gate (CONTEXT-BOXED, pre-registered). corpus {n}x{dim}, pairs {n_pairs}"
    );
    println!("# Spearman rho vs FP32 cosine-distance (higher = better locality preservation)");
    println!("grid\tR\trho");

    // sign baseline (no R)
    let signs: Vec<Vec<u64>> = (0..n).into_par_iter().map(|i| sign_bits(row(i))).collect();
    let sign_d: Vec<f32> = pairs
        .par_iter()
        .map(|&(i, j)| hamming(&signs[i], &signs[j]))
        .collect();
    let rho_sign = spearman(&sign_d, &cos_dist);
    println!("sign\t-\t{rho_sign:.4}");

    let mut best_golden = f32::MIN;
    for &r in &[8usize, 32, 128] {
        for (name, grid) in [
            (
                "golden",
                (0..r).map(|i| spiral_point(i, r)).collect::<Vec<_>>(),
            ),
            ("random-pts", random_points(r)),
        ] {
            let addrs: Vec<Vec<u16>> = (0..n)
                .into_par_iter()
                .map(|i| address(row(i), dim, &grid))
                .collect();
            let ad: Vec<f32> = pairs
                .par_iter()
                .map(|&(i, j)| addr_dist(&addrs[i], &addrs[j]))
                .collect();
            let rho = spearman(&ad, &cos_dist);
            if name == "golden" && rho > best_golden {
                best_golden = rho;
            }
            println!("{name}\t{r}\t{rho:.4}");
        }
    }
    println!("\n# best golden rho = {best_golden:.4}");
    println!("# PRE-REGISTERED: PASS if >=0.50 and beats controls by >=0.05; FAIL if <0.30 or <= a control.");
}
