//! TwoNN intrinsic-dimension estimator (Facco et al. 2017) for embedding corpora.
//!
//! For each point, μ = r2/r1 = ratio of 2nd to 1st nearest-neighbour distance.
//! Under locally-uniform density of intrinsic dimension d, μ ~ Pareto(d), so
//! the linear fit  log(1 - F(μ_i))  =  -d * log(μ_i)  through the origin
//! recovers d. No binning, no manifold reconstruction — two distances/point.
//!
//! Metric: embeddings are L2-normalized and retrieval is ANGULAR, so NN
//! distances here use chord / Euclidean distance between unit vectors
//! (`sqrt(2 - 2cos)`). Cosine distance (`1 - cos`) is squared-angle locally
//! and biases TwoNN estimates downward.
//!
//! Validation control: the synthetic corpus is a latent_dim-D manifold
//! projected into dim-D, so TwoNN should recover ID ≈ latent_dim. Run the
//! default and check it lands near 64 before trusting it on real .npy data.
//!
//! Run (synthetic control):  cargo run --release --example twonn_id
//! Run (real corpus):        cargo run --release --example twonn_id -- --corpus-npy emb.npy
//! No external data, no BLAS.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

const SEED: u64 = 1;

struct Cfg {
    dim: usize,
    n: usize,
    latent_dim: usize,
    n_clusters: usize,
    corpus_npy: Option<String>,
    isotropic: bool,
}

fn parse() -> Cfg {
    let mut c = Cfg {
        dim: 256,
        n: 20_000,
        latent_dim: 64,
        n_clusters: 200,
        corpus_npy: None,
        isotropic: false,
    };
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        match x.as_str() {
            "--dim" => c.dim = a.next().unwrap().parse().unwrap(),
            "--n" => c.n = a.next().unwrap().parse().unwrap(),
            "--latent" => c.latent_dim = a.next().unwrap().parse().unwrap(),
            "--clusters" => c.n_clusters = a.next().unwrap().parse().unwrap(),
            "--corpus-npy" => c.corpus_npy = Some(a.next().unwrap()),
            "--isotropic" => c.isotropic = true,
            other => panic!("unknown arg {other}"),
        }
    }
    c
}

fn gauss(rng: &mut ChaCha8Rng) -> f32 {
    let u1: f32 = rng.random_range(1e-9..1.0);
    let u2: f32 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Minimal NumPy v1/v2 .npy reader for 2-D little-endian f32 C-order arrays.
fn load_npy_f32(path: &str) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).expect("read npy");
    assert!(bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY", "not a numpy file");
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
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
    };
    assert!(hstart + hlen <= bytes.len(), "truncated numpy header");
    let header = std::str::from_utf8(&bytes[hstart..hstart + hlen]).expect("utf8 header");
    assert!(header.contains("'descr': '<f4'"), "expected <f4 dtype");
    assert!(header.contains("'fortran_order': False"), "expected C order");
    let after = &header[header.find("'shape':").expect("shape")..];
    let inner = &after[after.find('(').unwrap() + 1..after.find(')').unwrap()];
    let dims: Vec<usize> = inner.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    assert_eq!(dims.len(), 2, "expected 2-D array");
    let (n, dim) = (dims[0], dims[1]);
    let dstart = hstart + hlen;
    assert_eq!(bytes.len() - dstart, n * dim * 4, "data length mismatch");
    let mut out = vec![0.0f32; n * dim];
    for (i, ch) in bytes[dstart..].chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
    }
    (out, n, dim)
}

/// L2-normalize each row in place (cosine geometry).
fn l2_normalize_rows(v: &mut [f32], dim: usize) {
    let n = v.len() / dim;
    for i in 0..n {
        let row = &mut v[i * dim..(i + 1) * dim];
        let nrm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if nrm > 0.0 {
            for x in row.iter_mut() {
                *x /= nrm;
            }
        }
    }
}

/// Returns (corpus, n, dim). Synthetic = same low-rank clustered construction
/// as bench_rank (latent_dim manifold in dim-space → TwoNN should recover
/// ~latent_dim). Real = whatever the .npy holds, L2-normalized.
fn load_corpus(cfg: &Cfg) -> (Vec<f32>, usize, usize) {
    if let Some(ref path) = cfg.corpus_npy {
        let (mut v, n, dim) = load_npy_f32(path);
        l2_normalize_rows(&mut v, dim);
        return (v, n, dim);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let (d, l) = (cfg.dim, cfg.latent_dim);
    let mut a = vec![0.0f32; d * l];
    for x in a.iter_mut() {
        *x = gauss(&mut rng);
    }
    let mut protos = vec![0.0f32; cfg.n_clusters * l];
    for x in protos.iter_mut() {
        *x = gauss(&mut rng);
    }
    let make = |proto: &[f32], noise: f32, rng: &mut ChaCha8Rng| -> Vec<f32> {
        let mut z = vec![0.0f32; l];
        for j in 0..l {
            z[j] = proto[j] + noise * gauss(rng);
        }
        let mut out = vec![0.0f32; d];
        for i in 0..d {
            let mut acc = 0.0f32;
            for j in 0..l {
                acc += a[i * l + j] * z[j];
            }
            out[i] = acc;
        }
        out
    };
    let mut corpus = vec![0.0f32; cfg.n * d];
    for i in 0..cfg.n {
        let e = if cfg.isotropic {
            let mut v = vec![0.0f32; d];
            for x in v.iter_mut() {
                *x = gauss(&mut rng);
            }
            v
        } else {
            let c = rng.random_range(0..cfg.n_clusters);
            make(&protos[c * l..(c + 1) * l], 0.3, &mut rng)
        };
        corpus[i * d..(i + 1) * d].copy_from_slice(&e);
    }
    l2_normalize_rows(&mut corpus, d);
    (corpus, cfg.n, d)
}

/// For sampled anchors, compute μ = r2/r1 over COSINE distance (1 - cos),
/// searching each anchor against the full corpus. Anchors are sampled (the
/// Pareto fit is robust to it) but each search is exact, so r1/r2 are exact.
/// Returns the μ values (>1, with degenerate/duplicate cases dropped).
fn nn_ratios(corpus: &[f32], n: usize, dim: usize) -> Vec<f64> {
    // sample up to 3000 anchors deterministically (stride sampling, seed-free)
    let n_anchors = n.min(3000);
    let stride = (n / n_anchors).max(1);
    let anchors: Vec<usize> = (0..n).step_by(stride).take(n_anchors).collect();
    anchors
        .par_iter()
        .filter_map(|&ai| {
            let a = &corpus[ai * dim..(ai + 1) * dim];
            // Track the two smallest chord distances to OTHER points.
            let (mut d1, mut d2) = (f64::INFINITY, f64::INFINITY);
            for bi in 0..n {
                if bi == ai {
                    continue;
                }
                let b = &corpus[bi * dim..(bi + 1) * dim];
                let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                // Euclidean / chord distance between unit vectors:
                // sqrt(2 - 2cos) ∝ sin(θ/2), locally LINEAR in the angle.
                // (Cosine distance 1-cos ≈ θ²/2 is a *squared* distance and
                // halves the TwoNN estimate — TwoNN needs a locally-linear metric.)
                let dist = (2.0 - 2.0 * dot as f64).max(0.0).sqrt();
                if dist < d1 {
                    d2 = d1;
                    d1 = dist;
                } else if dist < d2 {
                    d2 = dist;
                }
            }
            // drop degenerate anchors (exact duplicate neighbour => r1≈0)
            if d1 > 1e-12 && d2.is_finite() {
                let mu = d2 / d1;
                if mu > 1.0 + 1e-9 {
                    return Some(mu);
                }
            }
            None
        })
        .collect()
}

/// TwoNN Pareto fit. Sort μ ascending, empirical CDF F_i = (i+1)/N, then fit
/// d through the origin on  y = -log(1 - F)  vs  x = log μ.  Robust variant:
/// discard the top 10% of μ (Facco et al.'s recommended tail trim) and use the
/// closed-form slope d = Σ x·y / Σ x².
fn twonn_fit(mus: &[f64]) -> f64 {
    let mut m: Vec<f64> = mus.to_vec();
    m.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = m.len();
    let keep = ((n as f64) * 0.9) as usize; // trim top 10% tail
    let (mut sxx, mut sxy) = (0.0f64, 0.0f64);
    for (i, &mu) in m.iter().enumerate().take(keep) {
        let f = (i + 1) as f64 / n as f64; // empirical CDF
        let x = mu.ln();
        let y = -(1.0 - f).ln();
        sxx += x * x;
        sxy += x * y;
    }
    sxy / sxx
}

fn main() {
    let cfg = parse();
    let (corpus, n, dim) = load_corpus(&cfg);
    eprintln!("# twonn_id: n={n} dim={dim} source={}",
        if cfg.corpus_npy.is_some() { "npy" } else if cfg.isotropic { "isotropic" } else { "clustered" });
    let mus = nn_ratios(&corpus, n, dim);
    let d = twonn_fit(&mus);
    println!("TwoNN intrinsic dimension estimate: {d:.2}  (ambient dim {dim})");
    if cfg.corpus_npy.is_none() && !cfg.isotropic {
        println!("(synthetic control: expect ~{} = latent_dim)", cfg.latent_dim);
    }
}
