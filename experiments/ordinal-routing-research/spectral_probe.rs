//! Spectral / number-variance probe for the "prime mile-marker" conjecture.
//!
//! Decisive cheap experiment #1: is a 1-D routing key over an embedding
//! corpus *Poisson* (number variance grows like the window length L) or
//! *rigid* (variance grows like log L, as GUE-spectra do)?
//!
//! If Poisson, there is nothing for spectral / prime-gap structure to grip
//! and plain quantile bucketing is the whole story. Rigidity (sub-linear
//! number variance) would be the only empirical opening for the conjecture.
//!
//! Key subtlety handled below: number variance is only meaningful after
//! *unfolding* by the SMOOTH density, not the empirical CDF. Unfolding by
//! rank trivially yields a perfect lattice (Sigma^2 = 0) and measures
//! nothing. A random projection r . x of an L2-normalized vector is
//! approximately N(0, 1/dim) by the CLT, so we unfold with the matching
//! Gaussian CDF Phi and measure the residual fluctuation.
//!
//! Run (synthetic, self-contained):
//!     cargo run --release --example spectral_probe
//! Run (real corpus):
//!     cargo run --release --example spectral_probe -- --corpus-npy emb.npy
//!
//! No external data, no BLAS. Reuses the same construction as bench_rank.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 1;

struct Cfg {
    dim: usize,
    n: usize,
    latent_dim: usize,
    n_clusters: usize,
    n_keys: usize, // number of independent random-projection keys to average over
    corpus_npy: Option<String>,
    isotropic: bool, // control: N(0,I) corpus, no clusters (true Poisson expected)
    unfold_empirical: bool, // exact-rank unfold (lattice; demo only, not a rigidity test)
    unfold_smooth_knots: usize, // K>0 => smooth empirical unfold with K knots (correct method)
}

fn parse() -> Cfg {
    let mut c = Cfg {
        dim: 256,
        n: 200_000,
        latent_dim: 64,
        n_clusters: 200,
        n_keys: 8,
        corpus_npy: None,
        isotropic: false,
        unfold_empirical: false,
        unfold_smooth_knots: 0,
    };
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        match x.as_str() {
            "--dim" => c.dim = a.next().unwrap().parse().unwrap(),
            "--n" => c.n = a.next().unwrap().parse().unwrap(),
            "--latent" => c.latent_dim = a.next().unwrap().parse().unwrap(),
            "--clusters" => c.n_clusters = a.next().unwrap().parse().unwrap(),
            "--keys" => c.n_keys = a.next().unwrap().parse().unwrap(),
            "--corpus-npy" => c.corpus_npy = Some(a.next().unwrap()),
            "--isotropic" => c.isotropic = true,
            "--unfold-empirical" => c.unfold_empirical = true,
            "--unfold-smooth" => c.unfold_smooth_knots = a.next().unwrap().parse().unwrap(),
            "--rigid-selftest" => {} // handled in main, bypasses corpus
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

/// Phi(z) = 0.5 (1 + erf(z / sqrt 2)). Standard normal CDF.
fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

fn main() {
    let cfg = parse();
    // Instrument self-test: feed a perfectly evenly-spaced (unfolded) key
    // directly, bypassing corpus + projection. This is a maximally rigid 1-D
    // sequence; the estimator MUST report Sigma^2/L << 1 (falling). Proves the
    // probe can SEE rigidity when it is present, so a Poisson reading elsewhere
    // is a real finding, not instrument blindness.
    if std::env::args().any(|a| a == "--rigid-selftest") {
        let n = 50_000usize;
        let key: Vec<f64> = (0..n).map(|i| i as f64).collect(); // perfect lattice
        println!("# RIGID SELF-TEST: perfectly even key, expect Sigma^2/L << 1");
        println!("L\tSigma2_mean\tSigma2/L");
        report(&[key], n);
        return;
    }
    let (keys_per_proj, n) = build_keys(&cfg);
    println!("# spectral_probe: n={n} keys={}", keys_per_proj.len());
    println!("# unfolded by Gaussian CDF; Poisson => Sigma^2(L) ~ L, rigid => ~log L");
    println!("L\tSigma2_mean\tSigma2/L\tslope_hint");
    report(&keys_per_proj, n);
}

/// Minimal NumPy v1/v2 .npy reader for 2-D little-endian f32 C-order arrays
/// (same format contract as bench_rank's --corpus-npy).
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

/// Project a real corpus into the same per-projection unfolded keys the
/// synthetic path produces. L2-normalizes rows first (cosine geometry).
fn build_keys_from_corpus(cfg: &Cfg, corpus: &[f32], n: usize, d: usize) -> (Vec<Vec<f64>>, usize) {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    // unit-normalize rows in a local copy
    let mut c = corpus.to_vec();
    for i in 0..n {
        let row = &mut c[i * d..(i + 1) * d];
        let nrm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if nrm > 0.0 {
            for x in row.iter_mut() {
                *x /= nrm;
            }
        }
    }
    let mut out = Vec::with_capacity(cfg.n_keys);
    for _ in 0..cfg.n_keys {
        let mut r = vec![0.0f32; d];
        for v in r.iter_mut() {
            *v = gauss(&mut rng);
        }
        let rn: f32 = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in r.iter_mut() {
            *v /= rn;
        }
        let mut keys: Vec<f64> = (0..n)
            .map(|i| {
                let doc = &c[i * d..(i + 1) * d];
                r.iter().zip(doc).map(|(a, b)| (a * b) as f64).sum()
            })
            .collect();
        if cfg.unfold_empirical {
            keys.sort_by(|x, y| x.partial_cmp(y).unwrap());
            for (i, kv) in keys.iter_mut().enumerate() {
                *kv = i as f64;
            }
        } else {
            let sigma = (1.0 / d as f64).sqrt();
            for kv in keys.iter_mut() {
                *kv = normal_cdf(*kv / sigma) * n as f64;
            }
            keys.sort_by(|x, y| x.partial_cmp(y).unwrap());
        }
        out.push(keys);
    }
    (out, n)
}

/// Returns (one unfolded key-position list per projection, n).
fn build_keys(cfg: &Cfg) -> (Vec<Vec<f64>>, usize) {
    if let Some(ref path) = cfg.corpus_npy {
        let (corpus, n, d) = load_npy_f32(path);
        eprintln!("# loaded {n} x {d} from {path}");
        return build_keys_from_corpus(cfg, &corpus, n, d);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let d = cfg.dim;
    let l = cfg.latent_dim;
    // low-rank clustered corpus, same construction as bench_rank
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
        let nrm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        if nrm > 0.0 {
            for v in out.iter_mut() {
                *v /= nrm;
            }
        }
        out
    };
    let mut corpus = vec![0.0f32; cfg.n * d];
    for i in 0..cfg.n {
        let e = if cfg.isotropic {
            // control: draw directly in R^d as N(0,I), normalize. No cluster
            // structure, no low-rank latent -> the projection key should be a
            // clean Poisson process after Gaussian unfold.
            let mut v = vec![0.0f32; d];
            for x in v.iter_mut() {
                *x = gauss(&mut rng);
            }
            let nrm: f32 = v.iter().map(|t| t * t).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= nrm;
            }
            v
        } else {
            let c = rng.random_range(0..cfg.n_clusters);
            make(&protos[c * l..(c + 1) * l], 0.3, &mut rng)
        };
        corpus[i * d..(i + 1) * d].copy_from_slice(&e);
    }
    // random projection keys
    let mut out = Vec::with_capacity(cfg.n_keys);
    for _k in 0..cfg.n_keys {
        let mut r = vec![0.0f32; d];
        for v in r.iter_mut() {
            *v = gauss(&mut rng);
        }
        let rn: f32 = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in r.iter_mut() {
            *v /= rn;
        }
        // key value for each doc = r . x ; for L2-normalized x and unit r,
        // distributed ~ N(0, 1/dim) under the CLT for isotropic x.
        let mut keys: Vec<f64> = (0..cfg.n)
            .map(|i| {
                let doc = &corpus[i * d..(i + 1) * d];
                let dot: f32 = r.iter().zip(doc).map(|(a, b)| a * b).sum();
                dot as f64
            })
            .collect();
        keys.sort_by(|x, y| x.partial_cmp(y).unwrap());
        if cfg.unfold_smooth_knots > 0 {
            // SMOOTH EMPIRICAL unfold (correct method; removes the Gaussian-
            // mismatch confound). Fit the CDF with K coarse knots placed at
            // rank-quantiles, linear between. This subtracts the large-scale
            // marginal density of ANY distribution WITHOUT collapsing local
            // fluctuation to a lattice (which exact-rank unfolding does).
            // Number variance must then be read at window scales L << n/K so
            // the knot smoothing does not wash out the signal being measured.
            let k = cfg.unfold_smooth_knots.min(cfg.n);
            let n = cfg.n;
            // Precompute the k+1 knots once: (position p_j = rank j*n/k, value
            // keys[p_j]). Avoids recomputing knot_rank (div/mul/min) and
            // re-indexing keys inside the per-key binary search (review).
            let kr = |j: usize| -> usize { (j * n / k).min(n - 1) };
            let knot_pos: Vec<f64> = (0..=k).map(|j| kr(j) as f64).collect();
            let knot_val: Vec<f64> = (0..=k).map(|j| keys[kr(j)]).collect();
            let unfolded: Vec<f64> = keys
                .iter()
                .map(|&x| {
                    // bracketing knot index by value (knots sorted ascending)
                    let mut lo = 0usize;
                    let mut hi = k;
                    while lo < hi {
                        let mid = (lo + hi) / 2;
                        if knot_val[mid] < x {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    let j = lo.clamp(1, k);
                    let (v0, v1) = (knot_val[j - 1], knot_val[j]);
                    let (p0, p1) = (knot_pos[j - 1], knot_pos[j]);
                    if (v1 - v0).abs() < 1e-30 {
                        p0
                    } else {
                        p0 + (x - v0) / (v1 - v0) * (p1 - p0)
                    }
                })
                .collect();
            let mut u = unfolded;
            u.sort_by(|a, b| a.partial_cmp(b).unwrap());
            out.push(u);
        } else if cfg.unfold_empirical {
            // exact-rank unfold: assigns rank i. Trivially a lattice (Sigma^2=0)
            // for ANY input — this is NOT a rigidity test, only a demonstration
            // that exact quantile tiling balances occupancy. Kept for that use.
            for (i, kv) in keys.iter_mut().enumerate() {
                *kv = i as f64;
            }
            out.push(keys);
        } else {
            // fixed-Gaussian unfold: ONLY valid for the isotropic marginal.
            // CONFOUNDED for non-isotropic corpora (see ADVERSARIAL_REVIEW.md);
            // prefer --unfold-smooth.
            let sigma = (1.0 / d as f64).sqrt();
            for kv in keys.iter_mut() {
                *kv = normal_cdf(*kv / sigma) * cfg.n as f64;
            }
            out.push(keys);
        }
    }
    (out, cfg.n)
}

/// Number variance Sigma^2(L): variance of the count of points in a window
/// of length L, over many window placements, averaged across projections.
fn report(keys_per_proj: &[Vec<f64>], n: usize) {
    let ls = [2.0_f64, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0];
    let n_windows = 4000usize;
    for &lwin in ls.iter() {
        if lwin >= n as f64 {
            break;
        }
        let mut acc_var = 0.0f64;
        for positions in keys_per_proj {
            // place n_windows windows uniformly in [0, n-L], count via binary search
            let mut counts = Vec::with_capacity(n_windows);
            for w in 0..n_windows {
                let start = (w as f64) * ((n as f64 - lwin) / n_windows as f64);
                let end = start + lwin;
                let lo = positions.partition_point(|&p| p < start);
                let hi = positions.partition_point(|&p| p < end);
                counts.push((hi - lo) as f64);
            }
            let mean = counts.iter().sum::<f64>() / counts.len() as f64;
            let var = counts.iter().map(|c| (c - mean).powi(2)).sum::<f64>()
                / counts.len() as f64;
            acc_var += var;
        }
        let sigma2 = acc_var / keys_per_proj.len() as f64;
        let log_l = lwin.ln();
        println!(
            "{lwin:.0}\t{sigma2:.4}\t{:.4}\t(L={lwin:.0}, lnL={log_l:.3})",
            sigma2 / lwin
        );
    }
}
