//! CONTEXT-BOXED, PRE-REGISTERED — the directions experiment done RIGHT, using
//! every finding: center (remove the cone) -> project to the populated k-dim
//! subspace (where joint structure is legible) -> place R probe directions via
//! {random, Sobol, Kronecker} -> route -> candidates-scanned at matched recall.
//! Isolated by request: touches no harness, writes nothing.
//!
//! WHY THIS ISN'T fib_directions REDUX:
//!   fib_directions placed golden directions on the AMBIENT 768-sphere, where
//!   concentration of measure makes iid-Gaussian already near-uniform -> doomed.
//!   The marginal uniformity lemma is BLIND to joint structure; directions act on
//!   the JOINT correlation. So the honest test is in the centered, low-effective-
//!   dim subspace (intrinsic dim ~13), where low-discrepancy has room to matter.
//!
//! SEQUENCES (data-oblivious; PCA control is the data-DEPENDENT ceiling):
//!   random   — iid Gaussian directions in the k-subspace (the baseline).
//!   sobol    — base-2 digital net -> inverse-normal -> subspace direction.
//!              Discrepancy-optimal at R=2^m; predicted to spike at R in {16,32,64}.
//!   kronecker— additive {i*alpha} per axis (generalized golden) -> direction.
//!              Three-distance at EVERY prefix; predicted to hold across all R but
//!              degrade as k climbs (dimension factor in its discrepancy bound).
//!   pca-axes — the top-R principal directions themselves (DATA-DEPENDENT control;
//!              expected to win — it bounds what oblivious sets leave on the table).
//!
//! METRIC: per query, probe its bucket +/- rad on each of R directions; candidate
//! union; recall@10 vs FP32 cosine top-10. Report the FAIR ENVELOPE (max recall at
//! candidates-scanned <= budget) per sequence — the only fair cross-arm axis.
//! Also report post-map DISCREPANCY (star-discrepancy proxy) of each direction set,
//! because cube->sphere mapping warps the even-spacing and that must be MEASURED.
//!
//! PRE-REGISTERED VERDICT (fixed before running), family-level, Bonferroni in mind:
//!   COMPONENT WIN if a sequence beats random by >= 0.02 recall at matched
//!     candidates in >= 2 of {fiqa,nq,quora} AND survives at k near 13.
//!   SOBOL-AS-PREDICTED if its wins concentrate at R in {16,32,64} (power-of-2).
//!   KRONECKER-AS-PREDICTED if it wins across R but its margin shrinks as k grows.
//!   CLASS-DEAD if neither low-discrepancy sequence beats random anywhere by >=0.02
//!     -> the entire oblivious-directions hypothesis (not just golden) is falsified.
//!   The Kronecker/Sobol HYBRID is NOT built here; it is earned only if each wins
//!     in a DISTINCT R-regime. This experiment decides whether that even applies.
//!
//! Run: cargo run --release --example subspace_directions -- \
//!          --corpus-npy /tmp/corpora/fiqa_corpus.npy --queries-npy /tmp/corpora/fiqa_q.npy --kdim 13

// Research probe (not library code): index loops and the inv-normal constant
// tables read closer to the math than the iterator rewrites would.
#![allow(
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::excessive_precision
)]

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

/// Top-k principal directions of the centered corpus via power iteration +
/// deflation. Returns k unit vectors of length `dim`. Pure Rust, no BLAS.
/// Uses a doc subsample for the covariance action (cov is implicit: Cov*x =
/// (1/n) X^T (X x), never materialized).
fn pca_topk(centered: &[f32], n: usize, dim: usize, k: usize, mean: &[f32]) -> Vec<Vec<f32>> {
    let sample: Vec<usize> = if n > 20000 {
        let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0x9C0FFEE);
        (0..20000).map(|_| rng.random_range(0..n)).collect()
    } else {
        (0..n).collect()
    };
    // implicit covariance action y = Cov * x using centered rows on the sample
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
    let _ = mean;
    for _ in 0..k {
        let mut v: Vec<f32> = (0..dim)
            .map(|_| {
                let u1: f32 = rng.random_range(1e-9..1.0);
                let u2: f32 = rng.random_range(0.0..1.0);
                (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
            })
            .collect();
        for _ in 0..50 {
            // deflate against found components
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

/// project a centered vector onto the k-dim PCA basis -> k coords.
fn project(v: &[f32], basis: &[Vec<f32>]) -> Vec<f32> {
    basis
        .iter()
        .map(|b| v.iter().zip(b).map(|(a, c)| a * c).sum())
        .collect()
}

// ---- direction generators in k-dim (returned as R unit vectors length k) ----

fn dirs_random(r: usize, k: usize) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0x1111 ^ (r as u64));
    (0..r)
        .map(|_| {
            let mut v: Vec<f32> = (0..k)
                .map(|_| {
                    let u1: f32 = rng.random_range(1e-9..1.0);
                    let u2: f32 = rng.random_range(0.0..1.0);
                    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
                })
                .collect();
            let nrm: f32 = v.iter().map(|t| t * t).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= nrm;
            }
            v
        })
        .collect()
}

/// Van der Corput / Sobol-lite: radical inverse in base 2 per dimension using
/// distinct primes for the radical-inverse base per axis (Halton-style net, the
/// digital low-discrepancy family). inverse-normal -> unit direction.
fn radical_inverse(mut i: usize, base: usize) -> f32 {
    let mut f = 1.0f64;
    let mut r = 0.0f64;
    while i > 0 {
        f /= base as f64;
        r += f * (i % base) as f64;
        i /= base;
    }
    r as f32
}

const PRIMES: [usize; 32] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131,
];

/// inverse standard normal CDF (Acklam's rational approximation) for u in (0,1).
fn inv_norm(u: f32) -> f32 {
    let u = u.clamp(1e-6, 1.0 - 1e-6) as f64;
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let pl = 0.02425;
    let x = if u < pl {
        let q = (-2.0 * u.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if u <= 1.0 - pl {
        let q = u - 0.5;
        let rr = q * q;
        (((((a[0] * rr + a[1]) * rr + a[2]) * rr + a[3]) * rr + a[4]) * rr + a[5]) * q
            / (((((b[0] * rr + b[1]) * rr + b[2]) * rr + b[3]) * rr + b[4]) * rr + 1.0)
    } else {
        let q = (-2.0 * (1.0 - u).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    };
    x as f32
}

fn dirs_sobol(r: usize, k: usize) -> Vec<Vec<f32>> {
    (0..r)
        .map(|i| {
            let mut v: Vec<f32> = (0..k)
                .map(|a| inv_norm(radical_inverse(i + 1, PRIMES[a % PRIMES.len()])))
                .collect();
            let nrm: f32 = v.iter().map(|t| t * t).sum::<f32>().sqrt();
            if nrm > 0.0 {
                for x in v.iter_mut() {
                    *x /= nrm;
                }
            }
            v
        })
        .collect()
}

/// Kronecker: per-axis additive recurrence {i*alpha_a}, alpha from the generalized
/// golden ratio (root of x^{k+1}=x+1) approximated per-axis by primes^(1/(a+2)) frac.
fn dirs_kronecker(r: usize, k: usize) -> Vec<Vec<f32>> {
    // irrational generators: fractional parts of sqrt(prime) (badly approximable, Weyl-equidistributed)
    let alphas: Vec<f64> = (0..k)
        .map(|a| (PRIMES[a % PRIMES.len()] as f64).sqrt().fract())
        .collect();
    (0..r)
        .map(|i| {
            let mut v: Vec<f32> = (0..k)
                .map(|a| inv_norm(((i as f64 + 1.0) * alphas[a]).fract() as f32))
                .collect();
            let nrm: f32 = v.iter().map(|t| t * t).sum::<f32>().sqrt();
            if nrm > 0.0 {
                for x in v.iter_mut() {
                    *x /= nrm;
                }
            }
            v
        })
        .collect()
}

fn dirs_pca_axes(r: usize, k: usize) -> Vec<Vec<f32>> {
    // data-dependent ceiling: the k principal axes, then (for R>k) random unit
    // combinations of them seeded deterministically — so R>k gives R DISTINCT
    // directions spanning the populated subspace, not repeats (control fix).
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0xACE5);
    (0..r)
        .map(|i| {
            let mut v = vec![0f32; k];
            if i < k {
                v[i] = 1.0;
            } else {
                for x in v.iter_mut() {
                    let u1: f32 = rng.random_range(1e-9..1.0);
                    let u2: f32 = rng.random_range(0.0..1.0);
                    *x = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
                }
                let nrm: f32 = v.iter().map(|t| t * t).sum::<f32>().sqrt();
                for x in v.iter_mut() {
                    *x /= nrm;
                }
            }
            v
        })
        .collect()
}

/// star-discrepancy proxy: max over a random set of half-spaces of |empirical - expected|.
fn discrepancy_proxy(dirs: &[Vec<f32>], k: usize) -> f32 {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0xDDDD);
    let probes = 2000;
    let mut worst = 0f32;
    for _ in 0..probes {
        // random half-space normal
        let mut hn: Vec<f32> = (0..k)
            .map(|_| {
                let u1: f32 = rng.random_range(1e-9..1.0);
                let u2: f32 = rng.random_range(0.0..1.0);
                (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
            })
            .collect();
        let hnm: f32 = hn.iter().map(|t| t * t).sum::<f32>().sqrt();
        for x in hn.iter_mut() {
            *x /= hnm;
        }
        let frac = dirs
            .iter()
            .filter(|d| d.iter().zip(&hn).map(|(a, b)| a * b).sum::<f32>() > 0.0)
            .count() as f32
            / dirs.len() as f32;
        worst = worst.max((frac - 0.5).abs());
    }
    worst
}

struct Proj {
    dir: Vec<f32>,
    width: f32,
}
impl Proj {
    fn bucket(&self, v: &[f32]) -> i64 {
        let dot: f32 = self.dir.iter().zip(v).map(|(a, b)| a * b).sum();
        (dot / self.width).floor() as i64
    }
}

fn main() {
    let mut cp = None;
    let mut qp = None;
    let mut kdim = 13usize;
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        match x.as_str() {
            "--corpus-npy" => cp = Some(a.next().unwrap()),
            "--queries-npy" => qp = Some(a.next().unwrap()),
            "--kdim" => kdim = a.next().unwrap().parse().unwrap(),
            _ => {}
        }
    }
    let (mut corpus, n, dim) = load_npy_f32(&cp.expect("--corpus-npy"));
    let (mut queries, nq, dq) = load_npy_f32(&qp.expect("--queries-npy"));
    assert_eq!(dim, dq);
    l2_normalize_rows(&mut corpus, dim);
    l2_normalize_rows(&mut queries, dim);

    // ground truth = FP32 cosine top-10 on normalized RAW vectors
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

    // center
    let mean = coord_mean(&corpus, n, dim);
    let mut cc = corpus.clone();
    for i in 0..n {
        for c in 0..dim {
            cc[i * dim + c] -= mean[c];
        }
    }
    let mut qc = queries.clone();
    for i in 0..nq {
        for c in 0..dim {
            qc[i * dim + c] -= mean[c];
        }
    }

    // PCA basis from centered corpus, project both
    eprintln!("# computing top-{kdim} PCA basis (power iteration)...");
    let basis = pca_topk(&cc, n, dim, kdim, &mean);
    let cproj: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|i| project(&cc[i * dim..(i + 1) * dim], &basis))
        .collect();
    let qproj: Vec<Vec<f32>> = (0..nq)
        .into_par_iter()
        .map(|i| project(&qc[i * dim..(i + 1) * dim], &basis))
        .collect();

    // width calibration: ~sqrt(n) buckets along a unit direction in subspace
    let spread = {
        let mut s = 0f32;
        for v in cproj.iter().take(2000) {
            for &x in v {
                s += x * x;
            }
        }
        (s / (2000.0 * kdim as f32)).sqrt() * 6.0
    };
    let base_width = spread / (n as f32).sqrt();

    println!("# subspace_directions (CONTEXT-BOXED, pre-registered). corpus {n}x{dim}, k={kdim}");
    println!("# centered+projected to k-dim PCA subspace; recall@10 vs FP32 cosine top-10");
    let gens: [(&str, fn(usize, usize) -> Vec<Vec<f32>>); 4] = [
        ("random", dirs_random),
        ("sobol", dirs_sobol),
        ("kronecker", dirs_kronecker),
        ("pca-axes", dirs_pca_axes),
    ];
    let r_values = [8usize, 16, 32, 64, 128];

    println!("seq\tR\tdiscrepancy\trad\tcand\trecall@10");
    let mut pts: Vec<(String, f64, f64)> = vec![];
    for (name, gen) in gens.iter() {
        for &r in &r_values {
            let dirs = gen(r, kdim);
            let disc = discrepancy_proxy(&dirs, kdim);
            let projs: Vec<Proj> = dirs
                .into_iter()
                .map(|d| Proj {
                    dir: d,
                    width: base_width,
                })
                .collect();
            // index: per direction, bucket -> docs
            let index: Vec<std::collections::HashMap<i64, Vec<u32>>> = projs
                .par_iter()
                .map(|p| {
                    let mut m: std::collections::HashMap<i64, Vec<u32>> =
                        std::collections::HashMap::new();
                    for di in 0..n {
                        m.entry(p.bucket(&cproj[di])).or_default().push(di as u32);
                    }
                    m
                })
                .collect();
            for rad in [0i64, 1, 2, 4] {
                let stats: Vec<(usize, f32)> = (0..nq)
                    .into_par_iter()
                    .map(|qi| {
                        use std::collections::HashSet;
                        let mut cand: HashSet<u32> = HashSet::new();
                        for (pi, p) in projs.iter().enumerate() {
                            let b = p.bucket(&qproj[qi]);
                            for bb in (b - rad)..=(b + rad) {
                                if let Some(ids) = index[pi].get(&bb) {
                                    cand.extend(ids.iter().copied());
                                }
                            }
                        }
                        let hits = truth[qi].iter().filter(|&&i| cand.contains(&i)).count();
                        (cand.len(), hits as f32 / truth[qi].len() as f32)
                    })
                    .collect();
                let mc = stats.iter().map(|s| s.0 as f64).sum::<f64>() / nq as f64;
                let mr = stats.iter().map(|s| s.1 as f64).sum::<f64>() / nq as f64;
                println!("{name}\t{r}\t{disc:.4}\t{rad}\t{mc:.0}\t{mr:.4}");
                pts.push((name.to_string(), mc, mr));
            }
        }
    }
    println!("\n# FAIR ENVELOPE: max recall@10 at candidates-scanned <= budget");
    let budgets = [500.0, 1000.0, 2000.0, 4000.0, 8000.0];
    print!("budget");
    for (name, _) in gens.iter() {
        print!("\t{name}");
    }
    println!();
    for &bud in &budgets {
        print!("{bud:.0}");
        for (name, _) in gens.iter() {
            let best = pts
                .iter()
                .filter(|(nm, c, _)| nm == name && *c <= bud)
                .map(|(_, _, r)| *r)
                .fold(0.0, f64::max);
            print!("\t{best:.4}");
        }
        println!();
    }
    println!("\n# PRE-REGISTERED: COMPONENT WIN if sobol/kronecker beat random by >=0.02 at matched budget in >=2 corpora;");
    println!("#   sobol-as-predicted if wins concentrate at R in {{16,32,64}}; kronecker if across R but shrinks as k grows;");
    println!("#   CLASS-DEAD if neither beats random by >=0.02 anywhere. Hybrid earned only on distinct-regime component wins.");
}
