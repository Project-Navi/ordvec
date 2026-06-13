//! Corpus zoo generator — writes diverse embedding geometries as .npy so the
//! conjecture probes (spectral_probe, shard_recall, twonn_id, bench_rank) can
//! be verified across distributions, not just the one synthetic clustered
//! Gaussian. Decouples generation from measurement and dogfoods the .npy path.
//!
//! Kinds (each stresses a different failure mode / mimics a real pathology):
//!   clustered   — low-rank Gaussian clusters (the existing baseline)
//!   isotropic   — N(0,I) on the sphere (Poisson NULL control)
//!   anisotropic — power-law singular spectrum sigma_k ~ k^-alpha (real
//!                 embedding decay; Gao et al. narrow-cone)
//!   rogue       — a few dimensions with huge variance (Timkey rogue dims)
//!   manifold    — nonlinear low-D manifold (sinusoidal) in high-D: the one
//!                 place angular RIGIDITY could plausibly hide
//!   lattice     — jittered 1-D lattice projected up: POSITIVE rigidity
//!                 control. If the number-variance probe never reports
//!                 sub-Poisson, it can't see rigidity; this proves it can.
//!
//! Run:  cargo run --release --example gen_corpus -- --kind anisotropic \
//!          --n 50000 --dim 256 --out corpus_aniso.npy
//! Also emits a matched query file with --queries-out (smaller-noise draws).

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

struct Cfg {
    kind: String,
    n: usize,
    n_queries: usize,
    dim: usize,
    latent: usize,
    clusters: usize,
    alpha: f32,
    rogue: usize,
    out: String,
    queries_out: Option<String>,
    seed: u64,
}

fn parse() -> Cfg {
    let mut c = Cfg {
        kind: "clustered".into(),
        n: 50_000,
        n_queries: 200,
        dim: 256,
        latent: 64,
        clusters: 200,
        alpha: 1.0,
        rogue: 3,
        out: "corpus.npy".into(),
        queries_out: None,
        seed: 1,
    };
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        match x.as_str() {
            "--kind" => c.kind = a.next().unwrap(),
            "--n" => c.n = a.next().unwrap().parse().unwrap(),
            "--queries" => c.n_queries = a.next().unwrap().parse().unwrap(),
            "--dim" => c.dim = a.next().unwrap().parse().unwrap(),
            "--latent" => c.latent = a.next().unwrap().parse().unwrap(),
            "--clusters" => c.clusters = a.next().unwrap().parse().unwrap(),
            "--alpha" => c.alpha = a.next().unwrap().parse().unwrap(),
            "--rogue" => c.rogue = a.next().unwrap().parse().unwrap(),
            "--out" => c.out = a.next().unwrap(),
            "--queries-out" => c.queries_out = Some(a.next().unwrap()),
            "--seed" => c.seed = a.next().unwrap().parse().unwrap(),
            other => panic!("unknown arg {other}"),
        }
    }
    c
}

/// Inverse standard-normal CDF (Acklam's rational approximation, |err|<1e-9).
fn inv_norm_cdf(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    let b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01];
    let c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    let d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00];
    let plow = 0.02425;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= 1.0 - plow {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

fn gauss(rng: &mut ChaCha8Rng) -> f32 {
    let u1: f32 = rng.random_range(1e-9..1.0);
    let u2: f32 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Write a 2-D row-major f32 buffer as a NumPy v1 .npy (<f4, C order).
fn write_npy(path: &str, data: &[f32], n: usize, dim: usize) {
    use std::io::Write;
    let header_body = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({n}, {dim}), }}"
    );
    // total header must make (10 + len) a multiple of 64; pad with spaces + \n
    let mut hb = header_body.into_bytes();
    let unpadded = 10 + hb.len() + 1; // +1 for trailing newline
    let pad = (64 - (unpadded % 64)) % 64;
    hb.extend(std::iter::repeat(b' ').take(pad));
    hb.push(b'\n');
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("create npy"));
    f.write_all(b"\x93NUMPY").unwrap();
    f.write_all(&[1u8, 0u8]).unwrap(); // version 1.0
    f.write_all(&(hb.len() as u16).to_le_bytes()).unwrap();
    f.write_all(&hb).unwrap();
    for &v in data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    eprintln!("# wrote {path}: {n} x {dim} f32");
}

fn l2(v: &mut [f32]) {
    let nrm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if nrm > 0.0 {
        for x in v.iter_mut() {
            *x /= nrm;
        }
    }
}

/// Generate `count` L2-normalized rows of dimension cfg.dim for the chosen
/// geometry. `is_query` reduces noise so queries sit near corpus structure.
fn generate(cfg: &Cfg, rng: &mut ChaCha8Rng, count: usize, is_query: bool) -> Vec<f32> {
    let d = cfg.dim;
    let l = cfg.latent;
    let noise = if is_query { 0.1 } else { 0.3 };
    // shared random projection A: d x l  (reused across rows in this call)
    let mut a = vec![0.0f32; d * l];
    for x in a.iter_mut() {
        *x = gauss(rng);
    }
    // cluster prototypes (latent space)
    let mut protos = vec![0.0f32; cfg.clusters * l];
    for x in protos.iter_mut() {
        *x = gauss(rng);
    }
    // power-law per-latent scaling for "anisotropic": sigma_k ~ (k+1)^-alpha
    let scale: Vec<f32> = (0..l)
        .map(|k| ((k + 1) as f32).powf(-cfg.alpha))
        .collect();

    let mut out = vec![0.0f32; count * d];
    for i in 0..count {
        let row = &mut out[i * d..(i + 1) * d];
        match cfg.kind.as_str() {
            "isotropic" => {
                for x in row.iter_mut() {
                    *x = gauss(rng);
                }
            }
            "clustered" => {
                let c = rng.random_range(0..cfg.clusters);
                let mut z = vec![0.0f32; l];
                for j in 0..l {
                    z[j] = protos[c * l + j] + noise * gauss(rng);
                }
                for ii in 0..d {
                    let mut acc = 0.0f32;
                    for j in 0..l {
                        acc += a[ii * l + j] * z[j];
                    }
                    row[ii] = acc;
                }
            }
            "anisotropic" => {
                // power-law spectrum: latent coords scaled by k^-alpha
                let mut z = vec![0.0f32; l];
                for j in 0..l {
                    z[j] = scale[j] * gauss(rng);
                }
                for ii in 0..d {
                    let mut acc = 0.0f32;
                    for j in 0..l {
                        acc += a[ii * l + j] * z[j];
                    }
                    row[ii] = acc;
                }
            }
            "rogue" => {
                // baseline isotropic + a few dims with ~10x variance
                for x in row.iter_mut() {
                    *x = gauss(rng);
                }
                for r in 0..cfg.rogue.min(d) {
                    row[r] = 10.0 * gauss(rng);
                }
            }
            "manifold" => {
                // nonlinear low-D manifold: latent t in R^m mapped through
                // sinusoids of varying frequency, then projected up. Angular
                // structure here is the best chance for rigidity to appear.
                let m = l.min(8).max(2);
                let mut t = vec![0.0f32; m];
                for tj in t.iter_mut() {
                    *tj = rng.random_range(0.0..std::f32::consts::TAU);
                }
                // build a smooth high-D embedding: row[ii] = sum_j sin((j+1) t_j + phase_ii)
                for ii in 0..d {
                    let mut acc = 0.0f32;
                    for j in 0..m {
                        let phase = a[ii * l + j]; // reuse A as fixed phases
                        acc += ((j + 1) as f32 * t[j] + phase).sin();
                    }
                    row[ii] = acc + noise * gauss(rng);
                }
            }
            "projected-rigid" => {
                // STRONGEST rigidity control: place a quantile-spaced target on
                // a FIXED direction u (= col 0 of A, unit-normalized once) and
                // make it dominate. x_i = t_i * u  (+ tiny ortho jitter). Then
                // for ANY probe direction r, r.x_i = t_i * (r.u), a scaled copy
                // of the rigid sequence — so the projected key inherits the
                // quantile spacing (up to L2 distortion). This tests whether the
                // random-projection + normalize pipeline PRESERVES rigidity that
                // genuinely lives in the key, closing the selftest gap.
                let t = inv_norm_cdf((i as f64 + 0.5) / count as f64) as f32;
                for ii in 0..d {
                    row[ii] = t * a[ii * l] + 0.001 * gauss(rng);
                }
            }
            "lattice" => {
                // POSITIVE rigidity control. Each row gets a DISTINCT, evenly
                // spaced position pos = i/count in [-1,1] (used once — no
                // duplicates). Embedded on a small ARC: a constant base
                // direction (col 0 of A) plus a small pos-scaled component along
                // an orthogonal direction (col 1). The constant base survives
                // L2 normalization (a pure line would collapse to +/- one
                // vector); projecting onto the pos-direction recovers the
                // evenly-spaced sequence -> SUB-Poisson (rigid) number variance.
                // Gaussian-QUANTILE-spaced positions: pos_i = Phi^-1((i+0.5)/count).
                // The marginal is then exactly N(0,1), matching the probe's
                // Gaussian unfold, so any residual structure is REAL. Evenly
                // spaced in probability = maximally regular = rigid.
                let pos = inv_norm_cdf((i as f64 + 0.5) / count as f64) as f32;
                for ii in 0..d {
                    let base = a[ii * l];
                    let along = a[ii * l + 1];
                    row[ii] = base + 0.35 * pos * along + 0.01 * gauss(rng);
                }
            }
            other => panic!("unknown kind {other}"),
        }
        l2(row);
    }
    out
}

fn main() {
    let cfg = parse();
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let corpus = generate(&cfg, &mut rng, cfg.n, false);
    write_npy(&cfg.out, &corpus, cfg.n, cfg.dim);
    if let Some(ref qp) = cfg.queries_out {
        let q = generate(&cfg, &mut rng, cfg.n_queries, true);
        write_npy(qp, &q, cfg.n_queries, cfg.dim);
    }
}
