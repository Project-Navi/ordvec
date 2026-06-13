//! R-projection shard-recall experiment for the training-free routing layer.
//!
//! Open empirical question from the prime/index conjecture investigation:
//! can a *data-oblivious* router (R random 1-D projections, each bucketed by
//! a uniform grid) capture true neighbours well enough to route, and does
//! making the grid periods pairwise COPRIME beat independent random OFFSETS
//! at decorrelating bucket seams?
//!
//! Method (per agent-5 spec): for each query, the router probes a budget of
//! buckets across the R projections; shard-recall@k = fraction of the FP32
//! cosine top-k that land in the probed union. We plot recall vs
//! CANDIDATES-SCANNED (not vs probe radius), so arms with different bucket
//! counts are compared at equal work — the only fair axis.
//!
//! Arms:
//!   coprime        — R grids, pairwise-coprime periods
//!   aligned        — R grids, identical period (worst case: seams stack)
//!   random-offset  — R grids, same period, independent random phase
//!   both           — coprime periods AND random offsets
//!
//! Decision rule: if coprime ~= random-offset within noise, coprimality adds
//! nothing (justified only on build-determinism). If coprime materially beats
//! random-offset at equal candidates-scanned and survives reseeding, that is
//! the publishable surprise.
//!
//! Run (synthetic): cargo run --release --example shard_recall
//! No external data, no BLAS.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

const SEED: u64 = 1;

struct Cfg {
    dim: usize,
    n: usize,
    n_queries: usize,
    k: usize,
    latent_dim: usize,
    n_clusters: usize,
    r_values: Vec<usize>, // projection counts to sweep
    isotropic: bool,
    corpus_npy: Option<String>,
    queries_npy: Option<String>,
}

fn parse() -> Cfg {
    let mut c = Cfg {
        dim: 256,
        n: 50_000,
        n_queries: 200,
        k: 10,
        latent_dim: 64,
        n_clusters: 200,
        r_values: vec![1, 2, 4, 8, 16],
        isotropic: false,
        corpus_npy: None,
        queries_npy: None,
    };
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        match x.as_str() {
            "--dim" => c.dim = a.next().unwrap().parse().unwrap(),
            "--n" => c.n = a.next().unwrap().parse().unwrap(),
            "--queries" => c.n_queries = a.next().unwrap().parse().unwrap(),
            "--k" => c.k = a.next().unwrap().parse().unwrap(),
            "--latent" => c.latent_dim = a.next().unwrap().parse().unwrap(),
            "--clusters" => c.n_clusters = a.next().unwrap().parse().unwrap(),
            "--isotropic" => c.isotropic = true,
            "--corpus-npy" => c.corpus_npy = Some(a.next().unwrap()),
            "--queries-npy" => c.queries_npy = Some(a.next().unwrap()),
            other => panic!("unknown arg {other}"),
        }
    }
    c
}

/// Minimal NumPy v1/v2 .npy reader for 2-D little-endian f32 C-order arrays
/// (same format contract as bench_rank's --corpus-npy / --queries-npy).
fn load_npy_f32(path: &str) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).expect("read npy");
    assert!(bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY", "not a numpy file");
    let major = bytes[6];
    let (hlen, hstart) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
    };
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

fn gauss(rng: &mut ChaCha8Rng) -> f32 {
    let u1: f32 = rng.random_range(1e-9..1.0);
    let u2: f32 = rng.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Low-rank clustered corpus + matched queries (same construction as bench_rank).
/// Returns (corpus, queries) as flat row-major dim-strided buffers.
fn make_corpus(cfg: &mut Cfg) -> (Vec<f32>, Vec<f32>) {
    // Real corpus path: both files required (need real queries for honest recall).
    if let (Some(cp), Some(qp)) = (cfg.corpus_npy.clone(), cfg.queries_npy.clone()) {
        let (mut corpus, n, d) = load_npy_f32(&cp);
        let (mut queries, nq, dq) = load_npy_f32(&qp);
        assert_eq!(d, dq, "corpus/query dim mismatch");
        l2_normalize_rows(&mut corpus, d);
        l2_normalize_rows(&mut queries, dq);
        cfg.dim = d;
        cfg.n = n;
        cfg.n_queries = nq;
        eprintln!("# loaded corpus {n}x{d}, queries {nq}x{dq}");
        return (corpus, queries);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let d = cfg.dim;
    let l = cfg.latent_dim;
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
    let iso = |rng: &mut ChaCha8Rng| -> Vec<f32> {
        let mut v = vec![0.0f32; d];
        for x in v.iter_mut() {
            *x = gauss(rng);
        }
        let nrm: f32 = v.iter().map(|t| t * t).sum::<f32>().sqrt();
        for x in v.iter_mut() {
            *x /= nrm;
        }
        v
    };
    let mut corpus = vec![0.0f32; cfg.n * d];
    for i in 0..cfg.n {
        let e = if cfg.isotropic {
            iso(&mut rng)
        } else {
            let c = rng.random_range(0..cfg.n_clusters);
            make(&protos[c * l..(c + 1) * l], 0.3, &mut rng)
        };
        corpus[i * d..(i + 1) * d].copy_from_slice(&e);
    }
    let mut queries = vec![0.0f32; cfg.n_queries * d];
    for i in 0..cfg.n_queries {
        let e = if cfg.isotropic {
            iso(&mut rng)
        } else {
            let c = rng.random_range(0..cfg.n_clusters);
            make(&protos[c * l..(c + 1) * l], 0.1, &mut rng)
        };
        queries[i * d..(i + 1) * d].copy_from_slice(&e);
    }
    (corpus, queries)
}

/// FP32 brute-force cosine top-k ground truth (corpus is L2-normalized already).
fn ground_truth(corpus: &[f32], queries: &[f32], dim: usize, k: usize) -> Vec<Vec<usize>> {
    let n = corpus.len() / dim;
    let nq = queries.len() / dim;
    (0..nq)
        .into_par_iter()
        .map(|qi| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let mut scored: Vec<(usize, f32)> = (0..n)
                .map(|di| {
                    let doc = &corpus[di * dim..(di + 1) * dim];
                    let dot: f32 = q.iter().zip(doc).map(|(a, b)| a * b).sum();
                    (di, dot)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored.into_iter().take(k).map(|(i, _)| i).collect()
        })
        .collect()
}

/// One projection: a unit random direction + a grid period + a phase offset.
/// A doc's bucket on this projection is floor((r.x - phase) / width).
struct Proj {
    dir: Vec<f32>,
    width: f32,
    phase: f32,
}

impl Proj {
    fn bucket(&self, v: &[f32]) -> i64 {
        let dot: f32 = self.dir.iter().zip(v).map(|(a, b)| a * b).sum();
        ((dot - self.phase) / self.width).floor() as i64
    }
}

/// Pairwise-coprime-ish period multipliers (distinct small primes). The grid
/// WIDTH is base_width / period, so more-divided grids have more, finer buckets.
/// Coprime periods => the seam sets (multiples of width) share no interior
/// coincidence until the lcm — the vernier effect.
const COPRIME_PERIODS: [u32; 16] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
];

#[derive(Clone, Copy)]
enum Arm {
    Coprime = 0,
    Aligned = 1,
    RandomOffset = 2,
    Both = 3,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Coprime => "coprime",
            Arm::Aligned => "aligned",
            Arm::RandomOffset => "random-offset",
            Arm::Both => "both",
        }
    }
}

/// Build R projections for one arm. `base_width` is calibrated so a single
/// grid has ~sqrt(n) buckets (a reasonable shard granularity); coprime arms
/// subdivide by the prime multiplier.
fn build_projs(cfg: &Cfg, arm: Arm, r: usize, base_width: f32, rng: &mut ChaCha8Rng) -> Vec<Proj> {
    let d = cfg.dim;
    (0..r)
        .map(|i| {
            let mut dir = vec![0.0f32; d];
            for x in dir.iter_mut() {
                *x = gauss(rng);
            }
            let nrm: f32 = dir.iter().map(|t| t * t).sum::<f32>().sqrt();
            for x in dir.iter_mut() {
                *x /= nrm;
            }
            let (width, phase) = match arm {
                // coprime: finer grid per projection by a distinct prime factor
                Arm::Coprime => (base_width / COPRIME_PERIODS[i] as f32, 0.0),
                // aligned: identical grid, zero phase — seams stack exactly
                Arm::Aligned => (base_width, 0.0),
                // random-offset: identical period, independent random phase
                Arm::RandomOffset => {
                    let p: f32 = rng.random_range(0.0..1.0);
                    (base_width, p * base_width)
                }
                // both: coprime period AND random phase
                Arm::Both => {
                    let p: f32 = rng.random_range(0.0..1.0);
                    let w = base_width / COPRIME_PERIODS[i] as f32;
                    (w, p * w)
                }
            };
            Proj { dir, width, phase }
        })
        .collect()
}

/// Assign every doc a bucket-key tuple (one i64 per projection), grouped into
/// an index: for each projection, map bucket-id -> list of doc ids.
type BucketIndex = Vec<std::collections::HashMap<i64, Vec<u32>>>;

fn build_index(projs: &[Proj], corpus: &[f32], dim: usize) -> BucketIndex {
    let n = corpus.len() / dim;
    projs
        .par_iter()
        .map(|p| {
            let mut m: std::collections::HashMap<i64, Vec<u32>> = std::collections::HashMap::new();
            for di in 0..n {
                let b = p.bucket(&corpus[di * dim..(di + 1) * dim]);
                m.entry(b).or_default().push(di as u32);
            }
            m
        })
        .collect()
}

/// For one query at a given probe radius `rad` (probe buckets b-rad..=b+rad on
/// each projection), return the candidate union and its size (candidates
/// scanned). Recall is measured against `truth_set`.
fn probe_recall(
    projs: &[Proj],
    index: &BucketIndex,
    q: &[f32],
    rad: i64,
    truth: &[usize],
) -> (usize, f32) {
    use std::collections::HashSet;
    let mut cand: HashSet<u32> = HashSet::new();
    for (pi, p) in projs.iter().enumerate() {
        let b = p.bucket(q);
        for bb in (b - rad)..=(b + rad) {
            if let Some(ids) = index[pi].get(&bb) {
                cand.extend(ids.iter().copied());
            }
        }
    }
    let truth_set: HashSet<u32> = truth.iter().map(|&i| i as u32).collect();
    let hits = truth_set.iter().filter(|i| cand.contains(i)).count();
    let recall = hits as f32 / truth_set.len().max(1) as f32;
    (cand.len(), recall)
}

fn run_arms(cfg: &Cfg, corpus: &[f32], queries: &[f32], truth: &[Vec<usize>]) {
    // Calibrate base_width so a single unit-direction grid yields ~sqrt(n)
    // buckets. r.x ~ N(0, 1/dim) for L2-normalized x, so its spread ~ 6/sqrt(dim)
    // across the corpus; divide that range into ~sqrt(n) cells.
    let spread = 6.0 / (cfg.dim as f32).sqrt();
    let base_width = spread / (cfg.n as f32).sqrt();
    let arms = [Arm::Coprime, Arm::Aligned, Arm::RandomOffset, Arm::Both];
    // Collect (arm, cand, recall) across all (R, rad) for the fair envelope.
    let mut pts: Vec<(&'static str, f64, f64)> = Vec::new();
    println!("# recall vs candidates-scanned (mean over queries), per arm per R");
    println!("arm\tR\trad\tcand_scanned\trecall@k");
    for &r in &cfg.r_values {
        for &arm in &arms {
            // Aligned/RandomOffset don't subdivide, so they only differ from
            // each other by phase; coprime/both subdivide per projection.
            let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ (r as u64) << 8 ^ arm as u64);
            let projs = build_projs(cfg, arm, r, base_width, &mut rng);
            let index = build_index(&projs, corpus, cfg.dim);
            for rad in [0i64, 1, 2, 4] {
                let stats: Vec<(usize, f32)> = (0..cfg.n_queries)
                    .into_par_iter()
                    .map(|qi| {
                        let q = &queries[qi * cfg.dim..(qi + 1) * cfg.dim];
                        probe_recall(&projs, &index, q, rad, &truth[qi])
                    })
                    .collect();
                let mean_cand =
                    stats.iter().map(|s| s.0 as f64).sum::<f64>() / stats.len() as f64;
                let mean_rec =
                    stats.iter().map(|s| s.1 as f64).sum::<f64>() / stats.len() as f64;
                println!(
                    "{}\t{r}\t{rad}\t{:.0}\t{:.4}",
                    arm.name(),
                    mean_cand,
                    mean_rec
                );
                pts.push((arm.name(), mean_cand, mean_rec));
            }
        }
    }
    // Fair envelope: best recall achievable per arm at or below each candidate
    // budget. This is the only honest cross-arm comparison — equal work.
    println!("\n# FAIR ENVELOPE: max recall@k at candidates-scanned <= budget");
    let budgets = [1000.0, 2000.0, 4000.0, 8000.0, 16000.0];
    print!("budget");
    for a in &arms {
        print!("\t{}", a.name());
    }
    println!();
    for &bud in &budgets {
        print!("{bud:.0}");
        for a in &arms {
            let best = pts
                .iter()
                .filter(|(name, c, _)| *name == a.name() && *c <= bud)
                .map(|(_, _, r)| *r)
                .fold(0.0f64, f64::max);
            print!("\t{best:.4}");
        }
        println!();
    }
}

fn main() {
    let mut cfg = parse();
    let (corpus, queries) = make_corpus(&mut cfg);
    let truth = ground_truth(&corpus, &queries, cfg.dim, cfg.k);
    let src = if cfg.corpus_npy.is_some() {
        "npy"
    } else if cfg.isotropic {
        "isotropic"
    } else {
        "clustered"
    };
    println!(
        "# shard_recall: n={} q={} dim={} k={} corpus={src}",
        cfg.n, cfg.n_queries, cfg.dim, cfg.k
    );
    run_arms(&cfg, &corpus, &queries, &truth);
}
