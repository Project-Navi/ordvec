//! Direction-PLACEMENT bake-off for the training-free routing layer.
//!
//! Open thread from the routing research (PR #211): `shard_recall.rs` always
//! draws iid-Gaussian projection directions — low-discrepancy *placement* of the
//! directions themselves was never an arm. This rig tests one hypothesis and is
//! built to KILL it cheaply if it's false:
//!
//!   Shared-index triple-tiled golden-angle directions (a 2-sphere spiral per
//!   coord-triple, all triples coupled by ONE Fibonacci index so they cohere on
//!   the permutahedron) need FEWER candidates-scanned for matched recall@k than
//!   iid-Gaussian — strongest at small R and at Fibonacci R, decaying as dim
//!   grows (concentration of measure).
//!
//! Two arms, EVERYTHING else held identical to shard_recall's "aligned" calib
//! (single shared base_width, phase=0). Direction construction is the ONLY var.
//!   gaussian   — iid Gaussian normalized (the control)
//!   fib-spiral — shared-index triple-tiled golden-angle, per-triple rotated
//!
//! Honesty checks baked in (printed): at R=1 the arms MUST coincide (one
//! direction cannot carry low-discrepancy structure) — divergence => harness
//! bug, not a result. SEED=1 => identical tables across runs.
//!
//! Run (synthetic dim=256): cargo run --release --example fib_directions
//! Run (real dim=768):      cargo run --release --example fib_directions -- \
//!                              --corpus-npy repo_real.npy --queries-npy repo_q.npy
//! No external data on the synthetic path, no BLAS.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Base seed; overridable with --seed for the reseed-stability honesty check.
/// A real effect must survive reseeding (changes gaussian directions AND the
/// fib/project per-triple rotations); a seed-specific gap is an artifact.
static SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
fn seed() -> u64 {
    SEED.load(std::sync::atomic::Ordering::Relaxed)
}

struct Cfg {
    dim: usize,
    n: usize,
    n_queries: usize,
    k: usize,
    latent_dim: usize,
    n_clusters: usize,
    cone: f32,
    r_values: Vec<usize>,
    no_triple_rotation: bool,
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
        // Cone anisotropy: real transformer embeddings are NOT gaussian/centered —
        // they sit in a narrow cone (a dominant shared mean direction). `cone` is
        // the weight of a fixed shared axis added to every vector before
        // normalization; 0.0 = centered (old behaviour), larger = tighter cone.
        // Default tuned so synthetic mean-resultant-length is in the ballpark of
        // nomic (printed at startup so it can be matched to real --corpus-npy data).
        cone: 3.0,
        // Fibonacci R values (1,2,3,5,8,13,21) interleaved with non-Fib (4,16)
        // so a Fibonacci-specific bump is legible against its neighbours.
        r_values: vec![1, 2, 3, 4, 5, 8, 13, 16, 21],
        no_triple_rotation: false,
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
            "--cone" => c.cone = a.next().unwrap().parse().unwrap(),
            "--seed" => SEED.store(
                a.next().unwrap().parse().unwrap(),
                std::sync::atomic::Ordering::Relaxed,
            ),
            "--no-triple-rotation" => c.no_triple_rotation = true,
            "--corpus-npy" => c.corpus_npy = Some(a.next().unwrap()),
            "--queries-npy" => c.queries_npy = Some(a.next().unwrap()),
            other => panic!("unknown arg {other}"),
        }
    }
    c
}

/// Minimal NumPy v1/v2 .npy reader for 2-D little-endian f32 C-order arrays
/// (same contract as shard_recall / bench_rank).
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
    assert!(
        header.contains("'fortran_order': False"),
        "expected C order"
    );
    let after = &header[header.find("'shape':").expect("shape")..];
    let inner = &after[after.find('(').unwrap() + 1..after.find(')').unwrap()];
    let dims: Vec<usize> = inner
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
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

/// Low-rank clustered corpus + matched queries (same construction as shard_recall).
fn make_corpus(cfg: &mut Cfg) -> (Vec<f32>, Vec<f32>) {
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
    let mut rng = ChaCha8Rng::seed_from_u64(seed());
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
    // Shared cone axis: one fixed unit direction added (weight `cone`) to every
    // vector before normalization, making the corpus anisotropic like real
    // embeddings (a dominant mean direction) rather than centered/gaussian.
    let mut cone_axis = vec![0.0f32; d];
    for x in cone_axis.iter_mut() {
        *x = gauss(&mut rng);
    }
    {
        let nrm: f32 = cone_axis.iter().map(|v| v * v).sum::<f32>().sqrt();
        for x in cone_axis.iter_mut() {
            *x /= nrm;
        }
    }
    let cone = cfg.cone;
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
            out[i] = acc + cone * cone_axis[i];
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
        let c = rng.random_range(0..cfg.n_clusters);
        let e = make(&protos[c * l..(c + 1) * l], 0.3, &mut rng);
        corpus[i * d..(i + 1) * d].copy_from_slice(&e);
    }
    let mut queries = vec![0.0f32; cfg.n_queries * d];
    for i in 0..cfg.n_queries {
        let c = rng.random_range(0..cfg.n_clusters);
        let e = make(&protos[c * l..(c + 1) * l], 0.1, &mut rng);
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

/// Mean resultant length R̄ = ‖ (1/n) Σ x_i ‖ over L2-normalized rows.
/// 0 ⇒ isotropic (directions cancel); →1 ⇒ tight cone (one shared direction).
/// Printed for BOTH synthetic and real corpora so the synthetic anisotropy can
/// be matched to nomic's — the whole point of the user's anisotropy note.
fn mean_resultant_length(corpus: &[f32], dim: usize) -> f32 {
    let n = corpus.len() / dim;
    let mut mean = vec![0.0f64; dim];
    for di in 0..n {
        let row = &corpus[di * dim..(di + 1) * dim];
        let nrm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if nrm > 0.0 {
            for (m, &x) in mean.iter_mut().zip(row) {
                *m += (x / nrm) as f64;
            }
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }
    (mean.iter().map(|m| m * m).sum::<f64>().sqrt()) as f32
}

/// The dominant cone axis of a corpus = its normalized mean direction. Returned
/// as a unit vector; used by the project-then-spiral arm to deflate the shared
/// component before placing directions (anisotropic data lives off this axis).
fn dominant_axis(corpus: &[f32], dim: usize) -> Vec<f32> {
    let n = corpus.len() / dim;
    let mut mean = vec![0.0f32; dim];
    for di in 0..n {
        let row = &corpus[di * dim..(di + 1) * dim];
        let nrm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if nrm > 0.0 {
            for (m, &x) in mean.iter_mut().zip(row) {
                *m += x / nrm;
            }
        }
    }
    let nrm: f32 = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
    if nrm > 0.0 {
        for m in mean.iter_mut() {
            *m /= nrm;
        }
    }
    mean
}

#[derive(Clone, Copy, PartialEq)]
enum DirArm {
    Gaussian,
    FibSpiral,
    ProjectSpiral,
}

impl DirArm {
    fn name(self) -> &'static str {
        match self {
            DirArm::Gaussian => "gaussian",
            DirArm::FibSpiral => "fib-spiral",
            DirArm::ProjectSpiral => "project-spiral",
        }
    }
}

/// A single golden-angle spiral point on S^2 for index `i` of `r` total points.
/// z evenly stepped in [-1,1], azimuth advanced by the golden angle π(3-√5).
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

/// Deterministic 3x3 rotation for triple `t`: built by Gram–Schmidt on three
/// Gaussian columns from a triple-indexed RNG. Data-oblivious and reproducible.
/// Identity when `no_rotation` (the pure shared-index variant).
fn triple_rotation(t: usize, no_rotation: bool) -> [[f32; 3]; 3] {
    if no_rotation {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed() ^ 0xA071_u64.wrapping_mul(t as u64 + 1));
    let col = |rng: &mut ChaCha8Rng| [gauss(rng), gauss(rng), gauss(rng)];
    let dot = |a: &[f32; 3], b: &[f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norm = |a: &mut [f32; 3]| {
        let n = dot(a, a).sqrt();
        if n > 0.0 {
            for x in a.iter_mut() {
                *x /= n;
            }
        }
    };
    let mut e0 = col(&mut rng);
    norm(&mut e0);
    let mut v1 = col(&mut rng);
    let p1 = dot(&v1, &e0);
    for i in 0..3 {
        v1[i] -= p1 * e0[i];
    }
    norm(&mut v1);
    // e2 = e0 × e1 (right-handed, guaranteed orthonormal)
    let e2 = [
        e0[1] * v1[2] - e0[2] * v1[1],
        e0[2] * v1[0] - e0[0] * v1[2],
        e0[0] * v1[1] - e0[1] * v1[0],
    ];
    [e0, v1, e2]
}

/// Build R unit directions of dimension `dim` for one arm.
/// - Gaussian: iid normal columns, normalized (the control).
/// - FibSpiral: tile `dim` into d/3 triples; direction i = concat over triples
///   of (rotation_t · spiral_point(i, R)); ONE shared index i couples triples.
/// - ProjectSpiral: same as FibSpiral but each direction is then deflated of its
///   component along `cone_axis` (the data's dominant direction) and renormalized,
///   so directions live OFF the shared cone where anisotropic data actually splits.
fn build_dirs(arm: DirArm, r: usize, dim: usize, no_rot: bool, cone_axis: &[f32]) -> Vec<Vec<f32>> {
    match arm {
        DirArm::Gaussian => {
            let mut rng = ChaCha8Rng::seed_from_u64(seed() ^ 0xD132_0000 ^ (r as u64));
            (0..r)
                .map(|_| {
                    let mut v = vec![0.0f32; dim];
                    for x in v.iter_mut() {
                        *x = gauss(&mut rng);
                    }
                    let nrm: f32 = v.iter().map(|t| t * t).sum::<f32>().sqrt();
                    for x in v.iter_mut() {
                        *x /= nrm;
                    }
                    v
                })
                .collect()
        }
        DirArm::FibSpiral | DirArm::ProjectSpiral => {
            let n_tri = dim.div_ceil(3); // pad the tail triple if dim % 3 != 0
            let rots: Vec<[[f32; 3]; 3]> = (0..n_tri).map(|t| triple_rotation(t, no_rot)).collect();
            (0..r)
                .map(|i| {
                    let p = spiral_point(i, r); // shared index i across all triples
                    let mut v = vec![0.0f32; dim];
                    for (t, rot) in rots.iter().enumerate() {
                        // rotated triple vector
                        let rv = [
                            rot[0][0] * p[0] + rot[1][0] * p[1] + rot[2][0] * p[2],
                            rot[0][1] * p[0] + rot[1][1] * p[1] + rot[2][1] * p[2],
                            rot[0][2] * p[0] + rot[1][2] * p[1] + rot[2][2] * p[2],
                        ];
                        for (c, &rvc) in rv.iter().enumerate() {
                            let idx = t * 3 + c;
                            if idx < dim {
                                v[idx] = rvc;
                            }
                        }
                    }
                    if arm == DirArm::ProjectSpiral {
                        // deflate the dominant cone component, then renormalize
                        let proj: f32 = v.iter().zip(cone_axis).map(|(a, b)| a * b).sum();
                        for (x, &c) in v.iter_mut().zip(cone_axis) {
                            *x -= proj * c;
                        }
                    }
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
    }
}

/// One projection grid: unit direction + grid width (phase=0 for all arms here).
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
    let hits = truth
        .iter()
        .filter(|&&i| cand.contains(&(i as u32)))
        .count();
    let recall = hits as f32 / truth.len().max(1) as f32;
    (cand.len(), recall)
}

fn run(cfg: &Cfg, corpus: &[f32], queries: &[f32], truth: &[Vec<usize>]) {
    // base_width: single unit grid → ~sqrt(n) buckets (shard_recall calibration).
    let spread = 6.0 / (cfg.dim as f32).sqrt();
    let base_width = spread / (cfg.n as f32).sqrt();
    let cone_axis = dominant_axis(corpus, cfg.dim);
    let arms = [DirArm::Gaussian, DirArm::FibSpiral, DirArm::ProjectSpiral];

    let mut pts: Vec<(&'static str, f64, f64)> = Vec::new();
    println!("# recall vs candidates-scanned (mean over queries), per arm per R");
    println!("arm\tR\trad\tcand_scanned\trecall@k");
    for &r in &cfg.r_values {
        for &arm in &arms {
            let dirs = build_dirs(arm, r, cfg.dim, cfg.no_triple_rotation, &cone_axis);
            let projs: Vec<Proj> = dirs
                .into_iter()
                .map(|dir| Proj {
                    dir,
                    width: base_width,
                })
                .collect();
            let index = build_index(&projs, corpus, cfg.dim);
            for rad in [0i64, 1, 2, 4] {
                let stats: Vec<(usize, f32)> = (0..cfg.n_queries)
                    .into_par_iter()
                    .map(|qi| {
                        let q = &queries[qi * cfg.dim..(qi + 1) * cfg.dim];
                        probe_recall(&projs, &index, q, rad, &truth[qi])
                    })
                    .collect();
                let mean_cand = stats.iter().map(|s| s.0 as f64).sum::<f64>() / stats.len() as f64;
                let mean_rec = stats.iter().map(|s| s.1 as f64).sum::<f64>() / stats.len() as f64;
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
    } else {
        "synthetic-cone"
    };
    let rbar_c = mean_resultant_length(&corpus, cfg.dim);
    let rbar_q = mean_resultant_length(&queries, cfg.dim);
    println!(
        "# fib_directions: n={} q={} dim={} k={} corpus={src} cone={}",
        cfg.n, cfg.n_queries, cfg.dim, cfg.k, cfg.cone
    );
    println!(
        "# ANISOTROPY mean-resultant-length: corpus={rbar_c:.4} queries={rbar_q:.4} \
         (0=isotropic, 1=single cone) — match synthetic to real before trusting synthetic"
    );
    if cfg.no_triple_rotation {
        println!("# fib/project arms: NO per-triple rotation (pure shared-index variant)");
    }
    run(&cfg, &corpus, &queries, &truth);
}
