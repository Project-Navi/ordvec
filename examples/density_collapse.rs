//! Density-collapse probe: what does ordvec lose in dense regions, and is the
//! lost signal recoverable from intra-bucket permutation order?
//!
//! Mechanism under test. RankQuant b=2 encodes each vector as which QUARTILE
//! each coordinate's rank falls in. Two vectors that share top-bucket membership
//! COLLIDE — b=2 cannot tell them apart. This is worst in tight semantic
//! clusters (near-parallel vectors share their largest dims → same top ranks).
//!
//! But the FULL rank vector (a permutation of 0..D) still orders the
//! coordinates WITHIN the top bucket; b=2 discards that. The question:
//!
//!   Within a collided bucket, do TRUE FP32 neighbors differ in Kendall-tau of
//!   their top-k coordinate order LESS than random pairs in the same bucket?
//!
//! If yes, intra-bucket order carries the signal b=2 threw away, and a
//! tau-rerank recovers it WITHOUT new storage (the order is in the Rank code).
//! Ground truth = FP32 cosine, so this cannot miscalibrate (unlike the
//! number-variance probe).
//!
//! Run: cargo run --release --example density_collapse
//!      cargo run --release --example density_collapse -- --noise 0.15   (tighter)
//! No external data, no BLAS. Uses ordvec::rank::rank_transform.

use ordvec::rank::rank_transform;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

const SEED: u64 = 1;

struct Cfg {
    dim: usize,
    n: usize,
    latent: usize,
    clusters: usize,
    noise: f32, // cluster tightness: smaller = denser = more collapse
    bits: u32,  // bucket bits (2 = quartiles)
    topk: usize, // # top coords used for the intra-bucket tau test
    corpus_npy: Option<String>, // real embeddings; overrides synthetic + dim/n
}

fn parse() -> Cfg {
    let mut c = Cfg {
        dim: 256,
        n: 30_000,
        latent: 64,
        clusters: 200,
        noise: 0.3,
        bits: 2,
        topk: 16,
        corpus_npy: None,
    };
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        match x.as_str() {
            "--dim" => c.dim = a.next().unwrap().parse().unwrap(),
            "--n" => c.n = a.next().unwrap().parse().unwrap(),
            "--latent" => c.latent = a.next().unwrap().parse().unwrap(),
            "--clusters" => c.clusters = a.next().unwrap().parse().unwrap(),
            "--noise" => c.noise = a.next().unwrap().parse().unwrap(),
            "--bits" => c.bits = a.next().unwrap().parse().unwrap(),
            "--topk" => c.topk = a.next().unwrap().parse().unwrap(),
            "--corpus-npy" => c.corpus_npy = Some(a.next().unwrap()),
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

/// Cosine of two unit vectors (corpus is L2-normalized).
fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Kendall-tau DISTANCE between the orderings the two vectors induce on a fixed
/// coordinate set `coords` (fraction of discordant pairs, in [0,1]). Uses the
/// raw values to decide order on each coordinate pair. 0 = identical order.
fn kendall_tau(a: &[f32], b: &[f32], coords: &[usize]) -> f32 {
    let m = coords.len();
    if m < 2 {
        return 0.0;
    }
    let mut disc = 0usize;
    let mut tot = 0usize;
    for x in 0..m {
        for y in (x + 1)..m {
            let (cx, cy) = (coords[x], coords[y]);
            let sa = (a[cx] - a[cy]).signum();
            let sb = (b[cx] - b[cy]).signum();
            if sa != sb {
                disc += 1;
            }
            tot += 1;
        }
    }
    disc as f32 / tot.max(1) as f32
}

/// Indices of the top-k coordinates of a vector by value (descending).
fn top_coords(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| v[j].partial_cmp(&v[i]).unwrap());
    idx.truncate(k);
    idx
}

/// Hamming distance between two equal-length bucket codes (# coords that
/// landed in a different bucket). This is what the b=2 SCORER conflates: small
/// Hamming distance = codes the kernel can barely separate = real collapse.
fn hamming(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).filter(|(x, y)| x != y).count() as u32
}

/// Report the code-distance structure: exact collisions are ~0 on continuous
/// data (constant-composition codes are length-D sequences), so the real
/// "collapse" is NEAR-collision — how many docs sit within small Hamming
/// distance of a probe. Sampled over probes to stay O(n * samples).
fn collision_report(cfg: &Cfg, codes: &[Vec<u8>]) {
    let n = cfg.n;
    let n_probes = 200.min(n);
    let stride = (n / n_probes).max(1);
    let probes: Vec<usize> = (0..n).step_by(stride).take(n_probes).collect();
    // for each probe, min Hamming distance to any other doc + count within 2*min
    let stats: Vec<(u32, usize)> = probes
        .par_iter()
        .map(|&p| {
            let cp = &codes[p];
            let mut min_h = u32::MAX;
            for j in 0..n {
                if j == p {
                    continue;
                }
                let h = hamming(cp, &codes[j]);
                if h < min_h {
                    min_h = h;
                }
            }
            let thresh = (min_h + min_h / 2).max(min_h + 1);
            let near = (0..n)
                .filter(|&j| j != p && hamming(cp, &codes[j]) <= thresh)
                .count();
            (min_h, near)
        })
        .collect();
    let mean_min = stats.iter().map(|s| s.0 as f64).sum::<f64>() / stats.len() as f64;
    let mean_near = stats.iter().map(|s| s.1 as f64).sum::<f64>() / stats.len() as f64;
    println!("# density_collapse: n={} dim={} bits={} noise={} topk={} (smaller noise=denser)",
        cfg.n, cfg.dim, cfg.bits, cfg.noise, cfg.topk);
    println!("## Code near-collision structure (b={} bucket code, length {})", cfg.bits, cfg.dim);
    println!("mean nearest-code Hamming dist = {mean_min:.1} of {} coords", cfg.dim);
    println!("mean #docs within ~1.5x that distance = {mean_near:.1}");
    println!("(small Hamming = codes the b={} scorer can barely separate = collapse)", cfg.bits);
}

/// Minimal NumPy v1/v2 .npy reader for 2-D little-endian f32 C-order arrays.
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
    // L2-normalize rows (cosine geometry)
    for i in 0..n {
        let row = &mut out[i * dim..(i + 1) * dim];
        let nrm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if nrm > 0.0 {
            for x in row.iter_mut() {
                *x /= nrm;
            }
        }
    }
    (out, n, dim)
}

fn main() {
    let mut cfg = parse();
    let corpus = if let Some(path) = cfg.corpus_npy.clone() {
        let (v, n, dim) = load_npy_f32(&path);
        eprintln!("# loaded REAL corpus {n} x {dim} from {path}");
        cfg.n = n;
        cfg.dim = dim;
        v
    } else {
        make_corpus(&cfg)
    };
    let d = cfg.dim;

    // b-bit bucket code: top bucket id per coordinate. rank r in [0,D) ->
    // bucket r / (D / 2^bits). The b=2 "top-bucket membership" code we hash on
    // is the multiset of bucket ids (order-free), exactly what RankQuant ties on.
    let nbuckets = 1usize << cfg.bits;
    let bucket_w = d / nbuckets;
    let codes: Vec<Vec<u8>> = (0..cfg.n)
        .into_par_iter()
        .map(|i| {
            let ranks = rank_transform(&corpus[i * d..(i + 1) * d]);
            ranks
                .iter()
                .map(|&r| (r as usize / bucket_w).min(nbuckets - 1) as u8)
                .collect()
        })
        .collect();

    collision_report(&cfg, &codes);
    tau_report(&cfg, &corpus, &codes);
}

/// THE TEST. For each probe, gather the M docs whose b=2 codes are
/// Hamming-CLOSEST (the "lookalikes" the scorer conflates with the probe).
/// Split them into the FP32-true neighbours (top half by cosine) and the
/// FP32-far lookalikes (bottom half). Question: does intra-bucket top-k
/// Kendall-tau SEPARATE these two groups — i.e. do the true neighbours have
/// lower tau than the false lookalikes that the b=2 code can't tell apart?
/// If yes, the fine permutation order recovers exactly the signal b=2 lost,
/// from the existing Rank code, no new storage. FP32 cosine = ground truth.
fn tau_report(cfg: &Cfg, corpus: &[f32], codes: &[Vec<u8>]) {
    let d = cfg.dim;
    let n = cfg.n;
    let m = 40usize; // size of the b2-lookalike neighbourhood per probe
    let n_probes = 300.min(n);
    let stride = (n / n_probes).max(1);
    let probes: Vec<usize> = (0..n).step_by(stride).take(n_probes).collect();

    // Per probe: (mean tau of true-neighbour half, mean tau of far-lookalike half,
    //             mean cosine of each half) — averaged over probes.
    let rows: Vec<(f32, f32, f32, f32)> = probes
        .par_iter()
        .map(|&p| {
            let pv = &corpus[p * d..(p + 1) * d];
            let cp = &codes[p];
            // M code-nearest docs by Hamming
            let mut by_h: Vec<(u32, usize)> = (0..n)
                .filter(|&j| j != p)
                .map(|j| (hamming(cp, &codes[j]), j))
                .collect();
            by_h.sort_by_key(|x| x.0);
            by_h.truncate(m);
            // among those lookalikes, rank by TRUE cosine to the probe
            let mut by_cos: Vec<(f32, usize)> = by_h
                .iter()
                .map(|&(_, j)| (cos(pv, &corpus[j * d..(j + 1) * d]), j))
                .collect();
            by_cos.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let half = by_cos.len() / 2;
            let coords = top_coords(pv, cfg.topk);
            let tau_of = |slice: &[(f32, usize)]| -> (f32, f32) {
                let mut st = 0.0f32;
                let mut sc = 0.0f32;
                for &(c, j) in slice {
                    st += kendall_tau(pv, &corpus[j * d..(j + 1) * d], &coords);
                    sc += c;
                }
                let k = slice.len().max(1) as f32;
                (st / k, sc / k)
            };
            let (tau_near, cos_near) = tau_of(&by_cos[..half]);
            let (tau_far, cos_far) = tau_of(&by_cos[half..]);
            (tau_near, tau_far, cos_near, cos_far)
        })
        .collect();

    let mean = |f: &dyn Fn(&(f32, f32, f32, f32)) -> f32| {
        rows.iter().map(f).sum::<f32>() / rows.len().max(1) as f32
    };
    let tau_near = mean(&|r| r.0);
    let tau_far = mean(&|r| r.1);
    let cos_near = mean(&|r| r.2);
    let cos_far = mean(&|r| r.3);
    let wins = rows.iter().filter(|r| r.0 < r.1).count();
    println!("\n## Intra-code Kendall-tau test (top-{} coords, M={m} b2-lookalikes/probe, {} probes)",
        cfg.topk, rows.len());
    println!("among b2-lookalikes:   mean cosine   mean top-k tau-distance");
    println!("  FP32-TRUE neighbours {cos_near:.4}        {tau_near:.4}");
    println!("  FP32-FAR lookalikes  {cos_far:.4}        {tau_far:.4}");
    println!("probes where true-neighbour tau < far-lookalike tau: {wins}/{} = {:.3}",
        rows.len(), wins as f64 / rows.len().max(1) as f64);
    let verdict = if tau_near + 0.005 < tau_far {
        "SIGNAL: fine permutation order separates true neighbours from b2-lookalikes -> recoverable, no new storage"
    } else if tau_near > tau_far + 0.005 {
        "INVERTED: true neighbours have HIGHER tau — order does not help here"
    } else {
        "NO SIGNAL: intra-code order does not separate true neighbours from b2-lookalikes"
    };
    println!("verdict: {verdict}");
}

/// Low-rank clustered corpus; `noise` is the density dial.
fn make_corpus(cfg: &Cfg) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let (d, l) = (cfg.dim, cfg.latent);
    let mut a = vec![0.0f32; d * l];
    for x in a.iter_mut() {
        *x = gauss(&mut rng);
    }
    let mut protos = vec![0.0f32; cfg.clusters * l];
    for x in protos.iter_mut() {
        *x = gauss(&mut rng);
    }
    let mut corpus = vec![0.0f32; cfg.n * d];
    for i in 0..cfg.n {
        let c = rng.random_range(0..cfg.clusters);
        let mut z = vec![0.0f32; l];
        for j in 0..l {
            z[j] = protos[c * l + j] + cfg.noise * gauss(&mut rng);
        }
        let row = &mut corpus[i * d..(i + 1) * d];
        for ii in 0..d {
            let mut acc = 0.0f32;
            for j in 0..l {
                acc += a[ii * l + j] * z[j];
            }
            row[ii] = acc;
        }
        let nrm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if nrm > 0.0 {
            for v in row.iter_mut() {
                *v /= nrm;
            }
        }
    }
    corpus
}
