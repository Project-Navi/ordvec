//! Matched-bytes bake-off: does a Kendall-tau rerank of b=2 survivors beat
//! simply spending the bits on b=4, on R@10 vs FP32 cosine?
//!
//! This is the decisive deployment question for the density-collapse finding.
//! It is a CEILING experiment: the tau-rerank uses the FULL stored rank order
//! (ignoring how compactly that order could be stored). If the idealized
//! tau-rerank cannot beat b=4, the answer is "just use b=4" and no codec work
//! is justified. If it can, a compact tau codec becomes worth designing.
//!
//! Arms (all scored against FP32 brute-force cosine top-k):
//!   b2            — RankQuant b=2 asymmetric          (dim/4 bytes/vec)
//!   b4            — RankQuant b=4 asymmetric          (dim/2 bytes/vec)
//!   b2+tau        — b=2 top-M candidates, reranked by Kendall-tau of the
//!                   query's top-k coord order vs each doc's stored rank order
//!   b2+fp32       — b=2 top-M candidates, reranked by exact FP32 cosine
//!                   (absolute ceiling / sanity — should approach FP32)
//!
//! Run: cargo run --release --example tau_rerank_bakeoff
//!      cargo run --release --example tau_rerank_bakeoff -- --corpus-npy emb.npy --queries-npy q.npy
//! No external data, no BLAS.

use ordvec::rank::rank_transform;
use ordvec::RankQuant;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

const SEED: u64 = 1;

struct Cfg {
    dim: usize,
    n: usize,
    n_queries: usize,
    latent: usize,
    clusters: usize,
    k: usize,
    m: usize,    // candidate-set size for the rerank arms
    topk: usize, // # query top coords used by the tau rerank
    corpus_npy: Option<String>,
    queries_npy: Option<String>,
}

fn parse() -> Cfg {
    let mut c = Cfg {
        dim: 256,
        n: 30_000,
        n_queries: 200,
        latent: 64,
        clusters: 200,
        k: 10,
        m: 200,
        topk: 32,
        corpus_npy: None,
        queries_npy: None,
    };
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        match x.as_str() {
            "--dim" => c.dim = a.next().unwrap().parse().unwrap(),
            "--n" => c.n = a.next().unwrap().parse().unwrap(),
            "--queries" => c.n_queries = a.next().unwrap().parse().unwrap(),
            "--latent" => c.latent = a.next().unwrap().parse().unwrap(),
            "--clusters" => c.clusters = a.next().unwrap().parse().unwrap(),
            "--k" => c.k = a.next().unwrap().parse().unwrap(),
            "--m" => c.m = a.next().unwrap().parse().unwrap(),
            "--topk" => c.topk = a.next().unwrap().parse().unwrap(),
            "--corpus-npy" => c.corpus_npy = Some(a.next().unwrap()),
            "--queries-npy" => c.queries_npy = Some(a.next().unwrap()),
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

fn l2_rows(v: &mut [f32], dim: usize) {
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
    assert!(header.contains("'descr': '<f4'"), "expected <f4");
    assert!(header.contains("'fortran_order': False"), "expected C order");
    let after = &header[header.find("'shape':").expect("shape")..];
    let inner = &after[after.find('(').unwrap() + 1..after.find(')').unwrap()];
    let dims: Vec<usize> = inner.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    assert_eq!(dims.len(), 2, "expected 2-D");
    let (n, dim) = (dims[0], dims[1]);
    let dstart = hstart + hlen;
    assert_eq!(bytes.len() - dstart, n * dim * 4, "len mismatch");
    let mut out = vec![0.0f32; n * dim];
    for (i, ch) in bytes[dstart..].chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
    }
    (out, n, dim)
}

fn main() {
    let mut cfg = parse();
    let (corpus, queries) = load_corpus(&mut cfg);
    let truth = fp32_topk(&corpus, &queries, cfg.dim, cfg.k);
    println!(
        "# tau_rerank_bakeoff: n={} q={} dim={} k={} M={} topk={}",
        cfg.n, cfg.n_queries, cfg.dim, cfg.k, cfg.m, cfg.topk
    );
    run(&cfg, &corpus, &queries, &truth);
}

fn recall_at_k(pred: &[usize], truth: &[usize]) -> f32 {
    use std::collections::HashSet;
    let t: HashSet<usize> = truth.iter().copied().collect();
    let hits = pred.iter().filter(|i| t.contains(i)).count();
    hits as f32 / truth.len().max(1) as f32
}

/// Top-k indices of a vector by value, descending.
fn top_coords(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| v[j].partial_cmp(&v[i]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    idx
}

/// Kendall-tau distance between the orderings q and the doc's stored RANKS
/// induce on `coords` (fraction of discordant pairs). Lower = more similar
/// order. `dranks[c]` is the stored rank of coord c for this doc.
fn tau_dist(q: &[f32], dranks: &[u16], coords: &[usize]) -> f32 {
    let m = coords.len();
    if m < 2 {
        return 0.0;
    }
    let (mut disc, mut tot) = (0usize, 0usize);
    for x in 0..m {
        for y in (x + 1)..m {
            let (cx, cy) = (coords[x], coords[y]);
            let sq = (q[cx] - q[cy]).partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal);
            // stored ranks are integers; higher rank = larger value
            let sd = dranks[cx].cmp(&dranks[cy]);
            if sq != sd {
                disc += 1;
            }
            tot += 1;
        }
    }
    disc as f32 / tot.max(1) as f32
}

fn run(cfg: &Cfg, corpus: &[f32], queries: &[f32], truth: &[Vec<usize>]) {
    let d = cfg.dim;
    let nq = cfg.n_queries;

    // Build b=2 and b=4 indices on identical data.
    let mut b2 = RankQuant::new(d, 2);
    b2.add(corpus);
    let mut b4 = RankQuant::new(d, 4);
    b4.add(corpus);
    eprintln!(
        "# b2 {} B/vec, b4 {} B/vec",
        b2.bytes_per_vec(),
        b4.bytes_per_vec()
    );

    // Precompute stored ranks per doc (the CEILING tau signal: full order).
    // This is what a compact tau codec would have to approximate.
    let doc_ranks: Vec<Vec<u16>> = (0..cfg.n)
        .into_par_iter()
        .map(|i| rank_transform(&corpus[i * d..(i + 1) * d]))
        .collect();

    // Per query: candidate set = b=2's own asymmetric top-M (the cheap stage).
    let b2_res = b2.search_asymmetric(queries, cfg.m);
    let b4_res = b4.search_asymmetric(queries, cfg.k);

    let mut r_b2 = 0.0f64;
    let mut r_b4 = 0.0f64;
    let mut r_tau = 0.0f64;
    let mut r_fp = 0.0f64;

    let per_q: Vec<(f32, f32, f32, f32)> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let q = &queries[qi * d..(qi + 1) * d];
            // b2 top-k (first k of its top-M)
            let b2_ids: Vec<usize> =
                b2_res.indices_for_query(qi).iter().take(cfg.k).map(|&i| i as usize).collect();
            let b4_ids: Vec<usize> =
                b4_res.indices_for_query(qi).iter().map(|&i| i as usize).collect();
            // candidate pool = b2 top-M
            let cands: Vec<usize> =
                b2_res.indices_for_query(qi).iter().map(|&i| i as usize).collect();
            let coords = top_coords(q, cfg.topk);
            // tau rerank: ascending tau distance
            let mut by_tau: Vec<(usize, f32)> = cands
                .iter()
                .map(|&di| (di, tau_dist(q, &doc_ranks[di], &coords)))
                .collect();
            by_tau.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let tau_ids: Vec<usize> = by_tau.iter().take(cfg.k).map(|&(i, _)| i).collect();
            // fp32 rerank: descending cosine (absolute ceiling)
            let mut by_fp: Vec<(usize, f32)> = cands
                .iter()
                .map(|&di| {
                    let dv = &corpus[di * d..(di + 1) * d];
                    (di, q.iter().zip(dv).map(|(a, b)| a * b).sum())
                })
                .collect();
            by_fp.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let fp_ids: Vec<usize> = by_fp.iter().take(cfg.k).map(|&(i, _)| i).collect();
            (
                recall_at_k(&b2_ids, &truth[qi]),
                recall_at_k(&b4_ids, &truth[qi]),
                recall_at_k(&tau_ids, &truth[qi]),
                recall_at_k(&fp_ids, &truth[qi]),
            )
        })
        .collect();
    for (a, b, c, e) in &per_q {
        r_b2 += *a as f64;
        r_b4 += *b as f64;
        r_tau += *c as f64;
        r_fp += *e as f64;
    }
    let nqf = nq as f64;
    println!("\narm              bytes/vec   R@{}", cfg.k);
    println!("b2 asym          {:>9}   {:.4}", b2.bytes_per_vec(), r_b2 / nqf);
    println!("b4 asym          {:>9}   {:.4}", b4.bytes_per_vec(), r_b4 / nqf);
    println!(
        "b2 + tau-rerank  {:>9}*  {:.4}   (*ceiling: full stored ranks)",
        b2.bytes_per_vec(),
        r_tau / nqf
    );
    println!("b2 + fp32-rerank {:>9}   {:.4}   (absolute ceiling)", "—", r_fp / nqf);
    println!("\nVERDICT:");
    let (tau, b4v) = (r_tau / nqf, r_b4 / nqf);
    if tau > b4v + 0.005 {
        println!("  tau-rerank BEATS b4 at half the bytes ({tau:.4} > {b4v:.4}) — codec work justified");
    } else if tau + 0.005 < b4v {
        println!("  b4 BEATS idealized tau-rerank ({b4v:.4} > {tau:.4}) — just use b4, no codec");
    } else {
        println!("  tau-rerank ~= b4 ({tau:.4} vs {b4v:.4}) — tie even at the ceiling; b4 simpler");
    }
}

/// FP32 brute-force cosine top-k (corpus + queries L2-normalized). Vec per query.
fn fp32_topk(corpus: &[f32], queries: &[f32], dim: usize, k: usize) -> Vec<Vec<usize>> {
    let n = corpus.len() / dim;
    (0..queries.len() / dim)
        .into_par_iter()
        .map(|qi| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let mut s: Vec<(usize, f32)> = (0..n)
                .map(|di| {
                    let d = &corpus[di * dim..(di + 1) * dim];
                    (di, q.iter().zip(d).map(|(a, b)| a * b).sum())
                })
                .collect();
            s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            s.into_iter().take(k).map(|(i, _)| i).collect()
        })
        .collect()
}

fn load_corpus(cfg: &mut Cfg) -> (Vec<f32>, Vec<f32>) {
    if let (Some(cp), Some(qp)) = (cfg.corpus_npy.clone(), cfg.queries_npy.clone()) {
        let (mut c, n, d) = load_npy_f32(&cp);
        let (mut q, nq, dq) = load_npy_f32(&qp);
        assert_eq!(d, dq, "dim mismatch");
        l2_rows(&mut c, d);
        l2_rows(&mut q, dq);
        cfg.n = n;
        cfg.dim = d;
        cfg.n_queries = nq;
        eprintln!("# loaded corpus {n}x{d}, queries {nq}x{dq}");
        return (c, q);
    }
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
    let mk = |proto: &[f32], noise: f32, rng: &mut ChaCha8Rng| -> Vec<f32> {
        let mut z = vec![0.0f32; l];
        for j in 0..l {
            z[j] = proto[j] + noise * gauss(rng);
        }
        let mut o = vec![0.0f32; d];
        for i in 0..d {
            let mut acc = 0.0f32;
            for j in 0..l {
                acc += a[i * l + j] * z[j];
            }
            o[i] = acc;
        }
        o
    };
    let mut corpus = vec![0.0f32; cfg.n * d];
    for i in 0..cfg.n {
        let c = rng.random_range(0..cfg.clusters);
        corpus[i * d..(i + 1) * d].copy_from_slice(&mk(&protos[c * l..(c + 1) * l], 0.3, &mut rng));
    }
    let mut queries = vec![0.0f32; cfg.n_queries * d];
    for i in 0..cfg.n_queries {
        let c = rng.random_range(0..cfg.clusters);
        queries[i * d..(i + 1) * d]
            .copy_from_slice(&mk(&protos[c * l..(c + 1) * l], 0.1, &mut rng));
    }
    l2_rows(&mut corpus, d);
    l2_rows(&mut queries, d);
    (corpus, queries)
}
